//
// Created by ruishengcai on 2023/12/18.
//

#include <stdio.h>
#include <iostream>
// #define NDEBUG
#include <assert.h>
#include "yolov8_seg.h"
#include <utility>
#include <unordered_map>
#include "cuda_runtime_api.h"
#include "../estimator/parameters.h"

// 将原始图像中的边界框角点坐标变换到 缩放裁减之后的坐标
// 这是在yolo_seg_decode.cu文件中直接定义的函数，且没有相应的.cuh文件可以加载，因此这里需要声明此函数以便可以（链接）使用
void affine_project(float* matrix, float x, float y, float* ox, float* oy);

namespace YoloV8{
    
    using namespace std;
    using namespace cv;
    using namespace tensorrt_common;
    using namespace tensorrt_log;
    using namespace tensorrt_buffer;

    // for debug
    bool succ = true;

    // 对output1进行解码，其内部会调用相关CUDA核函数
    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    // 仅填充预处理的版本
    // void decode_kernel_invoker_pure_padding(
    //     float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    //     int* invert_affine_matrix, float* parray,
    //     int max_objects, cudaStream_t stream
    // );

    void sorted_bbox_for_NMS(float* bbox_ptr, float* bbox_num_ptr, int nitems, cudaStream_t stream);

    // 对初步解码出的bbpx进行极大值抑制，其内部会调用相关CUDA核函数
    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

    // output2的解码。上面这些函数在cpu中被调用，但是其中包含在gpu上执行的操作
    void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                            int mask_width, int mask_height, unsigned char *mask_out,
                            int mask_dim, int out_width, int out_height, cudaStream_t stream);

    void build_full_seg_mask(float left, float top, unsigned char *full_seg_map_device, int full_seg_map_width, 
                            int full_seg_map_height, unsigned char *small_seg_map_device,
                            int small_seg_map_width, int small_seg_map_height, int full_map_bbox_width, 
                            int full_map_bbox_height, unsigned char cls_label, unsigned char obj_id, unsigned char is_valid_solid_obj, cudaStream_t stream);

    void sample_pixel_for_objs(int num_objs, int num_sampled_pixel, float* bbox_info_device,
                                float mMaxDepthObj, float mMinDepthObj, float mbf, 
                                unsigned char* seg_map, int full_seg_map_width, int seg_map_height, 
                                float* depth_map, float* buffer_device_pixel, float fx, float fy, 
                                float cx, float cy,  float left_shift, float top_shift, int W_dep_map,
                                int H_dep_map, float left_shift_dep, float top_shift_dep, cudaStream_t stream);
    
    // 分割掩码对象,构造函数
    InstanceSegmentMap::InstanceSegmentMap(int width_, int height_){
        this->width  = width_;
        this->height = height_;
        // 在cpu上分配规定大小的页锁内存，其中width和height是预处理后的图像缩放到mask图大小之后的对应的预测边界框的尺寸
        // mask中的数据为unsigned char类型。此为非分页内存。seg map的数据类型为unsigned char
        checkCudaRuntime(cudaMallocHost((void**)&this->data, width * height));
    }

    InstanceSegmentMap::~InstanceSegmentMap(){
        if(this->data){
            checkCudaRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width  = 0;
        this->height = 0;
    }

    void AffineMatrix::compute(const cv::Size& from, const cv::Size& to){
        int scale_x = (to.width - from.width)/32 + 1;  // should be 1
        int scale_y = (to.height - from.height)/32 + 1; // should be 1

        assert((to.width - from.width)>=0 && "dst_width must be no less than src_width");

        assert((scale_x == 1) && "dst_width - src_width must be less than 32");

        assert((to.height - from.height)>=0 && "dst_height must be no less than src_height");

        assert((scale_y == 1) && "dst_height - src_height must be less than 32");

        int shift_x_left = (to.width - from.width) / 2;

        // 设置Yolov8的预处理方式与rapidflow的一样，可以减少总的预处理时间和内存消耗
        // int shift_y_top  = (to.height - from.height)/ 2;
        int shift_y_top = 0;

        // 最后一列进行四舍五入
        i2d[0] = (float)scale_x;  i2d[1] = 0.0;  i2d[2] = (float)shift_x_left;
        i2d[3] = 0.0;  i2d[4] = (float)scale_y;  i2d[5] = (float)shift_y_top;
        // 最后两个元素是为了凑32字节内存，无意义
        i2d[6] = 0.0;  i2d[7] = 0.0;

        d2i[0] = (float)scale_x;  d2i[1] = 0.0;  d2i[2] = -1.0 * i2d[2];
        d2i[3] = 0.0;  d2i[4] = (float)scale_y;  d2i[5] = -1.0 * i2d[5];
        // 最后两个元素是为了凑32字节内存，无意义
        d2i[6] = 0.0;  d2i[7] = 0.0;
    }

    Mat AffineMatrix::i2d_mat(){
        return Mat(2, 3, CV_32F, i2d);
    }
    
    static float iou(const Box& a, const Box& b){
        float cleft 	= max(a.left, b.left);
        float ctop 		= max(a.top, b.top);
        float cright 	= min(a.right, b.right);
        float cbottom 	= min(a.bottom, b.bottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
        float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
        return c_area / (a_area + b_area - c_area);
    }

    // YolosegConfig类需要定义，在read_configs.h中
    // Yoloseg::Yoloseg(const YolosegConfig &yolov8_seg_config, const cudaStream_t stream_common)
    //     : yolov8_seg_config_(yolov8_seg_config), engine_(nullptr){

    //     use_multi_stream_ = yolov8_seg_config_.use_multi_stream;

    //     setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    //     normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
    //     confidence_threshold_ = yolov8_seg_config_.confidence_threshold;
    //     nms_threshold_        = yolov8_seg_config_.nms_threshold;
    //     max_objects_          = yolov8_seg_config_.max_objects;
    //     max_pre_nms_          = yolov8_seg_config_.max_pre_max;

    //     assert(stream_common != nullptr && "common stream for infer should be provided!");

    //     stream_ = stream_common;
    // }

    // YolosegConfig类需要定义，在read_configs.h中
    Yoloseg::Yoloseg(const YolosegConfig &yolov8_seg_config, const cudaStream_t stream_common, 
                    const cudaStream_t stream_cpy_post)
        : yolov8_seg_config_(yolov8_seg_config), engine_(nullptr){

        use_multi_stream_ = yolov8_seg_config_.use_multi_stream;

        input_width_  = yolov8_seg_config_.inputW;
        input_height_ = yolov8_seg_config_.inputH;
        size_full_seg_map = input_width_ * input_height_ * sizeof(unsigned char);

        setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
        normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        confidence_threshold_ = yolov8_seg_config_.confidence_threshold;
        nms_threshold_        = yolov8_seg_config_.nms_threshold;
        max_objects_          = yolov8_seg_config_.max_objects;
        max_pre_nms_          = yolov8_seg_config_.max_pre_max;

        assert(stream_common != nullptr && "infer stream for yolov8seg should be provided!");
        
        stream_ = stream_common;

        if (use_multi_stream_ && stream_cpy_post){
            stream_cpy_post_   = stream_cpy_post;
            common_stream_cpy_ = true;
        }
    }

    Yoloseg::~Yoloseg(){
        // 等待gpu上与此stream_流相关的所有操作执行完。主要是buffers对象很可能还被使用
        // 这部分同步阻塞可以放在调用三个推理模型的host线程中
        //GPU_CHECK(cudaStreamSynchronize(stream_cpy_post_));
        //GPU_CHECK(cudaStreamSynchronize(stream_));

        if(raw_input_device != nullptr)
        {
            cudaFree(raw_input_device);
            raw_input_device = nullptr;
        }

        if(output_array_device != nullptr)
        {
            cudaFree(output_array_device);
            output_array_device = nullptr;
        }
        
        if(output_array_host != nullptr)
        {
            cudaFreeHost(output_array_host);
            output_array_host = nullptr;
        }

        if(full_seg_map_host != nullptr)
        {
            cudaFreeHost(full_seg_map_host);
            full_seg_map_host = nullptr;
        }

        if (small_seg_map_device != nullptr){
            cudaFree(small_seg_map_device);
            small_seg_map_device = nullptr;
        }
        if (small_seg_map_host != nullptr){
            cudaFreeHost(small_seg_map_host);
            small_seg_map_host = nullptr;
        }

        if (num_pre_deco_host != nullptr){
            cudaFreeHost(num_pre_deco_host);
            num_pre_deco_host = nullptr;
        }
        
        raw_image_device = nullptr;
        full_seg_map_device = nullptr;
        // full_seg_map_host   = nullptr;

        engine_.reset();

        context_.reset();

        buffers.reset();

        if (!common_stream_cpy_ && use_multi_stream_){
            cudaStreamDestroy(stream_cpy_post_);
        }
        else{
            stream_cpy_post_ = nullptr;
        }

        // 这里应该需要明确一下，Yolov8的infer stream是从外部传入的，所以这里只需要赋为空
        stream_ = nullptr;

        if(buffer_device_bbox_info != nullptr)
        {
            cudaFree(buffer_device_bbox_info);
            buffer_device_bbox_info = nullptr;
        }
    }

    bool Yoloseg::build() 
    {
        
        if (use_multi_stream_ && stream_cpy_post_ == nullptr){
            checkCudaRuntime(cudaStreamCreate(&stream_cpy_post_));
		    if(stream_cpy_post_ == nullptr) return false;
        }
        
        if(deserialize_engine())
        {
            printf("deserialize engine of YOLOv8 succeeded!\n");
            input_dims_   = engine_->getBindingDimensions(engine_->getBindingIndex(yolov8_seg_config_.input_tensor_names[0].c_str()));
            ASSERT(input_dims_.nbDims == 4);
            output1_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(yolov8_seg_config_.output_tensor_names[0].c_str()));
            ASSERT(output1_dims_.nbDims == 3);
            output2_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(yolov8_seg_config_.output_tensor_names[1].c_str()));
            ASSERT(output2_dims_.nbDims == 4);

            seg_map_width   = output2_dims_.d[3];
            seg_map_height  = output2_dims_.d[2];
            seg_map_scale_x = seg_map_width / (float)input_width_;
            seg_map_scale_y = seg_map_height / (float)input_height_;
            size_small_seg_map = seg_map_width * seg_map_height * sizeof(unsigned char);
            return true;
        }

        auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if (!builder) {
            return false;
        }
        // 如果想要选择显式批处理模式（这是使用onnx parser解析和编译engine时必须的模式），那么就必须使用kEXPLICIT_BATCH)这一flag来声明。隐式模式用0U来代替1U
        // 隐式或者显式的批处理模式的选择必须在创建INetworkDefinition对象时指出 
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
        const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) {
            return false;
        }
        auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }
        auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
                nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
        if (!parser) {
            return false;
        }
        
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            return false;
        }

        // 由于导出的Yolov8的onnx模型的宽高是动态的，因此这里需要设置范围，这范围是随便给定还是需要按照实际？
        // 默认此TensorRT仅用于KITTI数据集，并且模型的输入统一为1248*384
        // 可以尝试使用640*224的网络输入，但是相对应的仿射矩阵计算需要重新编写
        profile->setDimensions(yolov8_seg_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMIN, Dims4(1,  3, input_height_, input_width_));
        profile->setDimensions(yolov8_seg_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kOPT, Dims4(1,  3, input_height_, input_width_));
        profile->setDimensions(yolov8_seg_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMAX, Dims4(1,  3, input_height_, input_width_));
        config->addOptimizationProfile(profile);

        auto constructed = construct_network(builder, network, config, parser);
        if (!constructed) {
            return false;
        }
        auto profile_stream = makeCudaStream();
        if (!profile_stream) {
            return false;
        }
        config->setProfileStream(*profile_stream);
        TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            return false;
        }
        TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
        if (!runtime) {
            return false;
        }
        engine_ = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine_) {
            return false;
        }

        // yolov-seg的输入变量为1，输出变量为2（一个是检测输出1*8400*116，另一个是掩膜1*32*160*160，其中第一个输出为了内存连续性而进行了后两维的交换）
        assert(engine_->getNbIOTensors() == 3);

        save_engine();
        ASSERT(network->getNbInputs() == 1);
        input_dims_ = network->getInput(0)->getDimensions();
        ASSERT(input_dims_.nbDims == 4);
        ASSERT(network->getNbOutputs() == 2);
        output1_dims_ = network->getOutput(0)->getDimensions();
        ASSERT(output1_dims_.nbDims == 3);
        output2_dims_ = network->getOutput(1)->getDimensions();
        ASSERT(output2_dims_.nbDims == 4);

        seg_map_width   = output2_dims_.d[3];
        seg_map_height  = output2_dims_.d[2];
        seg_map_scale_x = seg_map_width / (float)input_width_;
        seg_map_scale_y = seg_map_height / (float)input_height_;
        size_small_seg_map = seg_map_width * seg_map_height * sizeof(unsigned char);
        
        return true;
    }

    bool Yoloseg::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                    TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                    TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                    TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
        auto parsed = parser->parseFromFile(yolov8_seg_config_.onnx_file.c_str(),
                                            static_cast<int>(gLogger.getReportableSeverity()));
        if (!parsed) {
            return false;
        }
        // 部署时不需要加载gpu版pytorch，这减少了很多内存占用
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1.5*1073741824);
        config->setFlag(BuilderFlag::kFP16);
        // dla_core在config文件中的给定值是-1，意味这不启用DLA？只有Neston设备的显卡有DLA core
        // enableDLA(builder.get(), config.get(), yolov8_seg_config_.dla_core);
        return true;
    }

    //! 推理的输出应该是boxarray类型，需要自己定义
    //! image应该是原始图像，在infer函数中先后进行预处理、推理和后处理全部操作
    bool Yoloseg::preproc_infer(const Mat &image, cudaEvent_t stop_infer_seg) {
        
        // 网络只有一个输入变量
        ASSERT(yolov8_seg_config_.input_tensor_names.size() == 1);
        if (!preprocess_input(image)) {
            cout << "Preprocessing in YOLOV8 failed!" <<endl;
            return false;
        }

        // cout << "mask map width: " << output2_dims_.d[3];
        // cout << "mask map height: " << output2_dims_.d[2];
        
        // 调用enqueueV3在stream上默认进行异步推理（即不阻塞CPU上后续操作），多个任务加入到stream队列中等待依次执行
        //！ 从TensorRT8.5开始enqueueV2()开始被弃用了！
        // 这里不让gpu阻塞cpu，是因为此推理线程为cpu上的子线程。executeV2函数则默认是同步执行，即该stream执行任务时会阻塞host线程
        // bool型返回变量只是用于指示是否成功将此任务加入指定stream的任务队列中，只要加入成功就立即返回，不会在这里等待直到推理任务完成
        bool status = context_->enqueueV3(stream_);
        
        // bool status = context_->executeV2(buffers.getDeviceBindings().data());
        // bool status = context_->enqueueV2(1, buffers->getDeviceBindings().data(), stream_, nullptr);

        if (!status) {
            cout << "Failed starting preferring of YOLOV8!" <<endl;
            return false;
        }
        // cout << "Succeed starting preferring of YOLOV8!" <<endl;
        if (stop_infer_seg != nullptr)
            checkCudaRuntime(cudaEventRecord(stop_infer_seg, stream_));
        return true;
    }

    // buffers用于管理与某一个模型推理时需要的内存，包括host上的页锁内存 和 device上的内存！image是原图像
    bool Yoloseg::preprocess_input(const Mat &image)
    {
        
        // cout << "raw image width " << image.cols << " height " << image.rows << " raw image channels " << image.channels() << endl;

        if(image.empty()){
                INFOE("Image is empty");
                return false;
            }
        
        if (image.channels() != 3){
            INFOE("The raw image should have 3 channels");
                return false;
        }
        
        if (!context_) {
            context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!context_) {
                return false;
            }
            // 这里应该给定的是预处理之后的图像的尺寸，即宽高都应该是32的整数倍！这里是原onnx模型采用了动态shape，需要指定动态shape的维度
            bool status = context_->setInputShape(yolov8_seg_config_.input_tensor_names[0].c_str(), Dims4(1, 3, input_height_, input_width_));
            assert(status && "Set InputShape failed!");
        }

        raw_row = image.rows;
        raw_col = image.cols;

        // Size类型的默认顺序为（width_size, height_size)
        Size size_src = image.size();
        Size size_dst(input_width_, input_height_);

        aff_mat.compute(size_src, size_dst);

        left_pad = aff_mat.i2d[2];
        top_pad  = aff_mat.i2d[5];

        // 注意，这里默认图像的每个通道应该是uint8数据类型，即只有1字节！
        size_t size_image      = image.cols * image.rows * 3;
        // 这里为仿射矩阵的6个int32或者float数分配了32个字节的内存，因为这样方便内存读取！
        size_t size_matrix     = iLogger::upbound(sizeof(aff_mat.d2i), 32);
        
        // 查看还有多少显存
        // size_t gpu_total_size, gpu_free_size;
        // cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size,&gpu_total_size);
        // if(cuda_status != cudaSuccess) cout << "Error: cudaMenGetInfo fails!" << endl;
        // cout << "剩余显存：" << double(gpu_free_size)/(1024.0*1024.0) << endl;

        // 每个binging变量应该都是按照fp32的大小来准备内存的，类型转换是自动发生在engine内部
        // 如果只用一个内存区域反复接收预处理之后的数据，且网络的输入的尺度（B、H 和 W）是允许变化的，那么应该重新分配此内存
        // 再加入8*sizeof(float)的内存大小，用来存储仿射矩阵
        //!! 每进行一次推理就要重新生成一次内存，这是没有必要的！将此buffers作为yoloseg类的成员，只有在系统结束运行时才析构它！
        // 由于有多个模型共用一张输入原始图像，因此原始图像从host到gpu的传输应该只进行一次，且应该是在模型推理的cpu子线程中，被多个模型所共用！
        if (buffers == nullptr)
        {
            // 采用 2 + MAX_IMAGE_BBOX结构是，before_NMS_counter + after_NMS_counter + bboxes
            // +6的目的是保证 (2 + 6) * sizeof(float) % 32 == 0 
            // before_NMS_counter_这个值理论上最大值为网络推理输出的预测bbox的个数，但这意味着每个预测框都是有效的（置信度够大），几乎不可能！
            // 因为后处理的数据类型应该都是float型的，counter的数据类型应该是int_32类型的，所以这里凑(2+6)*sizeof(int)%32 ==0就行
            int predicted_bbox = input_width_ / 32 * input_height_/ 32 + input_width_ / 16 * input_height_/ 16 + input_width_ / 8 * input_height_/ 8;
            // 不清楚初步解码得到的bbox个数会是多少个，所以直接分配和预测结果个数相等的内存
            int max_decoded_bbox = predicted_bbox;

            int bytesize_output_array = (2 + 6 + max_decoded_bbox * NUM_BOX_ELEMENT_) *sizeof(float);
            // 原始图像的通道值是uInt8型的。另外为了应对原始图像的尺寸可能会有小幅波动，使用规定的预处理之后的尺寸来分配内存。而且这个内存后面还能用来存放full_seg_map！！
            // int bytesize_raw_device   = input_width_ * input_height_ * 3 + size_matrix;
            int bytesize_raw_device   = input_width_ * input_height_ * 5 + size_matrix;
            // cout << "Byte size of memory for raw data image is " << bytesize_raw_device << endl;
            int Size_raw_post_var[2]  = {bytesize_raw_device, bytesize_output_array};

            // 1 raw input, 1 postprocessed output
            // buffers = make_shared<BufferManager>(engine_, 0, context_.get(), 1, 1, Size_raw_post_var);
            buffers = make_shared<BufferManager>(engine_, 0, context_.get(), 0, 0, Size_raw_post_var);

            for (int32_t i = 0; i < engine_->getNbIOTensors(); i++){
                auto const name = engine_->getIOTensorName(i);
                cout << name << endl;
                bool bind_ = context_->setTensorAddress(name, buffers->getBindingDeviceBuffer(name));
                if(!bind_) 
                    cout << "Failed bind buffer to binding name!" << endl;
            }

            // 不知道为什么在BufferManager中为raw data 和 post-process data分配的host和device memory在这里无法使用，因此只能在这里单独分配
            CHECK(cudaMalloc((void**)&raw_input_device, bytesize_raw_device));
            // if(raw_input_device == NULL) 
            //     cout<< "Failed to alloc device memory for raw data!" << endl;

            CHECK(cudaMalloc((void**)&output_array_device, bytesize_output_array));
            // if(output_array_device == NULL) 
            //     cout<< "Failed to alloc device memory for post-process data!" << endl;
            
            // void* ptr_out = malloc(bytesize_output_array);
            // assert(ptr_out != NULL && "Failed to malloc for full_seg_map_host!");
            // output_array_host = (float*)ptr_out;

            CHECK(cudaMallocHost((void**)&output_array_host, bytesize_output_array));
            assert(output_array_host != NULL && "Failed to malloc for full_seg_map_host!");

            // void* ptr_in = malloc(input_height_*input_width_*3);
            // assert(ptr_in != NULL && "Failed to malloc for full_seg_map_host!");
            // full_seg_map_host = (uchar*)ptr_in;

            // 这里多分配3个通道，通道1和2分别是cls map和id map，通道3用于保存各个像素点解码出来的置信度，通道4用于保存mask_for_bg，通道5用于保存mask_for_obj
            CHECK(cudaMallocHost((void**)&full_seg_map_host, input_height_*input_width_*5));
            assert(full_seg_map_host != NULL && "Failed to malloc for full_seg_map_host!");
        }
        // cout << "Succeeded create buffer for YOLOV8!" << endl;
        // 构造AutoDevice类对象时就会切换当前使用的device到指定的device（gpu_默认初始值是0）
        // 当此对象被析构时（应该是此预处理函数执行结束时），会自动切换回原先的device
        // CUDATools::AutoDevice auto_device(gpu_);

        // if(image.data == NULL) cout<< "Image ptr is NULL!" << endl;
        // 很奇怪，下面这些得到的ptr总是为nullptr！
        // 其实从CPU往device复制raw data时，只要能保证该host内存上所存数据在传输期间不会被修改，则可以不用是cudaMallocHost所分配的不可分页内存！
        // 而cv::Mat的内存是new得到的堆内存，在此期间不会改变。因此这里可以无需多一个从image内存复制到host内存的过程，直接将image复制到device上即可！
        // uchar* input_host_buffer = (uchar*)buffers->getVarHostBuffer(0);
        // if (input_host_buffer == NULL) cout << "input_host_buffer is nullptr!" << endl;
        // float* affine_matrix_host = (float*)input_host_buffer;
        // uchar* raw_image_host = input_host_buffer + size_matrix;
        // memcpy(affine_matrix_host, aff_mat.d2i, sizeof(aff_mat.d2i));
        // cout << "Succeeded copy affinx matrix data to host buffer!" << endl;
        // memcpy(raw_image_host, image.data, size_image, cudaMemcpyHostToHost);
        // cout << "Succeeded copy raw image to host buffer!" << endl;
        // 将affine matrix 和 raw data 从host复制到device
        // 此函数要给定需要复制的Bytes数，包括affine matrix 和 raw data（与其维度和数据类型相关）！此外，默认仿射矩阵的内存为32个bytes，但最后两个元素是无意义的！
        // buffers->copyInputToDeviceAsync(true, size_image + size_matrix, std::vector<std::string>(), stream_);
        
        // raw_input_device = (uchar*)buffers->getVarDeviceBuffer(0);
        // if(raw_input_device == NULL) cout << "input device buffer is NULL!" << endl;

        // float* output_host_buffer = buffers->getVarHostBuffer(1);
        // if (output_host_buffer == NULL) cout << "output_array_host is nullptr!" << endl;
        
        // output_array_device = (float*)buffers->getVarDeviceBuffer(1);
        // if (output_array_device == NULL) cout << "output_array_device is nullptr!" << endl;

        raw_image_device = (uint8_t*)raw_input_device + size_matrix;
        CHECK(cudaMemcpyAsync((void*)raw_input_device, (void*)aff_mat.d2i, sizeof(aff_mat.d2i), cudaMemcpyHostToDevice, stream_));
        CHECK(cudaMemcpyAsync(raw_image_device, image.data, size_image, cudaMemcpyHostToDevice, stream_));
        
        // cout << "Succeed copy iamge to device!" << endl;

        // 取用来存储预处理后的图像数据的gpu buffer，即inputbinding
        float* infer_device_buffer   = (float*)buffers->getBindingDeviceBuffer(yolov8_seg_config_.input_tensor_names[0]);
        // if(infer_device_buffer == NULL) cout << "Even the device memory for binding variable is NULL!!!" << endl;
        // 调用核函数进行预处理！此模型使用的预处理是仅进行原图像四周的填充！
        CUDAKernel::warp_pure_padding_bilinear_and_normalize_plane(
                raw_image_device,     image.cols * 3,    image.cols,    
                image.rows,     infer_device_buffer,     input_width_,
                input_height_, (float*)raw_input_device,  114, normalize_, stream_ );
        // cout << "Succeeded preprocess in YOLOV8!" << endl;
        return true;
    }

    // 尝试yolov8seg的后处理操作 与 rapidflow的推理 使用不同的stream进行并行操作，以节省整体耗时。
    // 不知道GPU资源（主要是指SMs而不是显存)够不够？ 有可能是不够的，可以比较一下计算时间
    bool Yoloseg::postprocess_output(map<int, YoloV8::Box> &masked_bboxes_seg_map, set<uchar> &solid_obj_cls, set<uchar> &deform_obj_cls, set<uchar> &small_solid_objs, cudaEvent_t stop_infer_seg, cudaEvent_t stop_post_proc)
    {
        solid_obj_cls.clear();
        deform_obj_cls.clear();
        small_solid_objs.clear();
        
        if (!masked_bboxes_seg_map.empty()) masked_bboxes_seg_map.clear();

        // 需要在seg的psot处理最开始等待seg的推理结束
        // 将事件阻塞放在后处理函数最前面的好处是允许在seg的infer期间先用stream_cpy进行flow和depth的与处理数据的传输！
        if(stop_infer_seg == nullptr) 
        {
            cout << "A cudaevent must be provided to make sure of the end of infer process!" << endl;
            exit(0);
        }
        // post_proc of yolov8seg in stream_cpy_ won't start until infer is finished
        checkCudaRuntime(cudaStreamWaitEvent(stream_cpy_post_, stop_infer_seg));
        
        // 因为是动态shape的输入，所以输出也是动态的，因此要获得具体的维度数值，必须从给定具体inputbinding维度的context中来获取各个binding的维度长度值！
        output1_dims_   = context_->getTensorShape(yolov8_seg_config_.output_tensor_names[0].c_str());// detection head 1x9828x43(对于自定义的KITTI分割数据集，只有7个类别)
        output2_dims_   = context_->getTensorShape(yolov8_seg_config_.output_tensor_names[1].c_str());// segment head 1x32x96x312 （原图384*1248，mask的大小为高宽各下采样4倍）
        int num_classes = output1_dims_.d[2] - 4 - output2_dims_.d[1];

        // 取device内存中用于保存网络输出的output1（即bbox head输出）的部分
        float* infer_output_bbox    = (float*)buffers->getBindingDeviceBuffer(yolov8_seg_config_.output_tensor_names[0]);
        // yolov8seg的第一個var是原始圖像，第二個var为后处理结果(弃用)
        // float* output_array_device  = (float*)buffers->getVarDeviceBuffer(1);
        // float* affine_matrix_device = (float*)buffers->getVarDeviceBuffer(0);
        
        float* affine_matrix_device = (float*)raw_input_device;

        // 内存段的第一个数表示初步后处理会得到多少个bbox，第二个数表示nms之后剩下多少个bbox。这里将它们初始化为0
        checkCudaRuntime(cudaMemsetAsync(output_array_device, 0, sizeof(int), stream_cpy_post_));
        checkCudaRuntime(cudaMemsetAsync(output_array_device+1, 0, sizeof(int), stream_cpy_post_));

        // for debug
        // printf("num of predicted bboxes: %d\n", output1_dims_.d[1]);

        // 对推理结果进行后处理得到bbox，结果保存在output_array_device中
        // 不限制解码bbox的数量，最大后处理成功的个数就是网络推理的输出个数！
        decode_kernel_invoker(infer_output_bbox, output1_dims_.d[1], num_classes, confidence_threshold_, affine_matrix_device, output_array_device, output1_dims_.d[1], stream_cpy_post_);

        // 对初步后处理得到的bbox根据置信度进行排序,然后只有前max_pre_nms个框会被用于进行NMS
        cudaDeviceProp deviceProp;
        int device = 0;
        
        GPU_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        if (!(deviceProp.major > 3 || (deviceProp.major == 3 && deviceProp.minor >= 5))) {
            printf("GPU %d - %s  does not support CUDA Dynamic Parallelism\n Exiting\n.", device, deviceProp.name);
            return false;
        }
        
        float* bbox_ptr = output_array_device + 2;
        //checkCudaRuntime(cudaStreamSynchronize(stream_cpy_post_));

        // 限制进入NMS操作的bbox的个数
        // 为了使用CUDA的动态并行操作（即在GPU上迭代地调用内核），需要在cmakeLists中为此文件设置编译和链接的特殊参数
        // https://zhuanlan.zhihu.com/p/579693025
        // https://forums.developer.nvidia.com/t/compile-cuda-program-with-dynamic-parallelism/55385
        // RTX_3070应该是sm_86构架，需要cuda11.1及之后版本
        // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        sorted_bbox_for_NMS(bbox_ptr, output_array_device, max_pre_nms_, stream_cpy_post_);

        // 先把后处理结果中用于记录bbox解码个数的数据传输到host中，这样后面可以精确确定需要进行多少个bbox的NMS，并且需要往host传输多少有效的字节信息
        // 传8个float数只是为了凑32bytes？
        size_t size_num_deco = 8 * sizeof(float);
        if (num_pre_deco_host == nullptr){
            checkCudaRuntime(cudaMallocHost(&num_pre_deco_host, size_num_deco));
            // cudaMemset只能用于设置device memory上的值，不能用于设置host memory上的吧？可以,且会阻塞当前host线程
            checkCudaRuntime(cudaMemset(num_pre_deco_host, 0, size_num_deco));
        }
        // 先等待初步解码结束才能访问device memory
        checkCudaRuntime(cudaStreamSynchronize(stream_cpy_post_));
        // 这个复制数据需要阻塞host线程
        checkCudaRuntime(cudaMemcpy(num_pre_deco_host, output_array_device, size_num_deco, cudaMemcpyDeviceToHost));
        
        // cout << "Succeeded sort bbox!" << endl;

        float* num_deco_ptr   = (float*)num_pre_deco_host;
        // 每张图像推理的后处理结果中的第一个数据表示进入NMS之前共有多少个候选的bbox
        int count_before_NMS  = (int)*num_deco_ptr;

        //cout << "Num of detected bbox before NMS: " << count_before_NMS << endl;

        int MAX_OBJ = min(count_before_NMS, max_objects_);
        if (MAX_OBJ == 0)
        {
            printf("num of detected object is: %d\n", 0);
            // 即使没有检测到物体，后面还需要构建mask
            // return true;
        }
        
        if (MAX_OBJ > 0) nms_kernel_invoker(output_array_device, nms_threshold_, MAX_OBJ, stream_cpy_post_);

        // printf("num of bbox before NMS: %d\n", count_before_NMS);

        // int Size_output_array = 1 + 31 + count  * NUM_BOX_ELEMENT;
        // 注意，此段内存中最后6个float长度的内存是无意义的，只是为了凑传输单位（32bytes）
        // 或者干脆直接传输 2 + 6 + max_pre_nms_ * NUM_BOX_ELEMENT_个float数据好了，因为反正进入nms的bbox不会超过max_pre_nms_个！
        // 但max_pre_nms_ = 1024还是挺大的，实际上经过conf过滤之后的物体个数一般只有2位数
        int size_bytes_output_array = (2 + 6 + count_before_NMS * NUM_BOX_ELEMENT_) * sizeof(float);

        // 此函数第一个参数，即需要复制的实际数据的Bytes数，跟初步后处理得到的bbox个数有关，这可以减少很多的传输需求
        // 可能还是直接全部内存进行复制更方便？不，数据传输的数据量应该尽可能地少！！！使用异步的数据传输操作
        // buffers->copyOutputToHostAsync(true, size_bytes_output_array, std::vector<std::string>(), stream_cpy_post_);
        cudaMemcpyAsync(output_array_host, output_array_device, size_bytes_output_array, cudaMemcpyDeviceToHost, stream_cpy_post_);
        
        checkCudaRuntime(cudaStreamSynchronize(stream_cpy_post_));

        //cout << "Succeeded copy infer output to host!" << endl;

        // float* output_array_host = (float*)buffers->getVarHostBuffer(1);

        // 每张图像推理的后处理结果中的第二个数据表示此图像中最终检测出了多少个bbox
        int count_after_NMS = (int)*(output_array_host + 1);

        if(MAX_OBJ == 0) 
            assert(count_after_NMS == 0);

        //printf("num of bbox after NMS: %d\n", count_after_NMS);

        // size_t gpu_total_size, gpu_free_size;
        // cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size,&gpu_total_size);
        // if(cuda_status != cudaSuccess) cout << "Error: cudaMenGetInfo fails!" << endl;
        // cout << "剩余显存：" << double(gpu_free_size)/(1024.0*1024.0) << endl;

        // full_seg_map_ 用于存储后处理得到的padded图像分辨率的seg map，每个点均为3通道的uchar型数据，通道1为该点所属物体的class，通道2为该点所属物体的id，通道3为该点为物体的概率值
        // 这里直接复用raw_image_host(后弃用）和raw_image_device，它们的内存比真正的raw image的要大，即和padded之后的图像大小相同
        // 此内存上的数据要被转化为CV_8UC3类型的Mat，这要求内存上每个点的3个通道是紧连着的！
        // if (full_seg_map_host == nullptr) full_seg_map_host = raw_image_host;
        if (full_seg_map_device == nullptr) full_seg_map_device = raw_image_device;
        // 需要3个通道。刚好使用raw image的device内存来作为full seg map，首先将所有点的各个通道值初始化为0
        
        checkCudaRuntime(cudaMemsetAsync(full_seg_map_device, 0, size_full_seg_map * 4, stream_));
        // 对于mask_for_bg，全部初始值设为255，那么这里是给定255这个值吗？还是用1（按bit?)？
        checkCudaRuntime(cudaMemsetAsync(full_seg_map_device+size_full_seg_map * 4, 255, size_full_seg_map, stream_));
        checkCudaRuntime(cudaEventRecord(stop_infer_seg, stream_));

        int count_valid_bbox = 0;
        if(count_after_NMS > 0)
        {
            // 注意，cudaMalloc的第一个参数是 void**类型的
            if (small_seg_map_device == nullptr)
            {
                // cout << "size of small seg map: " << size_small_seg_map << endl;
                checkCudaRuntime(cudaMalloc((void**)&small_seg_map_device, size_small_seg_map));
            }
            // 注意，cudaMemcpy()在为device内存赋值时是默认不阻塞host的！！因此如果既要给device内存赋值，又要阻塞host，则要用Async版本
            checkCudaRuntime(cudaMemsetAsync(small_seg_map_device, 0, size_small_seg_map, stream_));
            
            // 32*96*312  即预测的作为基底的mask共有32个，每一个的大小为（384/4）*（1248/4）
            float* infer_output_mask = (float*)buffers->getBindingDeviceBuffer(yolov8_seg_config_.output_tensor_names[1]); 
            
            // 非刚体目标的id从100开始计数，这部分区域之后是不被使用的
            int deform_obj_id = 100;
            // 刚体目标的id从1开始计数，id为0属于背景点
            int solid_obj_id = 1;

            num_valid_objs = 0;
            uchar is_valid_solid_obj;
            for(int i = 0; i < count_before_NMS; ++i)
            {
                if (count_valid_bbox >= count_after_NMS) break;
                // 记得要+2，因为前两个数据是bbox的个数
                float* pbox  = output_array_host + 2 + i * NUM_BOX_ELEMENT_;
                // 这个量表示该解码后的bbox在经过nms之后是否被保留
                int keepflag = pbox[6];
                if(keepflag == 1)
                {
                    is_valid_solid_obj = 1;

                    // 此工作中只检测7类物体，标签值从0到6
                    if (int(pbox[5]) < 0 || int(pbox[5]) > 6) continue;
                    
                    //if (int(pbox[5]) < 0 || int(pbox[5]) > 255) continue;
                    
                    // process mask
                    // reference: https://github.com/shouxieai/infer/blob/main/src/yolo.cu#L629
                    int row_index = pbox[7];
                    int mask_dim  = output2_dims_.d[1];
                    
                    float left, top, right, bottom;
                    
                    // 将bbox head后处理得到的原始图像中的边界框角点坐标变换到 缩放填充之后的坐标
                    float* i2d = aff_mat.i2d;
                    // 计算框的两个角点在pad后的图像中的坐标。得到的两个角点位置坐标虽然还是float，但是应该都是整数，因为缩放比例为1.0
                    affine_project(i2d, pbox[0], pbox[1], &left,  &top);
                    affine_project(i2d, pbox[2], pbox[3], &right, &bottom);
                    
                    float box_width          = right - left;
                    float box_height         = bottom - top;
                    float scale_to_predict_x = seg_map_scale_x;
                    float scale_to_predict_y = seg_map_scale_y;
                    // 然后再把pad后的图像上的边界框缩放到mask图的大小比例,边界框的长度和宽度很可能缩放之后不是整数，因此需要四舍五入
                    int mask_out_width       = box_width  * scale_to_predict_x + 0.5f;
                    int mask_out_height      = box_height * scale_to_predict_y + 0.5f;
                
                    // 应该不会有长度或宽度为0的bbox吧？原始比例的bbox的宽和高的乘积把又能太小，否则认为该物体太小，连采样最低个数的像素点都不行，则不算入当前帧待跟踪的物体（一般来说都是太远的物体才会在图像上显得太小）
                    if(mask_out_width > 0 && mask_out_height > 0)
                    {
                        count_valid_bbox+=1;
                        
                        // InstanceSegmentMap的构造函数中会在cpu上分配指定大小的页锁内存。这里必须用智能指针，这样方便InstanceSegmentMap没有被使用时可以自动释放此内存！
                        // 此cpu上的页锁内存需要临时分配，因为不知道每个图像具体会有多少个多大的bbox，如果直接按max_objects_个数来算，每个的大小按mask图的大小，那么极其耗费内存！
                        // 此处暂时不需要单独获取每个bbox在seg map上的区域
                        // result_object_box.seg  = make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);

                        // InstanceSegmentMap类中的data就是一个host页锁内存区域的指针
                        // unsigned char* small_mask_host   = result_object_box.seg->data;
                        
                        // 注意，由于yoloseg的标签值必须得从0开始，因为我们将所有label都+1，即从1开始，而将0作为背景点的标签值！!!!!
                        int cls_label = (int)pbox[5] + 1;
                        Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], cls_label);
                        // 非刚性目标的obj id从100开始计数
                        // 训练的YOLOV8检测器中规定，0 - person, 1 - rider, 2 - car(包含了轿车和van), 3 - bicycle, 4 - bus, 5 - truck, 6 - train
                        // 注意，上面我们对所有的原始cls_label都加了1！！因此后面car的cls label应该是3！
                        // if (cls_label==1 || cls_label==2 || cls_label==4)
                        if (cls_label==1 || cls_label==2 || cls_label==4 || cls_label==7)
                        {
                            deform_obj_cls.insert((uchar)cls_label);
                            result_object_box.id = deform_obj_id++;
                            is_valid_solid_obj = 0;
                        }
                        // 是否希望在这里把图像中bbox过小的物体先排除？可以限制是排除图像上半部分的小bbox？
                        // 然而可能有的物体本身不小，但只检测出了一小部分，而其上的特征点其实是可以被匹配（暴力或光流）到的。
                        // todo: 因此这里是否不先将其排除，而是后续利用特征点或者像素的深度值来排除太远的物体？（但是这部分太小的区域可能会导致FAST点检测函数的失效？)
                        else if(box_width * box_height < 35 * 35 && ((pbox[1] + pbox[3])/2.0 <= raw_row/2.0))
                        // else if(box_width * box_height < 1.5 * NUM_SAMPLED_PIXEL_OBJ)
                        {
                            // 不被跟踪的刚体的id也都用deform_obj_id来标识，这样剩下的刚体的局部id就是连续的
                            small_solid_objs.insert(deform_obj_id);
                            // 指定类别范围内的刚体，如果其bbox太小，则认为该物体的跟踪价值不大，则将放弃该物体。但是仍然需要将该物体的mask求解去除，以避免这些动态点对相机运动估计的影响
                            result_object_box.id = deform_obj_id++;
                            is_valid_solid_obj = 0;
                            // 这也是属于刚体，记录其真实类被
                            solid_obj_cls.insert((uchar)cls_label);
                            // 然后将此面积过小的物体的类别改为某个无效类别，这样它在seg_map上所显示的类别就是无效的类别，便于后续将其上的匹配或检测点排除
                            cls_label = 1;
                        }
                        else
                        {
                            // 注意，每一帧检测到的刚体物体实例的id是从1开始的。0代表的是背景实例点
                            result_object_box.id = solid_obj_id++;
                            solid_obj_cls.insert((uchar)cls_label);
                            ++num_valid_objs;
                            // cout << "Cls of valid solid obj:" << cls_label << endl;
                        }

                        uchar obj_id = (uchar)result_object_box.id;

                        float* mask_weights  = infer_output_bbox + row_index * output2_dims_.d[2] + num_classes + 4;

                        // 先确保上面device内存赋值已经完成
                        if(count_valid_bbox == 1) checkCudaRuntime(cudaStreamWaitEvent(stream_, stop_infer_seg));

                        // 从mask图的bbox区域内确定每个像素的mask值，结果保存在mask_out_device对应的gpu内存上
                        // mask解码后的存储是行优先，每一个点的2通道uchar数据是紧挨着的
                        decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                        infer_output_mask, output2_dims_.d[3], output2_dims_.d[2],
                                        small_seg_map_device, mask_dim, mask_out_width, mask_out_height, stream_);

                        //cout << "Succeeded decode single _mask!" << endl;

                        // 原本是在每个bbox对象中都保存一个InstanceSegmentMap，以便后续在CPU后构建full seg map。但是后面改成了直接在GPU上构建full seg map！
                        // small_seg = make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
                        // 注意，这里的等号其实有个float到int的隐式转换
                        // small_seg->left = left * scale_to_predict_x;
                        // small_seg->top  = top  * scale_to_predict_y;

                        // result_object_box.left_bbox_pad_img = left;
                        // result_object_box.top_bbox_pad_img = top;
                        
                        build_full_seg_mask(left, top, full_seg_map_device, input_width_, input_height_,
                                        small_seg_map_device, output2_dims_.d[3], output2_dims_.d[2], int(box_width), int(box_height),
                                        (uchar)cls_label, obj_id, is_valid_solid_obj, stream_);
                        
                        //cout << "Succeeded build decode single mask in full seg_map!" << endl;

                        // int pitch_dst = mask_out_width;
                        // int pitch_scr = output2_dims_.d[3];
                        // int left_drift = result_object_box.seg->left; 
                        // int top_drift  = result_object_box.seg->top; 
                        // unsigned char* small_mask_device = small_seg_map_device + left_drift + top_drift * output2_dims_.d[3];
                        // 需要用cudaMemcpy2DAsync来复制单个bbox的mask数据！因为mask是直接解码在96*312的seg_map上，但我们只需要复制上面物体边界框内部的数据！
                        // checkCudaRuntime(cudaMemcpy2DAsync(small_mask_host, pitch_dst, small_mask_device, pitch_scr, mask_out_width, mask_out_height, cudaMemcpyDeviceToHost, stream_cpy_post_));
                        
                        // result_object_box 每一个bbox结果中包含边界框在原始图像上的位置和尺寸信息，还有就是在mask图（96*312分辨率）中的边界框位置及框中的各像素的mask值
                        // 如果元素对象中有定义移动构造函数，并且emplace_back中给定右值时才进行移动拷贝。这里只是调用拷贝构造函数（由于没有转换构造函数，所以其跟push_back一样，如果没有编译器优化，则在拷贝时还是会创建临时对象）
                        // 但是由于bbox中的Seg元素是用shared_ptr指向的，因此拷贝构造时也只是多了个shared_ptr对象且对seg对象的引用+1而已
                        // https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
                        // 在离开此后处理函数之后，result_object_box这个临时对象就会被析构，其中对seg的引用计数就-1，只剩下Output中的shared_ptr的引用了
                        // postprocess_boxes.emplace_back(result_object_box);
                        //？？ 这里需要阻塞host等待cpy完成吗？不需要，bbox中的seg map的地址被智能指针所管理，这里引用+1，该地址所指内存仍然存在，后续只需要等待此event完成即可
                        
                        // masked_bboxes_seg_map[(int)obj_id] = result_object_box;
                        // 因为已经在GPU中构建full_seg_map了，因此这里就只保存最终考虑范围内的有效刚体的bbox信息
                        if(is_valid_solid_obj) masked_bboxes_seg_map[(int)obj_id] = result_object_box;
                    }
                }
            }
        }
        else
        {
            num_valid_objs = 0;
        }
        
        // 构建一个原图大小的单通道uchar的mask，其中属于类别1、2和4的点均为黑色（0），其他点均为白色（255）。可以使用raw_image_device和raw_image_host的第三个通道

        // 总的检测物体（包含有效和无效的物体）
        printf("final total num of detected objs: %d\n", count_valid_bbox);
        // 检测到的有效类别的物体的数量
        printf("final detected num of valid solid obj: %d\n", num_valid_objs);

        // 将gpu上的seg mag制到给定的cpu内存上。注意，我们是将前2个通道(class 和 id)紧挨在以其，前2个通道的map之后的map用来保存prob，最后的prob部分不需要传输到host
        checkCudaRuntime(cudaMemcpyAsync(full_seg_map_host, full_seg_map_device, size_full_seg_map * 5, cudaMemcpyDeviceToHost, stream_));

        // checkCudaRuntime(cudaEventRecord(stop_post_proc, stream_cpy_post_));
        // checkCudaRuntime(cudaEventSynchronize(stop_post_proc));

        // stream上所有任务执行结束前会一直阻塞host调用线程,保证完成所有后处理之后再共享数据
        // checkCudaRuntime(cudaStreamSynchronize(stream_cpy_post_));

        //！ 如果模型推理和后处理都是在子线程中进行，那么这里可以使用std::promise进行线程间的数据共享，在主线程中使用wait来进行阻塞；
        //!  当然也可以仅仅是将类成员函数作为GPU线程的执行函数,子线程直接使用类成员变量进行处理，当然这时就需要加锁
        //pro.set_value(postprocess_boxes);

        //for (int i = 0; i < result_object_box.size(); i++){
        //    seg_map.emplace_back(postprocess_boxes[i]);
        //}

        //! 如果主线程是使用shared_future来进行数据共享的话，那么对Output是进行深度拷贝，那么此是应该要在此处释放Output中所占据的内存
        //！ vector的clear函数会逐一地删除其中的元素，如果是int等内置类型，则直接释放其内存，如果是自定义对象，则调用其析构函数。
        //！ box类中的InstanceSegmentMap对象由智能指针管理着，智能指针对象在被销毁之前会调用所指对象的析构函数，从而释放InstanceSegmentMap对象所占的host内存！

        if (stop_post_proc != nullptr)
            checkCudaRuntime(cudaEventRecord(stop_post_proc, stream_));
        return true;
    }
    
    // 当前帧所有待匹配物体上的采样像素点
    bool Yoloseg::sample_pixel(float* array_pixel_objs, float* depth_map_device, float* buffer_device_pixel, map<int, YoloV8::Box> &masked_bboxes_seg_map, 
                                vector<int> &valid_objs, int W_dep_map, int H_dep_map, float left_shift_dep, float top_shift_dep, bool &done_sample, bool &check_total_lost_objs)
    {
        // 需要在host和device值减传输的数据大小。这里是+2，用于保存物体的局部id 和 采样像素点数量.
        int final_valid_num_objs = valid_objs.size();
        if(final_valid_num_objs == 0)
        { 
            done_sample = true;
            return true;
        }
        // 预分配的检测物体数最多为40
        assert(final_valid_num_objs <= 40);

        size_t buffer_for_pixel = final_valid_num_objs * (NUM_SAMPLED_PIXEL_OBJ * 3 + 2 + 7) * sizeof(float); 
        float bbox_info[final_valid_num_objs * 5];
        size_t buffer_for_bbox_info = final_valid_num_objs * 5 * sizeof(float);
        int index = 0;
        float left, top, right, bottom;

        // bbox_info[0] = (float)num_valid_objs;
        int id;
        // 采样的物体顺序也是按照valid_objs中的物体顺序来排列的
        for(int i = 0; i < final_valid_num_objs; ++i)
        {
            id = valid_objs[i];
            Box &bbox = masked_bboxes_seg_map[id];
            
            affine_project(aff_mat.i2d, bbox.left, bbox.top, &left, &top);
            // 对right和bottom其实不需要进行仿射变换，因为计算它们只是为了求得bbox的宽和高，而yolov8预处理只进行填充没有缩放，因此这里的仿射变换不影响宽度和高度的值
            affine_project(aff_mat.i2d, bbox.right, bbox.bottom, &right, &bottom);
            bbox_info[index++] = (float)bbox.id;
            bbox_info[index++] = left;
            bbox_info[index++] = top;
            // width and height of bbox
            bbox_info[index++] = right - left;
            bbox_info[index++] = bottom - top;
        }
        
        if(buffer_device_bbox_info == nullptr)
        {
            size_t buffer_pixel = 40 * 5 * sizeof(float);
            checkCudaRuntime(cudaMalloc((void**)&buffer_device_bbox_info, buffer_pixel));
        }
        
        // 物体上可能有效的像素点不足，因此提前赋值为0
        checkCudaRuntime(cudaMemset(buffer_device_pixel, 0, buffer_for_pixel));
        checkCudaRuntime(cudaMemcpy((void*)buffer_device_bbox_info, (void*)bbox_info, buffer_for_bbox_info, cudaMemcpyHostToDevice));
        
        sample_pixel_for_objs(final_valid_num_objs, NUM_SAMPLED_PIXEL_OBJ, buffer_device_bbox_info, 
                                mThDepthObj, mMinDepthPt, mbf, full_seg_map_device, input_width_, input_height_, depth_map_device, 
                                buffer_device_pixel, FOCAL_LENGTH_X, FOCAL_LENGTH_Y, SHIFT_X, SHIFT_Y, 
                                aff_mat.i2d[2], aff_mat.i2d[5], W_dep_map, H_dep_map, left_shift_dep, top_shift_dep, stream_);
        
        // 这里需要用stream_阻塞host吗？只要传输数据也用stream_不就行了吗？反正调用此sample_pixel函数也是用的子线程，这里都阻塞直到完成像素点采样，然后通知主线程
        checkCudaRuntime(cudaStreamSynchronize(stream_));
        // 等待obj_matching函数中完成物体匹配，查看是否有在当前帧完全漏检的物体，如果有，提前将其上一帧匹配物体的采样像素点复制走
        while(!check_total_lost_objs)
        {
            usleep(300);
        }
        checkCudaRuntime(cudaMemcpy(array_pixel_objs, buffer_device_pixel, buffer_for_pixel, cudaMemcpyDeviceToHost));
        checkCudaRuntime(cudaDeviceSynchronize());
        usleep(300);
        done_sample = true;
        cout << "Succeed sample pixel of objs!" << endl;
        return true;
    }

    cv::Size Yoloseg::full_seg_map_size()
    {
        return cv::Size(input_width_, input_height_);
    }

    int Yoloseg::get_num_valid_objs()
    {
        return num_valid_objs;
    }

    void* Yoloseg::get_full_seg_map_host()
    {
        return full_seg_map_host;
    }
    
    AffineMatrix &Yoloseg::get_affine_matrix()
    {
        return aff_mat;
    }

    // have to define static function draw_mask()
    // void Yoloseg::visualization(const std::string &image_name, const cv::Mat &image, BoxArray &boxes) {
    //     cv::Mat image_display;
    //     if(image.channels() == 1)
    //         cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    //     else
    //         image_display = image.clone();

    //     for(auto& obj : boxes){
    //         uint8_t b, g, r;
    //         // 模板函数tie返回一个tuple，元素就是b, g, r的引用，此tie函数的意思就是把这三个变量绑在一起形成tuple以便赋值？
    //         tie(b, g, r) = iLogger::random_color(obj.class_label);        
    //         cv::Scalar color(b, g, r);
    //         if(obj.seg){
    //             draw_mask(image_display, obj, color);
    //         }
    //     }

    //     for(auto& obj : boxes){
    //         uint8_t b, g, r;
    //         tie(b, g, r) = iLogger::random_color(obj.class_label);
    //         // 物体边界框        
    //         cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

    //         auto name    = cocolabels[obj.class_label];
    //         auto caption = iLogger::format("%s %.2f", name, obj.confidence);
    //         int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    //         // 物体的标注说明栏
    //         cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    //         // 需要给定的是文字string左下角的坐标，即距离标注说明栏左端3像素，距离下端5像素？
    //         cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    //     }
    //     // 取文件名的前缀，文件名应该是还会包含多级路径/的，仅取最后的不加后缀的文件名
    //     string file_name = iLogger::file_name(files[i], false);
    //     string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
    //     INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
    //     cv::imwrite(image_name + ".jpg", image_display);
    // }

    void Yoloseg::save_engine() {
        if (yolov8_seg_config_.engine_file.empty()) return;
        if (engine_ != nullptr) {
            nvinfer1::IHostMemory *data = engine_->serialize();
            ofstream file(yolov8_seg_config_.engine_file, ios::binary);
            if (!file) return;
            file.write(reinterpret_cast<const char *>(data->data()), data->size());
        }
    }

    bool Yoloseg::deserialize_engine() {
        ifstream file(yolov8_seg_config_.engine_file.c_str(), ios::binary);
        if (file.is_open()) {
            file.seekg(0, ifstream::end);
            std::size_t size = file.tellg();
            file.seekg(0, std::ifstream::beg);
            char *model_stream = new char[size];
            file.read(model_stream, size);
            file.close();
            IRuntime *runtime = createInferRuntime(gLogger);
            if (runtime == nullptr) {
                delete[] model_stream;
                return false;
            }
            engine_ = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
            delete[] model_stream;
            if (engine_ == nullptr) return false;
            return true;
        }
        return false;
    }

}






