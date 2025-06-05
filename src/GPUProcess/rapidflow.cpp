//
// Created by ruishengcai on 2023/12/20.
//

#include <stdio.h>
// #define NDEBUG
#include <assert.h>
#include "rapidflow.h"
#include <utility>
#include <unordered_map>
#include "cuda_runtime_api.h"

// 将原始图像中的边界框角点坐标变换到 缩放裁减之后的坐标
void affine_project(float* matrix, float x, float y, float* ox, float* oy);

namespace RapidFlow{
    using namespace std;
    using namespace cv;
    using namespace tensorrt_common;
    using namespace tensorrt_log;
    using namespace tensorrt_buffer;
     
    // for debug
    bool succ = true;
    
    // 对output进行后处理，其实就是unpad操作，其内部会调用相关CUDA核函数。
    // 没必要，将结果传回host，使用opencv中的函数即可
    // void decode_kernel_invoker(
    //     float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    //     float* invert_affine_matrix, float* parray,
    //     int max_objects, cudaStream_t stream
    // );

    // 带默认值的参数，其默认值不能同时在声明和定义中都出现，要么在声明，要么在定义中只给定一次默认值。且有默认值的参数应该放在无默认值参数的后面
    void InputPadder::pad(const Mat &raw_img, Mat &padded_img, const string mode){
        h_raw  = raw_img.rows;
        w_raw  = raw_img.cols;
        
        // size_raw.height = ht;
        // size_raw.width  = wd;

        int pad_ht = ((h_raw / 32 + 1) * 32 - h_raw) % 32;
        int pad_wd = ((w_raw / 32 + 1) * 32 - w_raw) % 32;

        Padder[0]  = 0;
        Padder[1]  = pad_ht;
        Padder[2]  = pad_wd / 2;
        Padder[3]  = pad_wd - pad_wd / 2;
        
        if (mode == "replicate"){
            int borderType = BORDER_REPLICATE;
            copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType);
        }
        else{
            int borderType = BORDER_DEFAULT;
            Scalar color = Scalar(114, 114, 114);
            copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType, color);
        }
    }

    void InputPadder::unpad(const Mat &padded_img, Mat &orig_img){
        int ht = padded_img.rows;
        int wd = padded_img.cols;
        
        cv::Mat temp_Mat = padded_img(Range(Padder[0], padded_img.rows-Padder[1]), Range(Padder[2], padded_img.cols-Padder[3]));
        // must use clone to copy data from a memory to another memory
        orig_img = temp_Mat.clone();
    }

    void AffineMatrix::compute(const cv::Size& from, const cv::Size& to){
        int scale_x = (to.width - from.width)/32 + 1;  // should be 1
        int scale_y = (to.height - from.height)/32 + 1; // should be 1

        assert((to.width - from.width)>=0 && "dst_width must be no less than src_width");

        assert(scale_x==1 && "dst_width - src_width must be less than 32");

        assert((to.height - from.height)>=0 && "dst_height must be no less than src_height");

        assert(scale_y==1 && "dst_height - src_height must be less than 32");

        int shift_x_left = (to.width - from.width) / 2;

        int shift_y_top  = 0;

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

    Mat AffineMatrix::i2d_mat()
    {
        return Mat(2, 3, CV_32F, i2d);
    }

    // Rapidflow::Rapidflow(const RapidflowConfig &rapidflow_config, const cudaStream_t stream_common)
    //     : rapidflow_config_(rapidflow_config), engine_(nullptr) {
        
    //     use_multi_stream_  = rapidflow_config_.use_multi_stream;

    //     setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    //     // rapidflow只要求将输入图像的channel值归一化到[-1,1]，不进行通道顺序变换
    //     normalize_ = CUDAKernel::Norm::alpha_beta(2 / 255.0f, -1.0f, CUDAKernel::ChannelType::None);

    //     // assert(stream_common != nullptr && "common stream for infer should be provided!");

    //     if (stream_common){
    //         stream_     = stream_common;
    //         common_stream_infer_ = true;
    //     }
    // }

    Rapidflow::Rapidflow(const RapidflowConfig &rapidflow_config, cudaStream_t stream_common, cudaStream_t stream_cpy)
        : rapidflow_config_(rapidflow_config), engine_(nullptr){
        
        use_multi_stream_  = rapidflow_config_.use_multi_stream;

        input_width_  = rapidflow_config_.inputW;
        input_height_ = rapidflow_config_.inputH;

        setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);

        normalize_  = CUDAKernel::Norm::alpha_beta(2 / 255.0f, -1.0f, CUDAKernel::ChannelType::None);
        
        // assert(stream_common != nullptr && "common stream for infer should be provided!");
        
        if (stream_common){
            stream_  = stream_common;
            common_stream_infer_ = true;
        }

        if (use_multi_stream_ && stream_cpy){
            stream_cpy_ = stream_cpy;
            common_stream_cpy_ = true;
        }
    }

    Rapidflow::~Rapidflow()
    {
        // 等待gpu上与此stream_流相关的所有操作执行完。主要是buffers对象很可能还被使用
        // 这部分同步阻塞可以放在调用三个推理模型的host线程中
        //GPU_CHECK(cudaStreamSynchronize(stream_cpy_));
        //GPU_CHECK(cudaStreamSynchronize(stream_));

        engine_.reset();

        context_.reset();

        buffers.reset();

        cur_input_buffer_device = nullptr;
        
        // 这里应该需要明确一下，如果stream_和stream_cpy_所指对象不是从类外部传入，而是在类内创建的，那么应该在此处destroy
        if (!common_stream_cpy_ && use_multi_stream_){
            cudaStreamDestroy(stream_cpy_);
        }
        else{
            stream_cpy_ = nullptr;
        }
  
        if (!common_stream_infer_){
           cudaStreamDestroy(stream_);
        }
        else{
            stream_ = nullptr;
        }
    }

    bool Rapidflow::build() 
    {
        if (stream_ == nullptr){
            //checkCudaRuntime(cudaStreamCreate(&stream_));
            checkCudaRuntime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
		    if(stream_ == nullptr) 
            {
                cout << "failed create stream!" << endl;
                return false;
            }
        }

        if (use_multi_stream_ && stream_cpy_ == nullptr){
            // checkCudaRuntime(cudaStreamCreate(&stream_cpy_));
            checkCudaRuntime(cudaStreamCreateWithFlags(&stream_cpy_, cudaStreamNonBlocking));
		    if(stream_cpy_ == nullptr) return false;
        }
        
        if(deserialize_engine())
        {
            printf("deserialize engine of Rapidflow succeeded!\n");
            input1_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(rapidflow_config_.input_tensor_names[0].c_str()));
            input2_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(rapidflow_config_.input_tensor_names[1].c_str()));
            ASSERT(input1_dims_.nbDims == 4);
            ASSERT(input2_dims_.nbDims == 4);
            output_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(rapidflow_config_.output_tensor_names[0].c_str()));
            // rapidflow的推理结果应该是 B*C*H*W，这里没有批量推理，所以B=1，每个像素有一个光流值，所以C=2。注意在复制结果时要
            ASSERT(output_dims_.nbDims == 4);
            return true;
        }
        
        auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
        if (!builder) 
        {
            cout << "failed create InferBuilder!" << endl;
            return false;
        }
        // 如果想要选择显式批处理模式（这是使用onnx parser解析和编译engine时必须的模式），那么就必须使用kEXPLICIT_BATCH)这一flag来声明。隐式模式用0U来代替1U
        // 隐式或者显式的批处理模式的选择必须在创建INetworkDefinition对象时指出 
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
        const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) 
        {
            cout << "failed create Network!" << endl;
            return false;
        }
        auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) 
        {
            cout << "failed create BuilderConfig!" << endl;
            return false;
        }
        auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
                nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
        if (!parser) 
        {

            return false;
        }
        
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            return false;
        }

        // 由于导出的Yolov8的onnx模型的宽高是动态的，因此这里需要设置范围，这范围是随便给定还是需要按照实际？
        // 默认此TensorRT仅用于KITTI数据集，并且模型的输入统一为1248*384
        // 可以尝试使用640*224的网络输入，但是相对应的仿射矩阵计算需要重新编写
        profile->setDimensions(rapidflow_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMIN, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(rapidflow_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kOPT, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(rapidflow_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMAX, Dims4(1, 3, input_height_, input_width_));

        profile->setDimensions(rapidflow_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kMIN, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(rapidflow_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kOPT, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(rapidflow_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kMAX, Dims4(1, 3, input_height_, input_width_));

        config->addOptimizationProfile(profile);
        cout << " Start constructing network!" << endl;
        auto constructed = construct_network(builder, network, config, parser);
        if (!constructed) 
        {
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

        assert(engine_->getNbBindings() == 3);

        save_engine();
        ASSERT(network->getNbInputs() == 2);
        input1_dims_ = network->getInput(0)->getDimensions();
        input2_dims_ = network->getInput(1)->getDimensions();
        ASSERT(input1_dims_.nbDims == 4);
        ASSERT(input2_dims_.nbDims == 4);
        ASSERT(network->getNbOutputs() == 1);
        output_dims_ = network->getOutput(0)->getDimensions();
        // rapidflow的推理结果应该是 B*C*H*W，这里没有批量推理，所以B=1，每个像素有一个光流值，所以C=2。注意在复制结果时要
        ASSERT(output_dims_.nbDims == 4);

        return true;
    }

    bool Rapidflow::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                    TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                    TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                    TensorRTUniquePtr<nvonnxparser::IParser> &parser) const 
    {
        auto parsed = parser->parseFromFile(rapidflow_config_.onnx_file.c_str(),
                                            static_cast<int>(gLogger.getReportableSeverity()));
        if (!parsed) {
            return false;
        }
        // 光流模型推理时需要较多的内存空间，因此给它分配2G。
        // 如果是需要从onnx初次转化得到engine，那么需要的内存就会比较大！对于Rapidflow而言，需要5G内存！
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1073741824*8);
        config->setFlag(BuilderFlag::kFP16);
        // dla_core在config文件中的给定值是-1，意味这不启用DLA？另外只有Jetson系列的gou上有DLA Core!
        // enableDLA(builder.get(), config.get(), yolov8_seg_config_.dla_core);
        return true;
    }

    //! flow_Mat should be passed from the main process thread and with dims size {2, input_height_, input_width_} and tyype CV32FC1;
    //! cudaEvent should be create in the sub thread
    bool Rapidflow::infer(cudaEvent_t stop_cpy_input_flow, cudaEvent_t stop_infer_flow) 
    {
        
        checkCudaRuntime(cudaEventRecord(stop_cpy_input_flow, stream_cpy_));
        
        // infer of rapidflow in stream_ won't start until input copy is finished
        checkCudaRuntime(cudaStreamWaitEvent(stream_, stop_cpy_input_flow));

        //！ 从TensorRT8.5开始enqueueV2()开始被弃用了！
        // 这里不让gpu阻塞cpu，是因为此推理线程为cpu上的子线程。executeV2函数则默认是同步执行，即该stream执行任务时会阻塞host线程
        bool status = context_->enqueueV3(stream_);
        // bool status = context_->executeV2(buffers->getDeviceBindings().data());
        if (!status) {
            return false;
        }

        checkCudaRuntime(cudaEventRecord(stop_infer_flow, stream_));

        return true;
    }

    // 光流模型的图像预处理步骤是在cpu上完成，这里只是将预处理结果从cpu复制到host再传输到device
    bool Rapidflow::preprocess_input(const Mat &image, cudaEvent_t stop_cpy_input_flow)
    {

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
            bool status0 = context_->setInputShape(rapidflow_config_.input_tensor_names[0].c_str(), Dims4(1, 3, input_height_, input_width_));
            bool status1 = context_->setInputShape(rapidflow_config_.input_tensor_names[1].c_str(), Dims4(1, 3, input_height_, input_width_));
            assert(status0 && "Set Shape of Input0 failed!");
            assert(status1 && "Set Shape of Input1 failed!");
        }

        if (buffers == nullptr)
        {
            buffers = make_shared<BufferManager>(engine_, 0, context_.get());
            for (int32_t i = 0; i < engine_->getNbIOTensors(); i++)
            {
                auto const name = engine_->getIOTensorName(i);
                context_->setTensorAddress(name, buffers->getBindingDeviceBuffer(name));
            }
        }
        // 光流网络有两个输入，input0是上一时刻的图像，input1是当前时刻的图像，而输出只有一个，即上一时刻图像的光流图
        // 因此每次输入的预处理图像应该是input1，后处理时再将当前图像复制到input0的位置上
        // 在host内存上多分配了一个图像的内存，不需要用，但这应该不影响系统运行
        float* preproc_img1_host = (float*)buffers->getBindingHostBuffer(rapidflow_config_.input_tensor_names[1]);
        
        // 预处理之后的图像每个通道应该是float数据类型
        std::size_t size_preproc_img = input_width_ * input_height_ * 3 * sizeof(float);
        
        Mat padded_img(input_height_, input_width_, CV_8UC3, Scalar(0, 0, 0));

        inputpadder.pad(image, padded_img);

        if (normalize_.channel_type == CUDAKernel::ChannelType::Invert){
            cvtColor(padded_img, padded_img, cv::COLOR_BGR2RGB);
        }
        Mat padded_img_;
        // convertTo只能改变通道的位深，不能改变通道数。此函数支持in-place操作
        padded_img.convertTo(padded_img_, CV_32FC3);
        // channels用vector更方便，因为后续不用手动给指明元素个数
        vector<Mat> channels;
        // split函数会根据channel顺序将Mat分解为C个（H*W)的Mat，并保存到channels数组或vector中
        // 但是Mat仅仅是矩阵头，各个Mat所指向的（H*W)矩阵在内存上不一定是连续的，而只有所有矩阵头Mat自身信息是内存连续的
        split(padded_img_, channels);
        //printf("split_img has dims : %d, has channels:%d, total %d split imgs\n", channels[0].dims, channels[0].channels(), channels.size());
        // 创建一个3*H*W的Mat，保证内存是连续的
        int dims[3] = {3, input_height_, input_width_}; 
        Mat joined(3, dims, CV_32F);

        for (int i = 0; i < channels.size(); i++){
            float* ptr = &joined.at<float>(i, 0, 0); // pointer to first element of slice i
            Mat destination(input_height_, input_width_, CV_32F, (void*)ptr);  // no reallocate, see the usage document
            if (normalize_.type == CUDAKernel::NormType::MeanStd){
                channels[i] = (channels[i] * normalize_.alpha - normalize_.mean[i]) / normalize_.std[i];
            }
            else if (normalize_.type == CUDAKernel::NormType::AlphaBeta){
                channels[i] = channels[i] * normalize_.alpha + normalize_.beta;
            }
            channels[i].copyTo(destination);
        }
        // printf("preproc_img has dims : %d, has channels:%d\n", joined.dims, joined.channels());
        // merge后默认还是只能得到H*W*C顺序的Mat，而需要传给binding的是C*H*W的内存顺序的数据
        // merge(channels, padded_img);
        memcpy(preproc_img1_host, joined.data, size_preproc_img);
        
        // 放弃在gpu上进行光流和双目匹配的预处理
        // rapidflow和CGI-Stereo的预处理在cpu进行就可以，包括传输在内所有耗时都可以被yoloveseg在gpu上的处理耗时所覆盖
        // size_t size_matrix     = iLogger::upbound(sizeof(aff_mat.d2i), 32);

        // uint8_t* image_device        =  (uint8_t*)raw_img_affine_device;
 
        // float* affine_matrix_device  = (float*)(raw_img_affine_device + size_raw_img);

        // 调用核函数进行预处理！此模型使用的预处理是仅进行原图像四周的填充！
        // CUDAKernel::warp_pure_padding_bilinear_and_normalize_plane(
        //         image_devic,     image.cols * 3,    image.cols,    image.rows, 
        //         infer_device_buffer,     input_width_,      input_height_, 
        //         affine_matrix_device,    -1, 
        //         normalize_, stream_ 
        //     );

        // 第2个input是当前帧图像，第1个input是上一帧的
        // 可以自己一个个地进行复制，也可以给定index，使用已定义函数一次性复制多个
        // inputbinding的index不一定就比outputbinding的index小，所以这里要根据binding的名字来取index！
        // float* infer_input1_device_   = (float*)buffers->getBindingDeviceBuffer(rapidflow_config_.input_tensor_names[1]);
        // CHECK(cudaMemcpyAsync(infer_input1_device_, preproc_img_host, size_preproc_img, cudaMemcpyHostToDevice, stream_cpy_));

        std::vector<std::string> index_bindings_cpy = {rapidflow_config_.input_tensor_names[1]};
        //！ 从host复制到device则可以使用与infer不同的stream，但是极端情况下可能出现infer开始了但是input还没传输到
        //! 函数里使用了显式stream进行数据传输，但是需要保证infer操作在复制结束后才执行，可以使用绑定在stream_cpy_上的cudavent
        buffers->copyInputToDeviceAsync(false, size_preproc_img, index_bindings_cpy, stream_cpy_);
        if (stop_cpy_input_flow != nullptr)
            checkCudaRuntime(cudaEventRecord(stop_cpy_input_flow, stream_cpy_));
        return true;
    }

    // 后处理函数。将input1复制到input0中，将推理结果从device复制到host，然后再将host上的内存指针共享给主线程
    bool Rapidflow::postprocess_output(Mat &unpadded_flow_map, const bool &copy_infer_output, cudaEvent_t stop_infer_flow, cudaEvent_t stop_post_proc)
    {
        // post_proc of rapidflow in stream_cpy_ won't start until infer is finished
        if (stop_infer_flow != nullptr)
            checkCudaRuntime(cudaStreamWaitEvent(stream_cpy_, stop_infer_flow));

        std::size_t byteSize_input_img   = input_width_ * input_height_ * 3 * sizeof(float);

        std::size_t byteSize_flow_img    = input_width_ * input_height_ * 2 * sizeof(float);

        void* infer_input0_device_  = buffers->getBindingDeviceBuffer(rapidflow_config_.input_tensor_names[0]);

        void* infer_input1_device_  = buffers->getBindingDeviceBuffer(rapidflow_config_.input_tensor_names[1]);

        cur_input_buffer_device = (float*)infer_input1_device_;

        //！ 要注意，后处理步骤的传输与网络推理必须使用同个stream，即两个操作必须同步！
        // 这里的Async指的是不阻塞host线程！
        CHECK(cudaMemcpyAsync(infer_input0_device_, infer_input1_device_, byteSize_input_img, cudaMemcpyDeviceToDevice, stream_cpy_));
        
        if (copy_infer_output)
        {
            std::vector<std::string> index_bindings_cpy = {rapidflow_config_.output_tensor_names[0]};

            buffers->copyOutputToHostAsync(false, byteSize_flow_img, index_bindings_cpy, stream_cpy_);
        
            // 必须等到后处理结束才能往主线程共享host上后处理结果的地址（虽然此地址是不变的，但是内容是会变的）
            checkCudaRuntime(cudaStreamSynchronize(stream_cpy_));

            // //！ can also use eventSynchronize instead of stremSynchronize ,this method can use to calcu time
            if (stop_post_proc != nullptr){
                checkCudaRuntime(cudaEventRecord(stop_post_proc, stream_cpy_));
                // checkCudaRuntime(cudaEventSynchronize(stop_post_proc));
            }
            
            float* flow_ptr = (float*)buffers->getBindingHostBuffer(rapidflow_config_.output_tensor_names[0]);
            // rapidflow的推理结果是B*C*H*W，其中B=1，C=2，其数据在内存的排列数序是先通道0的H*W，然后再是通道1的H*W，按通道依次排列。
            // 而CV_32FC2的Mat的数据顺序是H*W*C，其内存是H*W图上的每个点的2个通道数据都排列在一起。
            vector<Mat> split_mat;
            Mat x_flow(input_height_, input_width_,CV_32F,(void*)flow_ptr);
            Mat y_flow(input_height_, input_width_,CV_32F,(void*)flow_ptr + input_height_ * input_width_ * sizeof(float));
            split_mat.push_back(x_flow);
            split_mat.push_back(y_flow);
            Mat padded_flow_map;
            merge(split_mat,padded_flow_map);
            inputpadder.unpad(padded_flow_map, unpadded_flow_map);
            
            //！如果模型推理和后处理都是在子线程中进行，那么这里也可以用std::promise进行线程间的数据共享，在主线程中使用wait来进行阻塞
            //! 主线程应该对页锁内存上的数据进行深拷贝
            //pro_flow.set_value(flow_ptr);
        }
        return true;
    }

    // 返回device上用于保存当前帧预处理后图像的内存指针
    float* Rapidflow::get_binding_input_cur_device()
    {
        return cur_input_buffer_device;
    }

    void Rapidflow::save_engine() {
        if (rapidflow_config_.engine_file.empty()) return;
        if (engine_ != nullptr) {
            nvinfer1::IHostMemory *data = engine_->serialize();
            ofstream file(rapidflow_config_.engine_file, ios::binary);
            if (!file) return;
            file.write(reinterpret_cast<const char *>(data->data()), data->size());
        }
    }

    bool Rapidflow::deserialize_engine() {
        ifstream file(rapidflow_config_.engine_file.c_str(), ios::binary);
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
