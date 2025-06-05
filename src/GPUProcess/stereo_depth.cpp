//
// Created by ruishengcai on 2023/12/22.
//

#include <stdio.h>
// #define NDEBUG
#include <assert.h>
#include "stereo_depth.h"
#include <utility>
#include <unordered_map>
#include "cuda_runtime_api.h"

// 将原始图像中的边界框角点坐标变换到 缩放裁减之后的坐标
void affine_project(float* matrix, float x, float y, float* ox, float* oy);

namespace StereoDepth{
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

    void InputPadder::pad(const Mat &raw_img, Mat &padded_img, const string mode){
        h_raw  = raw_img.rows;
        w_raw  = raw_img.cols;

        int pad_ht = ((h_raw / 64 + 1) * 64 - h_raw) % 64;
        int pad_wd = ((w_raw / 64 + 1) * 64 - w_raw) % 64;

        // DPStereo只在上侧和右侧进行填充
        Padder[0]  = pad_ht;
        Padder[1]  = 0;
        Padder[2]  = 0;
        Padder[3]  = pad_wd;
        
        if (mode == "constant"){
            int borderType = BORDER_CONSTANT;
            Scalar color = Scalar(0, 0, 0);
            copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType, color);
        }
        else{
            int borderType = BORDER_DEFAULT;
            Scalar color = Scalar(114, 114, 114);
            copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType, color);
        }
    }

    void InputPadder::unpad(const Mat &padded_img, Mat &orig_img){
        // int ht = padded_img.rows;
        // int wd = padded_img.cols;
        // Is the Mat(range1, range2) is a shallow copy？ Range是左闭右开
        // 这种对Mat的局部区域的索引和=赋值，只是共享内存，并没有进行值的复制。
        cv::Mat temp_Mat = padded_img(Range(Padder[0], h_raw+Padder[0]), Range(Padder[2], w_raw+Padder[2]));
        // 用clone或者copyTo来复制数据到别的内存
        orig_img = temp_Mat.clone();
    }

    void AffineMatrix::compute(const cv::Size& from, const cv::Size& to){
        int scale_x = (to.width - from.width)/64 + 1;  // should be 1
        int scale_y = (to.height - from.height)/64 + 1; // should be 1

        assert((to.width - from.width)>=0 && "dst_width must be no less than src_width");

        assert(scale_x==1 && "dst_width - src_width must be less than 32");

        assert((to.height - from.height)>=0 && "dst_height must be no less than src_height");

        assert(scale_y==1 && "dst_height - src_height must be less than 32");

        int shift_x_left = to.width - from.width;

        int shift_y_top  = to.height - from.height;

        // 最后一列进行四舍五入
        // DPStereo的pad方式是在只在上边和右边进行填充，因此x的shift应该是0
        i2d[0] = (float)scale_x;  i2d[1] = 0.0;  i2d[2] = (float)shift_x_left * 0;
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

    DPStereo::DPStereo(const DPstereoConfig &DPStereo_config, cudaStream_t stream_common)
        : DPstereo_config_(DPStereo_config), engine_(nullptr) {
        
        use_multi_stream_  = DPstereo_config_.use_multi_stream;

        // TODO: 可以尝试使用640*224的网络输入，但是相对应的仿射矩阵计算需要重新编写
        input_width_  = DPstereo_config_.inputW;
        input_height_ = DPstereo_config_.inputH;

        setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);

        float mean[3] = {0.485, 0.456, 0.406};
        float std[3]  = {0.229, 0.224, 0.225};
        // CGIStereo要求首先将输入图像的channel值归一化到[0,1]（即除以255），然后再使用mean-std方式进行normalize
        // mean-std normalize的计算为 input[channel] =（input[channel]*alpha - mean[channel])/std[channel]
        normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);

        // assert(stream_common != nullptr && "common stream for infer should be provided!");

        if (stream_common)
        {
            stream_     = stream_common;
            common_stream_infer_ = true;
        }
    }

    DPStereo::DPStereo(const DPstereoConfig &DPStereo_config, cudaStream_t stream_common, cudaStream_t stream_cpy)
        : DPstereo_config_(DPStereo_config), engine_(nullptr){
        
        use_multi_stream_  = DPstereo_config_.use_multi_stream;

        input_width_  = DPstereo_config_.inputW;
        input_height_ = DPstereo_config_.inputH;

        setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);

        float mean[3] = {0.485, 0.456, 0.406};
        float std[3]  = {0.229, 0.224, 0.225};

        normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
        
        //assert(stream_common != nullptr && "common stream for infer should be provided!");

        if (stream_common){
            stream_     = stream_common;
            common_stream_infer_ = true;
        }
        
        if (use_multi_stream_ && stream_cpy){
            stream_cpy_ = stream_cpy;
            common_stream_cpy_ = true;
        }
    }

    DPStereo::~DPStereo(){
        // 等待gpu上与此stream_流相关的所有操作执行完。主要是buffers对象很可能还被使用
        //GPU_CHECK(cudaStreamSynchronize(stream_cpy_));
        //GPU_CHECK(cudaStreamSynchronize(stream_));

        engine_.reset();

        context_.reset();

        buffers.reset();

        // 这里应该需要明确一下，如果strem_和stream_cpy_所指对象不是从类外部传入，而是在类内创建的，那么应该在此处destroy
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

    bool DPStereo::build() {

        if (stream_ == nullptr){
            //checkCudaRuntime(cudaStreamCreate(&stream_));
            checkCudaRuntime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
		    if(stream_ == nullptr) return false;
        }

        if (use_multi_stream_ && stream_cpy_ == nullptr){
            //checkCudaRuntime(cudaStreamCreate(&stream_cpy_));
            checkCudaRuntime(cudaStreamCreateWithFlags(&stream_cpy_, cudaStreamNonBlocking));
		    if(stream_cpy_ == nullptr) return false;
        }
        
        if(deserialize_engine())
        {
            printf("deserialize engine of DP-Stereo succeeded!\n");
            input1_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(DPstereo_config_.input_tensor_names[0].c_str()));
            input2_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(DPstereo_config_.input_tensor_names[1].c_str()));
            ASSERT(input1_dims_.nbDims == 4);
            ASSERT(input2_dims_.nbDims == 4);
            output_dims_ = engine_->getBindingDimensions(engine_->getBindingIndex(DPstereo_config_.output_tensor_names[0].c_str()));
            // CGIStereo的网络输出是没有batch这一维度的，只有C*H*W,其中C=1，即视察值
            ASSERT(output_dims_.nbDims == 3);
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
        // auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
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
        profile->setDimensions(DPstereo_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMIN, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(DPstereo_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kOPT, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(DPstereo_config_.input_tensor_names[0].c_str(),
                            OptProfileSelector::kMAX, Dims4(1, 3, input_height_, input_width_));

        profile->setDimensions(DPstereo_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kMIN, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(DPstereo_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kOPT, Dims4(1, 3, input_height_, input_width_));
        profile->setDimensions(DPstereo_config_.input_tensor_names[1].c_str(),
                            OptProfileSelector::kMAX, Dims4(1, 3, input_height_, input_width_));

        config->addOptimizationProfile(profile);

        auto constructed = construct_network(builder, network, config, parser);
        if (!constructed) {
            printf("build engine failed! %d\n", succ);
            return false;
        }
        
        auto profile_stream = makeCudaStream();
        if (!profile_stream) {
            return false;
        }
        
        config->setProfileStream(*profile_stream);
        //printf("build engine failed! %d\n", succ);
        TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            return false;
        }
        //printf("build engine failed! %d\n", succ);
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
        // CGIStereo的网络输出是没有batch这一维度的，只有C*H*W,其中C=1，即视察值
        ASSERT(output_dims_.nbDims == 3);

        return true;
    }

    bool DPStereo::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                    TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                    TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                    TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
        auto parsed = parser->parseFromFile(DPstereo_config_.onnx_file.c_str(),
                                            static_cast<int>(gLogger.getReportableSeverity()));
        if (!parsed) 
        {
            printf("build engine failed! %d\n", succ);
            return false;
        }
        // 光流模型用于数据存储的内存最多是5G
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1073741824*2);
        config->setFlag(BuilderFlag::kFP16);
        // dla_core在config文件中的给定值是-1，意味这不启用DLA？
        // enableDLA(builder.get(), config.get(), yolov8_seg_config_.dla_core);
        return true;
    }

    bool DPStereo::infer(cudaEvent_t stop_infer_depth, cudaEvent_t stop_cpy_input_depth) 
    {
        if (stop_cpy_input_depth != nullptr)
            // infer in stream_ won't start until input copy is finished
            // checkCudaRuntime(cudaStreamWaitEvent(stream_cpy_, stop_cpy_input_depth));
            checkCudaRuntime(cudaEventSynchronize(stop_cpy_input_depth));

        //！ 从TensorRT8.5开始enqueueV2()开始被弃用了！
        // 这里不让gpu阻塞cpu，是因为此推理线程为cpu上的子线程。executeV2函数则默认是同步执行，即该stream执行任务时会阻塞host线程
        bool status = context_->enqueueV3(stream_);
        // bool status = context_->executeV2(buffers.getDeviceBindings().data());
        if (!status) {
            return false;
        }
        
        if (stop_infer_depth != nullptr)
            checkCudaRuntime(cudaEventRecord(stop_infer_depth, stream_));
        cout << "Start infer DP-Stereo!" << endl;
        return true;
    }

    // 深度模型的图像预处理步骤是在cpu上完成，再将预处理结果从cpu复制到host再传输到device
    bool DPStereo::preprocess_input(const Mat &image_left, const Mat &image_right, cudaEvent_t stop_cpy_input_depth)
    {

        if(image_left.empty()){
            INFOE("Left image is empty");
            return false;
        }

        if(image_right.empty()){
            INFOE("Right image is empty");
            return false;
        }

        if (image_left.channels() != 3){
            INFOE("The raw left image should have 3 channels");
            return false;
        }

        if (image_right.channels() != 3){
            INFOE("The raw right image should have 3 channels");
            return false;
        }

        // 貌似context可以是公用的?只有同一个网络模型才能共享
        if (!context_) {
            context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
            if (!context_) 
            {
                return false;
            }
            bool status0 = context_->setInputShape(DPstereo_config_.input_tensor_names[0].c_str(), Dims4(1, 3, input_height_, input_width_));
            bool status1 = context_->setInputShape(DPstereo_config_.input_tensor_names[1].c_str(), Dims4(1, 3, input_height_, input_width_));
            assert(status0 && "Set Shape of Input0 failed!");
            assert(status1 && "Set Shape of Input1 failed!");
        }

        if (buffers == nullptr){
            buffers = make_shared<BufferManager>(engine_, 0, context_.get());
            for (int32_t i = 0; i < engine_->getNbIOTensors(); i++){
                auto const name = engine_->getIOTensorName(i);
                context_->setTensorAddress(name, buffers->getBindingDeviceBuffer(name));
            }
        }
        // cout << "Succeeded create buffer for DP-Stereo!" << endl;

        // 立体深度网络有两个输入，input0是左图像，input1是右图像，输出只有一个，即视差图（左图像+视差图=右图像）
        float* preproc_img0_host = (float*)buffers->getBindingHostBuffer(DPstereo_config_.input_tensor_names[0]);
        float* preproc_img1_host = (float*)buffers->getBindingHostBuffer(DPstereo_config_.input_tensor_names[1]);

        // 预处理之后的图像每个通道应该是float数据类型（是把亮度值都归一化到[0,1]了吗？）
        std::size_t size_preproc_img = input_width_ * input_height_ * 3 * sizeof(float);
        
        vector<float*> list_img_host = {preproc_img0_host, preproc_img1_host};
        vector<Mat> list_img_raw     = {image_left, image_right};

        int dims[3] = {3, input_height_, input_width_}; 
        Mat joined_left(3, dims, CV_32F);
        Mat joined_right(3, dims, CV_32F);
        vector<Mat> list_joined{joined_left, joined_right};

        vector<Mat> channels;
        Mat padded_img;
        for (int j = 0; j < list_img_raw.size(); j++){
            Mat padded_img_(input_height_, input_width_, CV_8UC3, Scalar(0, 0, 0));
            inputpadder.pad(list_img_raw[j], padded_img_);
            if (normalize_.channel_type == CUDAKernel::ChannelType::Invert){
                cvtColor(padded_img_, padded_img_, cv::COLOR_BGR2RGB);
            }
            
            // convertTo只能改变通道的位深，不能改变通道数。此函数支持in-place操作
            padded_img_.convertTo(padded_img, CV_32FC3);
            split(padded_img, channels);
            // cout << "Succeeded split pad image for DP-Stereo!" << endl;
            for (int i = 0; i < channels.size(); i++)
            {
                float* ptr = &list_joined[j].at<float>(i, 0, 0); // pointer to first element of slice i
                Mat destination(input_height_, input_width_, CV_32F, (void*)ptr);  // no reallocate, see the useage document
                if (normalize_.type == CUDAKernel::NormType::MeanStd){
                    channels[i] = (channels[i] * normalize_.alpha - normalize_.mean[i]) / normalize_.std[i];
                }
                else if (normalize_.type == CUDAKernel::NormType::AlphaBeta){
                    channels[i] = channels[i] * normalize_.alpha + normalize_.beta;
                }
                channels[i].copyTo(destination);
            }

            // merge后默认还是只能得到H*W*C顺序的Mat，而需要传给binding的是C*H*W的内存顺序的数据
            // merge(channels, padded_img);
            memcpy(list_img_host[j], list_joined[j].data, size_preproc_img);
            channels.clear();
        }

        // 第2个input是当前帧图像，第1个input是上一帧的
        // 可以自己一个个地进行复制，也可以给定index，使用已定义函数一次性复制多个
        // inputbinding的index不一定就比outputbinding的index小，所以这里要根据binding的名字来取index！
        // float* infer_input1_device_   = (float*)buffers->getBindingDeviceBuffer(rapidflow_config_.input_tensor_names[1]);
        // CHECK(cudaMemcpyAsync(infer_input1_device_, preproc_img_host, size_preproc_img, cudaMemcpyHostToDevice, stream_cpy_));

        std::vector<std::string> index_bindings_cpy = {DPstereo_config_.input_tensor_names[0], DPstereo_config_.input_tensor_names[1]};
        //！ 从host复制到device则可以使用与infer不同的stream，但是极端情况下可能出现infer开始了但是input还没传输到
        //! 函数里使用了显式stream进行数据传输，但是需要保证infer操作在复制结束后才执行，可以使用绑定在stream_cpy_上的cudavent
        buffers->copyInputToDeviceAsync(false, size_preproc_img, index_bindings_cpy, stream_cpy_);
        
        if (stop_cpy_input_depth != nullptr)
            checkCudaRuntime(cudaEventRecord(stop_cpy_input_depth, stream_cpy_));

        // cout << "Succeeded preprocess for DP-Stereo!" << endl;
        return true;
    }

    // 后处理函数.DP-Stereo的推理结果应该再乘以256才是最终的视察值？不是，原工作共乘以256是为了可视化
    bool DPStereo::postprocess_output(Mat &unpadded_depth_map, cudaEvent_t stop_infer_depth, cudaEvent_t stop_post_proc)
    {
        // post_proc of rapidflow in stream_cpy_ won't start until infer is finished
        // checkCudaRuntime(cudaStreamWaitEvent(stream_cpy_, stop_infer_depth));
        if(stop_infer_depth != nullptr)
            checkCudaRuntime(cudaStreamWaitEvent(stream_cpy_, stop_infer_depth));
            // checkCudaRuntime(cudaEventSynchronize(stop_infer_depth));
        
        std::size_t byteSize_disp_img = input_width_ * input_height_ * sizeof(float);
        // cout << "byte size of disp map: " << byteSize_disp_img << endl;
        // 这里的Async指的是不阻塞host线程！
        std::vector<std::string> index_bindings_cpy = {DPstereo_config_.output_tensor_names[0]};

        // 用于记录所有模型处理和推理的时间
        // 必须等到后处理结束才能往主线程共享host上后处理结果的地址（虽然此地址是不变的）
        if (stop_post_proc != nullptr)
        {
            buffers->copyOutputToHostAsync(false, byteSize_disp_img, index_bindings_cpy, stream_cpy_);
            checkCudaRuntime(cudaStreamSynchronize(stream_cpy_));
            checkCudaRuntime(cudaEventRecord(stop_post_proc, stream_cpy_));

            // 也可以选择stream来阻塞host，但是为了记录时间，所以上面选用了event来阻塞
            //checkCudaRuntime(cudaStreamSynchronize(stream_cpy_));
        }
        // 如果不选择用event来阻塞当前线程，则需要用stream！
        else
        {
            // 选择直接用默认stream来复制数据，则默认阻塞当前线程
            buffers->copyOutputToHost(false, byteSize_disp_img, index_bindings_cpy);
            // buffers->copyOutputToHostAsync(false, byteSize_disp_img, index_bindings_cpy, stream_cpy_);
            // checkCudaRuntime(cudaStreamSynchronize(stream_cpy_));
        }
        
        float* depth_ptr = (float*)buffers->getBindingHostBuffer(DPstereo_config_.output_tensor_names[0]);
        // 注意，DPStereo的推理结果是B*C*H*W,其中B=1，C=1。这里由于是单通道Mat，因此数据的排列数序和CV_32FC1的Mat（H*W*C）实际上是一样的，可以直接用此内存的数据来形成depth_map
        Mat padded_depth_map(input_height_, input_width_, CV_32FC1, (void*)depth_ptr);
        inputpadder.unpad(padded_depth_map, unpadded_depth_map);

        //！ 如果模型推理和后处理都是在子线程中进行，那么可以使用std::promise进行线程间的数据共享，在主线程中使用wait来进行阻塞
        //! 主线程应该对页锁内存上的数据进行深拷贝
        //pro_depth.set_value(depth_ptr);

        return true;
    }

    float* DPStereo::get_binding_output_device()
    {
        float* ptr = (float*)buffers->getBindingDeviceBuffer(DPstereo_config_.output_tensor_names[0]);
        return ptr;
    }

    void DPStereo::save_engine() {
        if (DPstereo_config_.engine_file.empty()) return;
        if (engine_ != nullptr) {
            nvinfer1::IHostMemory *data = engine_->serialize();
            ofstream file(DPstereo_config_.engine_file, ios::binary);
            if (!file) return;
            file.write(reinterpret_cast<const char *>(data->data()), data->size());
        }
    }

    bool DPStereo::deserialize_engine() {
        ifstream file(DPstereo_config_.engine_file.c_str(), ios::binary);
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
