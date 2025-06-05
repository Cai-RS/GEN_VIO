//
// Created by ruishengcai on 2023/12/20.
//

#ifndef RAPIDFLOW_H_
#define RAPIDFLOW_H_

#include <vector>
//#include <string>
#include <cstring>
#include <memory>
#include <Eigen/Core>
#include <future>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <preprocess_kernel.cuh>
#include <cuda_tools.hpp>

// #include "Thirdparty/TensorRTBuffer/include/buffers.h"
#include "../TensorRT/include/buffers.h"
#include "read_configs.h"

namespace RapidFlow
{

    using namespace std;
    using tensorrt_common::TensorRTUniquePtr;
    using namespace tensorrt_buffer;
    using namespace cv;

    // 此仿射矩阵struct定义在各个模型的h文件中，因为各个模型所需的预处理方式可能是不同的！
    struct AffineMatrix{
        //int和float数据一样占据4字节。对于yoloseg模型而言，要去在原图像尺寸基础上向上取最小的32的整倍数，因此系数直接都取int型即可。
        // 仿射矩阵只需要2*3=6个元素，但是为了读取内粗方便，这里定义了8个数据用于凑32个字节的内存！因此最后两个元素都直接赋为0或0.0！
        float i2d[8];       // image to dst(network), 2x3 matrix
        float d2i[8];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to);

        cv::Mat i2d_mat();
    };

    struct InputPadder{
        // top, bottom, left, right
        int Padder[4];

        int h_raw, w_raw;
        string pad_mode = "replicate";
        
        void pad(const Mat &raw_img, Mat &padded_img, const string mode = "replicate");
        
        void unpad(const Mat &padded_img, Mat &orig_img);
    };

    // https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/rapidflow
    class Rapidflow {
    public:
        // explicit关键字表示此函数的参数禁止隐式转换
        //explicit Rapidflow(const RapidflowConfig &rapidflow_config, const cudaStream_t stream_common);

        // 传入共用的stream用于多个模型的同步推理
        explicit Rapidflow(const RapidflowConfig &rapidflow_config, cudaStream_t stream_common, cudaStream_t stream_cpy = nullptr);

        ~Rapidflow();

        bool build();

        bool infer(cudaEvent_t stop_cpy_input_flow, cudaEvent_t stop_infer_flow);

        bool preprocess_input(const Mat &image, cudaEvent_t stop_cpy_input_flow = nullptr);

        bool postprocess_output(Mat &unpadded_flow_map, 
                                const bool &copy_infer_output = true,
                                cudaEvent_t stop_infer_flow = nullptr,
                                cudaEvent_t stop_post_proc = nullptr);

        float* get_binding_input_cur_device();

        // void visualization(const std::string &image_name, const cv::Mat &image);

        void save_engine();

        bool deserialize_engine();

    private:
        RapidflowConfig rapidflow_config_;
        nvinfer1::Dims input1_dims_{};
        nvinfer1::Dims input2_dims_{};
        nvinfer1::Dims output_dims_{};

        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;

        bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                            TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                            TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                            TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;
        
        // 这个stream应该被各个模型推理任务所共享吧？因为每个模型的推理是同步的，所以使用一个stream就够了
        cudaStream_t stream_        = nullptr;
        cudaStream_t stream_cpy_    = nullptr;
        bool common_stream_infer_   = false;
        bool common_stream_cpy_     = false;
        int input_width_            = 1248;
        int input_height_           = 384;
        bool use_multi_stream_      = true;
        float* cur_input_buffer_device = nullptr;
        InputPadder inputpadder;
        // 仿射矩阵和InputPadder两者选择其一就可以了
        AffineMatrix aff_mat;
        CUDAKernel::Norm normalize_;
        // 内存管理器，一个网络模型在整个slam系统运行器件只用一个管理器
        shared_ptr<BufferManager> buffers = nullptr;
    };

} // namespace RapidFlow

typedef std::shared_ptr<RapidFlow::Rapidflow> RapidflowPtr;

#endif  // RAPIDFLOW_H_