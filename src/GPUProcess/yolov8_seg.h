//
// Created by ruishengcai on 2023/12/18.
//

#ifndef YOLOV8_SEG_H_
#define YOLOV8_SEG_H_

#include <vector>
#include <cstring>
#include <cstddef>
#include <unistd.h>
//#include <string>
#include <memory>
#include <Eigen/Core>
#include <future>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <preprocess_kernel.cuh>
#include <cuda_tools.hpp>

//#include "Thirdparty/TensorRTBuffer/include/buffers.h"
#include "../TensorRT/include/buffers.h"
#include "read_configs.h"

// https://github.com/ultralytics/ultralytics
namespace YoloV8
{
    using namespace std;
    using tensorrt_common::TensorRTUniquePtr;
    using namespace tensorrt_buffer;

    struct InstanceSegmentMap {
    int width = 0, height = 0;      // width % 8 == 0
    int left = 0, top = 0;          // 160x160 feature map
    unsigned char *data = nullptr;  // is width * height memory

    InstanceSegmentMap(int width, int height);
    virtual ~InstanceSegmentMap();
    };

    struct Box {
    float left, top, right, bottom, confidence;
    // float left_bbox_pad_img, top_bbox_pad_img;
    int class_label;
    int id;
    // std::shared_ptr<InstanceSegmentMap> seg;  // mask

    Box() = default;
    Box(float left_, float top_, float right_, float bottom_, float confidence_, int class_label_)
        : left(left_),
            top(top_),
            right(right_),
            bottom(bottom_),
            confidence(confidence_),
            class_label(class_label_) 
            {
                // left_bbox_pad_img = 0.0;
                // top_bbox_pad_img = 0.0;
            }
    };
    typedef std::vector<Box> BoxArray;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    // 计算用于图像缩放填充处理的仿射矩阵
    // 此仿射矩阵默认用于KITTI数据集，固定缩放后的图片尺寸为1248*384，且默认原图像与目标图像尺寸缩放比例为1.0，只需要进行填充
    // 此仿射矩阵struct定义在各个模型的h文件中，因为各个模型所需的预处理方式可能是不同的！
    struct AffineMatrix{
        //int和float数据一样占据4字节。对于yoloseg模型而言，要去在原图像尺寸基础上向上取最小的32的整倍数，因此系数直接都取int型即可。
        // 仿射矩阵只需要2*3=6个元素，但是为了读取内粗方便，这里定义了8个数据用于凑32个字节的内存！因此最后两个元素都直接赋为0或0.0！
        float i2d[8];       // image to dst(network), 2x3 matrix
        float d2i[8];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to);

        cv::Mat i2d_mat();
    };

    class Yoloseg 
    {
    public:

        int input_width_  = 1248;
        int input_height_ = 384;
        int left_pad = 0;
        int top_pad  = 0;

        // explicit关键字表示此函数的参数禁止隐式转换
        // 传入共用的stream用于多个模型的同步推理
        Yoloseg(const YolosegConfig &yolov8_seg_config, const cudaStream_t stream_common, 
                            const cudaStream_t stream_cpy_post = nullptr);

        ~Yoloseg();

        bool build();

        void save_engine();

        bool deserialize_engine();

        bool preproc_infer(const cv::Mat &image, cudaEvent_t stop_infer_seg = nullptr);

        bool postprocess_output(map<int, YoloV8::Box> &seg_map, set<uchar> &solid_obj_cls, set<uchar> &deform_obj_cls, 
                                set<uchar> &small_solid_objs, cudaEvent_t stop_infer_seg, cudaEvent_t stop_post_proc = nullptr);

        bool sample_pixel(float* array_pixel_objs, float* depth_map_device, float* buffer_device_pixel, map<int, YoloV8::Box> &masked_bboxes_seg_map, 
                            vector<int> &valid_objs, int W_dep_map, int H_dep_map, float left_shift_dep, float top_shift_dep, bool &done_sample, bool &check_total_lost_obj);
        //void visualization(const std::string &image_name, const cv::Mat &image, BoxArray &boxes);

        // 类外只能通过类的public函数来间接地访问类内的private函数或成员
        cv::Size full_seg_map_size();
        // 函数尽量不要使用和某个变量完全一致的名字（这里是num_valid_objs），尤其是函数即没有参数，返回类型还与同名变量的类型一致
        int get_num_valid_objs();

        // 函数可以返回引用！
        AffineMatrix &get_affine_matrix();

        void* get_full_seg_map_host();

    private:
        YolosegConfig yolov8_seg_config_;
        nvinfer1::Dims input_dims_{};
        nvinfer1::Dims output1_dims_{};
        nvinfer1::Dims output2_dims_{};
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;

        bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                            TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                            TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                            TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;


        // 需要逐像素地计算，因此还是用GPU更快！
        // 形参要不要使用 const shared_ptr& 类型呢？这样可以既不增加计数（但是不知道指向的对象内存何时会被释放），也无法重新设置资源
        // 不能重设资源是指智能指针不能指向新的对象内存，但是原本指向的对象内存上的内容时可以改变的，这与shared_ptr<const T>要区分开
        // ?? 使用智能指针的引用作为参数时无法保证所指对象在函数运行期间一直都存在，这是很危险的！！！所以一定要在此类的析构函数中保证此成员函数结束运行之后再销毁所指对象
        bool preprocess_input(const cv::Mat &image);

        // 从C++11之后就可以在类的定义中直接为普通成员变量直接声明和定义！
        // ?? 这个stream应该被各个模型推理任务所共享吧？因为每个模型的推理是同步的，所以使用一个stream就够了
        cudaStream_t stream_          = nullptr;
        cudaStream_t stream_cpy_post_ = nullptr;
        bool common_stream_cpy_       = false;
        
        int raw_col = 0;
        int raw_row = 0;

        float confidence_threshold_   = 0;
        float nms_threshold_          = 0;
        int max_pre_nms_              = 1024;  // max number of decoded bbox for NMS
        int max_objects_              = 100;
        int NUM_BOX_ELEMENT_          = 8;    // left, top, right, bottom, confidence, class,
                                            // keepflag, row_index(output)
        int seg_map_width             = 312;
        int seg_map_height            = 96;
        float seg_map_scale_x         = 1.0;
        float seg_map_scale_y         = 1.0;

        int num_valid_objs            = 0;
        
        // 使用不同的stream来进行 数据传输 和 推理（包括预处理。推理和后处理）？没必要，每一帧需要复制的数据不是特别多...
        bool use_multi_stream_ = true;
        AffineMatrix aff_mat;
        CUDAKernel::Norm normalize_;
        // BoxArray postprocess_boxes;
        // 内存管理器，一个网络模型在整个slam系统运行器件只用一个管理器
        shared_ptr<BufferManager> buffers = nullptr;
        uchar* raw_input_device = nullptr;
        uint8_t* raw_image_device = nullptr;
        float* output_array_device = nullptr;
        float* output_array_host = nullptr;
        void* num_pre_deco_host = nullptr;
        // 此内存用于存储每个后处理得到的small seg map
        uint8_t* small_seg_map_device = nullptr;
        uchar* small_seg_map_host   = nullptr;
        size_t size_small_seg_map;
        
        // 此内存用于存储每个后处理得到的full seg map（padded图分辨率），为2通道的unsigned char，第一个通道表示该点的class（0 - 6），第2个通道表示该点所属物体的id（从1开始）
        // 另外还需要一个图像内存来临时记录padded图像上每个点属于物体的概率，其为遍历过的物体中点插值概率最大的值，用于往full_seg_map_device中对应点存放相应的class label和实例id
        // 这里不必单独分配三个单通道的图像或者一个三通道的图像 的内存，而是使用yoloseg用来传输raw image的host和device内存！
        // full_seg_device直接复用raw_input_device和raw_input_host的内存即可
        uint8_t* full_seg_map_device  = nullptr;
        void* full_seg_map_host = nullptr;
        size_t size_full_seg_map;
        
        std::shared_ptr<InstanceSegmentMap> small_seg;

        float* buffer_device_bbox_info = nullptr;
    };

}; // namespace YoloV8

typedef std::shared_ptr<YoloV8::Yoloseg> YolosegPtr;

#endif //YOLOV8_SEG_H_
