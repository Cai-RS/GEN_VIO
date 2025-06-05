/*******************************************************
 * Copyright (C) 2023, BMSTU
 * 
 * This file is part of OVIO.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef SIFT_H_
#define SIFT_H_

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_tools.hpp>
#include <NvInfer.h>

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudautils.h"
#include "../utility/tic_toc.h"
#include "read_configs.h"
#include <memory>
#include <numeric>
#include <set>
#include "../estimator/parameters.h"

using namespace std;
using namespace cv;

class Sift
{
    public:

        SiftConfig sift_config_;
        
        explicit Sift(const SiftConfig &sift_config, cudaStream_t stream_common, cudaStream_t stream_cpy = nullptr);

        ~Sift();

        void Allocate(int img_W, int img_H, bool host = true, int max_valid_num_match = 1024, bool use_masked_img = true);

        void deallocate();

        void preprocess_input(const Mat &image_left, const Mat &image_right, void* host_mask_objs = nullptr, int mask_img_W = 0, int mask_img_H = 0, int left = 0, int top = 0);

        void detect_match_sift(bool first_frame = false, bool use_masked_img = true);

        void postprocess(float max_disp_y, bool first_frame = false, bool use_masked_img = true);

        void select_stereo_matching(float max_disp_y);

        // 形参为二维数组时，必须给定第2维的大小，而第一维可以不用给定（有可能在索引时超出实际该数组的行数）
        void select_flow_matching(float max_disp_y, const Mat &seg_map_prev, const Mat &seg_map_cur, Mat &mask_bg_prev, int num_flow_pt_in_bloc[][6], int num_temp_flow_pt_in_bloc[][6], int num_long_track_FAST_in_bloc[][6], 
                                vector<Point2f> &FAST_prev, const vector<int> &cnt_tracked, vector<int> &temp_flow_pt_id, const Mat &prev_l_img = cv::Mat(), bool use_masked_img = true);
        
        void Postprocess();
        
        void pad(const cv::Mat &raw_img, Mat &padded_img, int* Padder, const string mode = "");

        void unpad(const cv::Mat &padded_img, cv::Mat &raw_img, const int raw_w, const int raw_h, const int* Padder);

        bool is_border_mask(float x, float y);

        void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, int * ValidPts, int NumValid);

        void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

        // 注意，下面这些成员变量，最好是放在private域内，即类外部不能直接访问，需要通过类的成员函数才能访问，这些可以保护这些变量（免得被其他用户以自己的方式私自修改这些变量）！
        // 这里只是为了方便暂时放在public域！
        // 各个图像中需要在host和device上保存的数据，包括padded后的图像，以及match后的sift点数据。单纯检测到的sift点数据在上面的各个SiftData中
        // 增加img4、img5和对应的siftData，用于只有valid detected objs的mask图像的sift检测和匹配
        // （这里img4和5是mask后的左图像，由于右图像没有进行yolo分割，因此没法单独在物体上检测sift。实际上如果算力充足，则可对右图像进行分割、mask和检测isft）
        CudaImage img1, img2, img3, img4, img5;

        SiftData siftData1, siftData2, siftData3, siftData4, siftData5;

        vector<int> flow_sort_pt_id, stereo_sort_pt_id, prev_stereo_sort_pt_id;
        vector<float> flow_sort_pt_ambi, stereo_sort_pt_ambi, prev_stereo_sort_pt_ambi;
        float *flow_dev_socre_id_match = nullptr;
        float *stereo_dev_socre_id_match = nullptr;
        // 保存上一帧的立体匹配的信息（从score到match_ypos等5个量）,xpos和ypos不需要，因为和上一帧的flow match共享一套检测点，保存id即可
        float *prev_stereo_match_info = nullptr;

        set<int> prev_flow_pt_with_stereo, invalid_stereo_match;

    private:
        // C++11之后类的普通成员变量（即非用户自定义类的对象）可以在声明时直接定义！
        bool memory_sift_freed      = true;
        cudaStream_t stream_        = nullptr;
        cudaStream_t stream_cpy_    = nullptr;
        bool common_stream_infer_   = false;
        bool common_stream_cpy_     = false;
        bool use_multi_stream_      = true;

        bool use_mask_img_for_sift_ = false;
        
        float *memoryTmp;
        // 这两个是被分配在不可分页内存上的两个图像，用于存放每一帧padded后的图像，其内存指针就是对应CudaImage中的h_data
        Mat limg_padded, rimg_padded, masked_limg_padded;
        Mat raw_mask_map;
        int img_W, img_H, raw_img_W, raw_img_H;
        
        int Padder[4]; 
        
        cv::Mat tracked_fea_prev_img, stereo_fea_prev_img;
    
};

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
double ScaleUp(CudaImage &res, CudaImage &src);

typedef std::shared_ptr<Sift> SiftPtr;

#endif  // SIFT_H_