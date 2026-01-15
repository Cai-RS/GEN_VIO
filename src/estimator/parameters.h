/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

// #include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

// 这些const的变量都是被用来定义数组的大小！因为定义数组大小只能用常量！
const int WINDOW_SIZE = 2;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

// 每一帧中最多保留的有效待匹配物体（去除非考虑类别的物体，去除bbox太小的物体）
const int MAX_NUM_OBJS_FRAME = 40;
// 要在先保留的物体跟踪的帧数（用于BEV）
// const int NUM_FRAME_TRACK_OBJS = 3;

// 每一个待匹配物体上的采样像素点数
const int NUM_SAMPLED_PIXEL_OBJ = 60;

// 每个小bloc中最终保留的跟踪点的最大数量
const int NUM_FEA_IN_BLOC = 14;
// 每个大bloc中最终保留的跟踪点的最大数量。为NUM_FEA_IN_BLOC的 2 - 4 倍
const int NUM_FEA_IN_BIG_BLOC = 40;

// 只在图像的下2/3（或1/2)部分获取检测新的FAST跟踪点
const int start_row_bloc = 2;   // 2  3

// 全局变量。
// 在此h文件中声明为extern全局变量，当此h文件被多个c文件include，则在多个cpp文件中都声明这些全局变量。只需要在其中任一个cpp文件定义这些变量，就可以被多个文件共享使用。
// 使用全局变量是比较危险的，尤其是在多线程系统！!!应该尽量避免！

// 窗口内每一帧的（相机）时间戳
extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern float Cam_H;
extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_RESULT_PATH_OBJS;
extern std::string OUTPUT_FOLDER;
extern std::string OUTPUT_FOLDER_OBJS;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
// 相机的像素焦距，即f*lambda_x，f_lambda_y，对于一般的相机，lambda_x = lambda_y
extern float FOCAL_LENGTH_X;
extern float FOCAL_LENGTH_Y;
extern float SHIFT_X;
extern float SHIFT_Y;
extern float mbf;

extern float Y_shift_right_image;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string img0_warm_up_gpu;
extern std::string img1_warm_up_gpu;
extern std::string img2_warm_up_gpu;
extern std::string img3_warm_up_gpu;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;

extern int MIN_DIST_BG;
extern int MIN_DIST_OBJ;
extern int REJECT_WITH_F;
extern double F_THRESHOLD;
extern double H_THRESHOLD;
extern double Th_score;
extern int Cal_FH_after_IMU_init_succ;
extern int SHOW_TRACK;
extern int FLOW_BACK;

// NCC cal
extern int sort_by_NCC;
extern int Len_edge_win;
extern int check_detect_by_ambi_NCC;
extern int check_match_by_ambi_NCC;
extern int sort_all_sift_FAST;
extern int refine_matching_flow;
extern int refine_matching_stereo;

extern float mThDepthBg;
extern float mThDepthObj;
extern float mMinDepthPt;
extern float mDepthMapFactor;
extern int border_x;
extern int border_y;
extern int MAX_CNT_PTS_BG;
extern int MIN_CNT_PTS_OBJ;
extern int MIN_CNT_PTS_TRACK_BG;
extern int MIN_CNT_PTS_TRACK_OBJ;
extern int MAX_CNT_PTS_TRACK_BG;
extern int MAX_CNT_PTS_TRACK_OBJ;
extern float AVE_DIST_3D_PTS_THRES;

extern bool has_stereo_rectified;

extern int TH_NUM_FRAME_FOR_LBA;
extern int Min_num_old_track_per_frame;
extern int Use_LBA_for_puer_V;
extern int Th_num_fea_for_LBA_pure_V;

extern int Thres_num_track_cur;

extern float Thres_Ambiguity_Flow;
extern float Thres_Ambiguity_Stereo;

extern int use_motion_to_pred_fea_pos;
extern int use_motion_to_pred_fea_dep;

extern int use_pnp_after_imu_init;

extern int PnP_per_frame;

extern float Min_dist_flow;

extern float Th_epipolar_con;

extern float Th_homography_con;

extern int Check_flow_with_pred_motion;

extern int Min_num_bg_track_with_dep_prev;

extern int Limit_num_static_track;

extern int retain_marg_info;

extern int Use_5_pts;

extern int Res_non_planar_pt;

extern int Use_tria_for_2d2d;

extern int Cal_cur_dep_by_motion;

extern int Trust_dep_from_motion;

extern int Check_dep_with_reproj_err;

extern int Use_pred_dep_to_find_stereo_mtach;

extern float Th_dep_sta_obj_fea_to_add;

extern float Base_max_th_ambi_NCC;

extern int Min_total_near_3D2D_track;
extern int Min_total_3D2D_track;

extern Eigen::Matrix3d K;
extern Eigen::Matrix3d K_trans;
extern Eigen::Matrix3d K_inv;
extern Eigen::Matrix3d K_trans_inv;

extern int use_gt_to_show_match;

extern int trans_result_format;
extern int evaluate_reslut;
extern int plot_line;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
