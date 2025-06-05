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
const int WINDOW_SIZE = 7;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

// 每一帧中最多保留的有效待匹配物体（去除非考虑类别的物体，去除bbox太小的物体）
const int MAX_NUM_OBJS_FRAME = 40;
// 要在先保留的物体跟踪的帧数（用于BEV）
// const int NUM_FRAME_TRACK_OBJS = 3;

// 每一个待匹配物体上的采样像素点数
const int NUM_SAMPLED_PIXEL_OBJ = 60;

const int NUM_FEA_IN_BLOC = 10;

// 全局变量。
// 在此h文件中声明为extern全局变量，当此h文件被多个c文件include，则在多个cpp文件中都声明这些全局变量。只需要在其中任一个cpp文件定义这些变量，就可以被多个文件共享使用。
// 使用全局变量是比较危险的！
// extern int WINDOW_SIZE;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

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
extern double F_THRESHOLD;
extern double Th_score;
extern int SHOW_TRACK;
extern int FLOW_BACK;

extern float mThDepthBg;
extern float mThDepthObj;
extern float mMinDepthPt;
extern float mDepthMapFactor;
extern int border_x;
extern int border_y;
extern int MAX_CNT_PTS_BG;
extern int MAX_CNT_PTS_OBJ;
extern int MIN_CNT_PTS_TRACK_BG;
extern int MIN_CNT_PTS_TRACK_OBJ;
extern int MAX_CNT_PTS_TRACK_BG;
extern int MAX_CNT_PTS_TRACK_OBJ;
extern float AVE_DIST_3D_PTS_THRES;

extern bool has_stereo_rectified;

extern int TH_NUM_FRAME_FOR_LBA;
extern int Thres_num_track_cur;

extern float Thres_Ambiguity_Flow;
extern float Thres_Ambiguity_Stereo;

extern int use_motion_to_pred_fea_pos;
extern int use_motion_to_pred_fea_dep;

extern int use_pnp_after_imu_init;

extern int PnP_per_frame;

extern float Min_dist_flow;

extern float Th_epipolar_con;

extern int Min_num_bg_track_with_dep_prev;

extern int Limit_num_static_track;

extern int use_Marg;

extern Eigen::Matrix3d K;
extern Eigen::Matrix3d K_trans;
extern Eigen::Matrix3d K_inv;
extern Eigen::Matrix3d K_trans_inv;

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
