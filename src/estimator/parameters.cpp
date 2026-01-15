/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

// 代表的是从将点 从IMU坐标系下的表示 转换到 cam坐标系下的表示
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

// int WINDOW_SIZE;
float Cam_H;
double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string VINS_RESULT_PATH_OBJS;
std::string OUTPUT_FOLDER;
std::string OUTPUT_FOLDER_OBJS;
std::string IMU_TOPIC;
int ROW, COL;
float FOCAL_LENGTH_X;
float FOCAL_LENGTH_Y;
float SHIFT_X;
float SHIFT_Y;
float mbf;

float Y_shift_right_image;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
std::string img0_warm_up_gpu;
std::string img1_warm_up_gpu;
std::string img2_warm_up_gpu;
std::string img3_warm_up_gpu;

int MAX_CNT;

int MIN_DIST_BG;
int MIN_DIST_OBJ;
int REJECT_WITH_F;
double F_THRESHOLD;
double H_THRESHOLD;
double Th_score;
int Cal_FH_after_IMU_init_succ;
int SHOW_TRACK;
int FLOW_BACK;

int sort_by_NCC;
int Len_edge_win;
int check_detect_by_ambi_NCC;
int check_match_by_ambi_NCC;
int sort_all_sift_FAST;
int refine_matching_flow;
int refine_matching_stereo;

float mThDepthBg;
float mThDepthObj;
float mMinDepthPt;

float mDepthMapFactor;

int border_x;
int border_y;

int MAX_CNT_PTS_BG;
int MIN_CNT_PTS_OBJ;
int MIN_CNT_PTS_TRACK_BG;
int MIN_CNT_PTS_TRACK_OBJ;
int MAX_CNT_PTS_TRACK_BG;
int MAX_CNT_PTS_TRACK_OBJ;
float AVE_DIST_3D_PTS_THRES;

bool has_stereo_rectified;

int TH_NUM_FRAME_FOR_LBA;
int Min_num_old_track_per_frame;
int Use_LBA_for_puer_V;
int Th_num_fea_for_LBA_pure_V;

int Thres_num_track_cur;

float Thres_Ambiguity_Flow;
float Thres_Ambiguity_Stereo;

int use_motion_to_pred_fea_pos;
int use_motion_to_pred_fea_dep;

int use_pnp_after_imu_init;
int PnP_per_frame;

float Min_dist_flow;
float Th_epipolar_con;
float Th_homography_con;
int Check_flow_with_pred_motion;

int Min_num_bg_track_with_dep_prev;

int Limit_num_static_track;

int retain_marg_info;

int Use_5_pts;

int Res_non_planar_pt;

int Use_tria_for_2d2d;

int Cal_cur_dep_by_motion;

int Trust_dep_from_motion;

int Check_dep_with_reproj_err;

int Use_pred_dep_to_find_stereo_mtach;

float Th_dep_sta_obj_fea_to_add;

float Base_max_th_ambi_NCC;

int Min_total_near_3D2D_track;
int Min_total_3D2D_track;

Eigen::Matrix3d K;
Eigen::Matrix3d K_trans;
Eigen::Matrix3d K_inv;
Eigen::Matrix3d K_trans_inv;

int use_gt_to_show_match;

// post-process (visualization)
int trans_result_format;
int evaluate_reslut;
int plot_line;

// template <typename T>
// T readParam(ros::NodeHandle &n, std::string name)
// {
//     T ans;
//     if (n.getParam(name, ans))
//     {
//         ROS_INFO_STREAM("Loaded " << name << ": " << ans);
//     }
//     else
//     {
//         ROS_ERROR_STREAM("Failed to load " << name);
//         n.shutdown();
//     }
//     return ans;
// }

void readParameters(std::string config_file)
{
    
    // FILE *fh = fopen(config_file.c_str(),"r");
    // if(fh == NULL){
    //     // ROS_WARN("config_file dosen't exist; wrong config_file path");
    //     // ROS_BREAK();
    //     printf("config_file dosen't exist; wrong config_file path\n");
    //     abort();
    //     return;          
    // }
    // fclose(fh);
    
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    printf("Start reading param!\n");
    
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;

    string img0_warm_up, img1_warm_up, img2_warm_up, img3_warm_up;
    fsSettings["image0_warm_up"] >> img0_warm_up;
    fsSettings["image1_warm_up"] >> img1_warm_up;
    fsSettings["image2_warm_up"] >> img2_warm_up;
    fsSettings["image3_warm_up"] >> img3_warm_up;

    img0_warm_up_gpu = img0_warm_up + ".png";
    img1_warm_up_gpu = img1_warm_up + ".png";
    img2_warm_up_gpu = img2_warm_up + ".png";
    img3_warm_up_gpu = img3_warm_up + ".png";
    
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FOCAL_LENGTH_X = fsSettings["focal_length_x"];
    FOCAL_LENGTH_Y = fsSettings["focal_length_y"];
    SHIFT_X = fsSettings["shift_x"];
    SHIFT_Y = fsSettings["shift_y"];
    Y_shift_right_image = fsSettings["Y_shift_image_1"];
    printf("ROW: %d COL: %d\n", ROW, COL);
    // ROS_INFO("ROW: %d COL: %d ", ROW, COL);
    
    int has_stetreo = fsSettings["has_stereo_rectified"];
    if(has_stetreo) 
        has_stereo_rectified = true;
    else
        has_stereo_rectified = false;

    MAX_CNT = fsSettings["max_cnt"];

    // MIN_DIST = fsSettings["min_dist"];
    MIN_DIST_BG = fsSettings["min_dist_bg"];
    MIN_DIST_OBJ = fsSettings["min_dist_obj"];
    REJECT_WITH_F = fsSettings["reject_with_F"];
    F_THRESHOLD = fsSettings["F_threshold"];
    H_THRESHOLD = fsSettings["H_threshold"];
    Th_score = fsSettings["th_score"];
    Cal_FH_after_IMU_init_succ = fsSettings["cal_FH_after_IMU_init_succ"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    Len_edge_win = fsSettings["len_edge_win"];
    sort_by_NCC = fsSettings["sort_by_NCC"];
    check_detect_by_ambi_NCC = fsSettings["check_detect_by_ambi_NCC"];
    check_match_by_ambi_NCC = fsSettings["check_match_by_ambi_NCC"];
    sort_all_sift_FAST = fsSettings["sort_all_sift_FAST"];
    refine_matching_flow = fsSettings["refine_matching_flow"];
    refine_matching_stereo = fsSettings["refine_matching_stereo"];
    
    mThDepthBg = (float)fsSettings["ThDepthBG"];
    mThDepthObj = (float)fsSettings["ThDepthOBJ"];
    mMinDepthPt = (float)fsSettings["MinDepthPt"];
    mbf = (float)fsSettings["Camera_bf"];

    mDepthMapFactor = (float)fsSettings["DepthMapFactor"];

    border_x = (int)fsSettings["Border_width"];
    border_y = (int)fsSettings["Border_height"];

    MAX_CNT_PTS_BG = (int)fsSettings["max_cnt_pts_bg"];
    MIN_CNT_PTS_OBJ = (int)fsSettings["min_cnt_pts_obj"];
    MIN_CNT_PTS_TRACK_BG = (int)fsSettings["min_cnt_pts_track_bg"];
    MIN_CNT_PTS_TRACK_OBJ = (int)fsSettings["min_cnt_pts_track_obj"];
    MAX_CNT_PTS_TRACK_BG = (int)fsSettings["max_cnt_pts_track_bg"];
    MAX_CNT_PTS_TRACK_OBJ = (int)fsSettings["max_cnt_pts_track_obj"];

    AVE_DIST_3D_PTS_THRES = (float)fsSettings["ave_dist_3d_pts_thres"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }
    
    // WINDOW_SIZE = fsSettings["window_size"];
    TH_NUM_FRAME_FOR_LBA = fsSettings["th_num_frame_for_LBA"];
    Min_num_old_track_per_frame = fsSettings["min_num_old_track_per_frame"];
    Use_LBA_for_puer_V = fsSettings["use_LBA_for_puer_V"];
    Th_num_fea_for_LBA_pure_V = fsSettings["th_num_fea_for_LBA_pure_V"];

    Thres_num_track_cur = fsSettings["thres_num_track_cur"];

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    // 像素平面上的帧间点匹配视差的阈值
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    // 归一化平面上的帧间点匹配视差的阈值
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH_X;

    Thres_Ambiguity_Flow = fsSettings["thres_ambiguity_flow"];
    Thres_Ambiguity_Stereo = fsSettings["thres_ambiguity_stereo"];

    use_motion_to_pred_fea_pos = (int)fsSettings["use_motion_to_pred_fea_pos"];
    use_motion_to_pred_fea_dep = (int)fsSettings["use_motion_to_pred_fea_dep"];

    use_pnp_after_imu_init = fsSettings["use_pnp_after_imu_init"];
    PnP_per_frame = fsSettings["pnp_per_frame"];

    Min_dist_flow = fsSettings["min_dist_flow"];

    Th_epipolar_con = fsSettings["th_epipolar_con"];

    Th_homography_con = fsSettings["th_homography_con"];
    
    Check_flow_with_pred_motion = fsSettings["check_flow_with_pred_motion"];

    Min_num_bg_track_with_dep_prev = fsSettings["min_num_bg_track_with_dep_prev"];

    Limit_num_static_track = fsSettings["limit_num_static_track"];

    retain_marg_info = fsSettings["retain_marg_info"];

    Use_5_pts = fsSettings["use_5_pts"];
    
    Res_non_planar_pt = fsSettings["res_non_planar_pt"];

    Use_tria_for_2d2d = fsSettings["use_tria_for_2d2d"];

    Cal_cur_dep_by_motion = fsSettings["cal_cur_dep_by_motion"];

    Trust_dep_from_motion = fsSettings["trust_dep_from_motion"];

    Check_dep_with_reproj_err = fsSettings["check_dep_with_reproj_err"];

    Use_pred_dep_to_find_stereo_mtach = fsSettings["use_pred_dep_to_find_stereo_mtach"];

    Th_dep_sta_obj_fea_to_add = fsSettings["th_dep_sta_obj_fea_to_add"];

    Base_max_th_ambi_NCC = fsSettings["base_max_th_ambi_NCC"];

    Min_total_near_3D2D_track = fsSettings["min_total_near_3D2D_track"];
    Min_total_3D2D_track = fsSettings["min_total_3D2D_track"];
    
    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    fsSettings["output_path_objs"] >> OUTPUT_FOLDER_OBJS;
    VINS_RESULT_PATH_OBJS = OUTPUT_FOLDER_OBJS + "/vio.csv";
    std::cout << "result path for objs " << VINS_RESULT_PATH_OBJS << std::endl;
    std::ofstream fout_objs(VINS_RESULT_PATH_OBJS, std::ios::out);
    fout_objs.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        printf("have no prior about extrinsic param, calibrate extrinsic param\n");
        // ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        
        if(NUM_OF_CAM == 2)
        {
            RIC.push_back(Eigen::Matrix3d::Identity());
            TIC.push_back(Eigen::Vector3d::Zero());
        }
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    // 如果给定了IMU和相机的外参的标定结果
    else 
    {
        // 如果认为标定结果不够准确（或者认为外参在物体运动过程中会轻微改变？如果这样，则后续每次滑窗都需要估计外参），则需要在线优化估计外参
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            printf(" Optimize extrinsic param around initial guess!\n");
            // ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            printf(" fix extrinsic param!\n");
            // ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T_ic_0;
        cv::cv2eigen(cv_T, T_ic_0);
        Eigen::Matrix3d R_0 = T_ic_0.block<3, 3>(0, 0);
        Eigen::Vector3d P_0 = T_ic_0.block<3, 1>(0, 3);
        Eigen::Matrix3d R_ic = R_0.transpose();
        Eigen::Vector3d P_ic = -R_ic * P_0;
        RIC.push_back(R_ic);
        TIC.push_back(P_ic);
    }
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    Cam_H = fsSettings["cam_H"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T_ic_1;
        cv::cv2eigen(cv_T, T_ic_1);
        Eigen::Matrix3d R_1 = T_ic_1.block<3, 3>(0, 0);
        Eigen::Vector3d P_1 = T_ic_1.block<3, 1>(0, 3);
        Eigen::Matrix3d R_ic = R_1.transpose();
        Eigen::Vector3d P_ic = -R_ic * P_1;
        RIC.push_back(R_ic);
        TIC.push_back(P_ic);
    }
    
    INIT_DEPTH = fsSettings["init_depth"];
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        printf("Unsynchronized sensors, online estimate time offset, initial td: %f\n", TD);
        // ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        printf("Synchronized sensors, fix time offset: %f\n", TD);
        // ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    K << FOCAL_LENGTH_X, 0.0, SHIFT_X, 0.0, FOCAL_LENGTH_Y, SHIFT_Y, 0.0, 0.0, 1.0;
    K_inv = K.inverse();
    K_trans = K.transpose();
    K_trans_inv = K_trans.inverse();
    
    use_gt_to_show_match = fsSettings["use_gt_to_show_match"];

    trans_result_format = fsSettings["trans_result_format"];
    evaluate_reslut = fsSettings["evaluate_reslut"];
    plot_line = fsSettings["plot_line"];
    
    fsSettings.release();
}
