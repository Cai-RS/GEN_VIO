/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
// #include <std_msgs/Header.h>
// #include <std_msgs/Float32.h>
#include <unistd.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "utils.h"
#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

#include "../GPUProcess/rapidflow.h"
#include "../GPUProcess/yolov8_seg.h"
#include "../GPUProcess/stereo_depth.h"
#include "../GPUProcess/read_configs.h"

#include "../CudaSift/Sift.h"

//#include "sophus/so3.hpp"       // SO(3)李群
//#include "sophus/se3.hpp"       // SE(3)李代数

using namespace nvinfer1;
class Estimator
{

  public:
    // Eigen中的内存对齐 https://github.com/growinguptogether/EigenDocInChinese/blob/master/StructHavingEigenMembers.md 
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Estimator();
    ~Estimator();
    // __attribute__((no_sanitize("address"))) 
    void GPU_init_build(std::string &img0_for_gpu_build, std::string &img1_for_gpu_build, std::string &img2_for_gpu_build, std::string &img3_for_gpu_build);

    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const NumFeaObjFrame &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void Get_Process_IMU(double cur_cam_time);
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    // void processImage(ObjFeaNumFrame image, const double header);
    void processImage(const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // add by CRS
    void Fea_Obj_Extract_Track(double &t_vio, bool &init_succ, double t, cv::Mat &left_img, cv::Mat &right_img, cv::Mat &l_gray_img, cv::Mat &r_gray_img);
    void set_mask_objs_prev(double cur_time);
    
    void build_seg_map(const int &width, const int &height);
    void assign_sift_FAST(double &dt);
    void objs_matching(double dt, const vector<Vector3d> &Ps = vector<Vector3d>(), const vector<Matrix3d> &Rs = vector<Matrix3d>());

    // __attribute__((no_sanitize("address")))
    void sample_pixel_objs();

    // __attribute__((no_sanitize("address")))
    void GPUProcesImage(float &time_calcu);

    void CPU_Track_FAST(double _cur_time);
    
    void parallel_pose_objs_est(int prev_td, int cur_td_old, Matrix3d RCam_1, Vector3d PCam_1, Matrix3d RCam_2, Vector3d PCam_2, int num_inliers_PnP, bool initial_succ_prev = false);
    
    void intepolate_pose(int index_frame, double t1, double t2, double t_new, Matrix3d &R_new, Vector3d &P_new);
    void velocity_from_poses(const Matrix3d &R1, const Vector3d &p1, const Matrix3d &R2, const Vector3d &p2, const Matrix3d &gl_trans_R, const Vector3d &gl_trans_P, double &t, Eigen::Vector3d &l_vel, Eigen::Vector3d &ang_vel);
    bool pred_pose_with_vel(const Eigen::Matrix3d &R_1, const Eigen::Vector3d &p_1, const Eigen::Vector3d &ang_vel, const Eigen::Vector3d &l_vel, double t, Eigen::Matrix3d &R_2, Eigen::Vector3d &p_2);
    void updatePoseTransObjs();

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign(bool ForStereo = false);
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void get_dyn_objs_initial_pose_trans(vector<Vector3d> &delta_P, vector<Matrix3d> &delta_R);
    bool SolveObjPoseTransByPnP(const FeaFrame &fea_tracked_obj, Vector3d &delta_P, Matrix3d &delta_R, vector<float*> &pixel_lost_objs, const int &valid_objs, Vector3f &ave_3D_pts_obj, int gl_obj_id = 0);
    Vector3f sample_pixel_for_lost_obj(float* ptr_pix_prev, float* ptr_pix_cur, int &num_pixel, Vector3d &delta_P, Matrix3d &delta_R, bool cal_ave_3d_pts = false);
    
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T, bool pose_cam = false);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T, bool pose_cam = false);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
    void write_result_objs(FILE* outFile_cam, FILE* outFile_objs);
    
    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    cv::Mat _img, _img1, l_img_gray, r_img_gray;

    bool _shutdown;
    bool first_win, initial_succ_first_win;
    bool use_mask_img_for_sift;
    // 是否需要估计初始帧的位姿（即g在初始帧坐标系下的方向），如果汽车是在平地上行使，则可以不需要。
    // 但其实还需要估计初始帧的陀螺仪的bias。
    // 如果汽车是从静止开始的，则可以从静止状态完成初始化（估计初始帧的位姿），IMU和相机之间的标定一般需要线下完成
    bool NEED_ESTI_G;
    bool pred_cam_pose_with_IMU;
    bool can_change_seneor_type;
    bool done_cam_motion_pred;

    // 所有与跟踪相关的CPU线程上使用变量的设置 都需要对 mProcess加锁
    std::mutex mProcess;
    // GPU线程锁，可能需要对GPU推理所使用的一些参数进行改变，需要确保GPU中的工作已结束
    std::mutex mGPU;
    // 所有对测量队列（包括IMU，图像帧(包含特征点和检测对象，深度图））
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<RawImageData> raw_image_buffer;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<double> featureBuf;
    // This is estimated trigger time points for IMU
    double prevTime, curTime;
    int cur_img_rows, cur_img_cols;
    double delta_T_cam_prev, delta_T_cam_new, delta_T_imu_prev;
    // trigger time points for camera
    double prevTime_cam, curTime_cam;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    // GPU-related var
    double all_Time_GPU;

    cudaStream_t common_infer_stream;
    cudaStream_t cpy_post_stream;

    cudaEvent_t start_preproc_seg, stop_infer_seg, stop_post_seg, stop_cpy_input_flow, stop_infer_flow, stop_post_flow, stop_cpy_input_depth, stop_infer_depth, stop_post_depth;
    
    SiftConfig sift_config;
    YolosegConfig yolov8_seg_config;
    RapidflowConfig rapidflow_config;
    DPstereoConfig DPstereo_config;

    SiftPtr _sift_extract_match;
    YolosegPtr _yolov8seg;
    RapidflowPtr _rapidflow;
    DPStereoPtr _DPstereo;

    set<uchar> solid_obj_cls, deform_obj_cls, small_solid_objs;
    bool end_seg_post;
    bool end_flow_post;
    bool end_sift_post;
    bool end_stereo_post;
    bool end_FAST_track;
    bool done_sample;
    
    std::vector<int> ID_valid_match_stereo;
    std::vector<int> ID_valid_match_flow;
    // 上一帧和当前帧左图像中的各个对象的局部id，以及它们的bbox信息以及其small_mask_map
    std::map<int, YoloV8::Box> bbox_mask;
    Mat full_seg_map, full_seg_map_prev;
    Mat cls_map, id_map;
    int num_objs_frame, num_objs_prev_frame, num_solid_obj_frame, num_solid_obj_prev_frame;
    // 上一帧和当前帧的立体图像对的视差图
    Mat map_depth, map_depth_prev;
    // 当前帧与上一帧的光流图。（表示的都是 后一帧坐标-前一帧坐标）
    Mat map_flow;

    // purely CPU-related var
    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];
    
    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    // 当前帧和上一帧的 相机与最近的IMU的真实时间戳差值
    double td, prev_td;
    float prev_height;

    vector<int> id_frame_const_pose;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    // 保存上一帧中图像所对应的真实时间戳的相机估计位姿，因为上一帧的一开始td和滑窗估计的TD是不同的，因此认为上一帧估计出Rs对应的IMU时刻与图像时刻不是一致的，其时间差就是优化前后TD值的变化
    Matrix3d prev_cam_R, cur_cam_R;
    Vector3d prev_cam_P, cur_cam_P;

    Vector3d cam_motion_P_pred, cam_motion_P_pred_w;
    Matrix3d cam_motion_R_pred, cam_motion_R_pred_w;

    Vector3d prev_cam_P_using_imu[2];
    Matrix3d prev_cam_R_using_imu[2];
    
    // 这里定义的是速度变量，只是普通的李代数而已，所以用3维向量来存放即可
    Vector3d const_l_vel, const_ang_vel;

    // Sophus::SE3d const_vel;

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)] = {};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
    double last_t_prev, last_t_cur;
    Vector3d last_acc_prev, last_gyr_prev, last_acc_cur, last_gyr_cur;

    vector<Matrix3d> all_cam_R_before_init;
    vector<Vector3d> all_cam_P_before_init;

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    std::thread obj_motion_esti;
    // std::mutex fea_map_mutex;
    bool map_fea_optimized, map_fea_writen, tracker_pts_updated;

    //ObjFeaFrame Vec_Ptr_FeaObjFrame;
    SiftFrame siftfeaFrame;
    FASTFrame FastFeaFrame;
    
    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info = nullptr;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration = nullptr;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    bool initThreadFlag;

    vector<int> reserve_new_sift, reserve_new_FAST, direct_erase_fea;
    // state var for objs in lateset sliding window
    // 对于动态物体，最多只在线保留其在最近4帧中的观测和3次的位姿变换估计（用于BEV中的物体3D检测），同时逐帧地把更正后（使用LBA之后的最新两帧的相机位姿变换）的物体位姿存储到文件中
    vector<Vector3d> delta_P_objs, Vs_objs;
    vector<Matrix3d> delta_R_objs;
};
