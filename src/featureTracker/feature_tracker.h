/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
// #define NDEBUG
#include <assert.h>
#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <map>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../CudaSift/Sift.h"
#include "utils.h"
#include "hungarian_optimizer.h"
#include <omp.h>
#include <malloc.h>
#include <numeric>
#include <algorithm>
#include "../CudaSift/Sift.h"

#include "../GPUProcess/yolov8_seg.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

extern int NUM_THREADS;

extern int min_num_fea_track_obj;
extern int min_num_sift_track_obj;
extern int min_num_fea_track_bg;
extern int min_num_sift_track_bg;
extern int thres_num_fea_lose_obj;
// 对于当前帧的某个obj上的所有点，其所匹配的上一帧的物体中，匹配点最多的物体与第二多的物体间的匹配点数的比例，高于这个阈值则把当前帧物体匹配到最大匹配点数的上一帧物体
extern float thres_rel_ratio_match_1_2;

//bool inBorder(const cv::Point2f &pt);

// 此处是把函数模板的实现放在h文件中，好处是在cpp文件中调用此函数模板时可以按需实例化（显式或隐式），坏处是会把实现细节暴露给用户（因为h文件对用户可见）
// 其实可以把函数模板的声明和定义分离（声明在h文件，定义在某个cpp文件中并进行显式实例化声明，这样调用函数的cpp文件可以链接到此具体实例化的实现文件），就能避免用户知道具体的函数实现（把相关.cpp做成.lib或.dll）
template<typename T> void reduceVector(vector<T> &v, const vector<uchar> &status_)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status_[i])
            v[j++] = v[i];
    // resize会去除前j个元素之后的所有元素
    v.resize(j);
}

// void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status);

template<typename T>
inline int equals(T a, T b){
	return a == b;
}

template<typename T>
void calc_num_unique_value(vector<T> &src, vector<pair<T,int>> &dst)
{
    map<T,int> map_dst;
	for(int i = 0; i < src.size(); ++i)
    {
        // 这里用auto来给定iter的类别是最方便的，否则暂时不知道如何确切地定义其类别
        auto iter = map_dst.find(src[i]);
        if (iter != map_dst.end()) 
        {
            iter->second++;
        }
        else 
        {
            map_dst[src[i]] = 1;
        }
    }
    if (!dst.empty()) dst.clear();
    dst.assign(map_dst.begin(), map_dst.end());
}

template<typename T> void cal_centre_and_dist_pts(const vector<T> &pts, const set<int> &outliers, vector<float> &dist_pts, float factor_, bool has_cent, T &cent_pt);

template<typename T> float cal_rubust_norm_stderr_pts_dist(vector<float> &pts_dist, bool need_cal_dist, const vector<T> &pts, float factor_, set<int> &outliers_pts, bool has_cent, T &cent_pt, bool need_normalized = true, bool cal_std_dist = true);

void cal_MAD_value(vector<float> pts, float &dist_MAD, float &dist_median);

class FeatureTracker
{
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureTracker();
    ~FeatureTracker();
    void trackImage(int frame_count, const cv::Mat &_img, bool &end_flow_post, const cv::Mat &seg_map, bool &end_FAST_track, const cv::Mat &prev_dep_map = cv::Mat(), 
                    const cv::Mat &flow_map = cv::Mat(), const vector<Vector3d> &Ps = vector<Vector3d>(), const vector<Matrix3d> &Rs = vector<Matrix3d>());
    
    void reduce_invalid_fea(bool for_sift);

    void select_sift_V1(int frame_count, const cv::Mat &_img1, const cv::Mat &seg_map_prev, const cv::Mat &seg_map_cur, 
                    const cv::Mat &map_depth_prev, const cv::Mat &flow_map, bool &end_flow_post, bool inti_succ_IMU = false, bool use_mask_img = false);
    
    void select_sift_V2(int frame_count, const cv::Mat &_img1, const cv::Mat &seg_map_prev, const cv::Mat &seg_map_cur, 
        const cv::Mat &map_depth_prev, const cv::Mat &flow_map, bool &end_flow_post, bool inti_succ_IMU = false, bool use_mask_img = false);

    void det_new_FAST_objs(const int &frame_count, const int num_solid_obj, const cv::Mat &seg_map, const cv::Mat &cls_map, const cv::Mat &depth_map, bool initial_succ, const bool &marg_old_prev, bool &stereo_match_done, const cv::Mat &_img1 = cv::Mat());
    // 弃用
    void assign_FAST_objs(FASTFrame &FAST_frame, FeaObjFrame &TrackObjFeaFrame, FeaObjFrame &NewObjFeaFrame, vector<pair<int,int>> &num_obj_FAST);
    void assign_fea_objs(int frame_count, const Mat &dep_map, bool &end_flow_post, bool &end_stereo_post, const Mat &_img1, const vector<int> &valid_obj_id, const Mat &obj_id_map, map<int, YoloV8::Box> &bbox_mask);

    void draw_mask_fea_prev_obj(const double &cur_time);
    void setMask(bool initial_succ);
    // 新版的mask设置函数，不考虑是否使用IMU或者是否已经完成初始化
    void setMask(bool initial_succ, bool use_IMU);

    void set_new_fea_in_mask(bool initial_succ);
    
    void sorted_sift();
    
    void objs_matching_assign(int frame_count, double dt, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map_2, bool initial_succ,  
                                bool has_pred_motion_objs_cam = true, const vector<Vector3d> &Ps = vector<Vector3d>(), const vector<Matrix3d> &Rs = vector<Matrix3d>());
    
    bool check_match_two_objs(bool use_fea, bool cal_3D_pred_err, map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &fea_cur_obj, const int id_checked_prev_obj, const int id_checked_cur_obj,
                                map<int, pair<vector<Vec2f>,vector<Vec2f>>> &match_pts_of_prev_obj, map<int, pair<vector<float>,vector<float>>> &dep_match_pts_of_prev_obj,
                                vector<vector<int>> &final_assign_id_cur_objs, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map_2, const vector<Vector3d> &Ps, 
                                const vector<Matrix3d> &Rs, bool has_pred_motion, Vector3d &P_12, Matrix3d &R_12, bool &cur_sta_obj, float &ave_depth, 
                                map<int,vector<int>> &assign_prve_id, bool direct_erase_pts = false, bool cal_ave_depth_prev_pts = false);
                                
    // ave_depth_cur如果想要设置默认值，则前面必须要加const！！即缺省的引用只能用于读，不能写！！因为C ++不允许将临时量（例如参数的默认值）绑定到非const引用！
    bool check_stat_obj(bool use_fea, map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &fea_cur_obj, 
                                    const int id_checked_prev_obj, const vector<int> &matched_cur_objs, const Vector3d &P_12,
                                    const Matrix3d &R_12, float &ave_depth_cur, const cv::Mat &seg_map = cv::Mat(),
                                    const cv::Mat &flow_map = cv::Mat(), const cv::Mat &depth_map_2 = cv::Mat(), bool direct_erase_pts = true);

    int track_pixels_one_prev_obj(int prev_glob_obj_id, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map, 
                                    map<int, pair<vector<Vec2f>, vector<Vec2f>>> &match_pts, map<int, pair<vector<float>,vector<float>>> &match_pts_depth);

    // const map<int, YoloV8::BoxArray> &bbox_seg

    set<int> calcu_2d_3d_pts_dist(vector<float> &dist, bool pixel_pt, vector<Vec2f> &pts1, float &ave_depth_cur, vector<Vec2f> &pts2, bool cal_pts_var = false, 
                                bool cal_ave_3d_dist = false, const Vector3d &p_obj12 = Vector3d::Zero(), const Matrix3d &r_obj12 = Matrix3d::Identity(), 
                                const vector<float> &pts_depth_1 = vector<float>(), const vector<float> &pts_depth_2 = vector<float>(), const bool &cal_ave_depth_prev_pts = false);
    
    bool match_score_two_objs(const int id_checked_prev_obj, const vector<int> &id_checked_cur_objs, vector<float> &score_match, const cv::Mat &seg_map, const cv::Mat &flow_map, 
                                const cv::Mat &depth_map_2, const vector<Vector3d> &Ps, const vector<Matrix3d> &Rs);
    void Ptspredict_motion(const int &frame_count, const Matrix3d &RCam_motion, const Vector3d &PCam_motion, const Matrix3d &RCam_cur_pred, const Vector3d &PCam_cur_pred, bool pred_for_new_objs = true, bool use_motion = true);
    void Ptspredict_flow(const cv::Mat &flow_map, const cv::Mat &seg_map);
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    
    void UpdateCosts(const std::vector<std::vector<float>>& association_mat, SecureMat<float>* costs);
    
    int check_flow_with_F(const Matrix3d &F_cam, const Point2f &pt1, const Point2f &pt2);
    void DecomposeE(const Mat &E, Mat &R1, Mat &R2, Mat &t);
    // bool epipolarConstrain(const vector<Point2f> &kp1, const vector<Point2f> &kp2, const Eigen::Matrix3d& Mat_F, vector<uchar> &is_inlier);
    bool HomographyConstrain(vector<Point2f> &kp1, vector<Point2f> &kp2, const Eigen::Matrix3d& Mat_H, const Eigen::Matrix3d& Mat_H_inv, vector<uchar> &is_inlier, const float &Th_score, float &score, const float &Th_dist = 4.5);
    void rejectWithFV1(bool for_sift, bool inti_succ_IMU = false, const set<int> &pts_for_F = set<int>());
    void rejectWithFV2(bool inti_succ_IMU = false, const set<int> &pts_for_F = set<int>());
    // void recover_scale_t();
    
    // vector<cv::Point2f> undistortedPts(const vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    void undistortedPts(const vector<cv::Point2f> &pts, vector<cv::Point2f> &un_pts, camodocal::CameraPtr cam, const vector<uchar> &status_pts = vector<uchar>(), bool for_rigth_pts = false);
    // 直接在pts上进行修改，即像素点直接修改为归一化平面点
    void undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    void undistortedPts(vector<cv::Vec2f> &pts, camodocal::CameraPtr cam);
    void undistortedPts(const cv::Point2f &pts, cv::Point2f &un_pts, camodocal::CameraPtr cam);
    void spaceToPlane(const Eigen::Vector3d& P, Point2f& p, camodocal::CameraPtr cam);
    void ptsVelocity(vector<cv::Point2f> &vel_pts, vector<int> &ids, vector<cv::Point2f> &un_pts, map<int, cv::Vec4f> &prev_id_un_pts, 
                        bool cal_vel = true, bool cal_map = true, const vector<uchar> &status_pts = vector<uchar>(), bool for_right_pts = false);
    
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    float cal_dist_img_center_to_flow_line(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds, const vector<int> &reserve_sift, const vector<int> &reserve_FAST);
    void RemoveOutliers();
    void clear_var();
    cv::Mat getTrackImage();
    
    bool inBorder(const cv::Point2f &pt);
    
    int num_flow_pt_in_bloc[6][6];
    int num_temp_flow_pt_in_bloc[6][6];
    int num_long_track_FAST_in_bloc[6][6];
    int id_best_track_bloc[6][6];
    
    int total_frame;
    
    SiftPtr Sift_;

    // 根据规定的深度范围，计算出像素的平均立体匹配视差值，可以得到立体匹配中右图像点的初始值
    float ave_disp_obj, ave_disp_bg;
    
    // 特征点跟踪
    FeaObjFrame TrackObjFeaFrame;
    FeaObjFrame NewObjFeaFrame;
    // record the number of tracked and new detected sift points of every object (include the bg) in the currrent frame
    NumFeaObjFrame num_obj_sift, num_obj_FAST;

    // 物体跟踪匹配的相关变量
    // 静态跟踪点（包括纯背景点以及部分静态物体的特征点）
    FeaFrame TrackBgFea;
    FeaObjFrame FinalTrackObjFea;
    // 记录最终的上一帧的物体的关联情况，key为上一帧中的全局物体id，vector为与其关联的当前帧的临时物体id（其中还包含背景中漏检而在特征点关联中被发现的物体，其临时id为检测物体数递增）
    // 当前帧关联的物体最后还会包括被发现的漏检物体
    std::vector<pair<int,std::vector<int>>> FinalTrackObj;
    // 记录最终当前帧的所有有效（深度)的临时物体（不包含背景，也不包含之后发现的漏检物体）的关联情况，每个物体最多只与上一帧的一个非漏检物体相关联。如果有与上一帧背景的漏检点关联，则会记录在ParLostObjPrevBg或TotalLostObjPrevBg中
    std::map<int,int> FinalTrackCurObj;
    // 记录上一帧背景中哪些物体的部分（partial）或全部（total)被yolo漏检（通过与当前帧中的某个物体相关联来发现）
    // 部分漏检的上一帧物体的点是加入到FinalTrackObjFea对应的id物体中（方便位姿估计），这里只记录该物体id（后续从全局地图的背景中查找这些点）
    std::set<int> ParLostObjPrevBg;
    // 上一帧全部漏检的物体由于需要记录与当前帧所关联物体的局部id和特征点，因此采用FeaObjFrame数据类型。可以使用vector<map<int,vector<>>>吗？毕竟全漏检的物体应该不多？
    FeaObjFrame TotalLostObjPrevBg;
    // 记录在上一帧中漏检的物体（漏检但是通过特征点匹配找到了与其前一帧的关联物体）
    vector<int> detect_lost_objs_cur;
    // 上一帧中的物体在当前帧中最终没有完成关联的。这个有必要记录吗？该物体已经离开视野，那么就放弃对它的维护了吧？
    std::vector<int> FinalLostObjPrev;
    // 当前帧中的新物体的local id。对于当前帧中的新物体，直接把其上的所有跟踪特征点和新特征点都作为新的地图点！
    std::vector<int> new_objs_cur;
    // 上一帧与当前帧之间确定为静态物体的全局物体id（因为此时一定是完成了匹配，且上一帧的物体一定是静态物体）
    std::map<int,float> cur_stat_objs;
    // 记录当前帧所有完成关联的动态物体（包括那些上一帧是静态物体，但是在当前帧的静态检查中不通过的物体；不包括上一帧完全漏检的背景上的物体）
    // 注意，保存的id是FinalTrackObj中的序号，即所指的上一帧物体id是在FinalTrackObj第id个元素的first！
    std::set<int> cur_dyn_objs;
    // 记录当前帧背景中的漏检物体和对应的漏检点
    std::vector<int> lose_objs_cur_bg;
    // 记录当前背景中帧漏检的物体的被跟踪的特征点的id，与上面的lose_objs_cur_bg配对
    std::vector<std::vector<int>> fea_cur_lose_objs;
    // 记录当前帧中平均深度大于阈值的静态物体的全局id（指的是那些被当前帧漏检的物体，因为其大部分点可能是背景点，而背景点的深度是大于物体深度阈值的）
    set<int> invalid_stat_objs;
    // 记录当前帧背景中的跟踪点的最终匹配（上一帧）物体，方便后续对所有背景点修改信息或删除
    // map<int,int> match_prev_obj_cur_bg_fea;
    // 当前帧物体匹配阶段就确定为静态的物体，其上的只有两帧观测（上一帧的新点在当前帧跟踪到）的FAST点是否要提前加入静态点集参与相机运动估计（这其实只有在滑窗未满阶段才会需要在此阶段添加短期跟踪的点）
    set<int> added_short_track_Fea;
    // 当前帧中有效的检测物体（即最终会在当前帧中成为特征点数足够多 的 被跟踪物体，或者新物体，不包含当前帧完全漏检的物体）
    std::vector<int> valid_detect_obj;
    std::set<int> sta_obj_fea_in_map, sta_obj_fea_in_map_cur;
    
    bool check_total_lost_cur_objs;
    
    int num_sta_objs_found;
    set<int> id_gl_sta_obj;

    int frame_cnt;
    int row, col;
    // 左相机坐标系下的右相机视锥体的左侧面的方程，aX+bY+cZ+d=0。将左相机坐标系下估计的点（X，Y，Z）代入方程左边，如果结果小于0，则说明估计的点位于右相机视锥体之外
    double r_cam_3D_plane[4];
    float bg_left_border_left_img, obj_left_border_left_img, bg_right_border_right_img, obj_right_border_right_img;
    
    cv::Mat imTrack;
    cv::Mat tracked_fea_prev_img;
    cv::Mat mask_bg, mask_bg_prev, mask_bg_cur, mask_solid_objs, mask_prev_fea_objs;
    cv::Mat prev_mask_solid_objs;

    bool use_prev_fea;
    bool copy_mask_bg;
    bool done_select_sift, done_track_FAST;
    bool show_tracked_fea;
    bool has_lost_obj_prev;

    //cv::Mat fisheye_mask;
    cv::Mat prev_img, prev_img_r, cur_img, cur_img_r;
    cv::Mat prev_color_img_l, prev_color_img_r;
    int row_img_prev, col_img_prev;
    
    bool USE_TRIANGULATE_TWO_FRAME;
    bool reject_with_F, EASI_RANSAC_FH, has_valid_F, has_valid_H, fea_filtered, has_motion_pred_first_two_frame;
    set<int> reserve_bg_track_pt_id;
    bool only_use_track_sift_for_F;
    bool FAST_pred_motion, sift_pred_motion, cal_pred_for_sift_track;
    bool add_new_FAST_from_sift, add_new_sift_in_next_frame;
    // 上一帧和当前帧中的特征点（像素坐标）
    std::vector<cv::Point2f> prev_FAST, cur_FAST, cur_right_FAST, prev_sift, cur_sift, cur_right_sift;
    std::vector<int> FAST_no_stereo_bg, sift_no_stereo_bg;
    // 当前帧各特征点的全局id
    std::vector<int> ids_FAST, ids_FAST_right, ids_sift, ids_sift_right;
    // 当前帧各特征点在当前窗口内的被观测帧数（每一帧的左右图像观测只算一次）
    std::vector<int> track_cnt_FAST, track_cnt_sift;
    // 记录当前帧每个特征点的class label 和 obj id（一开始是当前帧的临时obj id，物体关联之后则变为全局obj id；其中最终的背景点和静态物体点都为0，动态背景点则为全局物体id）
    std::vector<std::pair<uchar, int>> obj_cls_id_FAST, obj_cls_id_sift;
    // 使用map结果保存点的全局id和其在cur_sift(加负号）或cur_FAST（正符号）中的序号
    std::map<int,int> gl_id_index_map;
    std::vector<int> id_bg_track_sift, id_bg_track_FAST;
    // map<int,int> gl_cls_index_map;
    // 上一帧和当前帧帧中特征点的深度值
    std::vector<float> prev_FAST_dep, cur_FAST_dep, prev_sift_dep, cur_sift_dep;
    // 记录cur_FAST和cur_sift中需要由depth_map获取立体匹配的特征点的id
    std::vector<int> id_FAST_no_depth, id_sift_no_depth;

    std::vector<cv::Point2f> n_FAST_bg;
    // vector<cv::Point2f> n_FAST_obj;
    // 当前帧新检测的特征点(暂时没使用)
    // vector<cv::Point2f> n_FAST_bg, n_FAST_obj, n_sift;
    
    // 记录当前帧保留的sift点 在 当前帧图像检测到的sift点集siftdata中的 序号
    std::vector<int> prev_sift_index, cur_sift_index;
    int num_sift_bg_prev, num_sift_bg_cur;
    // sift采用暴力匹配，因此不需要像素点的预测值。FAST的预测值不需要专门用一个变量来保存，在每一帧的开始用cur_FAST即可
    std::vector<cv::Point2f> predict_FAST;
    std::vector<float> predict_dep_FAST, predict_dep_sift;
    std::map<int,pair<float,int>> obj_fea_disp_num;
    double ave_dep_bg_cur_frame, ave_dep_bg_prev_frame;
    int num_bg_with_dep, num_bg_sift_with_dep;
    int id_new_track_sift;
    std::map<int,Vec<float,8>> new_sift_stereo_prev;
    std::vector<int> FAST_new_objs;
    // vector<cv::Point2f> predict_FAST_debug;
    // 记录当前帧所有FAST或sift点是否要保留。
    // status_FAST为上一帧的FAST点在当前帧的跟踪情况（1为跟踪到，0为否）；后面两个则是当前帧所有特征点（跟踪+新检测）最后是否要保留，1为保留，0为否。
    std::vector<uchar> status_FAST, statusLeftRIght, status_sift;
    // set<int> tracked_sift_with_depth;
    // int invalid_cur_FAST, invalid_cur_sift;

    // 记录每一帧中的cur_FAST和cur_sift最终的全局obj id，前面属于track的点的全局obj id其实是继承自上一帧的对应点，而后面新添加的点则是当前帧obj的临时id
    std::vector<int> prev_FAST_global_obj_id, prev_sift_global_obj_id;
    
    // vector<int> prev_obj_id_cur_track_FAST, prev_obj_id_cur_track_sift;
    // 记录上一帧 和 当前帧中 id最大的跟踪（自上一帧的）特征点； 当前帧中id最大的sift特征点（包含新特征点）； 当前帧中id最大的FAST跟踪点
    int last_id_track_fea_prev, last_id_track_fea_cur, last_id_sift_cur, last_id_track_FAST_cur;

    // 特征点的去畸变的归一化平面坐标
    std::vector<cv::Point2f> prev_un_FAST, cur_un_FAST, cur_un_right_FAST, prev_un_sift, cur_un_sift, cur_un_right_sift;
    // 当前帧和上一帧之间特征点的归一化平面上的2维速度
    std::vector<cv::Point2f> FAST_velocity, right_FAST_velocity, sift_velocity, right_sift_velocity;
        
    // 各特征点的全局id和归一化平面上的坐标
    // map<int, cv::Point2f> cur_un_FAST_map, prev_un_FAST_map, cur_un_sift_map, prev_un_sift_map, 
    // map<int, cv::Point2f> cur_un_right_FAST_map, prev_un_right_FAST_map, cur_un_right_sift_map, prev_un_right_sift_map;
    std::map<int, cv::Vec4f> prev_un_Fea_map, prev_un_r_Fea_map;
    // 当前帧右图像上的sift点对应的左图像点在左图像sift点集中的序号，方便后续根据左图像sift的更新信息来更新右图像上的sift点信息。弃用
    // vector<int> left_id_of_right_FAST, left_id_of_right_sift;
    std::map<int, cv::Point2f> prevLeftFeaMap, prevRightFeaMap;

    std::set<int> pts_for_cal_F;
    vector<int> temp_pts_for_F;
    int num_up_half;

    std::vector<camodocal::CameraPtr> m_camera;
    double prev_prev_time, prev_time, cur_time;
    bool stereo_cam;
    bool use_tria_stereo;
    bool no_add_new_sift;
    // 特征点全局id
    int n_id, n_obj_id;
    bool hasPrediction;
    // 跟踪点集 和 新检测点集 中属于背景点 的个数
    int num_track_sift_bg, num_new_sift_bg, num_track_FAST_bg, num_new_FAST_bg;
    // 跟踪点集中属于静态点（包含了背景和上一帧静态物体）的个数
    int num_track_sift_static, num_track_FAST_static, num_track_fea_static;
    // 当前帧的特征点中属于跟踪自上一帧的个数(指的是最终的有效跟踪点数)和总数
    int num_track_FAST, num_track_sift;
    // 记录当前所跟踪的静态点中，至少已跟踪2次（即连续观测3帧,包括当前帧）的点数，这个值影响当前是否要增加背景（和静态物体）中新检测FAST点的数量
    int num_sta_FAST_long_track, num_sta_sift_long_track, num_long_track_fea_stat;
    // 各物体的匹配点集中属于特征点的个数。如果需要使用物体上的像素点匹配的话。
    // vector<int> num_fea_objs;

    Mat Mat_F, Mat_H;
    Matrix3d R_from_E;
    Vector3d t_from_E;
    bool cal_Mat_F_H;
    
    // 每一帧应该至少要保留的在上一帧有深度值的静态跟踪点（背景点和物体点）
    int num_old_track_fea, num_track_fea_with_dep_prev, num_rest_track, num_rest_track_stereo;

    // 估计的上一帧与当前帧之间相机的运动，用于构建当前帧相机运动的F矩阵
    Matrix3d R_cam_motion;
    Vector3d P_cam_motion;

    int appro_num_track_stat_fea;

    // openMP中的变量锁
    omp_lock_t mylock;

    // 物体关联的相关变量。保存上一帧和当前帧中出现的各个物体的信息，包括对应的全局物体id，上一帧时刻的速度模型
    // 上一帧出现的所有物体（包括静态的和动态的）的全局id和全局类别，不包含背景
    std::vector<int> glob_obj_id_prev;
    std::vector<uchar> obj_cls_prev;
    // int指明上一帧的全局物体id(不包含背景运动), uchar代表该物体是否为已有的运动物体，是的话则RP_objs会有其速度模型。uchar 0代表旧物体且有运动，1代表旧物体但上一帧为静态，2代表为新物体！
    std::map<int, uchar> status_objs_prev;
    
    // 在上一帧的各个物体上采样的像素点坐标（注意是像素坐标，不是归一化坐标）及其估计深度值（从depth_map中获取，注意不是视差值！））；
    // 对于某一帧中完全漏检（但是又被特征点关联到）的物体，在此处保留它的密集像素中的内点（意味着把从上一帧的关联物体通过光流得到的像素点都加入位姿估计RANSAC中）
    // 二维数组在内存中是连续的！
    // map<int, vector<cv::Point3f>> pixel_objs_prev;
    std::map<int, float*> pixel_objs_prev;

    // 每一帧中每个考虑类别的刚体上采集的像素点数（3维点），每个点都是2维“像素“坐标和深度值紧挨着。
    // 每一行最前面的2个float分别保存该物体的全局类别（一开始为局部id，后面和全局物体关联后就修改为全局cls）和有效的采样点的点数，紧接是NUM_SAMPLED_PIXEL_OBJ个采样点之外（不一定都有效），最后5个元素是bbox的信息（左、右、上、下极限）和特征点的（相机坐标系下的）平均3D点坐标
    // 定义数组大小只能用常量，因此MAX_NUM_OBJS_FRAME和NUM_SAMPLED_PIXEL_OBJ必须是const型变量！
    float sampled_pixel[MAX_NUM_OBJS_FRAME][(NUM_SAMPLED_PIXEL_OBJ*3+2+7)];
    
    // 预测的当前帧和上一帧的各个动态物体的位姿“变换”，是把动态物体在上一帧的相机坐标下的点 变换到 当前帧相机坐标系下！
    std::map<int, std::pair<Matrix3d,Vector3d>> RP_objs_pred;

    // hungarian_optimizer
    HungarianOptimizer<float> optimizer;
    std::vector<std::pair<size_t, size_t>> assignments;
};
