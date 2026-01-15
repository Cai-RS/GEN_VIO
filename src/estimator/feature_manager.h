/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"
#include "../utility/tic_toc.h"
#include "utils.h"
#include "../featureTracker/feature_tracker.h"

// 此类表示某个特征地图点在其某个被观测帧中的观测信息
class FeaturePerFrame
{
  public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td)
    {
        // 左图像中的特征点在归一化平面中的坐标，因此原始VINS项目中z值恒为1
        // 但是在此项目中point(2)被存储了该点在当前帧相极坐标系下的深度（通过立体匹配）。注意x和y仍然是归一化平面坐标
        point.x() = _point(0);
        point.y() = _point(1);
        //point.z() = _point(2)
        point.z() = _point(2)/_point(2);
        // 特征点在左图像中的像素坐标
        uv.x() = _point(3);
        uv.y() = _point(4);
        // 特征点在归一化平面上的速度（当前帧相对于上一帧）
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        depth = -1.0;
        is_stereo = false;
    }
    void rightObservation(const Eigen::Matrix<double, 8, 1> &_point)
    {
        // 与上面左图像中的变量意义类似，只不过都是在右图像上（速度是相对于上一帧右图像）
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        // pointRight.z() = _point(2);
        pointRight.z() = _point(2)/_point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
    // 被观测帧的（左图像）的时间戳？还是说是IMU和相机帧之间的时间戳位移？是后者
    // 每一帧记录的td，其实是其上一个滑窗中所有帧所共同优化出的一个td值。因此，每一帧的这个td值都不相等，
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    // 如果有右观测点，则用立体三角化恢复出该帧下的深度
    double depth;
    bool is_stereo;
};

// 某地图点的被观测信息
class FeaturePerId
{
  public:
    // 该特征点的（滑窗内）全局Id
    const int feature_id;
    // 该地图点被观测到的首帧在当前滑窗中的id
    int start_frame;
    // 指的是该地图点在所有被观测帧上的观测信息（坐标，速度等）
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    // 该点在其观测首帧下的深度值
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    bool has_LBA; // has the point been in LBA
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0), has_LBA(false)
    {
    }

    int endFrame();
};

// 此类管理着滑窗内（准确来说应该是滑窗长度+1，包括即将要被marg掉的一帧）所有帧的特征地图点
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    int getFeatureCountLBAcur(int frameCnt);

    bool addFeatureCheckParallax(int frame_count, double Headers[], double prev_td, double td, FeatureTracker &tracker, int &num_track_add, int &num_3D2D_track, bool need_LBA);

    int addStaticFeature(int frame_count, int prev_td, double td, FeatureTracker &tracker, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], 
                          Matrix3d ric[], const vector<pair<int,int>> &id_fea, const int id_obj_cur, bool need_marg, bool marg_old, 
                          const vector<int> &reserve_new_sift = vector<int>(), const vector<int> &ignore_pts = vector<int>(), bool add_new_fea = false, int global_cls = 0);
    
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(int frameCnt, const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    
    int triangulate(int frameCnt, double Headers[], Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], FeatureTracker &tracker, int &num_track_3D_2D, bool good_est_RT = false, 
                    bool marg_old = true, bool before_PnP = false, bool try_tria = true, bool try_update_dep = false, bool LBA_succ = false, 
                    const Matrix3d &R_from_E = Matrix3d(), Vector3d *norm_t = nullptr, double scale = 1.0, float pred_dist_t = 0.0, const set<int> &reserve_bg_track_pt_id = set<int>());
    
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1, Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    
    float cal_ave_epi_line_dist_pts(const Matrix3d &R_cam_motion, const Vector3d &P_cam_motion, const vector<list<FeaturePerId>::iterator> &fea_iters, const vector<uchar> &status);
    
    bool initFramePoseByPnP(int frameCnt, FeatureTracker &tracker, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], Matrix3d &pred_R, Vector3d &pred_P, const Matrix3d &prev_cam_R, const Vector3d &prev_cam_P, 
                            float &ave_epi_dist, vector<int> &reserve_new_sift, vector<int> &reserve_new_FAST, int &num_track_cur_bg, int &num_inlier_fea_PnP, bool comp_with_prev_esti = false, bool initial_succ = false);
    
    bool solvePoseByPnP(int frame_count, Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, vector<cv::Point2d> &pts2D, vector<cv::Point3d> &pts3D, vector<int> &pts_id_vec, bool initial_succ);
    void removeBackShiftDepth(int frameCnt, Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P, FeatureTracker &tracker);
    void removeBackShiftDepth();
    void removeBack(int frame_cnt, FeatureTracker &tracker);
    void removeFront(int frame_cnt, FeatureTracker &tracker, double Headers[]);
    void removeOutlier(set<int> &outlierIndex);
    void find_long_track_fea_in_map(int frameCnt, set<int> &fea_with_more_frames_in_map, set<int> &fea_with_3_frames_in_map);

    // 这是最新窗口内的静态特征点
    // 每个元素FeaturePerId就是一个特征地图点，其中会记录观测到该点的所有帧中的信息
    // 添加新的特征地图点和增加某地图点的观测帧记录的操作都是在FeatureManager::addFeatureCheckParallax()函数中
    // 用map是不是更好？把点的全局id作为key，索引起来效率是否更高？但是内存要求会高很多！
    list<FeaturePerId> feature;
    
    // 存储最新窗口内的所有物体的特征点
    // map<int,list<FeaturePerId>> fea_objs_win;
    
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[2];
    
    int pts_id_bloc[36][NUM_FEA_IN_BIG_BLOC];
};

#endif