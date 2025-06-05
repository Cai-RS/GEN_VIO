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
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
// #include <ros/ros.h>
#include <map>
#include "../estimator/feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ImageFrame(){};
        ImageFrame(const FeaFrame& _points, double _t):t{_t},is_key_frame{false}
        {
            // 这里需要值复制吗？可以指针吗？
            points = _points;
        };
        FeaFrame points;
        double t;
        // 这里的R表示的是该帧相机时刻的IMU坐标系相对于世界坐标系（设定为首帧相机的坐标系）的姿态，T则是对应的位移
        Matrix3d R;
        Vector3d T;
        // 对pre_integration必须初始化，要么在这里，要么在构造函数中，如果没有初始化，则后续如果对其没有再赋值而使用，则会报错
        IntegrationBase *pre_integration = nullptr;
        bool is_key_frame;
};
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs);
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x, bool ForStereo = false);