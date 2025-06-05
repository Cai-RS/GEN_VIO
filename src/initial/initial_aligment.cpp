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

#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 请见深蓝VIO课程的第7讲PPT的第12页的公式10和11，优化目标为线性最小二乘
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        // tmp_b = 2 * (frame_j->second.pre_integration->delta_q * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    cout << "gyroscope bias initial calibration: " << delta_bg.transpose() << endl;
    // printf("gyroscope bias initial calibration: %lf, %lf, %lf\n",delta_bg(0),delta_bg(1),delta_bg(2));
    // ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());
    
    // 每一帧IMU的陀螺仪bias都是相同的优化量吗？
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x, bool ForStereo = false)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = 0;
    // 如果是单目+IMU，则还需要优化视觉尺度s
    if (!ForStereo){
        n_state = all_frame_count * 3 + 2 + 1;
    }
    else{
        n_state = all_frame_count * 3 + 2;
    }
    
    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 优化4次
    for(int k = 0; k < 8; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);
            MatrixXd tmp_A;
            if(!ForStereo){
                tmp_A.resize(6, 9);
            }
            else{
                tmp_A.resize(6, 8);
            }
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            if (!ForStereo)
            {
                tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
                // 残差项中，第一项为两帧之间的预积分值，而后面部分都是与视觉估计结果相关的，认为视觉估计的结果较为准确，这里就是要将通过调整IMU的参数和状态量来使IMU的估计结果与视觉的对齐     
                tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;
            }
            else
            {
                tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p - frame_i->second.R.transpose() * dt * dt / 2 * g0 - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
            }
            
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            if (!ForStereo)
            {
                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();
                A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            }
            else
            {
                A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
                b.tail<2>() += r_b.tail<2>();
                A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
                A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
            }
        }
        if(!ForStereo)
        {
            A = A * 1000.0;
            b = b * 1000.0;
        }   
        
        x = A.ldlt().solve(b);
        VectorXd dg;
        if (!ForStereo)
        {
            dg = x.segment<2>(n_state - 3);
            //double s = x(n_state - 1);
        }
        else
        {
            dg = x.segment<2>(n_state - 2);
        }
        g0 = (g0 + lxly * dg).normalized() * G.norm();
    }
    // 得到了东北天坐标系下的重力向量G（0，0，-9.8）在当前世界坐标系（由之前初始化首帧IMU的位姿时确定）下的表达   
    g = g0;
}

// 对于双目情况下的IMU和视觉的对齐，主要是估计各帧IMU的速度以及首帧相机坐标系下的g向量，不需要估计尺度s
bool LinearAlignmentForStereo(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    // 每帧相机时刻的IMU速度（3），首帧相机坐标系下的重力向量g（3维，此时没有约束它的模长）；双目情况下尺度已知，无需估计
    int n_state = all_frame_count * 3 + 3;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 9);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        // 位置预积分的误差相对于g的雅可比。
        // ！要注意，all_image_frame中每一帧中的R和P等来自于Rs和Ps，而Rs和Ps都是使用视觉约束得到的IMU在当前参考坐标系下的位姿估计，是较为准确的，而不是直接使用IMU测量来递推得到的位姿（此时Rs和g之间是不匹配的）
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // 残差项
        // tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;

        // 第i帧和第i+1帧之间的速度预积分（3维）的误差 相对于 各相关状态变量的 雅可比
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();
        // g放在状态变量的最后
        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        // 两前两帧IMU的速度变量与g之间的相关block
        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    // A = A * 1000.0;
    // b = b * 1000.0;
    // A = A * 1/100.0;
    // b = b * 1/100.0;
    x = A.ldlt().solve(b);
    g = x.segment<3>(n_state - 3);
    cout<<"result g norm: " << g.norm() << "vector: " << g.transpose() << endl;
    // ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 0.5)
    {
        RefineGravity(all_image_frame, g, x, true);
        if(fabs(g.norm() - G.norm()) > 0.5)
        {
            assert(false && "Failed to refine g!");
            return false;
        }
        
        cout<<"refine g norm: " << g.norm() << "vector: " << g.transpose() << endl;
    // ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    }
    return true;
}

// 之所以说是线性对齐，是因为这里最后所构建的方程组是线性最小二乘，有精确解析解
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    // 每帧相机时刻的IMU速度（3），首帧相机坐标系下的重力向量g（3维，此时没有约束它的模长），尺度
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        // 6是两帧之间的测量误差的维度，3维的位置预积分和3维的速度预积分。10是跟这两帧间误差相关的变量的维度，为这两帧的IMU速度（6）、g（3维）和尺度
        // 这里是选择只构建H中与当前两帧测量误差相关的状态变量的H和b，即总H和b中与当前误差相关的（不为0）block
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        // ImageFrame类中的R和T表示该帧相机时刻的IMU坐标系相对于世界坐标系（设定为初始帧相机坐标系）的位姿
        // 这一项是 第i帧和第i+1帧之间的位置预积分（3维）的误差 相对于 第i时刻的IMU坐标系速度的 雅可比
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        // 位置预积分的误差相对于g的雅可比
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // 下面公式括号中的原本应该是（第i帧的相机坐标系原点到第i+1帧的相机坐标系原点的位移向量（表达在世界坐标系下），但是由于IMU和相机都固连在刚体上，因此它们的进行纯平移时，在全局坐标系下的位移是相同的！
        // 此处为预积分的误差相对于视觉尺度s的雅可比。对于双目-IMU而言，尺度s应该是固定为1，所以可以将它从状态变量中去除，则下面这一项也可去除
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0; 
        // 残差项    
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        // 第i帧和第i+1帧之间的速度预积分（3维）的误差 相对于 各相关状态变量的 雅可比
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        // 这些误差的信息矩阵应该要怎么设置呢？这里可以根据动力学模型来推导信息矩阵吗？就是建立了误差的动力学方程,以传递估计方差
        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        // 将 第i帧和第i+1帧之间的预积分误差 所构造的H矩阵和b向量 加到总H矩阵
        // 目标函数为最小化 所有帧对的测量误差的方差（且为线性最小二乘）之和，求导可以直接得到理论解，求导后等式为 为各个 J‘J*x = -J‘f/2的累加，即H*x=b
        // 其实每两帧之间的误差都会有自己的完整H矩阵，但是每个误差只与很少的状态量相关，即每个分H都是很稀疏的，所以我们选择将不为0的块加到总H矩阵的相应部分
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();
        // g和s放在状态变量的最后
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 放大数值，减小计算精度误差
    A = A * 1000.0;
    b = b * 1000.0;
    // 使用矩阵分解来求解线性方程Ax=b的解，由于H是对称矩阵，所以使用Cholesky分解。注意，是线性方程组，所以并不需要给出x的初始值，非线性方程组才需要给初始值（因为没有精确解，需要采用数值算法进行迭代求解）
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    printf("estimated scale: %f\n", s);
    // ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    cout<<"result g norm: " << g.norm() << "vector: " << g.transpose() << endl;
    // ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    if(fabs(g.norm() - G.norm()) > 0.5 || s < 0)
    {
        return false;
    }
    cout<<"refine g norm: " << g.norm() << "vector: " << g.transpose() << endl;
    // ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x, bool ForStereo)
{
    solveGyroscopeBias(all_image_frame, Bgs);

    if (!ForStereo){
        if(LinearAlignment(all_image_frame, g, x))
            return true;
        else 
            return false;
    }
    else
    {
        if(LinearAlignmentForStereo(all_image_frame, g, x))
            return true;
        else 
            return false;
    }
}
