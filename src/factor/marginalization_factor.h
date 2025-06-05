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
// #include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS_MARG = 4;

// 单个残差块信息。接收不同具体类型的CostFunction（子类）或LossFunction，以及相关的真实的估计变量_parameter_blocks，以及指定哪些变量_drop_set即将会被marg
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    // 根据所给的CostFunction对象，确定其中的残差维度，估计变量个数和维度
    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

// 在进行marg时采用多线程，每个线程操作一部分的残差
struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 此类包含了总的要进行marg操作的信息，包括多个ResidualBlockInfo。此类中定义各个残差
class MarginalizationInfo
{
  public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MarginalizationInfo(){valid = true;};
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    // m表示当前要marg的变量的总维度，而n是剩下的变量的总维度
    int m, n;
    // 用数组double*（存放某个状态变量）的地址值(long类型）直接作为unorded_map的key！！！
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    // 每个优化变量的首维在序列中的位置序号id
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    // 对损失函数进行线性化之后的雅可比和引入的线性化残差
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
    bool valid;

};

// 定义一个继承自ceres::CostFunction的子类CostFunction
// 这是外界给定具体的 优化变量、残差和雅可比矩阵 的数据的调用接口，在成员函数Evalute中会将数据给到MarginalizationInfo成员对象以完成amrg操作
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
