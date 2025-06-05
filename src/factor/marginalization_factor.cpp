/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"

void ResidualBlockInfo::Evaluate()
{
    // 残差的总维度
    residuals.resize(cost_function->num_residuals());
    
    // 各个估计变量的维度大小
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    // 总的雅可比矩阵可以分成多个块列，每个块列的行数应该为残差的总维度，而列数与具体的估计变量的维度有关。这里先暂时用double数组来表示每个块列，下面会resize
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    // 设置与各个变量相关的残差的雅可比矩阵的大小
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        // 共享矩阵的内存
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    // 使用变量的新估计值来更新 残差值r 和 残差关于这些变量的雅可比J（雅可比和残差的值无关，但是和变量的值有关） 
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

// 添加一个新的测量因子
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);
    // 与此测量因子相关的变量，每个变量的值存储在指定的double *中
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // 每个变量的维度大小
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        // 用数组的地址值直接作为unorded_map的key！！！因为地址是独一无二的
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        // 将要被marg的变量的idx设为0？这个量在这里暂时起一个索引的作用，是知道某个变量(通过变量首元素地址addr作为唯一标识)是要被当前滑窗所marg掉的
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

// 更新参与marg的各个测量的残差和相关雅可比，然后为参与marg的相关剩余变量开辟堆内存保存其最新的估计值，以便下一帧进行LBA时可以更新先验残差rp的值
void MarginalizationInfo::preMarginalize()
{
    // 遍历所有要进行marg的测量（包括上一滑窗的先验）
    for (auto it : factors)
    {
        // 根据当前滑窗LBA最后一次优化后的变量估计，更新要marg的测量的 残差 以及 相关的雅可比矩阵（假如可以更新，如先验信息就无法更新J和H，只能更新r和b），因为在最后依次优化后就没有再更新变量的残差和雅可比了
        it->Evaluate();
        // 获取要marg的测量的各个相关变量的维度
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            // 查找当前滑窗中要marg的测量相关的变量的存储地址（变量数组的地址）
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            // 该变量的维度
            int size = block_sizes[i];
            // 为当前滑窗的先验信息的每一个相关变量（与任一要被marg的测量相关的变量）创建索引map，其索引key为该变量首元素在当前滑窗中的全局变量数组中的地址（与其旧的索引不同）
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                // 为这些参与当前滑窗marg的变量保存其在当前滑窗中优化后的估计值
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

// 真正执行marg操作的函数，在各个残差形成的H和b矩阵上进行Shur操作
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

// 正式进行marg，把被marg的量及其相关信息放在所有矩阵的开头，剩余的变量放在后面，然后进行舒尔补操作
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    for (auto &it : parameter_block_idx)
    {
        // first为该unorded_map中每一个元素的key，是该估计变量（当前或者之前）的内存地址，long型，是唯一的
        // 给parameter_block_idx的每个元素的value赋值(之前在创建此元素时只是暂时赋值为0），为该变量首元素在参与marg所有变量（逐元素排列）组成的数组中的序号
        it.second = pos;
        // localSize指的是某个变量在参与ceres优化时的维度，主要是针对位姿变量中的旋转，ceres保存估计后的旋转变量是用四元数，但是参与优化计算使用的是李代数
        pos += localSize(parameter_block_size[it.first]);
    }
    // 当前滑窗中要被marg的变量的总维度
    m = pos;

    // 这里说明parameter_block_idx中只保存了当前滑窗中要被marg的变量的标识（即其地址），而parameter_block_size保存了所有参与了被marg测量相关的变量的维度信息
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            // 把剩下的参与marg但是会被保留的变量添加到idx序列当中，key仍为其地址，vaule为其首元素在参与marg所有变量（逐元素排列）组成的数组中的序号
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }
    // 此时pos为参与当前滑窗marg操作的所有相关变量的总维度；而n是参与了marg但会被保留的其他变量的总维度
    n = pos - m;
    //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
    // 没有要被marg的变量？应该不会是发生在marg滑窗首帧的情况，因为滑窗首帧不至于与后面的任何一帧都没有任何的视觉共视或者IMU关联吧？
    // 那么这种情况只会发生在要marg次新帧，而次新帧的marg流程中直接舍弃了次新帧的视觉观测残差，IMU预积分直接融合，只会选择marg从上一滑窗继承来的先验残差（且必须该先验残差与上一滑窗的最新帧（即当前的次新帧）相关）
    // 则当前滑窗m==0说明上一个滑窗的先验信息与当前的次新帧（即上一个滑窗的最新帧）无关。则其实当前帧是没有添加任何要被marg的测量残差和变量的（因为marg次新帧时只选择marg先验信息，且其必须与次新帧变量有关）！
    // 那么至少说明次新帧（或者次次新帧，即上一滑窗marg掉了次新帧）和上一滑窗（或上上滑窗）的首帧没有共同视觉约束，这里说“至少”是因为往前推至少得有一个滑窗marg了最老帧，且当时最老帧与其最新帧之间没有共视关系。
    // 虽然多次marg旧帧之后可能会在最新的滑窗之后的首尾帧建立间接的联系（通过中间帧和新帧之间的共视），但是这种先验是比较脆弱和不准确的，还是需要直接的首尾帧的共视关系来保证稳定的追踪！
    if(m == 0)
    {
        valid = false;
        // 因为这种情况只可能发生在marg次新帧的时候，因此这里的不稳定跟踪是指至少（可能是更早的滑窗）上一帧滑窗的首尾两帧已经没有共视点了
        printf("unstable tracking...\n");
        return;
    }

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    // ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    // 使用4个CPU子线程来进行marg操作。这里要考虑CPU的核心数是否足够，除了这里要进行的多线程，在物体的位姿估计线程中也采用了多线程，因此要考虑总的线程数是否足够！
    pthread_t tids[NUM_THREADS_MARG];
    ThreadsStruct threadsstruct[NUM_THREADS_MARG];
    int i = 0;
    // 将要marg的残差平均分配给各个线程。这里是强行各个线程分配了指定的任务，因为这里是自己管理线程（的开启），而不是相openMP那样根据for循环来自动获取任务。
    // 另外，手动开启子线程时，这里的主线程是可以继续往下执行的（除非join阻塞了当前主线程）；而openMP中执行任务的多个线程是包含了其主线程的！
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS_MARG;
    }
    for (int i = 0; i < NUM_THREADS_MARG; i++)
    {
        // TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        // 子线程运行的函数为ThreadsConstructA，给定其实参
        // 每个子线程计算所负责的残差的marg后得到的A和b
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            printf("pthread_create error\n");
            abort();
            // ROS_WARN("pthread_create error");
            // ROS_BREAK();
        }
    }
    // 将各个线程得到的A和b累加
    for( int i = NUM_THREADS_MARG - 1; i >= 0; i--)  
    {
        // 逐个地等待子线程完成，将其marg后的A再加到总的A和b矩阵上
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    //TODO
    // 获取剩余相关变量的信息矩阵A和b，进行A和b的分解，得到等价的雅可比J_p和线性化残差r_p作为相关变量在下一滑窗优化时的先验约束（即这些变量的更新方程要受到这部分先验残差的影响）
    // marg后得到的Amm不一定是半正定的，因此这里使用这个操作使得这部分矩阵一定是半正定的
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
    // 选择A矩阵的特征值中大于eps的保留，而小于eps的特征值用0代替。用这种方式来计算Amm的逆，避免了数值不稳定
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());
    
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    // 计算marg后的密集子矩阵
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;
    // 埃尔米特矩阵的特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // 选择A矩阵大于eps(1e-8)的特征值，小于阈值的特征值用0代替。为什么要这么操作？这样是说A矩阵接近于非满秩，不如直接将某些特征值直接置为0，这样减少后面计算的精度问题
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    // 选择A矩阵大于eps(1e-8)的特征值，小于阈值的特征值用0代替，并计算非零特征值的倒数，这用于形成不满秩的J的逆
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
    // 对VectorXd逐元素计算平方根
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    // 注意，对于marg后产生的H和b，分解得到J和r时，我们认为分布的协方差矩阵为I，这样分解时就无需考虑协方差矩阵，否则两个方程无法确定三个变量
    // 由于H为对称矩阵，所以可以进行特征值分解 J‘*J = H = Q*X*Q',其中Q为正交方阵(Q'Q=QQ‘=I），X为元素皆为非负的对角线矩阵，则可以得到J = cwisesqrt(X)*Q‘； J的转置的逆为(J‘)^(-1)=（cwisesqrt(X)^(-1)*Q'
    // 用上面的S_sqrt形成的对角线矩阵可能是不满秩的（接近非满秩时人为使某些特征值为0了），则此时J的转置的逆不能用此对角线矩阵得，即用上面的方法先求非0特征值的倒数，再形成逆矩阵。
    // saes2.eigenvectors()即为特征值分解之后的Q矩阵
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // 这样得到的线性化残差的维度是r*1，r为参与marg的剩余变量的总维度！这个残差要参与到下一滑窗中这些变量的约束中去（即下一滑窗的LBA的第一次迭代，这些变量的初始值为它们在当前滑窗优化后的估计值，构建误差线性方程时关于这些变量首先就有了这个残差和J的约束）！
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

// 当前滑窗marg完成后，根据剩下的所有变量在下一滑窗期间会在全局内存（para_Pose等容器）中的地址，提前更新当前滑窗先验信息的相关变量（参与了marg但被保留）的标识key，以便下一滑窗可以用其估计的相关变量的更新值来更新先验信息中的残差r_p！
// 下一帧进行LBA时，先验残差rp也是会随着相关变量的估计值更新而更新，虽然其雅可比J不再变化，但是线性化点与最新估计之间的差值会变化，因此rp会变化。而这里的线性化点就是当前帧相关变量在LBA后的最新估计值
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {   
        // 逐一获取当前滑窗marg后剩余变量
        if (it.second >= m)
        {
            // 当前帧被marg后的每一个剩余变量在各个keep数组中是通过其唯一识别码it.first（其实就是未移动剩余变量在滑窗变量数组中的位置之前的各变量的真实地址）来索引
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            // 此为参与了当前滑窗marg的所有剩余变量在当前滑窗中优化后的估计值
            keep_block_data.push_back(parameter_block_data[it.first]);
            // 剩余变量在滑窗变量数组（是一个全局变量，其地址在整个系统运行期间不改变）中移动后的新地址（其实就是下一帧中各变量的存储地址），目的是能够找到下个滑窗中对应变量的值
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    // 这个值记录总共到保留的变量的总维度
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    // 要保留的残差（或者说测量）的总个数（注意，不是总的维度）
    set_num_residuals(marginalization_info->n);
};

// 对上一个滑窗的先验信息根据其相关变量（参与了marg的剩余变量）在当前滑窗的最新估计值来更新 它的先验残差r和b
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    // n是marg后剩余变量的总维度
    int n = marginalization_info->n;
    // m是被marg的变量的总维度
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        // 被marg的变量会放在变量序列的前面，因此marg后的每个剩余变量的首维在序列中的位置序号id就要减去被marg的变量总维度m
        // 上一帧滑窗中的keep_block_idx是记录marg剩余变量的首维在原始（marg前）变量数组中的序号，即默认所有变量的所有维度线性排列成一个数组
        int idx = marginalization_info->keep_block_idx[i] - m;
        // 从给定的参数数组中按顺序取出size个值组成VectorXd对象
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        // 上一个滑窗优化后的参与了marg的剩余变量的估计值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // marg之后会保留的残差，根据给定的VectorXd，取其元素形成residuals * n的Matrix（按行优先）
    // Map本身是个模板类，指定Eigen::VectorXd类型使其偏特例化（即该Map特例会使用构造函数中传入的数组指针的数据来构造和赋值VectorXd类型的变量）（其实也不算偏特例化，因为另外两个模板参数有默认值或者是可选的）
    // residuals, n则是特例化类Map的某个构造函数的参数，residuals是用于保存数据的数组（即指针），n是构造的vector对象的元素长度（会从residuals中按顺序取出或放入n个元素）
    // marginalization_info和linearized_residuals从上一个滑窗先验信息中解算出来的雅可比J_p和残差r_p。
    // 由于上一帧被marg掉的测量已经不在，所以当前帧中虽然剩余变量的值有更新，我们也无法更新这些变量对应的J_p，但r_p是可以更新的！
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                // 如果是位姿变量，则将它从7维变量转成6维
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                // 此Matrix在内存中的存储顺序是按行优先存储，即矩阵首地址之后紧接着存放的是矩阵第1行第2列的元素
                // 将jacobians[i]所指示的地址开始的连续n*size个double型变量的地址长度，用来组成一个维度为n*size（按行优先）的Matrix对象jacobian，即jacobian和jacobians[1]其实是共享内存的！
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                // 将该地址上所有变量置为0
                jacobian.setZero();
                // jacobian的数据在内存中是按行优先存放的，这里是要设置最左边的几列，那么先前按列优先存放是不是更方便些？
                // 取上一滑窗marg后的先验信息反解出来的J中属于各个相关变量的部分
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
