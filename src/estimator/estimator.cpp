/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
// #include "../utility/visualization.h"

using namespace cv;
using namespace std;
using namespace YoloV8;
using namespace RapidFlow;
using namespace StereoDepth;

// Rs是一个Matrix3d的数组，长度为WINDOW_SIZE + 1
// 在参数列表中使用 转换构造函数（只有一个形参）来构造临时变量并赋值初始化成员变量f_manager
Estimator::Estimator(): _shutdown(false), f_manager{Rs}
{
    printf("init begins\n");
    // ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    // 如果是多线程（后端在子线程中执行），则等待子线程结束，才能结束主线程
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
    // checkCudaRuntime(cudaStreamSynchronize(common_infer_stream));
    // checkCudaRuntime(cudaStreamSynchronize(cpy_post_stream));
    safeCall(cudaDeviceSynchronize());
    if(start_preproc_seg != NULL) cudaEventDestroy(start_preproc_seg);
	if(stop_infer_seg != NULL) cudaEventDestroy(stop_infer_seg);
    if(stop_cpy_input_flow != NULL) cudaEventDestroy(stop_cpy_input_flow);
    if(stop_infer_flow != NULL) cudaEventDestroy(stop_infer_flow);
    if(stop_cpy_input_depth != NULL) cudaEventDestroy(stop_cpy_input_depth);
    if(stop_infer_depth != NULL) cudaEventDestroy(stop_infer_depth);
    if(stop_post_flow != NULL) cudaEventDestroy(stop_post_flow);
    
    cudaStreamDestroy(common_infer_stream);
    cudaStreamDestroy(cpy_post_stream);
    
    clearState();
}

void Estimator::clearState()
{
    // 因为是系统的初始化，所以需要清空这些变量（可能是上一个测试序列时的存留）
    // lock() 调动后，如果发现 mutex 已经上锁，会等待它直到它解锁，这是阻塞的做法；
    // 也可以使用try_lock()无阻塞，上锁失败同步返回false，如果成功上锁则返回true。还可以设置等待时长或时间点，try_lock_for() 和 try_lock_until() 。
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    use_mask_img_for_sift = false;
    first_win = true;
    initial_succ_first_win = false;
    map_fea_optimized = false;
    map_fea_writen = false;
    tracker_pts_updated = true;
    prevTime = -1;
    curTime = 0;
    // 不估计外参
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    pred_cam_pose_with_IMU = true;
    NEED_ESTI_G = false;
    
    can_change_seneor_type = false;
    done_cam_motion_pred = false;

    delta_T_cam_new = 0;
    curTime_cam  = 0;
    prevTime_cam = 0;
    // 这里先保留系统最开始的WINDOW_SIZE次位姿转换，在第一次相机的LBA之后对每个物体的各帧位姿变换进行校正。
    // 后续由于滑窗可能marg掉次新帧，因此每次只校正最新的物体位姿变换，则对这里各变量再resize成NUM_FRAME_TRACK_OBJS，再减小预分配内存
    // delta_P_objs.resize(WINDOW_SIZE);
    // delta_R_objs.resize(WINDOW_SIZE);
    // Vs_objs.resize(WINDOW_SIZE);

    // 每次进行BA优化都是对一个滑动窗口内的关键帧进行，这里要同时保留的关键帧的个数为窗口长度+1，因为会有最新的关键帧和需要marg的关键帧同时存在，被marg的帧的信息要用来构建prior
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 为每一帧最多预留MAX_NUM_OBJS_FRAME个物体位姿信息的内存
        delta_P_objs.reserve(MAX_NUM_OBJS_FRAME);
        delta_R_objs.reserve(MAX_NUM_OBJS_FRAME);
        // Vs_objs.reserve(MAX_NUM_OBJS_FRAME);
        
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        // IMU的数据缓存队列
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (int i = 0; i < 2; i++)
    {
        prev_cam_P_using_imu[i] = Vector3d::Zero();
        prev_cam_R_using_imu[i] = Matrix3d::Identity();
    }
    
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;

    if(!all_image_frame.empty()) all_image_frame.clear();
    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;
    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    if(!last_marginalization_parameter_blocks.empty()) last_marginalization_parameter_blocks.clear();
    // 清空保存特征地图点的list
    f_manager.clearState();
    
    featureTracker.clear_var();
    failure_occur = 0;

    mProcess.unlock();

    cout << "Estimator init finished!" << endl;
}

// 初始化GPU上的操作所需要的变量，它们是作为Estimator类的成员，这样就不需要重复每次都重复创建和初始化。
// __attribute__((no_sanitize("address")))
void Estimator::GPU_init_build(std::string &img0_for_gpu_build, std::string &img1_for_gpu_build, std::string &img2_for_gpu_build, std::string &img3_for_gpu_build)
{
    all_Time_GPU    = 0;
    end_seg_post    = false;
    end_flow_post   = false;
    end_sift_post   = false;
    end_stereo_post = false;
    end_FAST_track  = false;
    done_sample     = true;
    checkCudaRuntime(cudaStreamCreateWithFlags(&common_infer_stream, cudaStreamNonBlocking));
    checkCudaRuntime(cudaStreamCreateWithFlags(&cpy_post_stream, cudaStreamNonBlocking));

    cudaEventCreate(&start_preproc_seg);
    cudaEventCreate(&stop_infer_seg, cudaEventDisableTiming);
    cudaEventCreate(&stop_post_seg, cudaEventDisableTiming);
    cudaEventCreate(&stop_cpy_input_flow, cudaEventDisableTiming);
    cudaEventCreate(&stop_infer_flow, cudaEventDisableTiming);
    cudaEventCreate(&stop_post_flow);
    cudaEventCreate(&stop_cpy_input_depth, cudaEventDisableTiming);
    cudaEventCreate(&stop_infer_depth, cudaEventDisableTiming);
    cudaEventCreate(&stop_post_depth);
    
    // _yolov8seg等变量的类型为shared_ptr<T>，new返回的类型为T*，两者不能直接赋值
    // _yolov8seg = shared_ptr<YoloV8::Yoloseg>(new Yoloseg(yolov8_seg_config, common_infer_stream, cpy_post_stream));
    // _rapidflow = shared_ptr<RapidFlow::Rapidflow>(new Rapidflow(rapidflow_config, nullptr, cpy_post_stream));
    // _DPstereo  = shared_ptr<StereoDepth::DPStereo>(new DPStereo(DPstereo_config, nullptr, cpy_post_stream));
    _yolov8seg = make_shared<Yoloseg>(yolov8_seg_config, common_infer_stream, cpy_post_stream);
    _rapidflow = make_shared<Rapidflow>(rapidflow_config, nullptr, cpy_post_stream);
    _DPstereo  = make_shared<DPStereo>(DPstereo_config, nullptr, cpy_post_stream);
    _sift_extract_match = make_shared<Sift>(sift_config, nullptr, cpy_post_stream);

    if (!_yolov8seg->build())
    {
        std::cout << "Error in Yoloseg building" << std::endl;
        exit(0);
    }

    if (!_rapidflow->build()){
        std::cout << "Error in Rapidflow building" << std::endl;
        exit(0);
    }

    if (!_DPstereo->build()){
        std::cout << "Error in DPStereo building" << std::endl;
        exit(0);
    }

    cout << "Reading image0 (first left image) for warm-up: " << img0_for_gpu_build << endl;
    // 注意，opencv读取的图像是彩色图，通道顺序是BGR！
    Mat left_img_0  = imread(img0_for_gpu_build);
    if(left_img_0.empty())
        printf("Failed to read left image!\n");
    // assert(left_img.channels() == 3);
    if(left_img_0.type() != CV_8UC3) cout << "Wrong data type of image!" << endl;
    if(!left_img_0.isContinuous()) cout << "Image memory is not continuous!" << endl;

    Mat left_img_0_gray = imread(img0_for_gpu_build, 0);

    cout << "Reading image1 (right image) for warm-up: " << img1_for_gpu_build << endl;
    Mat right_img_0 = imread(img1_for_gpu_build);
    if(right_img_0.empty())
        printf("Failed to read right image!\n");

    Mat right_img_0_gray = imread(img1_for_gpu_build, 0);

    cout << "Reading image2 (second left image) for warm-up: " << img2_for_gpu_build << endl;
    Mat left_img_1 = imread(img2_for_gpu_build);
    if(left_img_1.empty())
        printf("Failed to read right image!\n");

    Mat left_img_1_gray = imread(img2_for_gpu_build, 0);

    cout << "Reading image3 (second right image) for warm-up: " << img3_for_gpu_build << endl;
    Mat right_img_1 = imread(img3_for_gpu_build);
    if(right_img_1.empty())
        printf("Failed to read right image!\n");
    
    Mat right_img_1_gray = imread(img3_for_gpu_build, 0);

    // true表示需要在host上分配不可分页内存
    _sift_extract_match->Allocate(left_img_0_gray.cols, left_img_0_gray.rows, false, 2048, false);
    
    // warm-up of GPU infer! 如果不是连续地运行推理，warm-up似乎意义不大？
    for (int i = 0; i < 15; ++i)
    {
        if (i == 0)
        {
            _yolov8seg->preproc_infer(left_img_0, stop_infer_seg);

            _DPstereo->preprocess_input(left_img_0, right_img_0, stop_cpy_input_depth);

            _DPstereo->infer(stop_infer_depth, stop_cpy_input_depth);

            // 在GPU进行DPStereo的推理期间，在CPU上进行光流估计和Sift检测匹配的预处理，即各自进行pad处理和数据传输，这部分时间可以被GPU所覆盖
            _rapidflow->preprocess_input(left_img_0);
            
            // sift的预处理阻塞host线程
            _sift_extract_match->preprocess_input(left_img_0_gray, right_img_0_gray);
            
            //cout << "SiftCUda model is preferring!" << endl;
            // sift的GPU操作是否要用公用的stream？可以不用，用默认的stream就行，因为其他的stream都是Nonblocking的，其上的操作不会互相阻塞
            _sift_extract_match->detect_match_sift(true, false);
            // 分割和深度的后处理中不需要加入 后处理处理结束（主要是数据传输）的event，所有后处理的device-host数据传输操作统一用cpy_post_stream来进行
            _yolov8seg->postprocess_output(bbox_mask, solid_obj_cls, deform_obj_cls, small_solid_objs, stop_infer_seg);
            
            // SIFT的后处理全都在当前CPU线程中，而yoloseg的后处理中都会短暂地阻塞CPU（用于获取推理后的结果）
            _sift_extract_match->postprocess(-Y_shift_right_image/mMinDepthPt, true, false);
            //cout << "cudaFift post-process succeeded!" << endl;

            _rapidflow->postprocess_output(map_flow, false);
            
            _DPstereo->postprocess_output(map_depth, stop_infer_depth);

        }
        else
        {
            bbox_mask.clear();
            _yolov8seg->preproc_infer(left_img_1, stop_infer_seg);

            _DPstereo->preprocess_input(left_img_1, right_img_1, stop_cpy_input_depth);

            _DPstereo->infer(stop_infer_depth, stop_cpy_input_depth);

            // 在GPU进行DPStereo的推理期间，在CPU上进行光流估计和Sift检测匹配的预处理，即各自进行pad处理和数据传输，这部分时间可以被GPU所覆盖
            _rapidflow->preprocess_input(left_img_1);

            // 当前帧左图像的实例分割结果
            _yolov8seg->postprocess_output(bbox_mask, solid_obj_cls, deform_obj_cls, small_solid_objs, stop_infer_seg);


            _DPstereo->postprocess_output(map_depth, stop_infer_depth);

            safeCall(cudaDeviceSynchronize());

            _sift_extract_match->preprocess_input(left_img_1_gray, right_img_1_gray);

            // 选择在flow的infer之前进行sift的检测和match，这样sift的后处理中match的过滤（主要在CPU上进行）耗时就可以被flow的infer操作所覆盖
            _sift_extract_match->detect_match_sift(false, false);

            // cout << "Succeeded cudaSift detect and match in round 2!" << endl;

            _rapidflow->infer(stop_cpy_input_flow, stop_infer_flow);

            // SIFT的后处理全都在当前CPU线程中
            _sift_extract_match->postprocess(-Y_shift_right_image/mMinDepthPt, false, false);
            
            _rapidflow->postprocess_output(map_flow, true, stop_infer_flow, stop_post_flow);
        }
        // 出个深度图和光流图，看一下结果的复制过程是否正确！主要问题可能在于TensorRT的结果保存 和 Mat的数据保存 顺序不是一致的，即矩阵的行优先或列优先问题！
    }
    // 阻塞此host线程直到后处理完成
    safeCall(cudaDeviceSynchronize());
    _sift_extract_match.reset();
    // 得到推理结果之后，应该要立即将其中位于页锁内存的数据拷贝到其他内存中！
    std::cout<<"Warm-up of GPU succeed!"<<endl;
}

// 设置系统的内外参并且开启VIO后端操作的子线程（如果需要的话）
void Estimator::setParameter()
{
    mProcess.lock();
    // 双目相机的NUM_OF_CAM为2，左相机和右相机 相对 IMU的 有各自的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 相机和IMU的外参可能标定值（则后续不需要再优化估计外参）或较为准确的初始估计值（需要在此基础上进行优化估计），也可能完全没给定任何值（则初始化为了I和0），三种情况下ESTIMATE_EXTRINSIC分别为0、1和2。
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    // FeatureManager类管理着滑窗内（每个滑窗的长度为WINDOW_SIZE+1，包括最新进来的一帧）所有帧的特征地图点的信息（包括深度和观测）以及 完成这些地图点在滑窗内的删减、各帧中的投影计算
    f_manager.setRic(ric);
    // 设置这些factor类中的静态变量。FOCAL_LENGTH指的是像素焦距，即f*lambda_x，焦距f为米制度，lambda_x指的是图像平面上物理长度1m包含的像素数
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH_X / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH_X / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH_X / 1.5 * Matrix2d::Identity();
    // 没有“是否需要估计time shift”的flag吗？一定要估计time shift吗？正常是需要的，除非标定得非常准确且不会变化
    prev_td = TD;
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    // 类FeatureTracker用于管理最新两帧图像之间的特征点关联和新特征点的检测等相关变量和函数
    // 读入相机的内参。
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    // 能否得到前2帧之间的相机运动的预测，取决于是否有IMU，且使用IMU的积分来预测相机运动，且不需要估计初始帧IMU坐标系的位姿
    featureTracker.has_motion_pred_first_two_frame = (USE_IMU && pred_cam_pose_with_IMU && !NEED_ESTI_G);
    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << endl;

    // 如果要开启另一个线程来进行VIO的后端操作（即processMeasurements函数中所进行的IMU预积分、根据前端的视觉匹配来估计当前帧的位姿、VI联合初始化、视觉惯性LBA、marg操作和滑窗等）
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        // 对数据（这里的数据包括IMU测量，和经过处理的当前帧图像特征检测和匹配）进行进一步的处理（预积分，PnP估计当前帧位姿、VI初始化、滑窗内LBA和marg），在子线程processThread中进行
        // 注意，对于类成员函数作为线程函数，除了该函数的显式参数（刚好此函数为显式形参），还需要给定类对象作为隐藏参数！
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
    
    mGPU.lock();
    // printf是C语言格式的信息输出函数，不能直接输出string对象，而是必须将其转化为C格式的字符串
    // printf("Image0 for GPU warm-up: %s\nImage1 for GPU warm-up: %s\n", img1_warm_up_gpu.c_str(), img2_warm_up_gpu.c_str());
    GPU_init_build(img0_warm_up_gpu, img1_warm_up_gpu, img2_warm_up_gpu, img3_warm_up_gpu);
    mGPU.unlock();
}

// 改变系统的配置，use_imu表示是否有IMU，use_stereo表示是否有双目相机
void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    // 不能出现仅用 单目 的情况
    if(!use_imu && !use_stereo)
    {
        printf("at least use two sensors! \n");
        abort();
    }  
    else
    {
        // 是否需要改变IMU的配置
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            // 从原本没有IMU 变为 使用IMU，则需要重启系统
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            // 如果是从原本使用IMU 变为 停止使用IMU。不需要重启系统，只是抛弃当前滑窗的先验信息，从下一个滑窗开始构建新的纯视觉的先验信息
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        // 双目视觉的配置
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

// VINS处理工作的开始。往原始图像的buf中插入新的测量
// 注意，KITTI数据集中只有相机0和1经过严格的立体校正，校正后的相机2和3只是投影面共面，但是两者的光心连线并不完全沿相机0-1的x轴方向
// 因为相机2和3的光心均有全局y轴方向的偏置，相机3比相机2的光心要高n个像素点（n取决于该点在相机坐标系中的深度值，呈负相关，即点越远，左右图像的同一点的v坐标越接近）。
// 这里我们不修改相机2的左图像，只对数据集中相机3的右图像再进行校正，截去上面几行的像素，剩下的整体上移，最下面填补几行全黑像素。
// 也可以选择同时裁减掉左图像中最下面的几行，而右图像裁减掉最上面几行，但这会改变原始图像的尺寸。
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;

    RawImageData data;
    data.index = inputImageCnt;
    data.time = t;
    data.image_left = _img;
    
    // 这里不对右图像进行调整，因为深度大于10m的点，其在左右图像中的v坐标最多相差不到0.4个像素（具体值为Y_shift_right_image/Z），如果这里按照平均考察深度来计算偏移像素值，对于远处的物体的深度估计精度影响较大，不如不要进行校正。
    // if(0)
    // {
    //     int y_shift = 0;
    //     // 注意，这里给的原始图像_img和_img1应该是8UC3！
    //     cv::Mat rect_right_img = cv::Mat::zeros(ROW,COL,CV_8UC3);
    //     if (Y_shift_right_image > 0) 
    //     {
    //         y_shift = (int)(Y_shift_right_image/(0.5*mThDepthObj+0.5*mMinDepthPt) + 0.5);
    //         rect_right_img.rowRange(0,ROW-y_shift) = _img1.rowRange(y_shift,ROW).clone();
    //         // 这里如果要对_img1进行修改，则其类型不能为const的
    //         _img1 = rect_right_img.clone();
    //     }
    // }
    data.image_right = _img1;
    
    while(raw_image_buffer.size() >= 3 && !_shutdown){
        usleep(2000);
    }
    cout << "succeeded push new image!" << endl;
    mBuf.lock();
    // 应该要给定RawImageData的拷贝构造函数！（或者移动构造函数？但是原图像不一定会一直保存着）
    raw_image_buffer.push(data);
    mBuf.unlock();
}

// 设置上一帧物体特征点的mask，并且计算上一帧所有已知运动模型的物体上的特征点在当前帧中的预计位置

void Estimator::set_mask_objs_prev(double cur_time)
{
    // 如果每一帧的所有待跟踪FAST点已经在上一帧确定了，则标注它们在上一帧中的mask中的位置
    featureTracker.draw_mask_fea_prev_obj(cur_time);
    
    double delta_t = cur_time - featureTracker.prev_time;
    Matrix3d RCam_1, RCam_2;
    Vector3d PCam_1, PCam_2;

    bool change_state = false;
    // if(USE_IMU && (solver_flag == NON_LINEAR || !NEED_ESTI_G))
    if(USE_IMU && solver_flag == NON_LINEAR)
    {
        Matrix3d motion_R = Rs[frame_count].transpose() * Rs[frame_count-1];
        Vector3d motion_P = Rs[frame_count].transpose() * (Ps[frame_count] - Ps[frame_count-1]);

        if(can_change_seneor_type)
        {
            Quaterniond delta_Q(motion_R);
            double delta_angle = acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0;
            if (abs(delta_angle) >= 2.0)
            {
                printf(" big delta_angle!\n");
                // ROS_INFO(" big delta_angle ");
                //return true;
            }
            double dealt_P = motion_P.norm();
            
            // 近乎静止,则改为使用纯双目配置
            // 此项目只会从IMU-Stereo转为Stereo，不会从Stereo转为IMU，因为这涉及到VI初始化以及全局坐标系的改变
            if(abs(delta_angle) < 0.1 && dealt_P < 0.01)
            {
                change_state = true;
                // if(USE_IMU && !NEED_ESTI_G) 
                // {
                //     featureTracker.has_motion_pred_first_two_frame = true;
                //     featureTracker.R_cam_motion = motion_R;
                //     featureTracker.P_cam_motion = motion_P;
                // }
                changeSensorType(0, 1);
            }
        }
        
        if(!change_state)
        {
            RCam_1 = prev_cam_R_using_imu[0];
            PCam_1 = prev_cam_P_using_imu[0];
            RCam_2 = prev_cam_R_using_imu[1];
            PCam_2 = prev_cam_P_using_imu[1];
        }
    }
    
    // if((!USE_IMU && STEREO) || (USE_IMU && STEREO && NEED_ESTI_G && solver_flag == INITIAL))
    if((!USE_IMU && STEREO) || (USE_IMU && solver_flag == INITIAL))
    {
        if(frame_count > 1)
        {
            // cout << "Ps[frame_count-2]: " << Ps[frame_count-2].transpose() << endl;
            // cout << "Ps[frame_count-1]: " << Ps[frame_count-1].transpose() << endl;
            
            // Rs和Ps中的值是经过视觉估计的结果影响的
            RCam_1 = Rs[frame_count-2] * ric[0];
            PCam_1 = Rs[frame_count-2] * tic[0] + Ps[frame_count-2];
            RCam_2 = prev_cam_R;
            PCam_2 = prev_cam_P;
            
            // Vector3d P = Rs[frame_count-1] * tic[0] + Ps[frame_count-1];
        }
    }
    
    // 如果使用IMU
    // if(USE_IMU && pred_cam_pose_with_IMU && ((solver_flag == NON_LINEAR) || (!NEED_ESTI_G)))
    if(USE_IMU && pred_cam_pose_with_IMU && (solver_flag == NON_LINEAR))
    {
        cur_cam_R = Rs[frame_count] * ric[0];
        cur_cam_P = Ps[frame_count] + Rs[frame_count] * tic[0];
    }
    // 如果不用IMU积分来预测相机位姿，则只能用相机的恒速模型
    else
    {
        // 第3帧开始才会有相机恒速模型
        if(frame_count > 1)
        {
            // VI初始化之后，这里其实得到的不是图像真实快门触发时刻的相机位姿，而是与IMU记录时刻相对应的相机位姿，但是用来计算相机的运动速度是无妨的，只要用对了delta_T即可
            // 计算的是世界坐标下的两帧间相机运动
            // Matrix3d R_cam_motion_pred_w;
            // Vector3d P_cam_motion_pred_w;
            
            // delta_T_cam_prev保存在estimator对象中，则是默认tracker和后端优化是在同一个主线程顺序完成！
            velocity_from_poses(RCam_1, PCam_1, RCam_2, PCam_2, MatrixXd::Identity(3,3), Vector3d::Zero(), delta_T_cam_prev, const_l_vel, const_ang_vel);
            
            // 对于静态点，是应该只计算两帧图像之间的预测运动，还是要计算出当前帧的位姿预测？如果使用的全部是上一帧点的深度值，那么只需要计算运动变换
            // pred_pose_with_vel(MatrixXd::Identity(3,3), Vector3d::Zero(), const_ang_vel, const_l_vel, delta_t, R_cam_motion_pred_w, P_cam_motion_pred_w);
            
            // 注意，这里得到的R_cam_motion_pred_w和P_cam_motion_pred_w是表达在世界坐标系下的两帧间的位姿变换；而IMU积分得到的运动变换是表达在上一帧的IMU坐标下
            // 计算当前帧相机的位姿预测
            // cur_cam_R = R_cam_motion_pred_w * prev_cam_R;
            // cur_cam_P = R_cam_motion_pred_w * prev_cam_P + P_cam_motion_pred_w;

            pred_pose_with_vel(prev_cam_R, prev_cam_P, const_ang_vel, const_l_vel, delta_t, cur_cam_R, cur_cam_P);
            // 无论如何，保持当前帧都有IMU坐标系全局位姿的预测
            Rs[frame_count] = cur_cam_R * ric[0].transpose();
            Ps[frame_count] = cur_cam_P - Rs[frame_count] * tic[0];
        }
    }

    // featureTracker中所需要的是表达在后一帧相机坐标系下的相机运动（从前一帧变换到后一帧）
    // 只有在使用IMU且不需要优化g（不需要估计首帧的位姿）的情况下才能得到在第1帧与第2帧之间相机的运动预测
    // 使用IMU积分得到的相机帧间运动应该相对比较准确！前提是已经完成了VI的初始化！
    // if(frame_count > 1 || (USE_IMU && pred_cam_pose_with_IMU && (!NEED_ESTI_G || solver_flag == NON_LINEAR)))
    if(frame_count > 1)
    {
        // 此相机运动变换是表达在后一帧相机坐标系下的，用于直接将两帧相机坐标下的对应静态点进行变换
        cam_motion_R_pred = cur_cam_R.transpose() * prev_cam_R;
        cam_motion_P_pred = cur_cam_R.transpose() * (prev_cam_P - cur_cam_P);

        // 此相机运动变换是表达在世界坐标系下的
        // cam_motion_R_pred_w = cur_cam_R * prev_cam_R.transpose();
        // cam_motion_P_pred_w = -cam_motion_R_pred_w * prev_cam_P + cur_cam_P;

        featureTracker.R_cam_motion = cam_motion_R_pred;
        featureTracker.P_cam_motion = cam_motion_P_pred;
    }

    done_cam_motion_pred = true;

    // 物体的运动模型也至少得第3帧起才会有
    if(frame_count > 1)
    {
        Matrix3d R_pred;
        Vector3d P_pred;
        for(auto &iter: featureTracker.RP_objs_pred)
        {
            // 只能是全局坐标系下假设物体为匀速，不能是相机坐标系下
            velocity_from_poses(RCam_1, PCam_1, RCam_2, PCam_2, iter.second.first, iter.second.second, delta_T_cam_prev, const_l_vel, const_ang_vel);
            // 对于动态物体是计算两帧之间的运动变换，因此起始姿态用单位矩阵。计算出来的是全局坐标系下的帧间运动变换
            pred_pose_with_vel(MatrixXd::Identity(3,3), Vector3d::Zero(), const_ang_vel, const_l_vel, delta_t, R_pred, P_pred);
            // 再将运动变换转换到表达在当前帧相机坐标系下
            iter.second.first  = cur_cam_R.transpose() * R_pred * prev_cam_R;
            iter.second.second = cur_cam_R.transpose() * (P_pred + R_pred * prev_cam_P - cur_cam_P);
        }
    }

    // 如果每一帧的所有待跟踪FAST点已经在上一帧确定了，则在这里给定它们在当前帧图像的预测位置
    if(!featureTracker.add_new_sift_in_next_frame)
    {
        // 如何设置当前帧跟踪点的预测位置，是使用恒速运动模型 还是 等待并使用flow_map
        if(use_motion_to_pred_fea_pos)
        {
            // 系统第二帧也进入该函数
            // 系统第二帧中对上一帧的新点使用光流网络的结果来作为预测值（假设IMU不需要初始化，则第2帧静态点的预测可以有由IMU积分得到的相机运动来计算，但是动态物体就无法进行，所以还是需要用flow_map）
            if (frame_count <= 1)
                featureTracker.Ptspredict_motion(frame_count, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, false);
            else
                // 之后的帧中对上一帧的新物体点也选择使用相机的运动模型来来计算点的预测位置
                // featureTracker.Ptspredict_motion(frame_count, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P);
                featureTracker.Ptspredict_motion(frame_count, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, false);
        }
        else
        {
            featureTracker.Ptspredict_motion(frame_count, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, false);
        }
    }
    cout << "finish set_mask_objs_prev and set predict for fea!" << endl;
}

// 将yolo_seg的full_seg_map所指的host内存上的数据转化为CV_8UC3的Mat,然后再取前2通道的数据作为最终的seg_map
void Estimator::build_seg_map(const int &raw_img_width, const int &raw_img_height)
{
    while(!featureTracker.copy_mask_bg)
    {
        usleep(300);
    }
    
    // 将full_seg_map第一个通道的map取出，将其中cls值为1、2和4的点以及类别正确但是物体太小的点都设置为黑色，其他的点设置为白色。将该mask作为新的FAST点检测时的mask
    if (!featureTracker.mask_solid_objs.data || featureTracker.row_img_prev != raw_img_height || featureTracker.col_img_prev != raw_img_width)
    {
        featureTracker.mask_solid_objs.create(raw_img_height, raw_img_width, CV_8UC1);

        featureTracker.mask_bg.create(raw_img_height, raw_img_width, CV_8UC1);
    }
    // featureTracker.mask_solid_objs.setTo(255);
    // featureTracker.mask_bg.setTo(255);
    
    // Mat mask_solid_objs = featureTracker.mask_solid_objs;
    // Mat mask_bg = featureTracker.mask_bg;
    
    TicToc t_mask;

    cv::Size size_full_seg_map = _yolov8seg->full_seg_map_size();

    int pad_full_seg_map_size = yolov8_seg_config.inputW * yolov8_seg_config.inputH;

    void* host_seg_map =  _yolov8seg->get_full_seg_map_host();
    // cout << "Start copy data to padddd seg map!" << endl;
    Mat padded_full_mask_map = Mat(size_full_seg_map, CV_8UC2, host_seg_map);
    
    int left = _yolov8seg->get_affine_matrix().i2d[2];
    int top  = _yolov8seg->get_affine_matrix().i2d[5];
    // left + raw_img_width-1? Range表示所取范围start ～ end，包含start但不包含end
    Range x_range(left, left + raw_img_width);
    Range y_range(top, top + raw_img_height);

    // 这种拷贝方法会不会导致full_seg_map是与临时物体共享内存（=号的赋值）？貌似不会，opencv应该对此表达式进行默认优化。想要更安全的话，可以用copyTo()?
    Mat unpad_seg_map = padded_full_mask_map(y_range, x_range);
    unpad_seg_map.copyTo(full_seg_map);
    cout << "Succeeded copy seg map!" << endl;

    // 既然这里需要分开，为何不直接在_yolov8seg的seg_map中把这两个通道分开存放？然后在上面分开复制到cls_map和id_map。是还需要用到两通道的full_seg_map吗？
    vector<Mat> channels;
    split(full_seg_map, channels);
    channels[0].copyTo(cls_map);
    channels[1].copyTo(id_map);
    
    uint8_t* host_mask_for_obj = (uint8_t*)host_seg_map + 3 * pad_full_seg_map_size;
    Mat padded_mask_for_obj = Mat(size_full_seg_map, CV_8UC1, (void*)host_mask_for_obj);
    featureTracker.mask_solid_objs = padded_mask_for_obj(y_range, x_range).clone();

    // cv::imshow("mask for solid objs", featureTracker.mask_solid_objs);
    // waitKey(0);

    uint8_t* host_mask_for_bg = host_mask_for_obj + pad_full_seg_map_size;
    Mat padded_mask_for_bg = Mat(size_full_seg_map, CV_8UC1, (void*)host_mask_for_bg);
    featureTracker.mask_bg = padded_mask_for_bg(y_range, x_range).clone();
    
    if(featureTracker.add_new_sift_in_next_frame)
        featureTracker.mask_bg_cur = featureTracker.mask_bg.clone();
    else
        featureTracker.mask_bg_cur = featureTracker.mask_bg;

    // cv::imshow("mask for bg", featureTracker.mask_bg);
    // waitKey(0);

    // 最后还是把这部分构建mask的工作放到yoloseg的后处理函数中，由GPU去完成！因为这里计算太耗时了（需要47毫秒）！！！
    // 注意，opencv中Size类型的声明为 Size_(_TP _width, _TP _height)，即创建它时分别给的是图像的width和height！！即Size的参数先后是图像的列和行！！
    // 这与直接定义Mat的尺寸是相反的，即Mat(rows,cols,type)!
    // https://blog.csdn.net/CYummy/article/details/82983677

    //channels[1].copyTo(id_map);
    // cout << "Succeeded split channels of seg map!" << endl;

    // cv::Mat seg_mask(raw_img_height, raw_img_width, CV_8UC1);

    // 先将形变物体区域去除
    // if(!deform_obj_cls.empty())
    // {
    //     for (const auto &element:deform_obj_cls){
    //         // 遍历cls_map中各个点，如果与element值相等，则seg_map中对应点被设为255，否则被设为0
    //         cv::compare(cls_map, element, seg_mask, cv::CMP_EQ);
    //         // seg_mask中值不为0的点，在mask中对应点的值被设为0，否则不变。
    //         mask_solid_objs.setTo(0, seg_mask);
    //         mask_bg.setTo(0, seg_mask);
    //     }
    // }
    // // 再去除面积过小的刚体的区域
    // if(!small_solid_objs.empty())
    // {
    //     for (const auto &element:small_solid_objs)
    //     {
    //         cv::compare(id_map, element, seg_mask, cv::CMP_EQ);
    //         // seg_mask中值不为0的点，在mask中对应点的值被设为0，否则不变。
    //         mask_solid_objs.setTo(0, seg_mask);
    //         mask_bg.setTo(0, seg_mask);
    //     }
    // }
    // cout << "Succeeded set mask for bg!" << endl;

    // // mask_solid_objs中再去除背景点。这个mask是用于检测物体的新特征点，因此需要把背景区域去除。用于光流追踪时不需要使用此mask
    // cv::compare(cls_map, 0, seg_mask, cv::CMP_EQ);
    // mask_solid_objs.setTo(0, seg_mask);
    // cout << "Succeeded set mask for solid obj!" << endl;

    // // mask_bg中再去除刚体目标点
    // for (const auto& element:solid_obj_cls){
    //     cv::compare(cls_map, element, seg_mask, cv::CMP_EQ);
    //     mask_bg.setTo(0, seg_mask);
    // }

    printf("Build mask for FAST detection costs: %fms\n", t_mask.toc());
    // ROS_DEBUG("Build mask for FAST detection costs: %fms", t_mask.toc());
}

void Estimator::assign_sift_FAST(double &dt)
{
    // 只考虑特定类别的刚体目标，即去除形变物体 和 面积过小的刚体
    num_solid_obj_frame = bbox_mask.size();
    num_objs_frame = num_solid_obj_frame + deform_obj_cls.size() + small_solid_objs.size();
    vector<int> valid_solid_obj;
    if(num_solid_obj_frame > 0)
    {
        for(auto &bbox: bbox_mask)
        {
            valid_solid_obj.push_back(bbox.first);
        }
    }
    
    // 根据当前帧sift点的检测和匹配结果，确认跟踪自上一帧有效sift点的匹配，并且添加当前帧新的sift点。结合seg_map，暂时确定各个sift点的class label和物体id
    bool initial_IMU_succ = (solver_flag == NON_LINEAR);
    // if(!NEED_ESTI_G) initial_IMU_succ = true;
    featureTracker.select_sift_V2(frame_count, r_img_gray, full_seg_map_prev, full_seg_map, map_depth_prev, map_flow, end_flow_post, initial_IMU_succ, use_mask_img_for_sift);
    
    while (!end_FAST_track)
    {
        usleep(300);
    }
    end_FAST_track = false;
    // 基于跟踪到的FAST/sift点以及新检测的sift点，设置mask，然后检测新的FAST点(左和右图像中）并将当前帧跟踪到的和新检测到的FAST点按要求加入到各个obj集合中（包括bg的）
    // 如果是系统初始帧，则需要等待depth_map估计完成
    bool marg_old_prev = (marginalization_flag == MARGIN_OLD);
    featureTracker.det_new_FAST_objs(frame_count, num_solid_obj_frame, full_seg_map, cls_map, map_depth, initial_IMU_succ, marg_old_prev, end_stereo_post, r_img_gray);
    // 可能需要等待GPU上的立体匹配完成
    featureTracker.assign_fea_objs(frame_count, map_depth, end_flow_post, end_stereo_post, _img1, valid_solid_obj, id_map, bbox_mask);
}

// 对当前帧和上一帧的所有物体进行关联。关联的策略首先是基于特征点的匹配，如果特征点不够，再使用密集的像素点和flow_map以及物体运动假设：
// 1. 对当前帧各个刚体目标（包括背景），统计其上的特征点 所对应的上一帧的特征点中属于各个全局obj_id的数量和比例，具体再分为以下情况：
// 1.1 对于当前帧的所有背景特征点，如果其中对应于上一帧的某个非背景的obj_id的点数大于某个阈值（且这些点都比较集中），则可以认为当前帧的yoloseg没有检测到该物体（模型漏检，或者该物体被一些背景遮挡了，如挡板，但是这样不应该出现大量特征点匹配），
//  则可以假设把这些点都归为此全局物体，再从上一帧中用flow_map来计算属于该物体的大量像素在当前帧的2D匹配（对于每一帧的每一个物体，我们都采样大概100个像素点和其深度值保存着，以备下一帧出现漏检！），再使用这些点的depth_map得到的深度与上一帧的点通过运动假设模型得到的深度进行比较，如果大部分点的2D区域与当前的FAST点接近，且3D点与匹配的3D点也接近，则可以确认该物体是漏检；否则就是被背景所遮挡，开启遮挡模式。
// 1.2 对于当前帧的某个obj上的特征点（首先点数得足够多，需要大于阈值），如果其最大部分或第二大部分（大于一定比例阈值）点对应的上一帧的背景，第二大部分点（与最大部分的比例应该大于一定阈值）或最大部分点对应特征点属于上一帧的某个obj_id（设为obj k）：
// 1.2.1 则要么上一帧漏检（根据1.1，虽然上一帧的漏检物体的部分FAST点和采样像素点可以根据匹配结果被校正为物体k（除非是系统首帧就出现了漏检，则我们直接放弃首帧的该物体），但上一帧它的新特征点仍然被当作背景点（这部分FAST点会比较少，因为规定背景区域FAST点间隔较大，而且由于这些点在当前帧的位置预测采用的是背景的运动，则这部分FAST点其实也很难被正确匹配到。然而SIFT匹配可能反而比较多,这点可以作为标准），如果上一帧的这些背景FAST点和物体k的像素平均3D距离小于阈值(且该物体k为静态的，即为背景），则认为上一帧的这些FAST也属于物体k（其实因为点比较少，也可以直接抛弃掉这些点）！否则转入1.2.2
// 1.2.2.要么该物体在上一帧中没有出现，在当前帧出现和上一帧背景区域的错误匹配（可能性很小）。这类情况更多是在当前帧该物体和上一帧基本找不到匹配点，需要加入二分图匹配！
//  如果当前帧物体位于图像边缘，则可能是 :
//      a.新物体挡住了上一帧背景点在当前帧所预测的位置而出现误匹配，但应该也只是少量（且应该不会有SIFT匹配），直接抛弃这些关联点，当前物体设为新物体；
//      b.原有被遮挡的物体重新出现且挡住了上一帧背景点在当前帧所预测的位置而出现误匹配，但已经来到相机视野边缘，同样应该很少FAST匹配，且应该不会有SIFT点匹配，同样抛弃这些关联点，同样暂时设为新物体。对于a和b的所谓新物体要判断是否与先前维持的被遮挡物体的轨迹相重合！
// 如果当前物体位于图像中间，则与上面b类似可能是被挡物体重新出现，按照b的方法执行。
// 1.3 如果当前帧的某个obj上的特征点集与上一帧的多个obj都相关，且第一大相关没有很大的优势，则加入二分图匹配。
// 2. 将静态物体上的点归到背景点集合中，修改featureTracker中各个点的cls label和obj_id

// 参数中需要给定前后两帧之间相机和各个动态物体的相对运动(相机的运动由IMU或视觉估计提供，物体的运动是用再往前2帧的速度，假设为恒速运动），以及左相机和IMU之间的外参
// 在C++中，函数参数不能真正地是数组，数组参数会自动退化为指向其第一个元素的指针。这种行为称为“数组退化为指针”
void Estimator::objs_matching(double dt, const vector<Vector3d> &Ps, const vector<Matrix3d> &Rs)
{
    bool initial_succ = (solver_flag == NON_LINEAR);
    // 对于系统前2帧，此时无法提供当前相机的位姿估计。物体的匹配只使用前后两帧像素点或特征点集的方差的相似性。或者在IMU未初始化之前，不进行动态物体的位姿估计，只进行每一帧中动态点的剔除？
    // if (frame_count < 2 && NEED_ESTI_G) 
    if (frame_count < 2) 
    {
        featureTracker.objs_matching_assign(frame_count, dt, full_seg_map, map_flow, map_depth, initial_succ, false);
    }
    else 
    {
        featureTracker.objs_matching_assign(frame_count, dt, full_seg_map, map_flow, map_depth, initial_succ, true, Ps, Rs);
    }
}

// 开始进行视觉前端的处理，包括CPU线程中的FAST特征点检测和跟踪，GPU线程中的SIFT点跟踪以及光流、深度和实例分割
void Estimator::Fea_Obj_Extract_Track(double &t_vio, bool &init_succ, double t, cv::Mat &left_img, cv::Mat &right_img, cv::Mat &l_gray_img, cv::Mat &r_gray_img)
{
    if(!_shutdown)
    {
        TicToc t_all;

        // while(raw_image_buffer.empty())
        // {
        //     usleep(2000);
        // }

        // RawImageData raw_image;
        // mBuf.lock();
        // 注意，进行立体深度估计的图像需要是经过立体校正和去畸变的！！最好选择能提供去畸变图像的数据集！
        // raw_image = raw_image_buffer.front();
        // raw_image_buffer.pop();
        // mBuf.unlock();

        // double cur_time_ = raw_image.time;
        // Mat left_img = raw_image.image_left;
        // Mat right_img = raw_image.image_right;
        
        _img  = left_img; 
        _img1 = right_img;
        l_img_gray = l_gray_img;
        r_img_gray = r_gray_img;

        featureTracker.row = l_img_gray.rows;
        featureTracker.col = l_img_gray.cols;
        // cout << "rows of l_img: " << featureTracker.row << endl;
        // cout << "cols of l_img: " << featureTracker.col << endl;

        float t_GPU;
        double cur_time_ = t;
        cur_img_rows = left_img.rows;
        cur_img_cols = left_img.cols;

        // 创建临时子线程专门用于GPU的推理任务。t_GPU为此次GPU处理的耗时。
        // GPU推理所需要的其他变量和内存是作为Estimator类的成员，这里是用类成员函数来作为线程函数
        // 注意，在C++中std::thread中，为了保证线程安全，对于thread构造函数中传入的参数，在构造函数中都是对这些实参进行拷贝，保存在特定的内存空间中！！因此，正常传入的实参，其后续在子线程内的修改，不会反映到其他线程！
        // 如果想子线程对传入参数的修改会影响到原变量，那就需要用到C++11中的std::ref(T x)，它返回一个std::reference_wrapper<T>的变量，该变量内部保存的是对原变量x的引用（实际是指针）！
        // 而且这个返回变量是可以被拷贝的（因此也就拷贝了对原变量的引用）！而在把这个std::reference_wrapper<T>的变量传递给函数时，它又会自动地“去掉外壳”，变为其内部保存的变量，即对原变量的引用！！相应地，对于常值变量，则需要用cref()函数～
        // 但是在使用ref()将变量传给thread构造函数时。需要保证原变量的生命周期至少和子线程一样长，否则会导致悬空引用！！！
        // https://juejin.cn/post/7307472360533901348

        // 另外，thread函数和std::bind等需要用可调用函数对象作为参数的函数（即函数式编程或泛型编程，这类函数一般都是模板函数）相同，其所有参数的类型均为右值引用类型（即T&&）！！
        // 而C++中规定 一般情况下，左值引用T&只可以接收左值（常量左值引用const T&比较特殊，可以接收常量左值或右值），右值引用T&&只能接收右值，
        // 而且左值引用和右值引用本身都是左值变量（因此右值引用类型的变量不可以被赋予右值引用...）！！
        // 那么对于thread和bind等函数，当像上面正确传入左值引用类型（通过ref或者cref）的变量时，又怎么可以赋给给其右值引用类型的形参呢？而且如果可调用函数对象的形参是左值引用类型，thread或bind的右值引用类型形参又是如何能再赋值给可调用对象作为它的实参呢？
        // 那是因为对于模板函数，C++有完美转发机制和引用折叠规则：
        // https://wudaijun.com/2015/03/cpp-rvalue-referrence/
        // 即对于模板函数，T&&不直接表示右值引用类型，而是需要根据传入的参数的实际类型来推断和退化（接收左值则退化为左值引用，接收右值就保持为右值引用），而左值引用类型和右值引用类型的对象本身是左值，因此其各自保持为原来的类型（即左值引用类型和右值引用类型）！
        // 此外，对于模板函数内部在使用形参（例如thread或者bind函数内部会把形参再度赋给可调用函数对象作为其实参）时，都要对其用static_cast<>来使得传给thread或bind的实参再次转化为原有的类型，这对于类形为右值引用的原变量很重要，因为右值引用变量本身为左值！！
        
        cout << "Start GPU process!" << endl;
        if(_sift_extract_match == nullptr)
        {
            _sift_extract_match = make_shared<Sift>(sift_config, nullptr, cpy_post_stream);
            _sift_extract_match->Allocate(l_img_gray.cols, l_img_gray.rows, false, 2048, use_mask_img_for_sift);
        }
        
        // 共享
        featureTracker.Sift_ = _sift_extract_match;

        // 对于多线程，一般最好是使用join来保证子线程任务执行完成后，主线程再结束并执行完整的析构，
        // 除非需要更灵活并且想要独立地提供一种同步机制来等待线程完成，在这种情况下应该使用detach
        std::thread GPUprocess(&Estimator::GPUProcesImage, this, ref(t_GPU));
        GPUprocess.detach();
        
        // 清理上一帧featureTracker的变量
        if(frame_count > 0) featureTracker.clear_var();

        if(USE_IMU && solver_flag == NON_LINEAR)
        {
            Get_Process_IMU(cur_time_);
        }

        // 在完成seg之前，先用上一帧保留的特征点来制作上一帧物体特征点的mask，并计算各个特征点在当前帧图像的像素坐标预测值
        if(frame_count > 0)
        {
            // 防止下面build_seg_map的执行比这里的子线程更早
            featureTracker.copy_mask_bg = false;
            
            std::thread set_mask_predict_pts(&Estimator::set_mask_objs_prev, this, ref(cur_time_));
            set_mask_predict_pts.detach();
        }
        else
        {
            featureTracker.cur_time = cur_time_;
            done_cam_motion_pred = true;
        }
        
        // 在GPU运行期间同时在当前线程进行FAST特征点的匹配和检测，但是
        // FASTFrame FastFeaFrame;
        
        while(!end_seg_post)
        {
            // 每隔0.3ms查看yolo_seg后处理是否完成
            usleep(300);
        }
        // 根据得到的seg结果来得到原始图像尺寸的seg_map，每个像素有2个同道，首通道uchar为class label，二通道int为所属动态物体的全局id（静态物体跟背景一样都是0）
        build_seg_map(cur_img_cols, cur_img_rows);
        end_seg_post = false;
        
        // 每一帧跟踪FAST可以在SIFT的结果到来之前进行。需要seg_map来排除掉明显不对的匹配。
        // 而每一帧中新FAST点的检测除了要先获取FAST的跟踪结果，还需要等待sift的检测和跟踪结果（排除已有的特征点）。
        // 此函数用子线程来完成，这样后面主线程可以进行sift的分配以及新FAST的检测
        // 系统的第2帧需要等待flow_map来为上一帧的新物体提供在当前帧的位置预测值。如果为了提高每一帧的FAST特征点的CPU光流计算精度，那么每一帧都应该先等待flow_map的结果，以便为上一帧的新物体特征点提供跟踪预测值？
        featureTracker.cur_img = l_img_gray.clone();
        
        // 如果是使用相机的运动模型来为特征点提供位置预测，则这里可以先进行FAST点的track了；如果要使用flow_map，则需要等待光流网络的推理结束
        std::thread track_FAST(&Estimator::CPU_Track_FAST, this, cur_time_);
        track_FAST.detach();
        
        // 等待GPU上完成sift的检测和匹配
        // 这里sift检测匹配 和 flow_map的计算 的先后顺序可以再斟酌。
        // 更高效的做法，应该是先完成sift检测，以便快速进行FAST的新特征点的检测；至于用flow_map来提供上一帧新物体特征点的估计，这只有在系统的第2帧是必须的，之后的帧都可以假设新物体为静态，只利用相机的运动模型来提供预测像素坐标
        while(!end_sift_post)
        {
            usleep(300);
        }
        end_sift_post = false;
        
        // 将sift特征匹配和检测的分配到各个物体。然后再添加新的FAST点，并将FAST点分配到各个物体。
        // 此函数应该在当前主线程中进行
        assign_sift_FAST(cur_time_);
        
        all_Time_GPU += t_GPU;

        // 如果有IMU且不需要优化g，则从一开始就保留前后2帧的相机位姿，而这两个位姿之间的相对运动是来自于IMU的积分（不受视觉位姿估计的影响）
        // if(USE_IMU && ((solver_flag == NON_LINEAR) || !NEED_ESTI_G))
        if(USE_IMU && (solver_flag == NON_LINEAR))
        {
            if(frame_count < 2)
            {
                prev_cam_R_using_imu[frame_count] = Rs[frame_count] * ric[0];
                prev_cam_P_using_imu[frame_count] = Rs[frame_count] * tic[0] + Ps[frame_count];
            }
            else
            {
                prev_cam_R_using_imu[0] = prev_cam_R_using_imu[1];
                prev_cam_P_using_imu[0] = prev_cam_P_using_imu[1];
                
                prev_cam_R_using_imu[1] = Rs[frame_count] * ric[0];
                prev_cam_P_using_imu[1] = Rs[frame_count] * tic[0] + Ps[frame_count];
            }
        }
        
        while(!end_flow_post)
        {
            usleep(300);
        }
        end_flow_post = false;

        while(!end_stereo_post)
        {
            usleep(300);
        }
        end_stereo_post = false;

        // 这部分应该在sift和FAST都分配完成之后再进行。需要使用ROS
        // if (SHOW_TRACK)
        // {
        //     cv::Mat imgTrack = featureTracker.getTrackImage();
        //     pubTrackImage(imgTrack, cur_time_);
        // }

        std::thread sample_pts(&Estimator::sample_pixel_objs,this);

        // 进行物体关联。综合使用seg_map、flow_map和depth_map，并构建二分图匹配来进行物体的帧间关联。应该要先进行关联，然后再判断物体是否运动？
        // !!但是物体关联步骤需要先知道相机的位姿！

        // 局部vector存放指针，使用完也不需要逐个置空
        // ObjFeaNumFrame Vec_Ptr_NumFeaObjFrame;
        // Vec_Ptr_NumFeaObjFrame.push_back(&featureTracker.num_obj_sift);
        // Vec_Ptr_NumFeaObjFrame.push_back(&featureTracker.num_obj_FAST);

        // 是否有开启子线程进行后端处理。默认不开启（0），因为实时跑的时候同一帧的前后端必须紧接着执行才算是真正的执行时间！
        if(MULTIPLE_THREAD)  
        {   
            // 为什么多线程下需要每隔2帧才输入处理后的图像特征？那奇数帧的数据呢？应该是因为开启多线程就意味着追求实时性，而后端的频率较低，因此需要每隔2帧图像才进行一次后处理。
            // 因为VINS中要求对特征点的跟踪必须是连续的，每一帧的相机位姿只与其时间戳有关，在VI初始化前用和上一帧的特征匹配来计算运动；VI初始化后则是使用该点首帧的3D坐标来重投影到各帧，调整各帧的位姿！
            // 但是VINS-Mono又要求图像的频率最好大于10hz，因此对于KITTI数据集，这里是不宜再采用间隔一帧的后端处理方式
            // 多线程（即VIO的后端在单独的子线程中，区别于VIO前端子线程）的情况下，前端处理完了之后是否应该调用inputFeature()函数将特征数据作为消息发布出去？
            // 另外，后端和视觉处理前端的真的能并行处理吗？？后端包含了位姿估计，它跟视觉前端必须得是顺序关系呀！现实世界下每一帧的处理用时就应该是该帧的视觉处理+后端位姿估计的时间！
            //if(inputImageCnt % 2 == 0)
            {
                mBuf.lock();
                featureBuf.push(cur_time_);
                mBuf.unlock();
            }
        }
        else
        {
            mBuf.lock();
            featureBuf.push(cur_time_);
            mBuf.unlock();
            TicToc processTime;
            processMeasurements();
            sample_pts.join();
            printf("process time of backend: %f\n", processTime.toc());
        }
        
        featureTracker.row_img_prev = cur_img_rows;
        featureTracker.col_img_prev = cur_img_cols;
        
        map_depth_prev = map_depth.clone();
        // 当前帧的物体区域，用于下一帧均匀采样背景跟踪点
        if(featureTracker.add_new_sift_in_next_frame) 
        {
            // featureTracker.mask_bg_prev = featureTracker.mask_bg.clone();

            full_seg_map_prev = full_seg_map.clone();

            // if(featureTracker.show_tracked_fea) 
            {
                featureTracker.prev_color_img_l = _img.clone();
                // featureTracker.prev_color_img_r = _img1.clone();
            }
        }
        
        // 这个是以防需要像素点采样时，到此处子线程sample_pts还没完成（这只可能发生在系统首帧）
        // while(!done_sample)
        // {
        //     usleep(500);
        // }
        t_vio = t_all.toc();
        if (solver_flag == NON_LINEAR) init_succ = true;
    }
}

// 视觉前端的GPU部分，包含 当前帧和前一帧的左图像间、当前帧左右图像间的SIFT检测 以及 三个网络的推理
// __attribute__((no_sanitize("address")))
void Estimator::GPUProcesImage(float &time_calcu)
{
    // 不统计初始帧的GPU操作时间
    if (frame_count > 0)
    {
        bool adjust_para_sift = false;
        // 是否根据上一帧的sift跟踪点数量 和 新sift点数量 来调整当前帧sift检测和匹配的参数
        if(adjust_para_sift)
        { 
            // 还是使用num_new_sift_bg？num_new_sift_bg中包含 有和没 立体匹配的所有背景新sift点（没有立体匹配的新sift点会非常多）
            // 而num_bg_sift_with_dep包含有立体匹配的背景跟踪和新sift点
            // if(featureTracker.num_bg_sift_with_dep < 30 && _sift_extract_match->sift_config_.thres_Ambiguity_stereo < 0.95) 
            //     _sift_extract_match->sift_config_.thres_Ambiguity_stereo += 0.01;
            // else if(featureTracker.num_bg_sift_with_dep > 70)
            //     _sift_extract_match->sift_config_.thres_Ambiguity_stereo -= 0.01;
            
            // if(featureTracker.num_track_sift_bg < 40 && _sift_extract_match->sift_config_.thres_Ambiguity_flow < 0.92)
            //     _sift_extract_match->sift_config_.thres_Ambiguity_flow += 0.01;
            // else if(featureTracker.num_track_sift_bg > 200)
            //     _sift_extract_match->sift_config_.thres_Ambiguity_flow -= 0.01;
        }
        checkCudaRuntime(cudaEventRecord(start_preproc_seg, common_infer_stream));
    }

    _yolov8seg->preproc_infer(_img, stop_infer_seg);
    
    if (!deform_obj_cls.empty()) deform_obj_cls.clear();
    if (!small_solid_objs.empty()) small_solid_objs.clear();
    if (!solid_obj_cls.empty()) solid_obj_cls.clear();
    if (!bbox_mask.empty()) bbox_mask.clear();

    // 系统首帧不用进行flow_map的infer
    if (frame_count == 0)
    {
        _sift_extract_match->sift_config_.thres_Ambiguity_flow = Thres_Ambiguity_Flow;
        _sift_extract_match->sift_config_.thres_Ambiguity_stereo = Thres_Ambiguity_Stereo;

        _DPstereo->preprocess_input(_img, _img1, stop_cpy_input_depth);
        
        // 需要等待到yoloseg的后处理结束后再进行之后的GPU操作，在其他host线程中进行segmap画到大的image上，以便定点搜索
        _yolov8seg->postprocess_output(bbox_mask, solid_obj_cls, deform_obj_cls, small_solid_objs, stop_infer_seg, stop_post_seg);
        
        // stream_cpy会阻塞当前host线程直到上面所有GPU中的任务（推理和复制）都完成
        checkCudaRuntime(cudaEventSynchronize(stop_post_seg));
        
        // 通知主线程seg的后处理完成了
        end_seg_post = true;

        _sift_extract_match->preprocess_input(l_img_gray, r_img_gray, _yolov8seg->get_full_seg_map_host(), _yolov8seg->input_width_, 
                                                _yolov8seg->input_height_, _yolov8seg->left_pad, _yolov8seg->top_pad);

        // sift的gpu操作均用默认stream，会阻塞host线程
        _sift_extract_match->detect_match_sift(true);
        end_sift_post = true;

        // cout << "Succeed detect and match sift!" << endl;

        _rapidflow->preprocess_input(_img);
        
        _DPstereo->infer(stop_infer_depth, stop_cpy_input_depth);
        
        // SIFT的后处理全都在当前CPU线程中。最后将sift的match后处理放在了其他线程中，以便不阻碍这里GPU中其他任务的执行
        // _sift_extract_match->postprocess(-Y_shift_right_image/mMinDepthPt, true);
        // end_sift_post = true;

        _DPstereo->postprocess_output(map_depth, stop_infer_depth);
        end_stereo_post = true;
        // 系统首帧的情况下，光流模型的后处理仅仅是将预处理后的图像复制到前一帧的内存位置
        _rapidflow->postprocess_output(map_flow, false);
        
        // 此处的阻塞主要是为了等待首帧的flow的后处理，其内部没有阻塞（有可能yoloseg或DPStereo的后处理操作都还没完成）
        checkCudaRuntime(cudaStreamSynchronize(cpy_post_stream));
        end_flow_post = true;
        cout << "End processing in GPU for frame_count == 0!" << endl;
    }
    else
    {
        if(frame_count == 1) cout << "Start processing in GPU for frame_count > 0!" << endl;

        _rapidflow->preprocess_input(_img, stop_infer_flow);

        _yolov8seg->postprocess_output(bbox_mask, solid_obj_cls, deform_obj_cls, small_solid_objs, stop_infer_seg, stop_post_seg);

        // stream_cpy会阻塞当前host线程直到上面所有GPU中的任务（推理和复制）都完成
        checkCudaRuntime(cudaEventSynchronize(stop_post_seg));
        
        end_seg_post = true;

        // sift检测的预处理中的数据传输使用默认stream，会阻塞此处host线程
        _sift_extract_match->preprocess_input(l_img_gray, r_img_gray, _yolov8seg->get_full_seg_map_host(), _yolov8seg->input_width_, 
                                                _yolov8seg->input_height_, _yolov8seg->left_pad, _yolov8seg->top_pad);
        // sift的检测和匹配会阻塞当前host线程
        _sift_extract_match->detect_match_sift();
        end_sift_post = true;
        
        // flow的结果是32FC2，通道1为flow_x，通道2为flow_y，光流的定义为flow = pts_img2 - pts_img1，即后一帧点坐标减去前一帧点坐标
        _rapidflow->infer(stop_cpy_input_flow, stop_infer_flow);
        
        // 用flow的infer时间来覆盖sift的后处理和挑选匹配点与新点的时间（CPU上的操作）
        // _sift_extract_match->postprocess(-Y_shift_right_image/mMinDepthPt);
        // end_sift_post = true;
        
        // _rapidflow->postprocess_output(map_flow, true, stop_infer_flow, stop_post_flow);
        _rapidflow->postprocess_output(map_flow, true, stop_infer_flow);
        end_flow_post = true;
        // 把stereo任务放在sift检测之后，因为等到sift和FAST点跟踪和检测之后才需要用到depth_map（提供没有深度估计的跟踪点的深度值）
        // 这里不需要用stop_cpy_input_depth，因为上面与stream_cpy关联的stop_post_seg事件把当前host线程阻塞了
        // 立体视差的定义是 pts_dsip = pts_left - pts_right，因此每个点的视差一定是大于0的
        // stop_post_depth用于阻塞host线程，等待推理结果从device传到host
        _DPstereo->preprocess_input(_img, _img1, stop_cpy_input_depth);
        _DPstereo->infer(stop_infer_depth, stop_cpy_input_depth);
        _DPstereo->postprocess_output(map_depth, stop_infer_depth, stop_post_depth);
        
        // checkCudaRuntime(cudaEventSynchronize(stop_post_depth));
        end_stereo_post = true;
    }

    if(frame_count > 0)
    {
        // 用于计算时间间隔的两个Event可以不关联到同个stream上。不包括首帧的计算时间
        cudaEventElapsedTime(&time_calcu, start_preproc_seg, stop_post_depth);
        cout<< "Time for GPU process of a frame is" << time_calcu <<"ms"<<endl;
    }
    // 得到推理结果之后，应该要立即将其中位于页锁内存的数据拷贝到其他内存中，因为这些推理结果在GPU线程中不会一直被保存，下一次推理的后处理之前会被清除！
    // std::cout<<"Infer in GPU succeed!"<<endl;
}

// 传统的VINS视觉前端，这里只进行前后两帧左图像，以及当前帧左右图像间的 FAST点的检测和匹配！不涉及点的深度估计 和 相机位姿估计
void Estimator::CPU_Track_FAST(double _cur_time)
{
    cout << "Start track FAST in CPU!" << endl;
    if (frame_count == 0)
    {
        // 左右相机仅有部分视野重叠，需要确定左图像左侧中一定无法在右图像中找到匹配的区域的宽度，这部分的点在寻找有立体匹配的特征点时不需要考虑
        // 定义右相机视锥体左侧面的方程系数， aX+bY+cZ+d=0。
        // 对于有立体匹配的左图像的点，其3D坐标必须满足aX+bY+cZ+d>0，否则该点就只能在左图像中观测到，意味这立体匹配是错误的
        featureTracker.r_cam_3D_plane[0] = FOCAL_LENGTH_X/mbf;
        featureTracker.r_cam_3D_plane[1] = 0;
        featureTracker.r_cam_3D_plane[2] = COL/mbf/2.0;
        featureTracker.r_cam_3D_plane[3] = -1;
        // 物体的深度阈值为25m，背景的深度阈值为40m。
        // 假设右相机视锥体左侧面与深度40m处的垂直平面的交线上的点为（X，Y，40),则该点在左图像中的投影像素的横坐标为(X,Y,40)需满足aX+c*40-1=0
        // 同理深度25m处的交线上的点（X，Y，25）需满足aX+c*25-1=0。可以求得该交线在左图像中的位置（图像中的一条竖线）
        // 左图像中在该线左边的背景点或物体点，要么在右图像中没有相应的观测点；要么在右图像中也有观测，但是其深度超过了相应的阈值，不考虑该点。
        // 左图像中在该线右边的背景点或物体点，仍然有可能会在右图像中无法观测到，其左相机坐标系3D坐标需要满足上面的平面约束aX+bY+cZ+d>0
        double a = featureTracker.r_cam_3D_plane[0];
        double c = featureTracker.r_cam_3D_plane[2];
        featureTracker.bg_left_border_left_img    = (1 - c * mThDepthBg)/a/mThDepthBg * FOCAL_LENGTH_X + SHIFT_X;
        featureTracker.obj_left_border_left_img   = (1 - c * mThDepthObj)/a/mThDepthObj * FOCAL_LENGTH_X + SHIFT_X;
        if(featureTracker.bg_left_border_left_img < 0) featureTracker.bg_left_border_left_img = 0;
        if(featureTracker.obj_left_border_left_img < 0) featureTracker.obj_left_border_left_img = 0;
        // assert((featureTracker.bg_left_border_left_img > 0 && featureTracker.obj_left_border_left_img > 0) && "False border in left image!");
        
        // 同时，右图像中最右侧的部分区域的点是不可能在重叠视野内的，该区域大小与所考虑的场景深度有关
        featureTracker.bg_right_border_right_img  = (COL/FOCAL_LENGTH_X/2.0 - mbf/FOCAL_LENGTH_X/mThDepthBg) * FOCAL_LENGTH_X + SHIFT_X;
        featureTracker.obj_right_border_right_img = (COL/FOCAL_LENGTH_X/2.0 - mbf/FOCAL_LENGTH_X/mThDepthObj) * FOCAL_LENGTH_X + SHIFT_X;
        if(featureTracker.bg_right_border_right_img >= COL) featureTracker.bg_right_border_right_img = COL-1;
        if(featureTracker.obj_right_border_right_img >= COL) featureTracker.obj_right_border_right_img = COL-1;
        // assert((featureTracker.bg_right_border_right_img < COL && featureTracker.obj_right_border_right_img < COL) && "False border in right image!");

        // 物体像素点的加权平均视差值，认为太近的点不太可信（也不太容易匹配到），因此加大远点的视差值贡献。暂时不使用，因为深度的范围太大，导致视差的范围太大
        // featureTracker.ave_disp_obj = 2/3 * mbf/mThDepthObj + 1/3 * mbf/mMinDepthPt;
        // featureTracker.ave_disp_bg  = 2/3 * mbf/mThDepthBg + 1/3 * mbf/mMinDepthPt;

        cout << "Succeeded calcu border bg and img!" << endl;
    }
    
    while(!done_cam_motion_pred)
    {
        usleep(300);
    }

    vector<Vector3d> P_cam(2,Vector3d());
    vector<Matrix3d> R_cam(2,Matrix3d());
    R_cam[0] = prev_cam_R;
    P_cam[0] = prev_cam_P;
    
    R_cam[1] = cur_cam_R;
    P_cam[1] = cur_cam_P;

    done_cam_motion_pred = false;
    
    TicToc featureTrackerTime;
    
    // 每一次跟踪都需要用map_flow来为上一帧的新物体提供在当前帧的位置预测值？其实也可以不需要，即假设新物体在两帧间是静态的，则可以用相机的运动模型来预测其上特征点在当前帧的像素坐标
    featureTracker.trackImage(frame_count, l_img_gray, end_flow_post, full_seg_map, end_FAST_track, map_depth_prev, map_flow, P_cam, R_cam);
    
    printf("FAST Tracker time: %fms\n", featureTrackerTime.toc());
    
}

// 此函数是将从IMU传感器中传来的（应该是通过ROS系统从传感器获取，然后ROS再发布）的IMU测量信息插入到estiamtor的成员变量mBuf等中）
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();
    // 如果此前系统已经VI初始化成功（对于双目而言就是首个滑窗已满并且LBA成功）
    // VI初始化成功之后，当有新的IMU测量被插入消息队列时，用IMU测量来估计此时刻的IMU坐标下的全局位姿和速度
    if(0)
    {
        if (solver_flag == NON_LINEAR)
        {
            // 加锁，可以在不同线程中使用最新的IMU测量来预测系统当前最新的状态（指的是IMU坐标系的全局状态，latest_P等），例如此函数就是在IMU测量的接收
            mPropagate.lock();
            // 快速进行预积分计算，得到当前时刻的IMU坐标系的粗略位姿估计，用于实时视觉化
            fastPredictIMU(t, linearAcceleration, angularVelocity);
            // 发布位姿消息，用于实时视觉化.需要使用ROS
            // pubLatestOdometry(latest_P, latest_Q, latest_V, t);
            mPropagate.unlock();
        }
    }
}

// 每跟踪完一帧图像并获得该帧特征点(个数）的消息，则将其插入的特征队列中。
// 这里和VINS项目不一样，VINS中这里插入队列的是时间戳和特征点的具体信息，因为视觉跟踪前端与优化后端可能不是同步的。
// 而在此项目中默认是同步的，且复制所有特征点信息的代价太大，因此默认把点信息的map放在featureTracker对象中，这里只传递特征点的检测数量（如果要搞成异步，则必须复制所有特征点信息，不然要保存多帧的特征点信息）
// 此函数在此项目中其实未被用到
void Estimator::inputFeature(double t, const NumFeaObjFrame &featureFrame)
{
    mBuf.lock();
    // featureBuf.push(make_pair(t, featureFrame));
    featureBuf.push(t);
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

// 根据前后两帧相机的时间戳 和 当前估计的相机和IMU的触发time shift（td），来指定这两帧相机形成观测的时刻所应该对应的IMU测量的时间戳t0和t1
// 从IMU测量测量队列中获取这两个时刻间的所有测量 并 进行当前帧的预积分，而这两个时刻的IMU测量可能需要通过插值来估计
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    // 需要队列中最新的一个IMU的时刻比 要获取的时间区间的右边界时刻 要大，即更晚
    if(t1 <= accBuf.back().first)
    {
        double t0_ = t0;
        bool first_get = false;
        float tab_init = 0.0, tab_end = 0.0;
        float tinit = 0.0, t_end = 0.0;
        Vector3d acc_first, gyr_first, acc_end, gyr_end;
        if(frame_count == 0) 
            t0_ = t0 - 0.06;
        // else
        //     t0_ = t0 - 0.001;

        // 获取的首个IMU数据必须比上一帧图像的时间戳晚。
        // 为什么=t0的IMU也要抛弃？因为上一帧图像时刻对应的IMU测量已经在上一帧中被估计并保存了在acc_0和gro_0中，这里只需要在上一帧相机位姿的基础上获取后续的IMU测量进行预积分
        // TODO：上一帧如果进行了滑窗LBA，则time shift变量td也会被更新，则与上一帧视觉测量的发生时间 所对应的 IMU测量的时间戳则会改变（与td值有关），那它们的IMU测量（插值）估计值应该也是不一样的吧？
        // 当前帧并不会用上一帧LBA优化后的td去改变prev_time的值，因此此处的t0与上一帧的t1是一致的值。
        // 其实td改变的只是当前帧的截止帧IMU的时间戳t1，LBA会优化的是这个时刻的IMU的状态量；而当前帧被优化的td会决定用哪个时刻的插值图像的特征观测坐标（通过两观测图像间的光流值来插值其中间某个时刻的图像上的对应匹配点的坐标）来作为视觉观测约束！
        // ！注意，系统首帧时给定的上一帧时刻t0是-1，因此这里意味着比系统首帧t1早的IMU也都会被接受并插入accVector，用以初始化初始帧的姿态（近似）
        while (accBuf.front().first < t0_)
        {
            // if(frame_count > 0)
            // {
            //     last_t_prev = accBuf.front().first;
            //     last_acc_prev = accBuf.front().second;
            //     last_gyr_prev = gyrBuf.front().second;
            // }
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            // 获取第一个大于等于上一帧图像时间戳的IMU测量
            if(!first_get)
            {
                first_get = true;
                if(frame_count > 0)
                {
                    tinit = accBuf.front().first - t0;
                    if (tinit > 0)
                    {
                        // 只有相机时刻的测量值是需要插值的，其他时刻都不需要，只是时长需要注意
                        // tab_init = accBuf.front().first - last_t_prev;
                        // 线性插值。是否需要更加准确的插值，比如三次样条？或者这里直接计算区间的平均acc和gyr，后续也只使用欧拉积分而不是中值积分。
                        // acc_first = accBuf.front().second - (accBuf.front().second - last_acc_prev) * tinit/tab_init;
                        // gyr_first = gyrBuf.front().second - (gyrBuf.front().second - last_gyr_prev) * tinit/tab_init;
                        // accVector.emplace_back(accBuf.front().first, acc_first);
                        // gyrVector.emplace_back(accBuf.front().first, gyr_first);

                        accVector.push_back(accBuf.front());
                        gyrVector.push_back(gyrBuf.front());
                    }
                }
                else
                {
                    accVector.push_back(accBuf.front());
                    gyrVector.push_back(gyrBuf.front());
                }
                accBuf.pop();
                gyrBuf.pop();
            }
            else
            {
                last_t_cur = accBuf.front().first;
                last_acc_cur = accBuf.front().second;
                last_gyr_cur = gyrBuf.front().second;
                accVector.push_back(accBuf.front());
                accBuf.pop();
                gyrVector.push_back(gyrBuf.front());
                gyrBuf.pop();
            }
        }
        // 会获取大于或等于当前帧图像时刻的第一个IMU数据，而且不会把这个测量从对列中删去，下一帧图像的预积分计算会需要。
        // 插值得到当前帧图像时刻的IMU数据
        tab_end = accBuf.front().first - last_t_cur;
        t_end = t1 - last_t_cur;
        acc_first = (accBuf.front().second - last_acc_cur) * t_end/tab_end + last_acc_cur;
        gyr_first = (gyrBuf.front().second - last_gyr_cur) * t_end/tab_end + last_gyr_cur;
        accVector.emplace_back(t1, acc_first);
        gyrVector.emplace_back(t1, gyr_first);
        
        // last_t_prev = last_t_cur;
        // last_acc_prev = last_acc_cur;
        // last_gyr_prev = last_gyr_cur;
    }
    // 否则等待队列中放进时刻比t1更晚的IMU
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

// 调用GPU对当前帧各个待关联的刚体采样像素点和点深度
// __attribute__((no_sanitize("address")))
void Estimator::sample_pixel_objs()
{
    done_sample =false;
    float* flow_ptr = _rapidflow->get_binding_input_cur_device();
    float* depth_ptr = _DPstereo->get_binding_output_device();
    
    // 就算当前帧中除了背景外没有任何跟踪或检测的物体，也要进入obj_matching去完成背景点的信息修改
    featureTracker.check_total_lost_cur_objs = false;
    // 但是如果当前帧没有物体，则一定不需要进行像素点采样
    // if(featureTracker.valid_detect_obj.size() > 0)
    int W_dep_map = DPstereo_config.inputW;
    int H_dep_map = DPstereo_config.inputH;
    float left_shift_dep = _DPstereo->inputpadder.Padder[2];
    float top_shift_dep = _DPstereo->inputpadder.Padder[0];
    
    _yolov8seg->sample_pixel((float*)&featureTracker.sampled_pixel[0][0], depth_ptr, flow_ptr, bbox_mask, 
                            featureTracker.valid_detect_obj, W_dep_map, H_dep_map, left_shift_dep, 
                            top_shift_dep, done_sample, featureTracker.check_total_lost_cur_objs);
    
    // done_sample = true;
    // 等待device-host数据传输完成
    // while(!done_sample)
    // {
    //     usleep(500);
    // }
}   

void Estimator::Get_Process_IMU(double cur_cam_time)
{
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
    
    // td表示估计出的IMU时间戳与相机时间戳之间的触发时刻差
    // 则curTime表示的是当前帧相机所应该要对应的IMU时间戳（指是在真实时间上是同时测量到，td这个量是优化估计得到的）
    // curTime_cam = cur_cam_time;
    cout << "td in current frame: " << td << endl;
    curTime = cur_cam_time + td;
    
    while(1)
    {
        if (IMUAvailable(curTime))
            break;
        else
        {
            printf("wait for imu ... \n");
            // 不使用多线程时，就不允许当前帧图像跟踪完之后，当前帧图像时刻对应的IMU数据还没传进来吗？
            // 首先应该是不可能的，IMU的发布频率比相机高的多；其次，如果要等待IMU，就会影响到下一帧的图像的跟踪了（因为不知道IMU是否还会到来）！
            if (! MULTIPLE_THREAD)
                return;
            std::chrono::milliseconds dura(5);
            std::this_thread::sleep_for(dura);
        }
    }

    if(frame_count == 0) prevTime = curTime;
    mBuf.lock();
    
    // 得到的首个IMU数据的时间戳会晚于上一帧图像的时间戳（上一帧图像时刻的IMU已经在在上一帧的预积分期间内插得到了），但最后一个IMU会不早于当前帧图像的时间戳（当前帧图像时刻的IMU在当前帧预积分期间内插得到）
    // 必须得是系统第2帧图像开始才会进行IMU的处理，因为预积分是定义在两帧图像的后者中
    // 系统一开始prevTime初始值为-1
    getIMUInterval(prevTime, curTime, accVector, gyrVector);
    
    mBuf.unlock();
    
    if(!initFirstPoseFlag)
        // 得到系统首个IMU时刻的IMU坐标下相对于东北天坐标下的姿态的初始估计（实际上就是假定了一个世界坐标系，其与东北天坐标系还是有着roll和pitch角）
        initFirstIMUPose(accVector);
    // 通过IMU数据的积分来预测当前帧相机（或IMU）的位姿
    // 初始帧时刻的预积分是没有意义的！这里仍进行，只是为了创建一个非空对象指针
    // VINS原代码中这里用else，即首帧时刻不执行下面的代码，不生成预积分对象。但是在本项目中，需要插值得到首帧时刻的IMU值并需要将其赋给相关变量，因此需要执行下面的代码
    // else
        {
            for(size_t i = 0; i < accVector.size(); i++)
            {
                double dt;
                // 对于系统初始帧，prevTime等于curTime，这种情况在processIMU中有特殊处理
                if(i == 0)
                    dt = accVector[i].first - prevTime;
                else if (i == accVector.size() - 1)
                    dt = curTime - accVector[i - 1].first;
                else
                    dt = accVector[i].first - accVector[i - 1].first;
                // 将新时刻的IMU数据融合进预积分中
                processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
            }
        }
}

// 此函数是系统的后端（完成VI联合初始化与BA优化，滑窗更新），可以在主线程或子线程中被执行
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        // 某帧图像中跟踪或检测到的特征点，feature的first是该帧图像的时间戳；
        // map中的首个int表示的观测到特征地图点（在当前滑窗内）的全局ID；第2个int表示该地图点在该帧的该观测是位于左相机（0）还是右相机（1）（注意，对同一地图点的左右图像的观测可同时在vector中）
        double feature;
        // 浅拷贝，指针之间的赋值
        if(!featureBuf.empty()) feature = featureBuf.front();
        // td表示估计出的IMU时间戳与相机时间戳之间的触发时刻差
        // 则curTime表示的是当前帧相机所应该要对应的IMU时间戳（指是在真实时间上是同时测量到，td这个量是优化估计得到的）
        curTime_cam = feature;

        // 只有需要VI初始化且再未完成VI初始化时才在此处进行IMU的预计分
        // if(USE_IMU && ((solver_flag == INITIAL) && NEED_ESTI_G))
        if(USE_IMU && (solver_flag == INITIAL))
        {
            cout << "td in current frame: " << td << endl;
            curTime = curTime_cam + td;
            vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
            if(!featureBuf.empty())
            {
                while(1)
                {
                    if ((!USE_IMU || IMUAvailable(curTime)))
                        break;
                    else
                    {
                        printf("wait for imu ... \n");
                        // 不使用多线程时，就不允许当前帧图像跟踪完之后，当前帧图像时刻对应的IMU数据还没传进来吗？
                        // 首先应该是不可能的，IMU的发布频率比相机高的多；其次，如果要等待IMU，就会影响到下一帧的图像的跟踪了（因为不知道IMU是否还会到来）！
                        if (! MULTIPLE_THREAD)
                            return;
                        std::chrono::milliseconds dura(5);
                        std::this_thread::sleep_for(dura);
                    }
                }
                if(frame_count == 0) prevTime = curTime;

                mBuf.lock();
                // 得到的首个IMU数据的时间戳会晚于上一帧图像的时间戳（上一帧图像时刻的IMU已经在在上一帧的预积分期间内插得到了），但最后一个IMU会不早于当前帧图像的时间戳（当前帧图像时刻的IMU在当前帧预积分期间内插得到）
                // 必须得是系统第2帧图像开始才会进行IMU的处理，因为预积分是定义在两帧图像的后者中
                // 系统一开始prevTime初始值为-1
                getIMUInterval(prevTime, curTime, accVector, gyrVector);
                // feature元素中的ObjFeaNumFrame所指向的内存不是new出来的，而是类成员变量（栈内存上，对象消失则会释放），因此这里不需要对每个指针赋为NULL
                featureBuf.pop();
                mBuf.unlock();

                if(USE_IMU)
                {
                    if(!initFirstPoseFlag)
                        // 得到系统首个IMU时刻的IMU坐标下相对于东北天坐标下的姿态的初始估计（实际上就是假定了一个世界坐标系，其与东北天坐标系还是有着roll和pitch角）
                        initFirstIMUPose(accVector);
                    // 通过IMU数据的积分来预测当前帧相机（或IMU）的位姿
                    // 初始帧时刻的预积分是没有意义的！这里仍进行，只是为了创建一个非空对象指针
                    // else
                    {
                        for(size_t i = 0; i < accVector.size(); i++)
                        {
                            double dt;
                            // 对于系统初始帧，prevTime等于curTime，这种情况在processIMU中有特殊处理
                            if(i == 0)
                                dt = accVector[i].first - prevTime;
                            else if (i == accVector.size() - 1)
                                dt = curTime - accVector[i - 1].first;
                            else
                                dt = accVector[i].first - accVector[i - 1].first;
                            // 将新时刻的IMU数据融合进预积分中
                            processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                        }
                    }
                }
            }
        }
        else
        {
            if(!featureBuf.empty())
            {
                mBuf.lock();
                featureBuf.pop();
                mBuf.unlock();
            }
        }

        mProcess.lock();
        
        if(frame_count > 0) delta_T_cam_new = curTime_cam - prevTime_cam;
        
        //TicToc t_objs_matching;
        // if (frame_count < 2 && NEED_ESTI_G)
        if (frame_count < 2)
        {
            cout << "Start objs_matching!" << endl;
            objs_matching(delta_T_cam_new);
        }
        else
        {
            // 根据上一帧的相机的位姿和恒定速度假设，预测当前帧相机的位姿
            vector<Vector3d> P_cam(2,Vector3d());
            vector<Matrix3d> R_cam(2,Matrix3d());
            R_cam[0] = prev_cam_R;
            P_cam[0] = prev_cam_P;
            
            // 根据上一帧的相机速度，预估当前帧的相机位姿。速度量表示的是从下一帧的相机位姿转换到上一帧位姿（还需要考虑时间长度）
            // delta_T = T_i * (T_i-1)^-1，计算delta_T的轴角和位移，再除以上两帧的时间间隔，得到上上帧的body-centric constant velocity。
            // 用这个velocity乘以当前帧与上一帧的dt，计算出delta_T，在右乘上一帧的T，得到当前帧的估计T
            // 这个VI初始化前用恒速模型预估当前帧位姿的计算在前面已经完成了，这里直接用
            //pred_pose_with_vel(R_cam[0], P_cam[0], const_ang_vel, const_l_vel, delta_T_cam_new, R_cam[1], P_cam[1]);
            R_cam[1] = cur_cam_R;
            P_cam[1] = cur_cam_P;
            
            cout << "Start objs_matching!" << endl;
            objs_matching(delta_T_cam_new, P_cam, R_cam);
        }
        // printf("Objects matching cost time: %f ms\n", t_objs_matching.toc());

        // 根据当前帧和上一帧的跟踪特征点，和已经完成的预积分，进行视觉惯性的优化。
        // 对于MVIO，由于物体的状态确定和关联需要当前帧相机的位姿，因此在下面函数中还添加了物体关联模块
        // processImage(feature.second, feature.first);
        processImage(feature);
        // 保存这个用于VI初始化之后每一帧计算上一帧的本体速度，因为默认把IMU积分放在tracker函数之后
        delta_T_imu_prev = curTime - prevTime;
        prevTime = curTime;
        prevTime_cam = curTime_cam;
        delta_T_cam_prev = delta_T_cam_new;
        
        // printStatistics(*this, 0);

        // 用于视觉化
        // std_msgs::Header header;
        // header.frame_id = "world";
        // header.stamp = ros::Time(feature.first);
        // // 用于视觉化
        // pubOdometry(*this, header);
        // pubKeyPoses(*this, header);
        // pubCameraPose(*this, header);
        // pubPointCloud(*this, header);
        // pubKeyframe(*this);
        // pubTF(*this, header);
        mProcess.unlock();
        
        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// 根据初始帧之后几帧IMU的加速度测量 来初始化首帧IMU坐标系在东北天坐标系下的姿态（设定没有yaw偏航角为0）
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    // TODO：为什么用多帧的加速度计测量值的平均向量方向来近似作为初始帧帧IMU坐标系下g的向量方向？是因为要求初始时相机是静止的吗？其实用什么都无所谓
    Matrix3d R0 = Utility::g2R(averAcc);
    // 下面这两步其实没必要，因为在g2R函数中已经剔除了yaw角旋转，即此处计算的yaw会是0
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // Rs、Ps和Vs存储的是各相机帧时刻的IMU预积分（都是表达在前一相机帧时刻的IMU坐标系下）
    // 首帧图像时刻是没有预积分的。但是我们需要确定首帧IMU坐标系相对于全局东北天坐标下的位姿！
    // 这里，Rs[0]代表的就是首帧IMU坐标系在全局东北天坐标下的姿态（只不过直接设定首帧IMU的yaw角为0，位移为0）！！
    Rs[0] = R0;
    // Rs[0] = Matrix3d::Identity();
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

// 设置初始帧相机时刻的IMU坐标系的相对于世界坐标系（设定为首帧相机坐标系）的位姿
void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

// 将新一时刻的IMU读数融合进IMU预积分中
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 如果是整个系统的首个IMU（是指从系统首帧图像之后的首个IMU），则直接将该时刻IMU数据作为初始位姿
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // acc_0, gyr_0此时指的是上一时刻IMU的读数。初始帧的预积分没有意义，不会被使用
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 如果不是首帧图像，首帧的IMU坐标系的位姿已经在专门的函数中设定了
    if (frame_count != 0)
    {
        // 预积分类的成员函数push_back中会进行预积分，采用的是中值积分
        // pre_integrations用于保存滑窗内所有相机帧时刻的预积分值
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            // 此变量保存当前帧和上一帧图像之间的预积分
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        // 注意，这里乘以的是当前帧IMU预积分的初始值！这个初始值的设置是在processImage()函数中，即将当前帧Rs等预积分的初始值赋值为上一帧处理之后的该变量，而由于Rs[0]的特殊性，所以后续每一帧在此处计算后的这些预积分变量都是表达在了当前世界坐标下！！
        // 因此这里计算的un_acc_0 是在当前世界坐标系（近似东北天坐标系下，还没有VI联合优化以对齐东北天） 的加速度和角速度值，且是带有噪声的（即作为预积分测量值，后续与视觉估计结果进行比较和优化）
        // 注意，Rs[]中表示的是对应帧相机时刻的IMU姿态预积分（参考系是当前假设的世界坐标系）！
        // 未VI初始化之前，g仍为(0,0,9.8)。VI初始化和估计g的方向之后，g的模应该仍为9.8，但是各分量不一定严格符合(0,0,9.8)，这取决于优化的精度！  
        // 此处使用g来参与推断预测当前帧IMU相对参考坐标系的位姿，虽然此时Rs和g(0，0，9.8)是不匹配的，但是得到的Rs等值并不直接拿来用，后续会使用视觉的观测来重新计算每帧IMU的Rs等值！
        // 但如果后续需要直接使用IMU的数据来大致预测物体的位姿，则必定要进行VI对齐初始化！这样才能使得g和Rs之间是匹配的（即表达在同一个全局参考坐标系下，且对于双目-IMU而言，不一定要是严格的东北天坐标系）！     
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        // 采用中值积分的方式计算旋转增量
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 更新旋转预积分
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // 用更新后的旋转预积分计算将当前时刻的加速度测量值转换到此预积分的初始时刻（即上一帧图像时刻）的IMU坐标系
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 采用中值积分的方式计算位置和速度增量
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 更新位置和速度预积分
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

// 根据当前帧和上一帧的跟踪特征点进行位姿估计，结合已经完成的预积分进行视觉惯性的优化
void Estimator::processImage(const double header)
{
    printf("new image coming ------------------------------------------\n");
    // printf("Adding objects (including bg) %lu\n", (*(image[0])).size());

    // ROS_DEBUG("new image coming ------------------------------------------");
    // ROS_DEBUG("Adding objects (including bg) %lu", (*(image[0])).size());
    // td为相机和IMU之间的时间戳差异，IMU真实对应时间戳=相机时间戳+td。
    // 注意，对于每一个滑动窗口，其内部所有的帧都使用同一个td，即在LBA优化时所有帧共同使用和优化一个td值！因此，这里td是上一个滑窗优化出来的TD值（作为当前帧的初始TD），而prev_td则是上上个滑窗优化的结果（作为上一帧的初始TD)
    // 往滑窗的特征地图点集feature中 添加新的特征地图点和增加某地图点的观测帧记录的操作 都是在此函数中，包括添加系统首帧中的所有新地图点
    // 另外根据次新帧和次次新帧之间匹配点的平均视差，来决定是否将次新帧作为关键帧，从而决定是要marg掉次新帧还是最老帧（前提是当前滑窗帧数已满）！
    // prev_td其实是上上帧滑窗优化出来的TD，其被赋予了上一帧的点；而td是上一帧滑窗优化出来的TD，其被赋予了当前帧的点！即每个滑窗优化出来的TD是给下一帧的点使用的！！

    int pnp_succ = 0;

    int num_track_cur_bg = 0;
    if (f_manager.addFeatureCheckParallax(frame_count, prev_td, td, featureTracker, num_track_cur_bg, true))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    // 如果平均视差比较小，则marg掉次新帧（即次新帧不重要）
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }
    
    printf("Solving frame %d\n", frame_count);
    printf("current frame is %s\n", marginalization_flag ? "Non-keyframe" : "Keyframe");
    
    // 当前帧下连续观察帧数大于规定值的点数，且在首观测帧下有深度值的点
    int num_pt_cur_frame_in_LBA = f_manager.getFeatureCountLBAcur(frame_count);
    printf("number of feature in cur frame for LBA : %d\n", num_pt_cur_frame_in_LBA);

    // int num_pt_in_LBA = f_manager.getFeatureCount();
    // printf("number of feature for LBA: %d\n", num_pt_in_LBA);
    // ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    // ROS_DEBUG("Solving %d", frame_count);
    // ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    
    Headers[frame_count] = header;

    if (frame_count == 0) 
    {
        initial_timestamp = header;
    }
    
    ImageFrame imageframe(featureTracker.TrackBgFea, header);
    // 将前面processIMU计算的当前帧和上一帧图像之间的预积分（表达在当前假设世界坐标系下）放进当前图像frame对象
    if(USE_IMU) imageframe.pre_integration = tmp_pre_integration;
    // 保存当前图像frame
    all_image_frame.insert(pair<double, ImageFrame>(header, imageframe));
    // 为下一帧图像创建好预积分对象，其起始时刻的IMU测量就是当前帧IMU预积分中的最后一个测量
    if(USE_IMU) tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 如果相机和IMU之间的外参未标定（即直接初始化为I和0），则从第2帧开始利用每帧的预积分来进行外参旋转变量估计
    // 注意，如果外参是有给定一个较为可靠的初始值，但是认为后续还需要优化估计，则ESTIMATE_EXTRINSIC == 1，在VI初始化阶段不会估计外参，而是之后的VI联合LBA才会。
    // 如果外参已经线下被准确地标定，则ESTIMATE_EXTRINSIC == 0
    if(ESTIMATE_EXTRINSIC == 2)
    {
        printf("calibrating extrinsic param, rotation movement is needed\n");
        // ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        // 最少只需要两帧图像即可以进行IMU和相机之间相对旋转的估计？采取的是逐帧的预积分和视觉位姿估计来逐渐地优化外参，需要到系统第WINDOW_SIZE+1帧（即需要WINDOW_SIZE个预积分）来迭代优化外参。
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 注意，至少需要一个滑窗的数量的帧来完成相机和IMU的旋转外参的优化估计
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                printf("initial extrinsic rotation calib success!\n");
                cout << "initial extrinsic rotation: " << endl << calib_ric;
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                // 1表示后续LBA再对外参进行联合优化
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
    
    Matrix3d pred_cam_R;
    Vector3d pred_cam_P;
    double scale_init = 1.0;
    bool has_t = true;
    Vector3d t_with_scale;

    if(STEREO || solver_flag == NON_LINEAR)
    {
        float pred_dist_t = 0.0;
        // 对于双目，无论有没有IMU，系统首帧是直接固定位姿，系统第2帧的初始值只能用初始帧的位姿
        if (STEREO && frame_count == 0) 
        {
            pred_cam_R = Rs[frame_count] * ric[0];
            pred_cam_P = Rs[frame_count] * tic[0] + Ps[frame_count];
            cur_cam_R = pred_cam_R;
            cur_cam_P = pred_cam_P;
        }
        // 这些情况下第二帧的位姿预测只能用上一帧的位姿
        // 对于纯视觉只有第3帧开始才能使用恒速模型预测相机的位姿。如果考虑估计IMU初始帧的位姿且IMU的参数标定不够准确，则IMU+STEREO在初始化完成前相当于是纯视觉
        // else if(STEREO && (!USE_IMU || NEED_ESTI_G) && frame_count == 1) 
        else if(STEREO && frame_count == 1) 
        {
            // pred_cam_R = Rs[frame_count] * ric[0];
            // pred_cam_P = Rs[frame_count] * tic[0] + Ps[frame_count];
            pred_cam_R = prev_cam_R;
            pred_cam_P = prev_cam_P;
            
            cur_cam_R = pred_cam_R;
            cur_cam_P = pred_cam_P;
        }
        // 包含单目且已初始化，纯双目第3帧开始，双目+IMU已初始化或可以固定IMU首帧位姿且标定参数可以信任
        else
        {
            pred_cam_R = cur_cam_R;
            pred_cam_P = cur_cam_P;
        }

        // 如果通过sift匹配计算了F矩阵，并分解得到R和归一化的t。则这里尝试用上一帧有深度值的跟踪点来估计t的尺度
        if(frame_count > 0 && (featureTracker.has_valid_F || featureTracker.has_valid_H) && featureTracker.cal_Mat_F_H)
        {
            if(featureTracker.has_valid_F)
                cout << "Has R and direction of P from estimation of F!" << endl;
            else
                cout << "Has R and direction of P from estimation of H!" << endl;
            
            t_with_scale = featureTracker.t_from_E;
            float norm_t = t_with_scale.norm();
            cout << "norm_t of t_with_scale: " << norm_t << endl;
            // 预测的两帧间相机运动(上一帧变换到当前帧)。如果是使用IMU，则在VI初始化之前应该使用恒速运动模型来预测此运动
            Vector3d pred_motion_P = cur_cam_R.transpose() * (prev_cam_P - cur_cam_P); 
            
            // 使用极线约束估计得到的当前帧相机方向
            // 如果是IMU且已经初始化，则使用IMU的积分来作为R和t。其他情况下可视为纯视觉，则需要用此H和F估计得到的R和t
            if(!(USE_IMU && solver_flag == NON_LINEAR))
            {
                pred_cam_R = prev_cam_R * featureTracker.R_from_E.transpose();
                // 提前给定基于视觉的IMU位姿的预测值
                Rs[frame_count] = pred_cam_R * ric[0].transpose();
                cur_cam_R = pred_cam_R;
            }
            
            // 如果有相机的运动预测，则用其提供位移尺度的初始值
            if(frame_count > 1 && norm_t > 0)
            {
                pred_dist_t = pred_motion_P.norm();
                cout << "norm_t of pred_dist_t: " << pred_dist_t << endl;
                // 用IMU积分或者恒速运动模型的预测值给定t的尺度的初始值
                scale_init = pred_dist_t/norm_t;
                // 预测的位移大小不太小，否则认为物体是几乎没有位姿的。
                // 但是这应该会受到加速度计静态偏差和噪声的影响？
                // 如果tracker里面估计了H矩阵，那么应该是可以给定t的方向向量，如果位移几乎为0，那么其方向向两应该是什么？可以用来作为判断吗？好像没办法
                if(pred_dist_t >= 0.06)
                {
                    has_t = true;
                }
                else
                {
                    scale_init = 1.0;
                    // 即使预测的相机运动显示没有位移，也可以尝试恢复P的尺度？毕竟通过极线约束恢复了R 和 t的方向，说明并非纯旋转？
                    // has_t = false;
                    cout << "nearly has no translation! scale of predict translation: " << pred_dist_t << endl;
                }
            }
        }
        else
        {
            has_t = false;
        }
        
        bool recover_scale_succ = false;
        // int num_LBA_fea_cur_frame;
        // 如果是使用双目相机，则进行上一帧部分有立体匹配 但是 还没有三角化估计深度值 的跟踪点 的深度估计
        if(has_t)
        {
            set<int> &reserve_bg_track_pt_id = featureTracker.reserve_bg_track_pt_id;
            // 同时用恢复深度的3D-2D匹配来恢复t的尺度（这个对单目+IMU也可以使用）
            recover_scale_succ = f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu,  
                                                        true, featureTracker.R_from_E, &(t_with_scale), scale_init, pred_dist_t, reserve_bg_track_pt_id);
        }
        else
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu, true);

        // 如果成功恢复了t的尺度，则基于此t计算当前帧相机和IMU的P预估值
        if(recover_scale_succ)
        {
            // todo:IMU且初始化后，F或H估计出来的P是否要使用？
            // if(!(USE_IMU && solver_flag == NON_LINEAR))
            {
                cout << "estimated motion P of cam is: " << t_with_scale.transpose() << endl;
                // 赋予t尺度，并作为P的初始估计值，防止参与PnP的点数不足而失败
                pred_cam_P = prev_cam_P - pred_cam_R * t_with_scale;
                cur_cam_P = pred_cam_P;

                Ps[frame_count] = pred_cam_P - Rs[frame_count] * tic[0];
            }
        }
    }
    
    // bool initial_succ_prev = (solver_flag == NON_LINEAR);

    // 对于单目-IMU而言，首个滑窗帧数刚满时不一定能VI初始化成功，但是一定会marg掉最老帧或次新帧（即滑窗是为了保持局部帧数，与初始化是否成功无关）。
    // 如果当前滑窗VI失败，则marg后获取下一帧并形成新的滑窗，再次尝试初始化，重复尝试直到成功。
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization  对于单目+IMU，由于视觉的尺度必须要依赖于IMU，而IMU要与视觉对齐就必须要确定初始帧下g在IMU坐标系的方向，因此需要优化g
        if (!STEREO && USE_IMU)
        {
            // 对于单目IMU而言，是无法消除物体尺寸（深度）和运动的尺寸的不确定性的！因此本系统默认只使用双目来完成物体的跟踪，单目+IMU只能进行相机自身的定位。
            // parallel_pose_objs_est();

            // 只要滑窗内帧数满了，无论初始化成不成功，都要滑窗！
            // 那单目的帧间位姿估计是在哪里完成？单目必须得累积WINDOW_SIZE+1帧之后再一次性在initialStructure()中进行所有帧的位姿估计并完成VI的联合初始化！！
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // 必须得先完成IMU和相机的外参估计，才能进行其他IMU初始化的估计！
                // 注意，VINS系统由于是跑在真实无人机上的，因此相机或IMU消息的header是从0开始的，这里initial_timestamp的默认初始值也是为0！
                // 如果是使用数据集来进行评测，则要注意header的起始值或者initial_timestamp的初始值！
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    // 进行视觉-惯性联合初始化
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    // VI初始化成功之后，联合所有VI数据对当前滑窗内所有待优化变量进行LBA，然后再marg掉某一帧得到当前滑窗的先验残差信息
                    optimization();
                    // 取IMU的测量queue中的所有新数据（大于或等与当前图像时间戳+td的时刻的IMU数据），持续积分得到最新的IMU全局状态量的预测（用于轨迹可视化？）
                    // updateLatestStates();
                    solver_flag = NON_LINEAR;
                    // 上面marg得到先验信息之后，在此函数内清理被marg的帧和相关的地图点的观测信息（还可能去掉某些无效地图点），更新受影响的地图点的深度估计值和观测信息，移动滑窗变量中各元素的位置
                    slideWindow();
                    printf("Initialization finish!\n");
                    // ROS_INFO("Initialization finish!");
                }
                // 单目情况下，如果没能完成VI初始化，则直接滑窗去除某一帧
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization  原代码中为什么双目-IMU不用进行g对齐？？？理论上只要使用到IMU，就必须进行g的对齐操作！
        // 原代码中对于双目-IMU，应该是默认有提供一个外参旋转的标定结果作为初始值，则无需想单目一样累积WINDOW_SIZE+1帧再一次性进行所有帧的位姿估计和VI初始化，而是从第2帧起就根据跟踪点来用PnP视觉估计当前帧的全局位姿！
        if(STEREO && USE_IMU)
        {
            //!!! 此处是否应该先判断一下上面IMU和相机相对旋转ric是否已初始估计成功？此工作采取的是逐帧的预积分和视觉位姿估计来逐渐地优化外参，需要到系统第WINDOW_SIZE+1帧（即需要WINDOW_SIZE个预积分）时才完成外参的初始估计。
            // 上面ESTIMATE_EXTRINSIC == 2是指外参ric完全没有任何的标定结果可以作为初始化值，而是直接以I作为初始值，才需要整个滑动窗口的帧来逐渐估计出一个初始值（这其实更多是针对单目IMU的情况，其一开始整个窗口的帧都没进行视觉的位姿估计，而是等待ric的初始估计完成）。
            // 而如果我们能给定一个线下的标定结果作为初始值，则不需要上面的ric初始值估计过程。
            // 另外这里可以直接用“不太准确”的ric初始值来参与每帧相机全局位姿的视觉估计，因为对于每帧计算时只要求将3D地图点转换到同一个统一的坐标系（ric的标定偏差会导致这个坐标系与当前假定的世界坐标系有偏差，但是只要初始化期间ric不变，则相当于只是再稍微改变了一下假定的世界坐标系！）
            // 因此必须注意，对于双目-IMU，如果ESTIMATE_EXTRINSIC == 2，则跟单目一样，也必须得先得到ric的初始值估计，才能一次性进行所有帧的视觉估计和VI对齐！！否则每帧都变化的ric值会对下面initFramePoseByPnP()中得到的视觉估计位姿产生影响，即各帧位姿的参考世界坐标系会不同！！
            // 为了防止操作失误，保持双目-IMU时外参初始值的默认设置，这里还是加一个assert比较好。
            assert(ESTIMATE_EXTRINSIC != 2 && "For Stereo-IMU please provide the initial estimated extrinsic rotation Ric before PnP for pose estimation!");
            
            // 下面用被当前帧跟踪到的3D点地图点来递推估计当前帧相机和IMU的全局位姿（表达在当前假定的世界坐标系下），注意，是利用视觉估计的相机全局位姿和较准确的外参ric，来得到视觉方法估计下的当前帧IMU的全局位姿
            // 如果当前帧和上一帧之间的PnP求解成功（跟踪点小于4个，或者调用opencv的函数求解失败，才算失败），则能得到当前帧相机的全局位姿估计，而其保存的其实是用ric来得到视觉估计下的当前帧IMU坐标系的全局位姿并替换掉当前帧的Rs和Ps，后续VI联合初始化时作为视觉的估计结果。
            // ！！！对于VIO，如果当前帧和上一帧之间的PnP求解失败，则应该要重启系统了！因为此时除了回环重定位，没有任何办法可以延续后面的帧和之前的帧的视觉估计的位姿一致性（前面是用纯视觉估计的位姿，之后参杂了IMU预积分的影响）！
            // 即跟单目IMU一样，用于VI联合初始化的WINDOW_SIZE+1帧都要连续视觉跟踪和位姿估计成功！否则从中断跟踪的帧开始，就无法提供足够多的准确的视觉位姿估计来进行VI联合初始化！
            

            // 基于跟踪点和给定的当前帧位姿预测值，估计当前帧相机的全局位姿，更新所有内点的在当前帧的深度估计值（深度值不满足要求的点删除）；
            // 对于背景中的外点sift，如果当前帧深度估计可靠，可以考虑保留为新点（最终放弃该想法），否则和FAST外点一样全部删除（跟踪记录和地图记录）
            // todo: 如果PnP失败怎么办？后续是否有失败检查和重启？VINS-Fusion采用了不会失败的PnP（即不管估计的位姿结果有多烂，除非匹配点数不够4个）...
            pnp_succ = f_manager.initFramePoseByPnP(frame_count, featureTracker, Ps, Rs, tic, ric, pred_cam_R, pred_cam_P, prev_cam_R, prev_cam_P, reserve_new_sift, num_track_cur_bg);
            
            // 如果视觉跟踪成功，则用估计的结果作为当前帧相机的位姿
            // 如果视觉跟踪失败，则用IMU积分预测的位姿作为当前帧相机的位姿
            if(pnp_succ)
            {
                cur_cam_R = pred_cam_R;
                cur_cam_P = pred_cam_P;
            }
            else
            {
                pnp_succ = num_track_cur_bg;
                // todo 如果视觉约束不够多，则在LBA中固定当前帧的位姿（使用IMU的积分预测的位姿）
                // id_frame_const_pose.push_back(frame_count);
            }
            
            if(frame_count < 2)
            {
                prev_cam_R_using_imu[frame_count] = pred_cam_R;
                prev_cam_P_using_imu[frame_count] = pred_cam_P;
            }
            else
            {
                prev_cam_R_using_imu[0] = prev_cam_R_using_imu[1];
                prev_cam_P_using_imu[0] = prev_cam_P_using_imu[1];
                
                prev_cam_R_using_imu[1] = pred_cam_R;
                prev_cam_P_using_imu[1] = pred_cam_P;
            }

            if(frame_count > 0) Vs[frame_count] = (Ps[frame_count] - Ps[frame_count-1])/delta_T_cam_new;
            if(frame_count == 1) Vs[0] = Vs[1];

            // 如果是IMU-Stereo,则选择记录IMU坐标系的位姿，因为BA优化中会优化系统首帧的位姿，而label中的初始时刻的IMU坐标系为全局坐标系
            all_cam_R_before_init.push_back(Rs[frame_count]);
            all_cam_P_before_init.push_back(Ps[frame_count]);

            // 当完成了相机-IMU的外参 以及 当前帧的相机位姿（在Rs和Ps中）的初步估计之后，则可以开始并行地估计相机的运动 和 每个动态物体的运动
            // 单独子线程进行。其实也可以边进行相机的运动初始化，边估计各物体的运动，但是函数中需要使用当前帧相机位姿的估计，可以使用基于恒速模型的预测值，但是这样不够精确；等到初始化成功之后，其实就可以直接使用IMU的推测值
            map_fea_optimized = true;

            // 因为线程函数parallel_pose_objs_est()的型参均不是左值引用，因此这里无需传入的参数使用ref()来变成左值引用
            // 即使是系统首帧，也要执行此函数，其中不仅会估计动态物体的运动，还会更新所有特征点的全局信息，删除无效的特征点，并保存各个全局物体的采样像素点。
            obj_motion_esti = std::thread(&Estimator::parallel_pose_objs_est, this, prev_td, td, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, pnp_succ, false);

            // 如果跟踪失败，则尝试重定位 或者 直接重启系统！
            // if(!pnp_succ && frame_count!=0) restart_or_relocate();
            
            // 三角化深度仍未知的地图点的坐标（针对当前帧新增加的有左右匹配的特征点，以及在观测首2帧内都没完成三角化的点（需要到其第4帧观测起才会再次三角化））。此时当前帧Ps和Rs都是视觉估计出来的IMU的全局位姿了！三角化得到的深度都是该点在其观测首帧相极坐标系下的深度。
            // 用于三角化上一帧与当前帧间的跟踪点的位姿变化，是采用IMU积分得到的值 还是 用上面视觉估计得到的位姿 （为了避免视觉估计质量差导致大量点的三角化测量失败，于是使用基于IMU积分的两帧位姿）
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu);
            
            // TODO：:如何才需要固定住当前的位姿？确实需要该帧有足够参与LBA的点的观测，但是不能只看该帧之前的点，而是还得看从该帧开始跟踪的点？
            // if(0 && frame_count >= TH_NUM_FRAME_FOR_LBA-1)
            // {
            //     if(num_LBA_fea_cur_frame < 6)
            //     {
            //         if(pnp_succ) id_frame_const_pose.push_back(frame_count);
            //     }
            // }
            
            // 双目-IMU也要在首个滑窗内帧数满足时进行VI的联合初始化！
            // 除了要估计加速度计的bias，还应该估计全局参考坐标系下的g的方向值，使得其与Rs等IMU全局姿态估计相匹配！！全局参考坐标系可以不严格对齐东北天，但是g和Rs等必须是在同一个全局坐标系下（起始后续还是要转换对齐到东北天，否则g的读数会出问题）
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                // 这里是保存了从系统运行开始处理的每一帧图像，不仅仅是留在滑窗内的帧（因为滑窗满了之后会开始去掉某些帧）
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    // 这里的Rs[i]表示的是“用视觉方法估计出来的”第i帧相机时刻的IMU坐标系相对于当前世界坐标系的全局姿态，Ps[i]则是对应的位移。此变量作为VI联合初始化时的真值，与IMU预积分作差
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                
                bool result = true;

                // 原项目中对于双目-IMU只估计了陀螺仪bias。是否应该改成调用visualInitialAlign()函数，即除了估计陀螺仪bias，还要进行g方向的估计（期间固定尺度s为1）和初始帧IMU坐标系对齐到东北天坐标系
                // solveGyroscopeBias(all_image_frame, Bgs);
                result = visualInitialAlign(true);
                
                // for (int i = 0; i <= WINDOW_SIZE; i++)
                // {
                //     pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                // }

                if (result)
                {
                    initial_timestamp = header; 
                    initial_succ_first_win = true;
                    // 此函数内联合所有VI数据对当前滑窗内所有待优化变量进行LBA，然后再marg掉某一帧得到当前滑窗的先验残差信息。但是由于此时solver_flag还未被设置为NON_LINEAR，因此当前帧环窗内不进行marg形成先验，而是下面直接滑窗去掉某一帧
                    optimization();

                    tracker_pts_updated = true;
                    
                    // 如果首个滑窗就VI初始化成功，则除了在visualInitialAlign函数中校正所有相机位姿的参考坐标系之后，在LBA之后还要更新所有相机的估计值
                    if(first_win)
                    {
                        all_cam_R_before_init.clear();
                        all_cam_P_before_init.clear();
                        for (int i = 0; i <= frame_count; i++)
                        {
                            // trans to w_T_cam
                            // 从下面的公式可以看出，就cam位姿的校正跟IMU的位姿校正相同，只需要在先前估计值左乘rot_diff即可。
                            // cam_R = Rs[i] * ric[0]; 
                            // cam_P = Rs[i] * tic[0] + Ps[i];
                            // all_cam_R_before_init.push_back(cam_R);
                            // all_cam_P_before_init.push_back(cam_P);

                            all_cam_R_before_init.push_back(Rs[i]);
                            all_cam_P_before_init.push_back(Ps[i]);
                        } 
                        first_win = false;
                    }
                    // 否则，由于前面的滑窗已经marg掉了一些帧，那些帧的位姿没法参与LBA，因此没法对所有相机帧位姿都进行更新，这里只更新最新的一帧
                    else
                    {
                        all_cam_R_before_init.pop_back();
                        all_cam_P_before_init.pop_back();

                        // cam_R = Rs[frame_count] * ric[0]; 
                        // cam_P = Rs[frame_count] * tic[0] + Ps[frame_count];
                        // all_cam_R_before_init.push_back(cam_R);
                        // all_cam_P_before_init.push_back(cam_P);

                        all_cam_R_before_init.push_back(Rs[frame_count]);
                        all_cam_P_before_init.push_back(Ps[frame_count]);
                    }

                    // LBA后的当前帧相机位姿
                    cur_cam_R = Rs[frame_count] * ric[0];
                    cur_cam_P = Rs[frame_count] * tic[0] + Ps[frame_count];

                    // 取IMU的测量queue中的所有新数据（大于或等与当前图像时间戳+td的时刻的IMU数据），持续积分得到最新的IMU全局状态量的预测
                    // 预测这些量的作用是什么？用于可视化轨迹吗？
                    //updateLatestStates();

                    // 为什么不在当前滑窗就进行marg形成先验残差？
                    solver_flag = NON_LINEAR;

                    // 清理上面被marg掉的帧的相关数据（包括去除marg后无法提供有效优化约束的地图点），更新受影响的地图点的深度估计值和观测信息，移动滑窗变量中各元素的位置
                    // 注意，滑窗只清除相机帧和相关静态点，更改地图点的帧信息；对于动态物体，在初始化成功之后，始终只保存最近的连续四帧的观测和三个位姿变换
                    slideWindow();
                    
                    // 是否需要固定某些帧的位姿
                    // if(!id_frame_const_pose.empty())
                    // {
                    //     int first_elem = 0;
                    //     int rest_elem = 0;
                    //     int num = id_frame_const_pose.size();
                    //     // set的迭代器默认是const类型的，即无法通过其默认迭代器对set中的元素进行修改
                    //     if(marginalization_flag == MARGIN_OLD)
                    //     {
                    //         if(id_frame_const_pose[0] == 0) first_elem = 1;
                    //         for(int i = first_elem; i < num; ++i)
                    //         {
                    //             id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                    //             ++rest_elem;
                    //         }
                    //         id_frame_const_pose.resize(rest_elem);
                    //     }
                    //     else
                    //     {
                    //         for(int i = 0; i < num; ++i)
                    //         {
                    //             if(id_frame_const_pose[i] != WINDOW_SIZE-1) 
                    //             {
                    //                 if(rest_elem < i) id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                    //                 ++rest_elem;
                    //             }
                    //         }
                    //         if(rest_elem < num) id_frame_const_pose.resize(rest_elem);
                    //     }
                    // }

                    // 等待物体运动估计和添加新的静态点观测 完成后，再进行物体的全局运动的更新，以及滑窗操作中的地图无效点的清除
                    obj_motion_esti.join();

                    // 滑窗marg之前，使用相机的全局位姿来更新计算两帧之间物体的绝对运动变换（即世界坐标系下的该物体的帧间位姿转换）
                    // 这个函数可以放在下一帧的开始，即覆盖GPU中seg和flow的时间？
                    // 其实时间也不会很长，因为只有初始化成功的这一帧需要全部帧的物体运动都更新；之后由于相机滑窗marg掉一些帧（尤其是次新帧），会导致marg帧及之前的所有帧都没法更新了
                    // 但似乎没有在这里求得物体的全部帧的全局运动变换？因为无论是KITTI的tracking数据集中，还是BEV算法中，都只需要知道物体在每一帧的相机坐标系下的位姿，运动变换也都是在相机坐标下的
                    // 这里只需要求得两帧间的物体点的运动变换矩阵（表达在后一帧的相机坐标系下），后续如果有某一帧被marg掉，其实有间隔的两帧的运动变换也是把多个变换矩阵相乘起来而以，然后就可以求得全局的运动变换了。
                    // updatePoseTransObjs();
                    printf("Initialization finish!\n");
                    // ROS_INFO("Initialization finish!");
                }
                // 在VINS-Fusion原代码中不存在双目-IMU初始化失败的考虑！！
                else
                {
                    printf("VIO-initialization failed!\n");
                    abort();
                    // ROS_INFO("misalign stereo visual structure with IMU");
                    tracker_pts_updated = true;
                    slideWindow();

                    // if(!id_frame_const_pose.empty())
                    // {
                    //     int first_elem = 0;
                    //     int rest_elem = 0;
                    //     int num = id_frame_const_pose.size();
                    //     // set的迭代器默认是const类型的，即无法通过其默认迭代器对set中的元素进行修改
                    //     if(marginalization_flag == MARGIN_OLD)
                    //     {
                    //         if(id_frame_const_pose[0] == 0) first_elem = 1;
                    //         for(int i = first_elem; i < num; ++i)
                    //         {
                    //             id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                    //             ++rest_elem;
                    //         }
                    //         id_frame_const_pose.resize(rest_elem);
                    //     }
                    //     else
                    //     {
                    //         for(int i = 0; i < num; ++i)
                    //         {
                    //             if(id_frame_const_pose[i] != WINDOW_SIZE-1) 
                    //             {
                    //                 if(rest_elem < i) id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                    //                 ++rest_elem;
                    //             }
                    //         }
                    //         if(rest_elem < num) id_frame_const_pose.resize(rest_elem);
                    //     }
                    // }

                    // 等待物体运动估计和添加新的静态点观测 完成后，再进行物体的全局运动的更新，以及滑窗操作中的地图无效点的清除
                    obj_motion_esti.join();
                }
            }
            else
            {
                tracker_pts_updated = true;
                obj_motion_esti.join();
            }
        }
        
        // stereo only initilization 对于双目而言，即使是首帧，也是可以添加3D点的
        if(STEREO && !USE_IMU)
        {
            Matrix3d orig_pred_cam_R = pred_cam_R;
            Vector3d orig_pred_cam_P = pred_cam_P;
            // 如果PnP失败怎么办？
            // if(frame_count > 1)
            if(frame_count > 0)
            {
                pnp_succ = f_manager.initFramePoseByPnP(frame_count, featureTracker, Ps, Rs, tic, ric, pred_cam_R, pred_cam_P, prev_cam_R, prev_cam_P, reserve_new_sift, num_track_cur_bg, false);
            }
            
            // 如果视觉跟踪成功，则用估计的结果作为当前帧相机的位姿
            // 如果视觉跟踪失败，则用运动恒速模型预测的位姿作为当前帧相机的位姿
            if(pnp_succ)
            {
                cur_cam_R = pred_cam_R;
                cur_cam_P = pred_cam_P;
            }
            else
            {
                pred_cam_R = orig_pred_cam_R;
                pred_cam_P = orig_pred_cam_P;

                pnp_succ = num_track_cur_bg;
                // todo: 如果视觉约束不够多，则在LBA中固定当前帧的位姿（使用IMU的积分预测的位姿）。
                // 视觉约束是否足够，不是在当前帧决定吧？还得看之后该帧中是否有新的LBA的点？
                // id_frame_const_pose.push_back(frame_count);
            }
            
            cout << "Start thread of obj motion est!" << endl;
            
            obj_motion_esti = std::thread(&Estimator::parallel_pose_objs_est, this, prev_td, td, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, pnp_succ, false);
            
            map_fea_optimized = true;
            
            // 纯双目的话就不使用相机预测值（相机的运动模型）来进行三角化了，而是直接使用视觉估计值
            if(frame_count < 2)
            {
                prev_cam_R_using_imu[frame_count] = pred_cam_R;
                prev_cam_P_using_imu[frame_count] = pred_cam_P;
            }
            else
            {
                prev_cam_R_using_imu[0] = prev_cam_R_using_imu[1];
                prev_cam_P_using_imu[0] = prev_cam_P_using_imu[1];
                
                prev_cam_R_using_imu[1] = pred_cam_R;
                prev_cam_P_using_imu[1] = pred_cam_P;
            }
            
            if(frame_count > 0) f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu);
            
            // 如何衡量某帧的视觉约束是否足够？
            // if(frame_count >= TH_NUM_FRAME_FOR_LBA-1)
            // {
            //     if(num_LBA_fea_cur_frame < 6)
            //     {
            //         if(pnp_succ) id_frame_const_pose.push_back(frame_count);
            //     }
            // }

            // 纯双目情况下滑窗帧数还未满时就可以进行联合优化了，只是其后不会进行marg而已！
            optimization();
            
            if(frame_count == WINDOW_SIZE)
            {
                // optimization();
                tracker_pts_updated = true;

                Matrix3d cam_R;
                Vector3d cam_P;
                all_cam_R_before_init.clear();
                all_cam_P_before_init.clear();
                for (int i = 0; i <= frame_count; ++i)
                {
                    // trans to w_T_cam
                    // 从下面的公式可以看出，就cam位姿的校正跟IMU的位姿校正相同，只需要在先前估计值左乘rot_diff即可。
                    cam_R = Rs[i] * ric[0]; 
                    cam_P = Rs[i] * tic[0] + Ps[i];
                    all_cam_R_before_init.push_back(cam_R);
                    all_cam_P_before_init.push_back(cam_P);
                }
                cur_cam_R = all_cam_R_before_init[frame_count];
                cur_cam_P = all_cam_P_before_init[frame_count];
                
                // updateLatestStates();
                solver_flag = NON_LINEAR;

                // 滑窗操作的最后需要等待 子线程中往地图中加入新观测完成之后 才进行地图点的信息更新和清理
                slideWindow();

                // 是否要固定某些观测不足的帧的位姿？
                // if(!id_frame_const_pose.empty())
                // {
                //     int first_elem = 0;
                //     int rest_elem = 0;
                //     int num = id_frame_const_pose.size();
                //     // set的迭代器默认是const类型的，即无法通过其默认迭代器对set中的元素进行修改
                //     if(marginalization_flag == MARGIN_OLD)
                //     {
                //         if(id_frame_const_pose[0] == 0) first_elem = 1;
                //         for(int i = first_elem; i < num; ++i)
                //         {
                //             id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                //             ++rest_elem;
                //         }
                //         id_frame_const_pose.resize(rest_elem);
                //     }
                //     else
                //     {
                //         for(int i = 0; i < num; ++i)
                //         {
                //             if(id_frame_const_pose[i] != WINDOW_SIZE-1) 
                //             {
                //                 if(rest_elem < i) id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
                //                 ++rest_elem;
                //             }
                //         }
                //         if(rest_elem < num) id_frame_const_pose.resize(rest_elem);
                //     }
                // }

                obj_motion_esti.join();
                // updatePoseTransObjs();
                printf("Initialization finish!\n");
                // ROS_INFO("Initialization finish!");
            }
            else
            {
                tracker_pts_updated = true;
                obj_motion_esti.join();
            }
        }

        cout << "estimated translation P of IMU is: " << Ps[frame_count].transpose() << endl;

        // 将下一帧的预积分的初始值设置为当前帧的这些量
        // Rs[0]和Ps[0]是初始帧IMU坐标系相对于假定的世界坐标系的位姿（此项目中是通过计算前几个加速度测量的平均来假定为初始帧IMU坐标下的g方向，并计算其相对于东北天坐标系的位姿）
        // 所以使用Rs[0]和Ps[0]只能将初始帧IMU坐标系变换到一个假定的世界坐标系（与东北天坐标系近似）
        // 这里将当前帧的这些值设置为下一帧的初始值，则之后每一帧在根据IMU测量积分出来的Rs、Ps和Vs都是 该帧IMU相对于 假定世界坐标系 的位姿！
        // 这里如果还没完成初始化，则应该继续预积分以待完成初始化
        // if(frame_count < WINDOW_SIZE)

        map_fea_writen = false;
        if(frame_count < WINDOW_SIZE)
        {
            // 首个滑窗未满时才会++，当首个滑窗已满后，之后每个新帧的frame_count都是WINDOW_SIZE
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
        
        prev_cam_R = cur_cam_R;
        prev_cam_P = cur_cam_P;
        prev_td = td;
        // cout << "Succeeded process backend of MVIO!" << endl;

        // 世界坐标系z轴为竖直向上
        // prev_height = prev_cam_P(2);
    }
    else
    {
        TicToc t_solve;

        Matrix3d orig_pred_cam_R = pred_cam_R;
        Vector3d orig_pred_cam_P = pred_cam_P;

        // 纯双目视觉的情况下，对于每一个新帧都要基于视觉跟踪约束使用PnP来求解当前帧的位姿，并保存仅Ps和Rs中
        // 而有IMU且已经完成VI初始化的系统，会直接使用IMU测量的积分来预测当前帧图像时刻对应的IMU坐标系的状态（认为短时间内的IMU积分足够准确），保存在Rs和Ps中，见processIMU()函数

        // int num_LBA_fea_cur_frame = 0;
        // 仅配备双目相机的情况下才需要进行PnP估计。为什么？单目在初始化之后每一帧也可以进行3D-2D的位姿估计了呀？
        if(!USE_IMU)
        {
            pnp_succ = f_manager.initFramePoseByPnP(frame_count, featureTracker, Ps, Rs, tic, ric, pred_cam_R, pred_cam_P, prev_cam_R, prev_cam_P, reserve_new_sift, num_track_cur_bg, false, true);
        }
        // stereo+IMU的模式下，在初始化之后是否也要进行2帧间的PnP？
        else if(STEREO)
        {   
            // IMU初始化之后，可以设置每帧均进行PnP;否则，如果之前已经通过F或H的估计对跟踪点的外点滤除，则这里才不进行PnP，否则还是要用PnP来滤除外点
            // todo:是否有些情况下连续跟踪点数太少，LBA难以进行，则是否要进行PnP?
            // if(use_pnp_after_imu_init)
            if(use_pnp_after_imu_init || !featureTracker.fea_filtered)
                pnp_succ = f_manager.initFramePoseByPnP(frame_count, featureTracker, Ps, Rs, tic, ric, pred_cam_R, pred_cam_P, prev_cam_R, prev_cam_P, reserve_new_sift, num_track_cur_bg, true, true);
            else
            {
                // 如果当前帧参与LBA的跟踪点数不足，后续不进行LBA，则在这里进行PnP
                if(num_pt_cur_frame_in_LBA < 3)
                {
                    pnp_succ = f_manager.initFramePoseByPnP(frame_count, featureTracker, Ps, Rs, tic, ric, pred_cam_R, pred_cam_P, prev_cam_R, prev_cam_P, reserve_new_sift, num_track_cur_bg, true, true);
                }
            }
        }
        
        if(pnp_succ)
        {
            cur_cam_R = pred_cam_R;
            cur_cam_P = pred_cam_P;
            if(!USE_IMU)
            {
                prev_cam_R_using_imu[0] = prev_cam_R_using_imu[1];
                prev_cam_P_using_imu[0] = prev_cam_P_using_imu[1];
                
                prev_cam_R_using_imu[1] = pred_cam_R;
                prev_cam_P_using_imu[1] = pred_cam_P;
            }
        }
        else
        {
            if(!USE_IMU)
            {
                prev_cam_R_using_imu[0] = prev_cam_R_using_imu[1];
                prev_cam_P_using_imu[0] = prev_cam_P_using_imu[1];
                
                prev_cam_R_using_imu[1] = orig_pred_cam_R;
                prev_cam_P_using_imu[1] = orig_pred_cam_P;
            }
            
            pnp_succ = num_track_cur_bg;

            // todo: 如果视觉约束不够多，则在LBA中固定当前帧的位姿（使用IMU的积分预测的位姿）。
            // 视觉约束是否足够，不是在当前帧决定吧？还得看之后该帧中是否有新的LBA的点？
            // id_frame_const_pose.push_back(frame_count);
        }
        
        Vs[frame_count] = (Ps[frame_count] - Ps[frame_count-1])/delta_T_cam_new;

        // 对于双目IMU，直接拿IMU的积分结果作为当前帧相机的位姿，完成物体的运动估计
        obj_motion_esti = std::thread(&Estimator::parallel_pose_objs_est, this, prev_td, td, prev_cam_R, prev_cam_P, cur_cam_R, cur_cam_P, pnp_succ, true);
        
        // 带有IMU且已经初始化完成之后，直接使用IMU的积分值来推断当前帧的全局位姿
        // 根据初步估计的当前帧IMU位姿 和 相机/IMU外参，三角化当前滑窗中还没有深度估计的地图点（主要是上一帧中那些被当前帧跟踪到的还没有深度估计（在上一帧中没有左右匹配）的新点，以及当前帧中有左右匹配的新点）
        // 此项目的初始版本中默认双目系统中每一帧的特征点都必须要有立体深度估计！因为一方面有些背景点可能是动态物体的漏检；另一方面静态物体在某些帧可能重新变为动态。这就要要求了最好直接所有的点都要求有立体深度
        // 对于静态跟踪点，其深度值会在LBA中被优化～
        // 但由于近处点的立体匹配不容易获取，因此后续背景中的点不要求要有立体深度，否则背景跟踪点太少了
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu);
        
        // 如何衡量某一帧的视觉约束是否足够？不能只看该帧之前的跟踪点，还得看该帧之后的跟踪情况吧？
        // if(0 && frame_count >= TH_NUM_FRAME_FOR_LBA-1)
        // {
        //     if(num_LBA_fea_cur_frame < 6)
        //     {
        //         if(pnp_succ) id_frame_const_pose.push_back(frame_count);
        //     }
        // }

        // 联合当前滑窗所有VI信息对所有变量进行LBA优化，并且marg掉某帧形成先验残差信息
        // 非纯视觉时，由于初始化之后没有PnP,因此执行LBA。纯视觉时是否也需要每帧LBA？
        // 此外，如果当前帧参与LBA的点数过少，则不进行LBA。但是上面如果既没有PnP,又没有估计F或H呢？无论如何，如果某一帧的约束太少，则不适合进行LBA。则直接采用恒速运动的预测 或者 IMU积分值 作为当前帧相机的位姿
        // if(USE_IMU) 
        if(num_pt_cur_frame_in_LBA >= 3)
        {
            optimization();
        }
        
        set<int> removeIndex;
        // 对当前滑窗内的所有参与了LBA的地图点（需要至少有4帧观测）进行重投影（将地图点从其被观测首帧的左图像 重投影到 其他被观测帧的左右图像和首观测帧的右图像），计算和观测值的距离作为误差，平均像素误差大于3则认为是优化外点
        // 其中还要保留那些在当前帧还有跟踪且深度估计较为可靠的点作为新特征点！
        outliersRejection(removeIndex);
        // 去除地图中的外点。注意，此时物体位姿估计线程中可能正往地图中添加某些地图点的新观测或者新地图点，这里应该和该子线程就地图变量feature加锁(用map_fea_optimized变量来代替mutex）！
        // 倾向于这里先进行移除LBA后的外点，然后子线程中再往地图加入新的观测！因为如果某个静态点在前面至少连续4帧都跟踪到，但是在LBA后又平均投影误差较大，则即使再加上当前帧的观测，其平均误差也不会变小多少！！
        f_manager.removeOutlier(removeIndex);
        // 设置此值为true，通知子线程可以进行地图的写入操作了！
        map_fea_optimized = true;
        
        if (!MULTIPLE_THREAD)
        {
            // 当前滑窗内某些地图点因为重投影误差大于阈值，已经被作为外点从滑窗的地图点集中去除了，下面需要在跟踪线程的变量（与最新帧和次新帧的匹配相关）中去除与其中某些点相关的信息
            // 这里面只进行两个status的修改，真正删除cur_sift等变量中的元素要在子线程中进行！这样可以覆盖后续操作的时间
            // reserve_new_sift中是那些在当前帧有跟踪到的点，但是其之前所有的观测记录已经被删除，后续会作为当前帧其所属物体的新点（则当前帧不会加入地图）
            featureTracker.removeOutliers(removeIndex,reserve_new_sift,reserve_new_FAST);
            
            tracker_pts_updated = true;
            // 确定次新帧和最新帧中保留的特征点观测之后，根据当前最新两帧的全局位姿和恒速运动假设，预测这些点在下一帧左相机坐标系中可能的3D坐标
            // 预测下一帧的点坐标。放到下一帧的最开始去完成，因为需要知道两帧间的时间差，这样预估才能比较准确，防止掉帧的情况出现
            // predictPtsInNextFrame();
        }
        // printf("solver costs: %fms\n", t_solve.toc());    
        // ROS_DEBUG("solver costs: %fms", t_solve.toc());
        // 根据当前帧和上一帧的关联点数、当前帧IMU的bias的估计值、当前帧和上一帧之间的相对位移或旋转 来决定当前帧是否跟踪失败
        if (failureDetection())
        {
            printf("failure detection!\n");
            abort();
            // ROS_WARN("failure detection!");
            failure_occur = 1;
            obj_motion_esti.join();
            // 系统变量全部重置，包括重设failure_occur = 0
            clearState();
            setParameter();
            printf("system reboot!\n");
            // ROS_WARN("system reboot!");
            return;
        }

        cout << "start slide window!" << endl;

        // 滑窗去除某一帧的信息，更新各地图点的信息（观测记录和深度）。该函数最后需要等待子线程完成往地图中添加所有剩余的的静态跟踪点
        slideWindow();
        map_fea_writen = false;
        // if(!id_frame_const_pose.empty())
        // {
        //     int first_elem = 0;
        //     int rest_elem = 0;
        //     int num = id_frame_const_pose.size();
        //     // set的迭代器默认是const类型的，即无法通过其默认迭代器对set中的元素进行修改
        //     if(marginalization_flag == MARGIN_OLD)
        //     {
        //         if(id_frame_const_pose[0] == 0) first_elem = 1;
        //         for(int i = first_elem; i < num; ++i)
        //         {
        //             id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
        //             ++rest_elem;
        //         }
        //         id_frame_const_pose.resize(rest_elem);
        //     }
        //     else
        //     {
        //         for(int i = 0; i < num; ++i)
        //         {
        //             if(id_frame_const_pose[i] != WINDOW_SIZE-1) 
        //             {
        //                 if(rest_elem < i) id_frame_const_pose[rest_elem] = id_frame_const_pose[i] - 1;
        //                 ++rest_elem;
        //             }
        //         }
        //         if(rest_elem < num) id_frame_const_pose.resize(rest_elem);
        //     }
        // }
        
        // 去除掉那些LBA优化和滑窗之后之后深度估计值为负的地图点及其观测记录
        f_manager.removeFailures();
        cout << "succeeded remove failue points!" << endl;
        obj_motion_esti.join();
        
        // prepare output of VINS
        // key_poses.clear();
        // for (int i = 0; i <= WINDOW_SIZE; i++)
        //     key_poses.push_back(Ps[i]);
        // 记录当前滑窗优化和marg后的首帧（如果marg的是首帧，则marg后的首帧是当前滑窗的第2帧）和 尾帧（虽然滑窗内各帧在Rs中移动位置了，但是最后一个元素永远和当前滑窗的最新帧的数据相等）的全局位姿
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        prev_cam_R = last_R * ric[0];
        prev_cam_P = last_R * tic[0] + last_P;
        // 取IMU的测量queue中的所有新数据（大于或等与当前图像时间戳+td的时刻的IMU数据），持续积分得到最新的IMU全局状态量的预测
        // 预测这些量的作用是什么？为什么要在当前帧的最后积分下一帧图像期间要干的事？
        // 猜测是因为相机的测量频率比IMU的低很多，且此VINS系统的处理帧率也比相机的测量频率高，则在等待下一帧图像到来之前，先尽可能地处理一些accBuf中的测量数据？
        // 因此作用是在图像处理线程中优化完当前滑窗中所有帧之后，根据queue中新的IMU测量来推测之后最新的IMU位姿，用于尽快视觉化展示
        // 另外，函数中也只是从accBuf和gyrBuf中复制数据，而没有pop掉它们中的数据，这样也无法较少其中等待被取的数据数量？
        // 暂时不使用
        // updateLatestStates();

        cout << "estimated translation P of IMU is: " << Ps[frame_count].transpose() << endl;
        // prev_height = prev_cam_P(2);
        cout << "Succeeded process backend of MVIO!" << endl;
    } 

    if (!deform_obj_cls.empty()) deform_obj_cls.clear();
    if (!small_solid_objs.empty()) small_solid_objs.clear();
    if (!solid_obj_cls.empty()) solid_obj_cls.clear();
    if (!bbox_mask.empty()) bbox_mask.clear();
    // featureTracker.last_id_track_fea_prev = featureTracker.last_id_sift_cur;
}

// 单目-IMU的VI联合初始化函数。
// 单目-IMU系统会等累积满一个滑窗的帧测量（特征点检测和前后帧间的特征光流追踪）之后,
// 再在此函数中尝试一次性求解滑窗中所有帧的相机假定全局位姿和三角化地图点的假定全局坐标（和在其首观测帧的逆深度），
// 然后用视觉估计结果来估计各帧IMU的参数，和估计首帧IMU相机坐标系下g的方向，根据g将系用的全局坐标系由原先假定的全局坐标系（首帧相机坐标系）变到东北天坐标系，调整相应变量的全局估计值
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    // TODO：计算在假定的世界坐标系下多帧的局部加速度（即表达在上一帧的坐标系下）的标准差，为什么用这个值来判断激励是否足够？
    // 假设物体在短时间内自身的真实加速度变化很小，则均方差主要是由于物体的旋转（尤其是roll和pitch）导致的实际g与假定的世界坐标系的g的偏差！
    // 即这个值主要是判断物体在这期间是否有足够的pitch或roll运动，只有这样才能有足够的观测来优化g的方向（否则相关观测量一直不变，没法估计，即能观性不足）！！
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            // delta_v为两个相机帧之间的速度预积分增量（预积分增量直接就是IMU测量值的积分，其中还包含了待预估的bias，当然这个值是受到g的影响的）
            // 这里是计算两帧相机之间的局部平均加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        // 各个相机帧之间的平均局部加速度的平均
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        // 各个相机帧之间的平均局部加速度的标准差，衡量的是局部加速度的变化大小
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 为什么用局部加速度的标准差来判断IMU激励是否足够？
        // 猜测：
        // 加速度测量中包含了g的影响，这里的速度预积分也是包含了其影响。则var值的大小会有以下三种互斥的情况：
        // 1. 物体坐标系在两（相机）帧之间一直没有发生自身旋转，而是只进行了直线的恒加速运动，则var值会很小（g和真实加速度的复合量在局部坐标系中一直不变）。这种情况下不能确定g的方向和场景尺度吗？其实应该是可以的，为什么要排除这种情况（这对于汽车来说比较常见）
        // 2. 物体坐标系一直没有发生自身旋转（注意，即使物体公转，其自身不一定会自转），但是其线性加速度在多帧之间有明显的变化（数值或者方向，这意味这物体一定有真实加速度），则测量的复合加速度也会改变，那么var值会比较大。
        // 3. 两帧之间物体发生了（相对于自身坐标系的）roll或者pitch时，物体的局部加速度测量就会发生改变，那么无论物体自身的真实加速度是否恒定（即使一直为0），测量的复合加速度至少在局部坐标系下的方向会变，则var值会比较大。
        // 对于2，这也是为什么ORB-SLAM3中同样是此处的判断条件，只有当汽车在发生转弯时才会开始进行联合初始化，因为此时发生的是局部真实加速度（的数值）变化明显（汽车转弯时一般要减速），而不是发生了明显的roll或pitch！
        // 那么问题来了，只有不为0的恒定加速度的情况下就不能进行VI初始化吗？也许这对于单目会影响尺度的优化？也许用var>0.25来判断只是为了保证期间加速度不为0或者发生了旋转，而放弃了仅恒定加速度这种情况？
        // TODO; 因为“仅有恒定加速度”这种情况下没有办法仅通过使用加速度测量的统计（这里的aver_g和var）来判断加速度是否一直为0（g的影响）！因此干脆直接放弃这种情况下（然而确是汽车的常见运动）的初始化，而是等到“有变化的加速度”或“有pitch或roll运动”的场景出现时再尝试初始化！
        if(var < 0.25)
        {
            // VINS-Mono这里即使判断激励不足，还是尝试进行VI初始化，因为这里的判断条件其实是必要不充分的！不满足这个条件时相机其实可能还是有恒定加速度的！
            printf("IMU excitation not enouth!\n");
            //ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        printf("Not enough features or parallax; Move device around!\n");
        //ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    // 用跟踪结果完成初始滑窗内所有帧相机的位姿估计，以初始帧相机坐标系作为全局参考坐标系!
    // construct函数中只是对当前滑窗内所有帧进行姿态估计和观测地图点的三角化
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        printf("global SFM failed!\n");
        // ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame 为什么还有多余的帧要进行PnP??
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            // 这里的Q是以初始帧的相机坐标系作为参考坐标系，因此Q[0]其实就是单位矩阵I，T[0]是0向量！而下面得到的是视觉估计后各个时刻IMU坐标系相对于初始帧相机坐标系的位姿！
            // 用视觉估计结果和外参初始估计值来得到各帧IMU坐标系相对于当前全局坐标系（对于单目IMU而言，此处R采用的全局参考系是首帧相机的坐标系）的位姿！
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            // 相机和IMU都是固联在机体上，因此全局坐标下的位移量是相同的。但是首帧的IMU到首帧相机坐标系的位移不可能为0吧？
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        // 怎么会有这种情况？？难道是Headers中只保存关键帧的时间戳吗？从上面的代码来看并不是呀！此处之后的代码应该是不会被执行的？
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            printf("Not enough points for solve pnp !\n");
            // ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            printf("solve pnp fail!\n");
            // ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // 用视觉估计结果和外参初始估计值来得到各帧IMU坐标系相对于当前全局坐标系（对于单目IMU而言，此处R采用的全局参考系是首帧相机的坐标系）的位姿！
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        printf("misalign mono visual structure with IMU\n");
        // ROS_INFO("misalign mono visual structure with IMU");
        return false;
    }
}

// 对原版函数进行修改，以便stereo+IMU情况下固定尺度，且可以进行g的方向估计和坐标系对齐
bool Estimator::visualInitialAlign(bool ForStereo)
{
    TicToc t_g;
    VectorXd x;
    Matrix3d cam_R;
    Vector3d cam_P;
    bool result = false;
    bool first_win_ = first_win;
    //solve scale
    // 会在VisualIMUAlignment函数中优化估计陀螺仪Biaa，并且估计在当前假定全局参考坐标系中的g的表达，用于与东北天坐标系进行校正
    if (!ForStereo)
    {
        result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    }
    else
    {
        // result = VisualIMUAlignment(all_image_frame, Bgs, g, x, true);
        solveGyroscopeBias(all_image_frame, Bgs);
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            // 为什么加速度计的bias还是设为0？应该是认为加速度计的bias一般很小，对计算的影响不大，不值得专门估计
            pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        return true;
    }

    if(!result)
    {
        printf("solve g failed!\n");
        // ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state  貌似VisualIMUAlignment的优化中没有直接对每一ImageFrame的R和T进行过修改呀？
    // 这是对于单目-IMU的操作，是在VisualIMUAlignment函数调用之前对ImageFrame的R和T有进行优化估计，且单目-IMU的R和T的初始全局参考坐标系是初始帧相机坐标系！！
    // 而且单目-IMU情况下，之前Rs等变量（通过预积分建立的IMU的全局位姿）和all_image_frame中的R等（通过视觉估计出的IMU坐标系的全局位姿）还没有建立联系
    for (int i = 0; i <= frame_count; i++)
    {
        if (!ForStereo)
        {
            // 将Rs和Ps中的量替换为校正后的视觉估计的结果。两个量的参考坐标系也不同，先前的参考系是随意选择的（初始时刻附近的平均g），后面则是以初始帧相机坐标系为参考系
            Rs[i] = all_image_frame[Headers[i]].R;
            Ps[i] = all_image_frame[Headers[i]].T;
        }
        // 每帧的滑窗内的所有帧都是关键帧
        all_image_frame[Headers[i]].is_key_frame = true;
    }
    
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 为什么加速度计的bias还是设为0？应该是认为加速度计的bias一般很小，对计算的影响不大，不值得专门估计
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    if (!ForStereo){
        double s = (x.tail<1>())(0);
        for (int i = frame_count; i >= 0; i--)
            Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    }
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        // 只优化了关键帧的Vs？因为VINS中最后只会保存滑窗内的帧的位姿，也只有它们会参与后续的LBA，因此这里只会保留它们的速度估计
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // R0是 先前假定的全局坐标系 到 东北天坐标系的 相对位姿。g2R()函数中会默认将相对yaw角置为0
    Matrix3d R0 = Utility::g2R(g);
    // R0 * Rs[0]则表示的是计算出的 系统首帧IMU的坐标系 到 东北天坐标系下的相对位姿，由于yaw角实际上是不可测的，因此取yaw角值用于消除yaw相对旋转
    // 对于单目IMU而言，此时Rs[0]其实是单位矩阵I？
    double yaw;

    //Matrix3d rot_diff = R0 * Rs[0].transpose();

    Matrix3d rot_diff;
    
   
    // 为什么最后的rot_diff是这样？VisualIMUAlignment函数中最后估计出来的g是在哪个坐标系下，取决于all_image_frame中各帧的R和T的参考世界坐标系
    // 对于单目IMU而言，用于VI对齐的ImageFrame的R和T的参考世界坐标系是首帧相机坐标系；而对于双目-IMU而言，R和T的值直接来自于Ps和Rs（代表的是IMU的全局位姿），其参考世界坐标系为Rs[0]和RIC所共同计算出来的坐标系！
    // 无论优化后的g表达在哪个参考世界坐标系下，上面计算出的R0都是该参考坐标系在东北天坐标系下的位姿！
    // 而乘以rot_diff之前Ps、Vs和Rs表示的是各帧IMU坐标系在该参考世界坐标系下的位姿，最终Ps、Vs和Rs中需要保存的是各IMU坐标系在东北天下的全局位姿
    // 鉴于上面yaw角值的计算方式以及最后R0的校正，对于首帧IMU而言直接rot_diff = R0即可（R0*Rs[0]得到首帧IMU在东北天坐标系下的位姿，再校正其yaw角）！
    // 而对于之后帧的IMU坐标系，理论上应该先将其转换到校正前的首帧IMU坐标系下，然后再通过已校正的Rs[0]转换到东北天坐标系下，如下面公式(1)所示，化简后其实就是R0！
    if(!STEREO)
    {
        yaw = Utility::R2ypr(R0 * Rs[0]).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        // 此处得到的g应该得是(0,0,9.8)才对，即世界坐标系下的g的值（准确地说应该是静止状态下且IMU坐标系保持水平的情况下的读数）
        g = R0 * g;
        rot_diff = R0;
        
        // 不论有没有成功初始化，都对全局参考坐标系进行修改？
        for (int i = 0; i <= frame_count; i++)
        {
            // 对齐过程中，是没有把Ps和Rs当作已知常量来参与计算的，因此其值不会改变。这里是改变全局参考系后，要对各帧的IMU坐标系进行校正
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
            if (first_win_)
            {
                cam_R = Rs[i] * ric[0]; 
                cam_P = Rs[i] * tic[0] + Ps[i];
                all_cam_R_before_init.push_back(cam_R);
                all_cam_P_before_init.push_back(cam_P);
            }
        }
    }
    // 双目情况下，此时Rs中仍为首帧body坐标系到假定世界坐标系的旋转
    else
    {
        yaw = Utility::R2ypr(R0).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        g = R0 * g;
        rot_diff = R0;
        for (int i = 0; i <= frame_count; i++)
        {
            // 对齐过程中，是没有把Ps和Rs当作已知常量来参与计算的，因此其值不会改变。这里是改变全局参考系后，要对各帧的IMU坐标系进行校正
            Ps[i] = rot_diff * Ps[i];
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];
            if (first_win_)
            {
                cam_R = Rs[i] * ric[0]; 
                cam_P = Rs[i] * tic[0] + Ps[i];
                all_cam_R_before_init.push_back(cam_R);
                all_cam_P_before_init.push_back(cam_P);
            }
        }
        
        if(!first_win_ && !all_cam_P_before_init.empty())
        {
            for(int i = 0; i < all_cam_P_before_init.size(); ++i)
            {
                cam_R = rot_diff * all_cam_R_before_init[i];
                cam_P = rot_diff * all_cam_P_before_init[i];
                all_cam_R_before_init[i] = cam_R;
                all_cam_P_before_init[i] = cam_P;
            }
            cam_R = Rs[frame_count] * ric[0]; 
            cam_P = cam_R * ric[0].transpose() * tic[0] + Ps[frame_count];
            // 保存当前最新帧的位姿估计
            all_cam_R_before_init.push_back(cam_R);
            all_cam_P_before_init.push_back(cam_P);
        }
    }
    
    // 对原变量first_win的修改不在这里
    // if(first_win_) first_win_ = false;

    cout << "g0:" << g.transpose() << endl;
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // TODO: 判断一下估计的g是否接近（0, 0, -9.8)，如果没有，则IMU坐标系的位姿估计Rs其实并没有真正表达在东北天坐标系下！
    // 即使没有将IMU坐标系的位姿真正表达在东北天坐标系下，对于双目-IMU而言，后续也可以使用IMU的测量来直接积分推断IMU的全局位姿，只不过此位姿是相对于一个假定的“接近于东北天”的参考坐标系（有多接近取决于估计的g有多接近东北天的(0,0,9.8)）！
    // 但如果单目-IMU估计后的g不够接近（0,0,9,8），则是否意味着尺度的估计应该也是不够准确的？！另外，如果要结合GPS等全局信息，不对齐到东北天坐标系时是否可能会有一些问题？
    assert((g[0]*g[0]+g[1]*g[1]+(abs(g[2])-9.8)*(abs(g[2])-9.8)<=0.25) && "Estimated g doesn't match the real one, VI alignment doesn't succee!");
    cout << "my R0:  " << Utility::R2ypr(Rs[0]).transpose() << endl;
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    // TODO：此处我们对双目-IMU是否也要执行此操作
    // 双目-IMU在VI对齐阶段对于视觉估计的影响只是改变了全局参考坐标系，即改变了相机的全局位姿，这对之前地图点在各首观测帧下的深度估计有影响吗？
    // 如果使用了立体校正的图像对，那应该不需要；如果没有使用立体图像对，但该点的观测首帧中有左右匹配，且左右相机之间的相对位姿已知且固定，则也不需要；
    // 否则，该点的三角化需要已知左相机（和右相机）的估计的全局位姿，则在重新估计了所有帧的全局位姿之后，需要重新进行所有点的三角化！！（不重新三角化是否可以？即认为位姿优化前后会相对位姿影响不大？重新三角化后深度值应该会更准确）
    if(!STEREO || !has_stereo_rectified)
    {
        // 清除所有点的深度估计？单目系统在成功完成VI初始化之前并没有对滑窗内的地图点进行过任何的深度估计呀？？？！双目系统倒是需要清除
        f_manager.clearDepth();
        int num_LBA_fea_cur_frame = 0;
        // 用校正之后的各帧位姿来重新得到各个特征地图点在全局坐标系（东北天坐标系）下的3D坐标！
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric, featureTracker, prev_cam_P_using_imu, prev_cam_R_using_imu, num_LBA_fea_cur_frame);
    }
    return true;
}

// 根据视觉匹配逐一求解当前最新帧和滑窗内之前的帧之间的相对位姿，如果找到某一帧满足“相对位姿求解成功且两帧之间的观测匹配点的平均视差大于阈值”，则直接将该帧作为搜寻结果
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 不但要求解成功，还要求两帧之间视觉匹配点的平均视差大于阈值
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure\n", average_parallax * 460, l);
                // ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

// 将VINS项目使用的变量存储容器（如Rs为vector<Eigen::MatrixXd>）的对应变量初始估计值 赋给 Ceres的LBA问题的相关待优化变量（保存在double型数组值中，如para_Pose）
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    
    para_Td[0][0] = td;
}

// 将Ceres的估计结果（保存在double型数组值中，如para_Pose）赋给VINS项目使用的变量存储容器（如Rs为vector<Eigen::MatrixXd>）的对应变量
void Estimator::double2vector()
{
    // 先保留未LBA之前的当前滑窗首帧的全局位姿
    // 在VI初始化时由于估计了g，对所有IMU的位姿均进行了校正（初始帧IMU位姿与东北天对齐）。
    // 之后每次LBA还是要对系统首帧进行估计，因为后面所有帧的位姿都是相对于初始帧的IMU坐标系，而有了IMU的参与则需要继续优化首帧位姿（其实就是继续优化首帧坐标系下的g变量）
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    // 什么时候在会在此函数内出现failure_occur=1的情况？？
    // 一旦某一帧跟踪被检测出现了失败，则failure_occur立马被设为1，但是紧接着就会重置系统，此时又会设置failure_occur=0了！所以此处下面的条件是永远不会出现的！
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        // LBA优化后得到的当前滑窗首帧的IMU坐标系的全局位姿
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        // 如果发生了欧拉角的奇异锁，则相当于直接固定当前滑窗的首帧IMU的位姿？
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {   
            printf("euler singular point!\n");
            // ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }
        // 为什么有IMU的情况下，每一次LBA前后，都是固定滑窗首帧IMU的全局yaw角值？
        // 因为整个系统的首帧的全局yaw角就是假设的（假设与东北天坐标系的相对yaw为0），且我们认为VI初始化之后的全局参考坐标系就是(或接近）真实的东北天坐标系（这取决于优估计和转换后的g是否为(0, 0, -9.8)
        // 则整个系统首帧IMU的姿态R中只有yaw角是固定的，roll和pitch是可以优化的（因为估计的g代表的是假定的全局坐标系和东北天坐标系的相对位姿，g固定只代表全局参考系固定，而首帧固定的只有其相对全局坐标系的yaw和位移）
        // 因此每个滑窗的首帧（包括首个滑窗的首帧，即系统的首帧）的全局位姿在LBA的前后，其yaw角都应该保持不变，以此作为连结来保证每个滑窗中的帧都使用同一个原始的全局参考系（即VI初始化后由估计的g和(0,0,9.8)决定的坐标系）
        // 因此跟VI联合初始化后的校正思路类似，都是首先使得优化后的滑窗首帧IMU的Rs表达在由VI初始化估计的g确定的全局坐标系下（这里是通过保证每个滑窗LBA前后其首帧IMU的全局位姿的yaw角不变，因为它是从首帧滑窗优化后一直传递下来的）
        // 而每个滑窗从第2帧开始，都先变换到LBA前的滑窗首帧IMU坐标系，然后再使用LBA和校正后的Rs[0]来转换到由VI初始化估计的g确定的全局坐标系下！然后简化公式，就可以得到下面对每个Rs都适用的校正公式！
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            // 位移则是先计算优化后的每帧到首帧的相对位姿，然后再将这个位移校正到表达在“由VI初始化估计的g确定的全局坐标系W下”，最后再加上首帧IMU到全局坐标系W的位移（
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        // 纯视觉的情况下，滑窗首帧的位姿在LBA优化时其实是被固定住了的，因此此处para_Pose[0]其实就是I，P[0]其实就是0
        // 可是上一个滑窗marg掉首帧之后的先验信息，不会对当前滑窗的首帧的位姿产生影响吗？
        // 首帧被marg掉之后，其实是将与首帧有共视关系的其他帧都关联了起来，这只会影响这些帧之间的相对位姿。
        // 而每一个滑窗的首帧的全局位姿其实都是无法被优化的，因为系统的首帧相机坐标系就是直接被作为了全局坐标系，后续所有的帧的位姿其实都是相对与首帧的位姿！
        // 又由于滑窗会将系统的首帧给marg掉，所以后续的滑窗的首帧的位姿都应该直接固定住，因为此时它就是与最开始的假定全局坐标系（系统首帧相机坐标系）的一个直接关联！不可以被优化！
        // 滑窗的首帧之后的帧都是通过与首帧的视觉约束，虽然计算的仍然是全局位姿，但是由于首帧是固定的，所以其实约束的是它们与该滑窗首帧之间的相对位姿！这样就实现了之后所有的帧都是相对于系统初始帧的假定坐标系！
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    // para_Feature的第一个维度的长度 和 f_manager的元素数量 是相同的，只不过其中的某些点可能不会参与LBA（比如其在当滑窗中的连续观测帧数小于4），也就没有深度的估计和更新
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    
    f_manager.setDepth(dep);
    
    if(USE_IMU)
    {
        prev_td = td;
        td = para_Td[0][0];
    }
}

// 根据当前帧和上一帧的关联点数、当前帧IMU的bias的估计值、当前帧和上一帧之间的相对位移或旋转 来决定当前帧是否跟踪失败
bool Estimator::failureDetection()
{
    return false;
    // 为什么和上一帧之间的关联点这么少，还可以继续？？如果VI初始化已经成功，那么类似ORB-SLAM3中的思路，可以使用IMU测量在短暂时间内来推测位姿，同时期待可以回环线程可以完成视觉的回环（以重新开始跟踪）
    // 如果没有回环线程，且如果VI还没初始化成功，则这里应该直接返回true了。然而此函数只有在VI初始化之后才会调用，因此可以使用IMU来预测位姿。
    if (f_manager.last_track_num < 2)
    {
        printf(" little feature %d!\n", f_manager.last_track_num);
        // ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        printf(" big IMU acc bias estimation %f!\n", Bas[WINDOW_SIZE].norm());
        // ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        printf(" big IMU gyr bias estimation %f!\n", Bgs[WINDOW_SIZE].norm());
        // ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        printf(" big translation!\n");
        // ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        printf(" big z translation!\n");
        // ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        printf(" big delta_angle!\n");
        // ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

// 根据动态物体的特征点在上一帧相机坐标系下的3D坐标，和其在当前帧相机坐标系下的归一化平面的匹配点坐标，计算两帧之间物体的复合运动变换（融合了相机的帧间运动）
bool Estimator::SolveObjPoseTransByPnP(const FeaFrame &fea_tracked_obj, Vector3d &delta_P, Matrix3d &delta_R, vector<float*> &pixel_lost_objs, const int &valid_objs, Vector3f &ave_3D_pts_obj, int gl_obj_id)
{
    vector<cv::Point2f> pts2D;
    // 上一帧的特征点在其相机坐标系中的3D点坐标
    vector<cv::Point3f> pts3D;
    set<int> pt_id;
    vector<int> pt_id_vec;

    map<int,int> &gl_id_index_map = featureTracker.gl_id_index_map;
    vector<float> &prev_FAST_dep = featureTracker.prev_FAST_dep;
    vector<float> &prev_sift_dep = featureTracker.prev_sift_dep;

    for (auto &pt:fea_tracked_obj)
    {
        const Vector8d &pt_info = pt.second[0].second;
        // 当前帧的特征点的归一化平面坐标
        pts2D.emplace_back(pt_info(0),pt_info(1));
        // 上一帧的匹配点的归一化平面坐标
        double prev_x_norm;
        double prev_y_norm;
        double prev_z = 0;
        // 参与运动估计的物体点集中，有可能部分是临时添加的像素点匹配，则只有左相机的观测信息,其中还把像素点的速度和obj_id替换为上一帧的归一化点坐标和深度，具体查看FeatureTracker::objs_matching_assign的最后
        if (pt.first >= 0)
        {
            prev_x_norm = pt_info(0) - pt_info(5) * delta_T_cam_new;
            prev_y_norm = pt_info(1) - pt_info(6) * delta_T_cam_new;
            // 物体点不一定有立体匹配
            // prev_z = pt.second[1].second(2);
            int index = gl_id_index_map[pt.first];
            if(index > 0)
            {
                prev_z = prev_FAST_dep[index-1];
            }
            else
            {
                prev_z = prev_sift_dep[-index];
            }

            // 物体点在每一帧下应该都要有深度
            // assert(prev_z > 1.0);

            if(prev_z <= 0)
            {
                int index = featureTracker.gl_id_index_map[pt.first];
                cout << "elem in gl_idnex_map: " << index << endl;
                cout << "last_id_track_fea_cur: " << featureTracker.last_id_track_fea_cur << " id of invalid fea: " << pt.first << endl;

                if(index <= 0)
                {
                    cout << "cls: " << (int)featureTracker.obj_cls_id_sift[-index].first  << " l_obj_id: " << featureTracker.obj_cls_id_sift[-index].second << endl;
                }
                else
                {
                    cout << "cls: " << (int)featureTracker.obj_cls_id_FAST[index-1].first  << " l_obj_id: " << featureTracker.obj_cls_id_FAST[index-1].second << endl;
                }

                if(gl_obj_id > 0)
                {
                    cout << "gl obj id: " << gl_obj_id << endl;
                    if(featureTracker.status_objs_prev.find(gl_obj_id) != featureTracker.status_objs_prev.end())
                        cout << "status of gl obj is: " << (int)featureTracker.status_objs_prev[gl_obj_id] << endl;
                }
                else
                {
                    cout << "gl obj is total lost in prev frame!" << endl;
                }

                assert(false);
                continue;
            }
        }
        else
        {
            prev_x_norm = pt_info(5);
            prev_y_norm = pt_info(6);
            prev_z = pt_info(7);
        }
        pts3D.emplace_back(prev_x_norm * prev_z,prev_y_norm * prev_z,prev_z);
        pt_id.insert(pt.first);
        pt_id_vec.push_back(pt.first);
    }
    cv::Mat r, rvec, t, D, tmp_r, inliers;
    cv::eigen2cv(delta_R, tmp_r);
    // 旋转矩阵变为旋转向量
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(delta_P, t);
    // 内参矩阵设置为单位矩阵，则说明pts2D不是像素点坐标，则是归一化平面坐标
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    // 投影误差阈值的经验值8.0是像素坐标系的，而这里要用的是归一化平面坐标
    // FOCAL_LENGTH_X == FOCAL_LENGTH_Y

    pnp_succ = cv::solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 4.0 / FOCAL_LENGTH_X, 0.95, inliers);
    // 尝试放宽条件
    if (!pnp_succ)
    {
        cv::Mat inliers_loose;
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(delta_P, t);
        // 此函数内应该是不会改变非空Mat的inliers的size的，如果直接使用上面的inliers，则两次估计时如果内点数不一样，则数量差异无法体现在inliers中！所以这里使用新的inliers_loose
        pnp_succ = cv::solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / FOCAL_LENGTH_X, 0.85, inliers_loose);
        if(!pnp_succ)
        {
            printf("pnp for object failed ! \n");
            //assert(pnp_succ && "there is something wrong about the object pose PnPRansac estimation!");
            return false;
        }
        // printf("pnp for object succeeded ! \n");
        // copyto works for same size and type matrices, if src and dst Mat have different size or type, before copy data dst Mat will be relocated using create(NewSize,NewType)
        inliers_loose.copyTo(inliers);
    }
    
    // 内点数不至于这么少吧？这个是针对上一帧完全漏检的情况，如果最终内点数太少，则不认为该漏检发现是有效的。
    if (inliers.rows < 4) return false;

    // 旋转向量变为旋转矩阵
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;

    cv::cv2eigen(r, delta_R);

    cv::cv2eigen(t, delta_P);

    vector<uchar> &statusLeftRIght = featureTracker.statusLeftRIght;
    vector<uchar> &status_sift = featureTracker.status_sift;
    vector<float> &cur_FAST_dep = featureTracker.cur_FAST_dep;
    vector<float> &cur_sift_dep = featureTracker.cur_sift_dep;
    vector<int> &id_sift_no_depth = featureTracker.id_sift_no_depth;
    vector<int> &id_FAST_no_depth = featureTracker.id_FAST_no_depth;
    
    vector<Point2f> &cur_un_FAST = featureTracker.cur_un_FAST;
    vector<Point2f> &cur_un_sift = featureTracker.cur_un_sift;
    vector<int>::iterator iter_end_FAST = id_FAST_no_depth.end();
    vector<int>::iterator iter_end_sift = id_sift_no_depth.end();

    Vector3d pt_prev, pt_cur;
    float cur_z;
    
    int num_inliers = inliers.rows;
    int num_track_fea = 0;
    // 更新内点中深度估计不是很可靠的点。
    for(int j = 0; j < num_inliers; j++)
    {
        int id_vec = inliers.at<int>(j);
        int id_pt = pt_id_vec[id_vec];
        // 如果是非像素点
        if(id_pt >= 0)
        {
            int index = gl_id_index_map[id_pt];
            // 修改运动估计内点在当前帧中的深度值
            // 如果当前帧该点的深度是通过CPU光流匹配获得的，则认为较为可靠，不需要更新
            if(index > 0)
            {
                index -= 1;
                if(std::find(id_FAST_no_depth.begin(),iter_end_FAST,index) == iter_end_FAST)
                {
                    cur_z = cur_FAST_dep[index];
                    ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + cur_un_FAST[index].x * cur_z;
                    ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + cur_un_FAST[index].y * cur_z;
                    ave_3D_pts_obj(2) = ave_3D_pts_obj(2) + cur_z;
                    ++num_track_fea;
                } 
                else
                {
                    pt_prev(0) = pts3D[id_vec].x;
                    pt_prev(1) = pts3D[id_vec].y;
                    pt_prev(2) = pts3D[id_vec].z;
                    
                    pt_cur = (delta_R * pt_prev + delta_P);
                    cur_z = pt_cur(2);
                    // if (cur_z < mThDepthObj && cur_z > mMinDepthPt) 
                    if (cur_z < mThDepthObj && cur_z > 1.0)
                    {
                        ++num_track_fea;
                        cur_FAST_dep[index] = cur_z;
                        // 当前帧该点的x和y应该用检测值（乘以更新后的深度）还是运动更新后的值？
                        // ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + pts2D[id_vec].x * cur_z;
                        // ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + pts2D[id_vec].y * cur_z;
                        ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + pt_cur(0);
                        ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + pt_cur(1);
                        ave_3D_pts_obj(2) = ave_3D_pts_obj(2) + cur_z;
                    }
                    // 如果更新后深度不满足条件，则当作外点！
                    else
                    {
                        statusLeftRIght[index] = 0;
                        --num_inliers;
                    }    
                }
            }
            else
            {
                index = -1 * index;
                if(std::find(id_sift_no_depth.begin(),iter_end_sift,index) == iter_end_sift)
                {
                    ++num_track_fea;
                    cur_z = cur_sift_dep[index];
                    ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + cur_un_sift[index].x * cur_z;
                    ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + cur_un_sift[index].y * cur_z;
                    ave_3D_pts_obj(2) = ave_3D_pts_obj(2) + cur_z;
                }
                else
                {
                    pt_prev(0) = pts3D[id_vec].x;
                    pt_prev(1) = pts3D[id_vec].y;
                    pt_prev(2) = pts3D[id_vec].z;
                    
                    pt_cur = (delta_R * pt_prev + delta_P);
                    cur_z = pt_cur(2);
                    // if (cur_z < mThDepthObj && cur_z > mMinDepthPt)
                    if (cur_z < mThDepthObj && cur_z > 1.0)
                    {
                        ++num_track_fea;
                        cur_sift_dep[index] = cur_z;
                        // ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + pts2D[id_vec].x * cur_z;
                        // ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + pts2D[id_vec].y * cur_z;
                        ave_3D_pts_obj(0) = ave_3D_pts_obj(0) + pt_cur(0);
                        ave_3D_pts_obj(1) = ave_3D_pts_obj(1) + pt_cur(1);
                        ave_3D_pts_obj(2) = ave_3D_pts_obj(2) + cur_z;
                    }
                    else
                    {
                        status_sift[index] = 0;
                        --num_inliers;
                    }
                }
            }
        }
        // 内点特征点完成记录，则将其从点集删除
        // 剩下的是需要被清除的外点。这里多用一个set，其删除元素比vector要方便，其次是vector中的值不好做标记（因为可能有像素点，其id值为负，因此不能再用负值来标记运动估计外点）
        pt_id.erase(id_pt);
    }

    // >0即可？
    // 特征点要足够多，才使用特征点的平均3D坐标
    if (num_track_fea >= 3)
        ave_3D_pts_obj = ave_3D_pts_obj/num_track_fea;
    else if(num_inliers < 5)
    {
        // 注意，num_track_fea是两个物体间的关联特征点内点数量，但是两个关联物体之间不一定有特征点关联！！它们可能是完全由像素点来关联的！！这里如果它们之间总的内点数小于5个，则认为运动估计失败
        printf("pnp for object failed ! \n");
        // 如果深度值更新后发现内点数不足，该怎么办？还是把该物体保留为动态物体？还是作为当前帧的新物体？这里倾向于作为新物体，而且作为新物体时，原本那些深度估计不可靠的点也会被抛弃，即上面的运动更新深度值不会被保留下来
        return false;
    }

    // 运动估计的外点
    for(auto &id: pt_id)
    {
        // 像素点先跳过。
        // 如果某个物体当前帧完全漏检，则后续需要寻找上一帧的密集采样像素点到当前帧的可靠光流匹配！
        if (id < 0) 
        {
            // 像素匹配外点忽略
            continue;
        }
        
        // assert(gl_id_index_map.find(id) != gl_id_index_map.end() && "There must be something wrong with var gl_id_index_map!");
        int index = gl_id_index_map[id];
        // index小于等于0的是sift
        if (index <= 0)
        {
            status_sift[-1*index] = 0;
        }
        // 注意，FAST点的序号在gl_id_index_map中是+1的
        else
        {
            statusLeftRIght[index-1] = 0;
        }
    }
    
    vector<int> &lost_objs_cur = featureTracker.detect_lost_objs_cur;
    vector<camodocal::CameraPtr> &cam = featureTracker.m_camera;
    // 如果当前帧该物体为完全漏检的物体，则通过估计的运动模型 和 flow map的结果 来筛选该物体的采样像素
    if(gl_obj_id > 0 && !lost_objs_cur.empty())
    {
        auto iter_lost = std::find(lost_objs_cur.begin(),lost_objs_cur.end(),gl_obj_id);
        if(iter_lost != lost_objs_cur.end())
        {
            assert((MAX_NUM_OBJS_FRAME-valid_objs)>=lost_objs_cur.size() && "Array sampled_pixel is not enough to store all found valid objects in current frame!");

            assert(featureTracker.pixel_objs_prev.find(gl_obj_id) != featureTracker.pixel_objs_prev.end());

            float* ptr_pix_prev = featureTracker.pixel_objs_prev[gl_obj_id];
            int id = distance(lost_objs_cur.begin(),iter_lost);
            // 注意，取二维数组首元素的指针，必须是先用[0][0]寻址，然后再用&取址,这样得到的才是单个元素的指针；而二维数组的变量名a，其本身代表的是一个一维数组，指向的a[0]。而a+i表示二维数组的第i行（也是个一维数组）!
            // 即对于一个n维数组a，变量名a本身是一个指针，其代表的是一个“n-1维数组类型”的指针，指向的地址是a的第一个n-1维子数组。那么a+1就是指向a的第2个n-1维的子数组（的首地址），即+1是直接偏移整个n-1维子数组的长度！！
            // https://blog.csdn.net/Zonggu/article/details/124051465
            // 此外，对于一个n维数组，a代表的是指向其第一个n-1维度子数组的指针，指针类型为n-1维数组，a+1一次性偏移一个n-1维子数组的长度；而auto p = &a则代表的是“n维数组"类型的指针，虽然p同样指向数组a的首个元素的地址，但p+1是一次性偏移整个a数组的长度！！
            // https://blog.csdn.net/oqqHuTu12345678/article/details/52605952
            // 例如，一维int数组int a[5]，a+1表示第2个int元素，int *p = a代表p是一个整数指针！ 
            // 而二维数组int a[5][5]，a+1表示第2行这个一维数组，而auto p = a代表p是一个一维数组指针，即int (*p)[]（注意，如果显式定义p为int*，则赋值时右边也要变成整数指针，例如int *p = (int*)a，表示的是取一维数字的第一个元素的地址）！
            // 
            // 下面标注的这个方式取地址是对的，可以取到二维数组中具体第m行第n列的元素的地址，且为float型指针，不是一维数组指针!
            // float* ptr_pix_cur = &(featureTracker.sampled_pixel[0][0]) + (valid_objs+id)*(NUM_SAMPLED_PIXEL_OBJ*3+2+5) + 2;
            // 上面的表达式可以等价于下面两个表达式。其中表达式1的大()前面必须加上取*，因为大()里面始终是个一维数组类型的指针，必须得加*符号才会得到其内容，即某一行的一维数组，而一维float数组变量名本身是一个float*指针，加法运算表示取该一维数组中的下一个元素！
            // 表达式2中的对指针变量的[]操作其实就相当于 （地址加减）寻址操作 + 取内容*操作！
            // https://stackoverflow.com/questions/66798668/why-an-array-pointer-with-index-returns-the-value-in-that-index-instead-of-retur

            // float* ptr_pix_cur = *(featureTracker.sampled_pixel + (valid_objs+id)) + 2
            float* ptr_pix_cur = featureTracker.sampled_pixel[(valid_objs+id)] + 2;
            int num_pixel_cur = 0;
            
            sample_pixel_for_lost_obj(ptr_pix_prev, ptr_pix_cur, num_pixel_cur, delta_P, delta_R);
            // 修改物体信息
            // 物体的全局cls继承自上一帧的匹配物体
            // auto iter = std::find(featureTracker.glob_obj_id_prev.begin(), featureTracker.glob_obj_id_prev.end(),gl_obj_id);
            // int index = distance(featureTracker.glob_obj_id_prev.begin(),iter);
            // ptr_pix_cur[-2] = featureTracker.obj_cls_prev[index];
            ptr_pix_cur[-1] = num_pixel_cur;
            // 最后一位保存平均深度
            ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3D_pts_obj(0);
            ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3D_pts_obj(1);
            ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3D_pts_obj(2);
            // 保存采样像素点的内存指针
            pixel_lost_objs[id] = ptr_pix_cur;
        }
    }
    return pnp_succ;
}

// 为当前帧完全漏检的物体采集像素点，使用flow_map匹配 和 基于运动模型的重投影 的对比 来获取有效的采样像素点
Vector3f Estimator::sample_pixel_for_lost_obj(float* ptr_pix_prev, float* ptr_pix_cur, int &num_pixel_cur, Vector3d &delta_P, Matrix3d &delta_R, bool cal_ave_3d_pts)
{
    vector<camodocal::CameraPtr> &cam = featureTracker.m_camera;
    Vector3f ave_3d_pts(0,0,0);
    float min_u = 0, max_u = 0, min_v = 0, max_v = 0;

    int num_pixel_prev = (int)ptr_pix_prev[-1];
    float flow_pred_u, flow_pred_v, motion_pred_u, motion_pred_v;
    Point2f tmp_pt, tmp_un_pt;
    Vector3d pt_prev, pt_cur;

    num_pixel_cur = 0;
    vector<float> pix_u, pix_v;

    for(int k = 0; k < num_pixel_prev; ++k)
    {
        tmp_pt.x = ptr_pix_prev[3*k];
        tmp_pt.y = ptr_pix_prev[3*k+1];
        
        flow_pred_u = map_flow.at<Vec2f>(tmp_pt.y,tmp_pt.x)(0) + tmp_pt.x;
        flow_pred_v = map_flow.at<Vec2f>(tmp_pt.y,tmp_pt.x)(1) + tmp_pt.y;

        featureTracker.undistortedPts(tmp_pt, tmp_un_pt, cam[0]);
        pt_prev(2) = ptr_pix_prev[3*k+2];
        pt_prev(0) = tmp_un_pt.x * pt_prev(2);
        pt_prev(1) = tmp_un_pt.y * pt_prev(2);
        pt_cur = delta_R * pt_prev + delta_P;
        float dep_cur = pt_cur(2);
        // if (dep_cur < mMinDepthPt || dep_cur > mThDepthObj) continue;
        if (dep_cur < 1.0 || dep_cur > mThDepthObj) continue;
        featureTracker.spaceToPlane(pt_cur,tmp_pt,cam[0]);
        
        // 是否还要查看投影点是否在图像范围内？有距离约束应该就不会出现这种情况
        // 相差不超过3个像素点，认为该点属于漏检物体的
        if((flow_pred_u-tmp_pt.x)*(flow_pred_u-tmp_pt.x) + (flow_pred_v-tmp_pt.y)*(flow_pred_v-tmp_pt.y) < 9)
        {
            ptr_pix_cur[(num_pixel_cur*3)]   = tmp_pt.x;
            ptr_pix_cur[(num_pixel_cur*3+1)] = tmp_pt.y;
            ptr_pix_cur[(num_pixel_cur*3+2)] = dep_cur;

            pix_u.push_back(tmp_pt.x);
            pix_v.push_back(tmp_pt.y);
            
            ++num_pixel_cur;

            if (cal_ave_3d_pts)
            {
                // 需要隐式转换double为float
                ave_3d_pts(0) = ave_3d_pts(0) + pt_cur(0);
                ave_3d_pts(1) = ave_3d_pts(1) + pt_cur(1);
                ave_3d_pts(2) = ave_3d_pts(2) + pt_cur(2);
            }
        }
    }
    
    if(num_pixel_cur > 0)
    {
        ave_3d_pts = ave_3d_pts/num_pixel_cur;
        // 当前帧漏检物体的cls label继承自上一帧的匹配物体的全局cls
        ptr_pix_cur[-2] = ptr_pix_prev[-2];
        int num_size = 3 * NUM_SAMPLED_PIXEL_OBJ;
        
        min_u = *(std::min_element(pix_u.begin(), pix_u.end()));
        max_u = *(std::max_element(pix_u.begin(), pix_u.end()));
        min_v = *(std::min_element(pix_v.begin(), pix_v.end()));
        max_v = *(std::max_element(pix_v.begin(), pix_v.end()));
        ptr_pix_cur[num_size++] = min_u;
        ptr_pix_cur[num_size++] = max_u;
        ptr_pix_cur[num_size++] = min_v;
        ptr_pix_cur[num_size++] = max_v;
    }
    
    return ave_3d_pts;
}

// 2000行的主要函数！！
// 使用openMP对多个物体同时进行RANSAC位姿变换估计
// 此函数是要作为线程调用函数的，而这里不需要对其实参进行修改并反映到子线程之外的原变量，因此形参全都不需要是左值引用类型，即thread函数对所有传入的参数进行值拷贝即可（要求原变量的内存占用也不大，否则还是用左值引用比较好）
// 另外，thread函数需要给定其可调用函数对象的所有形参的实参值，不能依靠可调用对象的默认参数值，因此这里函数声明时就无需对initial_succ_prev赋予默认值了（否则浪费。但是如果此函数不需要在子线程中执行，则设置默认参数还是有用的）
void Estimator::parallel_pose_objs_est(int prev_td, int cur_td_old, Matrix3d RCam_1, Vector3d PCam_1, Matrix3d RCam_2, Vector3d PCam_2, int num_inliers_PnP, bool initial_succ_prev)
{
    cout << "Start estimate motion of objs and process the info of features!" << endl;
    
    TicToc t_obj_motion_esti;

    // 注意，对于参与运动估计的物体关联而言，由于可能关联点数不足，因此其中会加入像素点！在后处理时要注意区分这些像素点！
    //const FeaObjFrame &FinalTrackObjFea = featureTracker.FinalTrackObjFea;
    std::vector<pair<int,std::vector<int>>> &FinalTrackObj = featureTracker.FinalTrackObj;
    // 将set转化为vector
    const vector<int> cur_dyn_objs(featureTracker.cur_dyn_objs.begin(),featureTracker.cur_dyn_objs.end());
    // 要注意，对于map，由于[]运算符会在索引key不存在的情况下自动创建该项，这会改变map本身。因此，对于const的map不能使用[]运算符！！
    const FeaObjFrame &TotalLostObjPrevBg = featureTracker.TotalLostObjPrevBg;
    map<int,int> &FinalTrackCurObj = featureTracker.FinalTrackCurObj;
    // int num_final_valid_obj = FinalTrackCurObj.size();
    map<int,uchar> FinalTrackCurObjCls;

    const FeaObjFrame &TrackObjFeaFrame = featureTracker.TrackObjFeaFrame;
    int &pt_id = featureTracker.n_id;
    int &n_obj_id = featureTracker.n_obj_id;
    map<int,int> &gl_id_index_map = featureTracker.gl_id_index_map;
    vector<uchar> &status_FAST = featureTracker.statusLeftRIght;
    vector<uchar> &status_sift = featureTracker.status_sift;
    vector<int> ids_FAST = featureTracker.ids_FAST;
    vector<int> ids_sift = featureTracker.ids_sift;
    vector<pair<uchar, int>> &obj_cls_id_FAST = featureTracker.obj_cls_id_FAST;
    vector<pair<uchar, int>> &obj_cls_id_sift = featureTracker.obj_cls_id_sift;
    vector<int> &track_cnt_FAST = featureTracker.track_cnt_FAST;
    vector<int> &track_cnt_sift = featureTracker.track_cnt_sift;
    vector<int> &prev_sift_global_obj_id = featureTracker.prev_sift_global_obj_id;
    vector<int> &prev_FAST_global_obj_id = featureTracker.prev_FAST_global_obj_id;
    vector<int> &detect_lost_objs_cur = featureTracker.detect_lost_objs_cur;

    vector<float> &prev_sift_dep = featureTracker.prev_sift_dep;
    vector<float> &prev_FAST_dep = featureTracker.prev_FAST_dep;
    vector<float> &cur_sift_dep = featureTracker.cur_sift_dep;
    vector<float> &cur_FAST_dep = featureTracker.cur_FAST_dep;
    map<int, cv::Vec4f> &prev_un_Fea_map = featureTracker.prev_un_Fea_map;
    map<int, cv::Point2f> &prevRightFeaMap = featureTracker.prevRightFeaMap;

    const set<int> &sta_obj_fea_in_map = featureTracker.sta_obj_fea_in_map;
    
    vector<float*> sampled_pixel_lost_obj(detect_lost_objs_cur.size(), nullptr);

    // 将vector转化为set，以便后续查询其中的元素
    // 当前帧中没有立体匹配的物体点
    set<int> id_sift_no_depth(featureTracker.id_sift_no_depth.begin(),featureTracker.id_sift_no_depth.end());
    set<int> id_FAST_no_depth(featureTracker.id_FAST_no_depth.begin(),featureTracker.id_FAST_no_depth.end());
    // 当前帧中没有立体匹配的背景点
    // set<int> sift_no_stereo_bg(featureTracker.sift_no_stereo_bg.begin(),featureTracker.sift_no_stereo_bg.end());
    // set<int> FAST_no_stereo_bg(featureTracker.FAST_no_stereo_bg.begin(),featureTracker.FAST_no_stereo_bg.end());

    // 尽量不要使用这样的迭代器变量，因为如果在此函数结束之前，id_sift_no_depth的内存位置发生了变化，则此处的迭代器变量会在函数结束时造成内存错误！！
    // set<int>::iterator sift_iter_end = id_sift_no_depth.end(); 

    // copy这两个上一帧物体的全局信息，原本的信息后续需要查询
    vector<int> glob_obj_id_prev = featureTracker.glob_obj_id_prev;
    vector<uchar> obj_cls_prev = featureTracker.obj_cls_prev;
    featureTracker.glob_obj_id_prev.clear();
    featureTracker.obj_cls_prev.clear();

    // 如果有某个物体没法完成运动估计，则把这个物体的所有跟踪的当前sift点或FAST点（仅限于物体）都作为新的特征点
    // 第一个int是修改前的旧id，用于从跟踪数据中索引；第二个int为修改后的新点的id，用于创建新的点
    vector<pair<int,int>> new_bg_sift, new_tracked_stat_fea, new_stat_obj_sift;
    vector<vector<pair<int,int>>> vec_new_tracked_stat_fea, vec_new_stat_obj_sift;
    vector<pair<int,int>> vec_cur_prev_obj_id;
    vector<uchar> vec_g_cls;

    bool marg_old = false;
    if (marginalization_flag == MARGIN_OLD) marg_old = true;

    bool add_new_sift_in_next_frame = featureTracker.add_new_sift_in_next_frame;

    map<int,uchar> status_objs_prev;
    // map<int,Vector3f> ave_3d_pts_objs;
    // 记录上一帧的物体状态信息，然后清空该变量
    if(featureTracker.status_objs_prev.size() != 0) 
    {
        // map之间的赋值是拷贝（不会是深拷贝，即如果值为指针，则只拷贝指针，不会拷贝指针所指的内容）
        status_objs_prev = featureTracker.status_objs_prev;
        featureTracker.status_objs_prev.clear();
    }

    // 极端情况下在此处调用Rs和Ps之前，LBA中就已经在优化相关变量了，因此这里使用提前获取的值
    // from w_T_imu_ to w_T_cam
    // Matrix3d RCam_1 = Rs[frame_count-1] * ric[0];
    // Vector3d PCam_1 = Rs[frame_count-1] * tic[0] + Ps[frame_count-1];
    // Matrix3d RCam_2 = Rs[frame_count] * ric[0];
    // Vector3d PCam_2 = Rs[frame_count] * tic[0] + Ps[frame_count];

    int num_lost = 0;
    Matrix3d Cam_R_Trans;
    Vector3d Cam_P_Trans;
    int _valid_objs = featureTracker.valid_detect_obj.size();
    int num_dyn = 0;
    if(frame_count > 0)
    {
        int num_static_obj = 0;
        map<int, pair<Matrix3d,Vector3d>> &init_delta_RP_objs = featureTracker.RP_objs_pred;

        int num_prev_lost_obj = TotalLostObjPrevBg.size();

        // cout << "num_prev_lost_obj: " << num_prev_lost_obj << endl;

        int num_gl_dyn_objs = cur_dyn_objs.size();

        // cout << "num_gl_dyn_objs: " << num_gl_dyn_objs << endl;

        int num_dyn_objs = num_gl_dyn_objs + num_prev_lost_obj;

        Cam_R_Trans = RCam_2.transpose() * RCam_1;
        Cam_P_Trans = RCam_2.transpose() * PCam_1 - RCam_2.transpose() * PCam_2;

        // cout << "Num of dynamic object: " << num_dyn_objs << endl;
        // 防止当前帧检测物体比上一帧的全局物体数量少时，当前帧漏检物体加入时会覆盖掉上一帧的一部分物体的采样点（下面还需要用到）
        int valid_objs = std::max(_valid_objs, (int)featureTracker.pixel_objs_prev.size());

        if(num_dyn_objs > 0)
        {
            vector<Vector3d> &delta_Ps = delta_P_objs;
            vector<Matrix3d> &delta_Rs = delta_R_objs;
            delta_Ps.resize(num_dyn_objs);
            delta_Rs.resize(num_dyn_objs);
            
            // initial guess of ck_G_k_k-1 = T_k' * w_H_k_k-1 * T_k-1, where w_H_k_k-1 = w_L_k * (w_L_k-1)' (pose transform of object expressed in world inference)
            // ck_G_k_k-1 can transfer point i of a dynamic object point from coord ck-1_m_i in cam ck-1 to coord ck_m_i in cam ck, this motion transform expressed in coord system ck
            for (int k = 0; k < num_dyn_objs; ++k)
            {
                if (k < num_gl_dyn_objs)
                {
                    int id_obj = FinalTrackObj[cur_dyn_objs[k]].first;
                    if (init_delta_RP_objs.find(id_obj) != init_delta_RP_objs.end())
                    {
                        // cout << "Get object motion guess!" << endl;
                        // 要注意，当前帧中featureTracker.RP_objs_pred的量已经在之前更新为 物体在相机坐标系下的两帧间的运动变换的预测，正是这里所要精确估计的量！
                        // delta_Rs[k] = RCam_2.transpose() * init_delta_RP_objs[id_obj].first * RCam_1;
                        // delta_Ps[k] = RCam_2.transpose() * (init_delta_RP_objs[id_obj].first*PCam_1 + init_delta_RP_objs[id_obj].second - PCam_2);
                        delta_Rs[k] = init_delta_RP_objs[id_obj].first;
                        delta_Ps[k] = init_delta_RP_objs[id_obj].second;
                    }
                    // 如果是上一帧中的新物体
                    else
                    {
                        // 没有运动先验的物体，则用相机的局部运动作为初始值
                        // delta_Rs[k] = RCam_2.transpose() * Matrix3d::Identity() * RCam_1;
                        // delta_Ps[k] = RCam_2.transpose() * (Matrix3d::Identity()*PCam_1 - PCam_2);
                        delta_Rs[k] = Cam_R_Trans;
                        delta_Ps[k] = Cam_P_Trans;
                    }
                    // 是否在当前帧完全漏检的全局物体
                    if(std::find(detect_lost_objs_cur.begin(),detect_lost_objs_cur.end(),id_obj) != detect_lost_objs_cur.end())
                        ++num_lost;
                }
                // 上一帧中完全漏检的物体，则也是属于没有运动先验的物体
                else
                {
                    // delta_Rs[k] = RCam_2.transpose() * Matrix3d::Identity() * RCam_1;
                    // delta_Ps[k] = RCam_2.transpose() * (Matrix3d::Identity()*PCam_1 - PCam_2);
                    delta_Rs[k] = Cam_R_Trans;
                    delta_Ps[k] = Cam_P_Trans;
                }
            }

            // 在C++中，vector<bool>是个特殊的类型，它不是STL容器！也就不具备STL容器的一些基本特性，如迭代器等，甚至无法定义指向其元素的指针！！
            // 究其原因就是从C++98开始就对vector<bool>进行特殊定义以节省内存空间，也其中每个元素实际使用1bit而不是1Byte的空间，而C++中不允许定义指向单个bit的指针！
            // 对vector<bool>变量用[]取值时返回的其实是个右值（具体地说是”std::vector< bool>:reference”类型的对象），因此无法定义指向它的元素的指针，也无法将其赋给某个bool类型的引用。
            // 用[]将元素取出并赋值给别的bool变量，会有隐式变换过程！赋值给auto类型的变量则相当于传递了元素的引用，可以通过它对元素进行修改！
            // 总之，如果不涉及[]取值后的赋值或者取地址等操作，使用vector<bool>一般不会有问题！但是建议使用deque<bool>或者bitset等来代替！
            // https://blog.csdn.net/DoronLee/article/details/78462208
            vector<bool> PnPSucc(num_dyn_objs,false);
            vector<bool> is_static(num_dyn_objs,false);
            vector<int> id_lost;
            vector<Matrix3d> esti_motion_R(num_dyn_objs);
            vector<Vector3d> esti_motion_P(num_dyn_objs);
            
            for (auto &iter: TotalLostObjPrevBg)
            {
                // 记录在上一帧中完全漏检的当前帧临时物体
                id_lost.push_back(iter.first);
            }
            
            int num_thread = num_dyn_objs <= 4? num_dyn_objs : 4;
            omp_set_num_threads(num_thread);

            // get_objs_initial_pose(delta_Ps, delta_Rs);

        #pragma omp parallel default(shared)
            {
        #pragma omp for
                for (int i = 0; i < num_dyn_objs; ++i)
                {
                    // Matrix3d R_init = delta_Rs[i];
                    // Vector3d P_init = delta_Ps[i];
                    bool succ = false;
                    // 保存运动估计成功的物体的3D特征点的平均坐标，用于与真值物体bbox进行匹配
                    Vector3f ave_3D_pts_obj(0,0,0);
                    // 给定位姿估计的初始值
                    if (i < num_gl_dyn_objs)
                    {
                        int prev_id = FinalTrackObj[cur_dyn_objs[i]].first;
                        // 还需要给定物体点的status，用于将外点给标记为待删除点！
                        succ = SolveObjPoseTransByPnP(featureTracker.FinalTrackObjFea[prev_id], delta_Ps[i], delta_Rs[i], sampled_pixel_lost_obj, valid_objs, ave_3D_pts_obj, prev_id);
                    }
                    else
                    {
                        int cur_id = id_lost[i-num_gl_dyn_objs];
                        if(featureTracker.TotalLostObjPrevBg.find(cur_id) == featureTracker.TotalLostObjPrevBg.end())
                        {
                            // cout << "cur_id: " << cur_id << endl;
                            cout << "Weired!" << endl;
                            abort();
                        }
                        
                        succ = SolveObjPoseTransByPnP(featureTracker.TotalLostObjPrevBg[cur_id], delta_Ps[i], delta_Rs[i], sampled_pixel_lost_obj, valid_objs, ave_3D_pts_obj);
                    }

                    // 如果PnP估计失败，则后续要怎么处理？直接认为该物体跟踪丢失，把当前帧该物体作为新物体.
                    PnPSucc[i] = succ;
                    
                    if (succ)
                    {
                        // 如果该物体的运动与相机的帧间运动非常相近，则认为该物体在当前两帧之间是静态的！
                        Matrix3d ReproEr_R = Cam_R_Trans.transpose() * delta_Rs[i];
                        Vector3d ReproEr_P = Cam_R_Trans.transpose() * delta_Ps[i] - Cam_R_Trans.transpose() * Cam_P_Trans;

                        double t_rpe = ReproEr_P.norm();
                        double trace_rpe = 0;
                        for (int k = 0; k < 3; ++k)
                        {
                            // 为什么？为了避免那些很接近1.0但是比1.0略大的计算结果，这是计算误差
                            if (ReproEr_R(k,k)>1.0)
                                trace_rpe = trace_rpe + 1.0-(ReproEr_R(k,k)-1.0);
                            else
                                trace_rpe = trace_rpe + ReproEr_R(k,k);
                        }
                        float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.1415926;
                        // cout << "the pose change error between camera and object, " << "t: " << t_rpe <<  " R: " << r_rpe << endl;
                        // TODO: 这个相对运动的误差的阈值要仔细地设置，12厘米和1.2度的差别是合适的吗？
                        if (t_rpe <= 0.15 && r_rpe <= 1)
                        {
                            is_static[i] = true;
                            cout << "Got a static object!" << endl;
                        }
                        // 动态物体
                        else
                        {
                            // featureTracker.RP_objs_pred中最终要存储的是 动态物体在表达在后一帧相机坐标系下的两帧间物体的运动变换。
                            // esti_motion_R[i] = RCam_2 * delta_Rs[i] * RCam_1.transpose();
                            // esti_motion_P[i] = RCam_2 * delta_Ps[i] + PCam_2 - esti_motion_R[i] * PCam_1;
                            esti_motion_R[i] = delta_Rs[i];
                            esti_motion_P[i] = delta_Ps[i];
                        }

                        // 保存运动估计成功的物体的3D特征点的平均值，后续覆盖到GPU采样像素点数组中。当前帧完全漏检的物体，如果运动估计成功，则其在当前帧中的3D特征点的平均值已经写入相关采样像素数组中了。这里还是保存了
                        // if(i < num_gl_dyn_objs)
                        // {
                        //     int prev_id = FinalTrackObj[cur_dyn_objs[i]].first;
                        //     ave_3d_pts_objs[prev_id] = ave_3D_pts_obj;
                        // }
                        // else
                        // {
                        //     int cur_id = id_lost[i-num_gl_dyn_objs];
                        //     // 上一帧的完全漏检物体，则暂时使用当前帧的局部物体id的负值作为key
                        //     ave_3d_pts_objs[(-1*cur_id)] = ave_3D_pts_obj;
                        // }
                    }
                }
            }
            
            if(!delta_Rs.empty()) delta_Rs.clear();
            if(!delta_Ps.empty()) delta_Ps.clear();

            // 结束多线程。这里是会隐式地堵塞，直到所有线程完成

            // 下面开始各个物体的特征点信息修改

            featureTracker.RP_objs_pred.clear();
            
            for(int i = 0; i< num_dyn_objs; ++i)
            {
                // 如果是动态物体
                if (PnPSucc[i] && !is_static[i]) 
                {
                    ++num_dyn;
                    if (i < num_gl_dyn_objs)
                    {
                        // cout << "Get a dynamic object in No." << frame_count << " frame!" << endl;
                        int id_tracked_dyn_objs = cur_dyn_objs[i];
                        int prev_id = FinalTrackObj[id_tracked_dyn_objs].first;
                        // vector<pair<int,std::vector<int>>>::iterator pair_iter;
                        // // 函数式编程
                        // pair_iter = std::find_if(FinalTrackObj.begin(),FinalTrackObj.end(),[&prev_id](const pair<int,std::vector<int>> &elem){
                        //     return elem.first == prev_id;}
                        // );

                        // 该物体是否在上一帧有部分漏检
                        bool par_lost_prev = (featureTracker.ParLostObjPrevBg.find(prev_id) != featureTracker.ParLostObjPrevBg.end());

                        auto id_iter = std::find(glob_obj_id_prev.begin(),glob_obj_id_prev.end(),prev_id);
                        int cls_index = std::distance(glob_obj_id_prev.begin(),id_iter);
                        int glo_obj_id = glob_obj_id_prev[cls_index];

                        if(glo_obj_id != prev_id) assert("Something wrong with FinalTrackObj and glob_obj_id_prev!");
                        
                        uchar cls_prev = obj_cls_prev[cls_index];

                        uchar state_prev_obj = status_objs_prev[prev_id];

                        for (auto &cur_id: FinalTrackObj[id_tracked_dyn_objs].second)
                        {
                            // if (!new_tracked_stat_fea.empty()) new_tracked_stat_fea.clear();
                            // if (!new_stat_obj_sift.empty()) new_stat_obj_sift.clear();

                            // 如果上一帧物体在当前帧中有漏检，则修改这些漏检点的信息.
                            // 另外该物体在当前帧还有可能完全漏检
                            if(cur_id == 0)
                            {
                                auto iter_lose = std::find(featureTracker.lose_objs_cur_bg.begin(),featureTracker.lose_objs_cur_bg.end(),prev_id);
                                int id_in_lost = std::distance(featureTracker.lose_objs_cur_bg.begin(), iter_lose);
                                // 注意，关联异常点不会加入fea_cur_lose_objs中，而是提前被排除了或者修改为背景上的新sift点
                                for(const auto &lost_pt_in_bg: featureTracker.fea_cur_lose_objs[id_in_lost])
                                {
                                    if(gl_id_index_map.find(lost_pt_in_bg) == gl_id_index_map.end())
                                    {
                                        assert(false);
                                    }
                                    // 对于当前帧背景中的漏检部分跟踪点而言，其与上一帧匹配点的cls是对齐的，因此不需要修改cls
                                    int index = gl_id_index_map[lost_pt_in_bg];

                                    if (index > 0)
                                    {
                                        index = index - 1;
                                        // 如果是背景区域的运动估计的外点，则跳过。
                                        if (status_FAST[index] == 0) continue;

                                        obj_cls_id_FAST[index].second = glo_obj_id;
                                        // 如果该物体上一帧为静态，则将上一帧的点作为该动态物体的初始观测，修改点的id为新点，修改其跟踪数为2。
                                        // 但是实际上对于该点我们只能正确保存1帧的点信息，因为这里只能修改当前帧该点的id，上一帧的该点就不去修改了，反正之后也不会保存
                                        // 之所以修改，是因为如果该物体之前一直是静态（所有的静态点记录可能已经都在地图，且当前帧暂时不将它们删除，因为它们有可能大于3帧，可用于LBA），仅在上一帧和当前帧之间为动态，而下一帧又恢复为静态(这样的概率基本为0，但可能运动估计失误)，
                                        // 为了避免此种极端情况下 下一帧又将当前帧的观测（当前帧应该还属于动态点）加入地图中该id点的记录（下一帧不是marg次新帧，则会两帧观测一起加入地图），这里干脆直接将点的id变为新的，则该新点一定不会在地图中！
                                        if (state_prev_obj == 1) 
                                        {
                                            track_cnt_FAST[index] = 2;
                                            // 这里修改该点的全局id，其实会造成其与之前的跟踪点的全局id不一致，但是由于其是动态物体了，不会被放进地图，因此不会再用到其之前帧的跟踪
                                            ids_FAST[index] = pt_id++;
                                        }
                                    }
                                    else
                                    {
                                        index = -1*index;
                                        // 运动估计外点sift，如果其深度估计是比较可靠的，且每一帧需要添加新点，则将其作为背景的新点
                                        if(status_sift[index] == 0)
                                        {
                                            if(!add_new_sift_in_next_frame)
                                            {
                                                // 如果该跟踪点的深度是使用depth_map来获得，则认为深度估计不是很可靠，则放弃该点作为新点
                                                // 注意，虽然当前帧该点在背景区域，但是其跟踪的是上一帧的物体点，所以其在当前帧是否有立体匹配是记录在id_sift_no_depth中的，即物体点的情况
                                                // if (id_sift_no_depth.find(index) != id_sift_no_depth.end() || featureTracker.cur_sift_dep[index] <= 0) 
                                                if (id_sift_no_depth.find(index) != id_sift_no_depth.end())
                                                {
                                                    // status_sift[index] = 0;
                                                    continue;
                                                }
                                                // obj_cls_id_sift[index].second = 0;
                                                obj_cls_id_sift[index].first = 0;
                                                status_sift[index] = 1;
                                                track_cnt_sift[index] = 1;
                                                ids_sift[index] = pt_id++;
                                                prev_sift_global_obj_id[index] = 0;
                                                // 新的背景点暂时不加入静态地图
                                                //new_bg_sift.push_back(std::pair<int,int>(old_pt_id,pt_id-1));
                                                // new_bg_sift.emplace_back(old_pt_id,pt_id-1);
                                            }
                                            continue;
                                        }

                                        obj_cls_id_sift[index].second = glo_obj_id;
                                        
                                        if (state_prev_obj == 1)
                                        {
                                            track_cnt_sift[index] = 2;
                                            ids_sift[index] = pt_id++;
                                        }
                                    }
                                }
                                // 该物体在当前帧中是否完全漏检
                                // auto iter_lost = std::find(detect_lost_objs_cur.begin(),detect_lost_objs_cur.end(),prev_id);
                                // if (iter_lost != detect_lost_objs_cur.end())
                                // {
                                //     featureTracker.pixel_objs_prev[prev_id] = ?
                                // }
                            }
                            else
                            {   
                                FeaFrame::iterator end = featureTracker.FinalTrackObjFea[prev_id].end();
                                for(const auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                                {
                                    // 临时的像素点匹配跳过。两个物体之间不一定有特征点的跟踪，它们之间的关联以及运动估计可以是完全通过像素点来关联的！
                                    if(pt.first < 0) continue;
                                    // 注意，这里改变某些特征点的全局id之后，这里的gl_id_index_map的该点的key理论上也要改变，但是由于改后的点的全局id在当前帧不需要用到（而且其他关于特征点信息的变量也是基于旧id，因此这里不修改）！
                                    int index = gl_id_index_map[pt.first];
                                    auto iter = featureTracker.FinalTrackObjFea[prev_id].find(pt.first);
                                    
                                    // 当前帧检测物体上的非关联点 或者 匹配异常点 如果没有立体匹配 则直接去除
                                    if (iter == end)
                                    {
                                        if (index > 0)
                                        {
                                            index -= 1;
                                            // 如果该点没有立体匹配
                                            if(status_FAST[index] != 1)
                                                status_FAST[index] = 0;
                                            else
                                            {
                                                track_cnt_FAST[index] = 1;
                                                // 对于上一帧完全漏检的物体，则使用当前帧的检测类别来作为全局类别。
                                                // 注意，该物体的这些点由于不是和上一帧的背景点关联，而是和别的物体相关联，因此其obj_cls_id_sift的first是与其关联物体的全局cls相关联了。这里改为其当前帧检测类别
                                                obj_cls_id_FAST[index].first = cls_prev;
                                                // 当前物体为新物体
                                                obj_cls_id_FAST[index].second = glo_obj_id;
                                                ids_FAST[index] = pt_id++;
                                                prev_FAST_global_obj_id[index] = glo_obj_id;
                                            }
                                        }
                                        // 该物体上的某些sift点虽然错误地关联到了上一帧别的物体上 或者是 异常匹配，但是这些sift点可以保留为新的特征点？取决于其深度估计是否可靠
                                        else
                                        {
                                            index = -1 * index;
                                            // 如果是错误匹配点，则看深度估计是否可靠。对于背景点，暂时不保留新点
                                            if(status_sift[index] != 1) 
                                            {
                                                status_sift[index] = 0;
                                                continue;
                                            }
                                            
                                            // 下面的则是当前物体上关联到上一帧错误物体 的 sift点，保留为新点
                                            // status_sift[index] = 1;
                                            // 该物体的新特征点
                                            track_cnt_sift[index] = 1;
                                            // 错误匹配的点的cls label需要修改
                                            obj_cls_id_sift[index].first = cls_prev;
                                            obj_cls_id_sift[index].second = glo_obj_id;
                                            ids_sift[index] = pt_id++;
                                            // 对于跟踪点prev_sift_global_obj_id这个量直接继承自上一帧，此处的点跟踪自上一帧错误的物体，因此这里需要修改！
                                            prev_sift_global_obj_id[index] = glo_obj_id;
                                            
                                            continue;
                                        }
                                    }
                                    else
                                    {
                                        // 完成关联的物体，其在上一帧 和 当前帧 中可能同时出现部分漏检；其中，当前帧漏检的点的cls label已经与上一帧的全局cls label对齐；而上一帧漏检的点在当前帧使用局部的检测类别
                                        // 为了方便，直接对当前帧的所有匹配点的cls label都重新赋值为上一帧该物体的全局类别
                                        if (index > 0)
                                        {
                                            index = index - 1;

                                            // 跟踪点的cls label需要改变
                                            // 静态物体的第二个变量与背景一样，都是0；动态物体则是global obj id
                                            obj_cls_id_FAST[index].second = glo_obj_id;
                                            
                                            // 如果上一帧该物体部分漏检，则当前帧的物关联体点可能会跟踪自上一帧的背景漏检点，则这里统一修改所有匹配点的这个量，不用验证是否跟踪自上一帧背景
                                            // if (par_lost_prev)
                                            {
                                                obj_cls_id_FAST[index].first = cls_prev;
                                                prev_FAST_global_obj_id[index] = glo_obj_id;
                                            }

                                            // 如果该特征点匹配在位姿估计中是外点，则已经被置为待删除。是否要将其修改为新点？
                                            // 如果其有立体匹配，则可以修改为新点
                                            if (status_FAST[index] == 0) 
                                            {
                                                // 需要该外点有立体匹配，才能保留为新点
                                                if (id_FAST_no_depth.find(index) != id_FAST_no_depth.end())
                                                {
                                                    continue;
                                                }

                                                status_FAST[index] = 1;
                                                // 修改为动态物体上的新特征点
                                                track_cnt_FAST[index] = 1;
                                                
                                                ids_FAST[index] = pt_id++;
                                                continue;
                                            }
                                            
                                            if (state_prev_obj == 1) 
                                            {
                                                track_cnt_FAST[index] = 2;
                                                ids_FAST[index] = pt_id++;
                                            }
                                        }
                                        else
                                        {
                                            index = -1 * index;
                                            obj_cls_id_sift[index].second = glo_obj_id;
                                            
                                            // if (par_lost_prev)
                                            {
                                                obj_cls_id_sift[index].first = cls_prev;
                                                prev_sift_global_obj_id[index] = glo_obj_id;
                                            }

                                            // 物体上(非漏检部分）的位姿估计sift外点可以保留为新点，前提是深度估计来自于立体匹配
                                            if (status_sift[index] == 0) 
                                            {
                                                // 需要该外点有立体匹配，才能保留为新点
                                                // if (id_sift_no_depth.find(index) != id_sift_no_depth.end() || featureTracker.cur_sift_dep[index] <= 0) 
                                                if (id_sift_no_depth.find(index)!= id_sift_no_depth.end())
                                                {
                                                    continue;
                                                }
                                                
                                                status_sift[index] = 1;
                                                // 修改为动态物体上的新特征点
                                                track_cnt_sift[index] = 1;
                                                
                                                ids_sift[index] = pt_id++;
                                                
                                                continue;
                                            }
                                            // 如果是内点，则查看其上一帧是否为静态物体
                                            if (state_prev_obj == 1) 
                                            {
                                                track_cnt_sift[index] = 2;
                                                ids_sift[index] = pt_id++;
                                            }
                                        }
                                    }
                                }
                                // 修改每个物体的跟踪阶段检测到的new fea的信息，这部分可以放到最后，所有当前帧的物体一起进行。注意，当前背景中不会有属于新物体的新的漏检点，则漏检点只可能是从已有物体跟踪来的。
                                FinalTrackCurObj[cur_id] = glo_obj_id;
                                FinalTrackCurObjCls[cur_id] = cls_prev;
                            }
                        }
                        // 记录当前帧物体的全局id和全局cls label
                        featureTracker.glob_obj_id_prev.push_back(glo_obj_id);
                        featureTracker.obj_cls_prev.push_back(cls_prev);
                        // 动态物体
                        featureTracker.status_objs_prev[glo_obj_id] = 0;  

                        // featureTracker.RP_objs_pred[glo_obj_id].first = esti_motion_R[i];
                        // featureTracker.RP_objs_pred[glo_obj_id].second = esti_motion_P[i];
                        pair<Matrix3d,Vector3d> temp_pair;
                        temp_pair.first = esti_motion_R[i];
                        temp_pair.second = esti_motion_P[i];
                        featureTracker.RP_objs_pred.insert(make_pair(glo_obj_id,temp_pair));
                    }
                    // 上一帧完全漏检的物体。这种情况下无法使用像素点匹配
                    // TODO：（其实如果有逆向光流估计的话，则可以根据当前帧的物体寻找上一帧的匹配背景点）
                    else
                    {
                        int cur_id = id_lost[i-num_gl_dyn_objs];
                        if(cur_id == 0)
                            assert(false && "current bg point shouldn't be in TotalLostObjPrevBg!");
                        
                        // 这么取STL的迭代器作为局部变量其实危险性很大，因为如果在此迭代器的生存期内，容器由于插入元素而重新分配内存，那么原本的这个迭代器便失效了！
                        // 而容器在重新分配内存时会自动将旧的内存回收，但是旧的迭代器在函数退出时要被销毁，而销毁迭代器同时默认销毁它所指向的内存的内容，但是该内存早已被demalloc！
                        // FeaFrame::iterator end = featureTracker.TotalLostObjPrevBg[cur_id].end();
                        // 随机取一个匹配点的像素坐标
                        float x = featureTracker.TotalLostObjPrevBg[cur_id].begin()->second[0].second(3);
                        float y = featureTracker.TotalLostObjPrevBg[cur_id].begin()->second[0].second(4);
                        uchar cls_cur = cls_map.at<uchar>(y,x);

                        for(const auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                        {
                            int index = gl_id_index_map[pt.first];
                            // 如果不是匹配物体之间的关联点，或者是 异常关联点，则去除FAST点，保留sift点作为新特征点
                            if (featureTracker.TotalLostObjPrevBg[cur_id].find(pt.first) == featureTracker.TotalLostObjPrevBg[cur_id].end())
                            {
                                if(index > 0)
                                {
                                    index -= 1;
                                    // 如果该点没有立体匹配
                                    if(status_FAST[index] != 1)
                                        status_FAST[index] = 0;
                                    else
                                    {
                                        track_cnt_FAST[index] = 1;
                                        // 对于上一帧完全漏检的物体，则使用当前帧的检测类别来作为全局类别。
                                        // 注意，该物体的这些点由于不是和上一帧的背景点关联，而是和别的物体相关联，因此其obj_cls_id_sift的first是与其关联物体的全局cls相关联了。这里改为其当前帧检测类别
                                        obj_cls_id_FAST[index].first = cls_cur;
                                        // 当前物体为新物体
                                        obj_cls_id_FAST[index].second = n_obj_id;
                                        ids_FAST[index] = pt_id++;
                                        prev_FAST_global_obj_id[index] = n_obj_id;
                                    }
                                }
                                else
                                {
                                    index = -1 * index;
                                    // 异常匹配 或 非最终关联物体的匹配 的sift点在物体匹配阶段已经处理（删除）
                                    // 需要有立体匹配
                                    if(status_sift[index] != 1) 
                                    {
                                        status_sift[index] = 0;
                                        continue;
                                    }
                                    
                                    // status_sift[index] = 1;
                                    // 新特征点
                                    track_cnt_sift[index] = 1;
                                    // 对于上一帧完全漏检的物体，则使用当前帧的检测类别来作为全局类别。
                                    // 注意，该物体的这些点由于不是和上一帧的背景点关联，而是和别的物体相关联，因此其obj_cls_id_sift的first是与其关联物体的全局cls相关联了。这里改为其当前帧检测类别
                                    obj_cls_id_sift[index].first = cls_cur;
                                    // 当前物体为新物体
                                    obj_cls_id_sift[index].second = n_obj_id;
                                    ids_sift[index] = pt_id++;
                                    prev_sift_global_obj_id[index] = n_obj_id;
                                    continue;
                                }
                            }
                            else
                            {
                                if (index > 0)
                                {
                                    index = index - 1;

                                    // 对于跟踪点，不改变其全局id
                                    // 点的cls label不需要改变，因为上一帧为完全漏检，则以当前帧的检测类别为准
                                    obj_cls_id_FAST[index].second = n_obj_id;
                                    //obj_cls_id_FAST[index].first = cls_cur;
                                    prev_FAST_global_obj_id[index] = n_obj_id;

                                    // 运动估计外点
                                    if (status_FAST[index] == 0) 
                                    {
                                        // 需要有立体匹配
                                        if (id_FAST_no_depth.find(index) != id_FAST_no_depth.end()) 
                                            continue;
                                        
                                        status_FAST[index] = 1;
                                        // 新特征点
                                        track_cnt_FAST[index] = 1;
                                        ids_FAST[index] = pt_id++;
                                        continue;
                                    }

                                    // 运动估计内点
                                    // 由于上一帧为背景点，可能该点在之前多帧都作为背景点被跟踪着（且已经加入地图），为了防止过两帧该物体变为静态后，此点由于id未变而加入地图后会造成该点的观测帧未连续的情况出现
                                    track_cnt_FAST[index] = 2;
                                    ids_FAST[index] = pt_id++;
                                }
                                else
                                {
                                    index = -1 * index;
                                    obj_cls_id_sift[index].second = n_obj_id;
                                    prev_sift_global_obj_id[index] = n_obj_id;
                                    //obj_cls_id_sift[index].first = cls_cur;
                                    // 把这个物体的位姿估计中的sift外点作为新的特征点
                                    if (status_sift[index] == 0) 
                                    {
                                        // 需要有立体匹配
                                        // if (id_sift_no_depth.find(index) != id_sift_no_depth.end() || featureTracker.cur_sift_dep[index] <= 0) 
                                        if (id_sift_no_depth.find(index) != id_sift_no_depth.end()) 
                                            continue;
                                        
                                        status_sift[index] = 1;
                                        // 新特征点
                                        track_cnt_sift[index] = 1;
                                        ids_sift[index] = pt_id++;
                                        continue;
                                    }

                                    // 如果是运动估计内点，则改变该点的id和跟踪次数
                                    track_cnt_sift[index] = 2;
                                    ids_sift[index] = pt_id++;
                                }
                            }
                        }

                        // 记录当前帧物体的最终匹配，已便后续统一修改各个临时物体上的新检测点
                        FinalTrackCurObj[cur_id] = n_obj_id;
                        FinalTrackCurObjCls[cur_id] = cls_cur;
                        // 记录当前帧新物体的全局id和全局cls label
                        featureTracker.glob_obj_id_prev.push_back(n_obj_id);
                        featureTracker.obj_cls_prev.push_back(cls_cur);
                        featureTracker.status_objs_prev[n_obj_id] = 0;

                        // featureTracker.RP_objs_pred[n_obj_id].first = esti_motion_R[i];
                        // featureTracker.RP_objs_pred[n_obj_id].second = esti_motion_P[i];
                        pair<Matrix3d,Vector3d> temp_pair;
                        temp_pair.first = esti_motion_R[i];
                        temp_pair.second = esti_motion_P[i];
                        featureTracker.RP_objs_pred.insert(make_pair(n_obj_id,temp_pair));

                        // 当前的物体为新的动态物体
                        ++n_obj_id;
                    }
                }
                // 如果是运动估计失败，则把当前帧的物体作为新物体，新物体的所有特征点（包括新的背景点）不会加入静态地图。
                else if(!PnPSucc[i])
                {
                    // 将上一帧的该物体置为跟踪结束，当前帧的物体置为新物体
                    if (i < num_gl_dyn_objs)
                    {
                        int id_tracked_dyn_objs = cur_dyn_objs[i];
                        int prev_id = FinalTrackObj[id_tracked_dyn_objs].first;

                        for (auto &cur_id: FinalTrackObj[id_tracked_dyn_objs].second)
                        {
                            // 上一帧的物体在当前帧中部分或全部漏检，且该漏检部分有特征点与上一帧完成了匹配
                            if (cur_id == 0) 
                            {
                                auto iter_lose = std::find(featureTracker.lose_objs_cur_bg.begin(),featureTracker.lose_objs_cur_bg.end(),prev_id);
                                int id_in_lost = std::distance(featureTracker.lose_objs_cur_bg.begin(), iter_lose);
                                // 只保留sift点，FAST点就不要了（本身这些点也是通过光流来获取的，另外这部分点应该是很少的）
                                for(const auto lost_pt_in_bg: featureTracker.fea_cur_lose_objs[id_in_lost])
                                {
                                    int index = gl_id_index_map[lost_pt_in_bg];
                                    // 放弃背景中的FAST跟踪点。注意，要用obj_cls_id_FAST的second，即obj_id来分别是否为背景点
                                    if (index > 0) 
                                    {
                                        // 删除该跟踪点
                                        status_FAST[index-1] = 0;
                                        continue;
                                    }
                                    else
                                    {
                                        // 是否将所有跟踪点保留为新特征点
                                        int index = -1 * index;
                                        if(add_new_sift_in_next_frame)
                                        {
                                            status_sift[index] = 0;
                                            continue;
                                        }

                                        // 既然是物体点，那么就一定有有效的深度值。如果其深度不是来自于立体匹配
                                        // if (id_sift_no_depth.find(index) != id_sift_no_depth.end() || featureTracker.cur_sift_dep[index] <= 0) 
                                        if (id_sift_no_depth.find(index) != id_sift_no_depth.end()) 
                                        {
                                            status_sift[index] = 0;
                                            continue;
                                        }
                                        
                                        // 有立体匹配
                                        status_sift[index] = 1;
                                        // 新特征点
                                        track_cnt_sift[index] = 1;
                                        
                                        // old_pt_id = ids_sift[index];
                                        ids_sift[index] = pt_id++;
                                        // 修改该点的cls为背景
                                        obj_cls_id_sift[index].first = 0;
                                        // 当前帧的背景点.其临时obj id就是0
                                        // obj_cls_id_sift[index].second = 0;

                                        prev_sift_global_obj_id[index] = 0;
                                        // new_bg_sift.emplace_back(old_pt_id,pt_id-1);
                                    }
                                }
                                // 当前帧部分漏检的物体要从这里清除吗
                                //if (FinalTrackObj[id_tracked_dyn_objs].second.size() == 1) 
                                    //detect_lost_objs_pre.erase(?);
                                continue;
                            }
                            // 把当前帧的该局部物体归为新物体，后续会对所有的该物体的跟踪特征点进行信息修改（把其上的跟踪点都改为新点）
                            // 该局部物体的所有新检测点也要修改。
                            featureTracker.new_objs_cur.push_back(cur_id);
                            // 当前帧局部物体改为没有匹配
                            //FinalTrackCurObj[cur_id] = -1;
                            // FinalTrackCurObjCls[cur_id-1] = cls_cur;
                        }
                        // erase上一帧该全局物体的匹配信息？不可以！如果这里删除了，则下一个动态物体无法正确地从FinalTrackObj中索引！！FinalTrackObjFea是map结构，倒是可以先删除元素。可以把内部元素clear
                        // FinalTrackObj.erase(FinalTrackObj.begin()+id_tracked_dyn_objs);
                        FinalTrackObj[id_tracked_dyn_objs].second.clear();
                        // FinalTrackObjFea.erase(prev_id);
                        // 将该全局物体加入跟踪丢失行列。对丢失物体不需要使用特别的处理，因为其上的点（指上一帧的某些背景点）暂时还没有添加观测信息
                        featureTracker.FinalLostObjPrev.push_back(prev_id);
                    }
                    // 如果是上一帧完全漏检的物体，则做法相同，把当前帧物体上所有深度可靠的sift跟踪点都作为新点。这种情况下当前帧的关联点中不会有背景点
                    else
                    {
                        int cur_id = id_lost[i-num_gl_dyn_objs];
                        featureTracker.new_objs_cur.push_back(cur_id);
                        // FinalTrackCurObj[cur_id] = -1;
                        // 不再用到了
                        // TotalLostObjPrevBg.erase(cur_id);
                    }
                }
                // 如果当前帧该物体被确定为静态物体，只需要加入静态队列；
                // 另外，静态物体的当前帧的特征点观测是否需要加入静态地图中？需要考虑上一帧的匹配点是否已经在地图中（即该物体在上一帧是否就已经被确认为静态物体）
                else
                {
                    ++num_static_obj;
                    // ++featureTracker.num_sta_objs_found;
                    if (i < num_gl_dyn_objs)
                    {
                        int id_tracked_dyn_objs = cur_dyn_objs[i];
                        int prev_id = FinalTrackObj[id_tracked_dyn_objs].first;
                        
                        // 该物体是否在上一帧有部分漏检
                        bool par_lost_prev = (featureTracker.ParLostObjPrevBg.find(prev_id) != featureTracker.ParLostObjPrevBg.end());

                        // vector<pair<int,std::vector<int>>>::iterator pair_iter;
                        // // 函数式编程
                        // pair_iter = std::find_if(FinalTrackObj.begin(),FinalTrackObj.end(),[&prev_id](const pair<int,std::vector<int>> &elem){
                        //     return elem.first == prev_id;}
                        // );

                        uchar state_prev_obj = status_objs_prev[prev_id];
                        // 如果当前帧显示要marg_old，则说明次新帧与次次新帧之间的视差比较大，容易跟丢，则这里就可能需要把余下的静态跟踪点观测都加入静态地图中，以便之后窗口有足够的点有至少连续4个最新帧的观测（只有这些点会参与LBA）！
                        // 这其中最重要的是旧的静态物体的跟踪，其之前的点观测可能已经加入地图（某个点要么不在地图，如果已经在地图，则至少是2帧观测）

                        auto id_iter = std::find(glob_obj_id_prev.begin(),glob_obj_id_prev.end(),prev_id);
                        int cls_index = std::distance(glob_obj_id_prev.begin(),id_iter);
                        // int glo_obj_id = featureTracker.glob_obj_id_prev[cls_index];
                        // assert(glo_obj_id == prev_id && "Something wrong with glob_obj_id_prev!");
                        uchar cls_prev = obj_cls_prev[cls_index];
                        
                        for(auto &cur_id: FinalTrackObj[id_tracked_dyn_objs].second)
                        {
                            if (!new_tracked_stat_fea.empty()) new_tracked_stat_fea.clear();
                            if (!new_stat_obj_sift.empty()) new_stat_obj_sift.clear();

                            // 如果上一帧物体在当前帧中有漏检。对于上一帧非完全漏检的物体，其在当前帧的漏检点似乎没有需要修改的地方（因为cls label与上一帧对齐，临时obj id又刚好是背景0）。
                            // 把在 物体关联阶段被当作异常点 和 在运动估计时被当作外点的 的背景sift跟踪点 改为当前帧背景中新的sift特征点。
                            if(cur_id == 0)
                            {
                                auto iter_lose = std::find(featureTracker.lose_objs_cur_bg.begin(),featureTracker.lose_objs_cur_bg.end(),prev_id);
                                int id_in_lost;
                                if(iter_lose != featureTracker.lose_objs_cur_bg.end())
                                    id_in_lost = std::distance(featureTracker.lose_objs_cur_bg.begin(), iter_lose);
                                else
                                    assert(false);
                                
                                // 放进fea_cur_lose_objs的点都是在objs-matching中作为有效匹配点的
                                for(const auto lost_pt_in_bg: featureTracker.fea_cur_lose_objs[id_in_lost])
                                {
                                    // 对于当前帧背景中的跟踪点而言，其与上一帧匹配点的cls是对齐的，因此不需要修改cls
                                    int index = gl_id_index_map[lost_pt_in_bg];
                                    if (index <= 0)
                                    {
                                        index = -1 *index;
                                        // 如果是匹配和运动估计内点
                                        if(status_sift[index] != 0)
                                        {
                                            // 只要当前帧不是marg次新帧，则无论上一帧是否已经VI初始化，只要PnP后静态跟踪内点数量不足，则添加当前帧静态跟踪点的观测。
                                            // 要考虑该物体的距离，太远的物体点也不适合加入地图
                                            if (frame_count < WINDOW_SIZE || marg_old)
                                            {
                                                // 当前帧PnP得到的静态跟踪内点是否足够多
                                                if(num_inliers_PnP < Thres_num_track_cur)
                                                {
                                                    float dep = prev_sift_dep[index];
                                                    // 如果marg最老帧，则说明次新帧与次次新帧之间的视差较大（或者跟踪点较少），总之就是之后有跟踪丢失的风险，则考虑把当前帧所有的跟踪点都加入到地图（后续加入多少，根据长距离点和跟踪点的数量来决定）
                                                    if(dep > 0 && dep <= 7) 
                                                    {
                                                        new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                        ++num_inliers_PnP;
                                                    }
                                                }

                                                // 该物体如果在上一帧为动态，则这些跟踪点只承认最新这两帧的观测为静态点的。
                                                // 对于上一帧部分漏检的点，因为它是从属于上一帧的该物体的，因此其运动属性应该也相同，则对这些漏检点的cnt操作与此相同。
                                                if (state_prev_obj == 0) track_cnt_sift[index] = 2;
                                            }
                                            // 滑窗已满且当前帧需要marg次新帧，则看上一帧是否已经完成了VI初始化
                                            else
                                            {
                                                // 如果该点是上一帧的物体新点，当前帧确认为静态点，且上一帧该点有立体深度估计，当前帧没有深度估计，但是又刚好要marg次新帧，则应该及时将上一帧的物体点加入地图！
                                                // 当前帧该点不可能没有深度估计，因为每一帧中没有深度估计的物体点不会被采用！没有立体匹配的背景点也不会被加入物体跟踪变量中！所以这里直接把上一帧该点的观测删除即可！
                                                // if(track_cnt_sift[i] == 2) 
                                                // {
                                                //     new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                // }
                                                
                                                // 如果当前帧marg次新帧，那么需要将该静态点的跟踪次数减1吗？
                                                // 减掉1代表着该点的上一帧观测不会被加入地图及参与当前帧的LBA，这样一来，track_cnt_sift就代表着一个静态点能够在当前地图中保留的观测帧数的最大值。
                                                // 但是对于静态物体点而言，如果其一开始没有被加入地图，而是跟踪几帧后才开始加入，那么这里减1似乎没什么意义？最后决定不减1，减少操作时间
                                                // track_cnt_sift[index] = track_cnt_sift[index] - 1;

                                                // 如果上一帧该物体是动态点，则其有效的静态观测为2帧
                                                if (state_prev_obj == 0) 
                                                {
                                                    // 又因为上一帧要被marg掉，则实际有效观测只剩下1帧。最后决定marg次新帧也不将静态点的cnt_track减1
                                                    // track_cnt_sift[index] = 1;
                                                    track_cnt_sift[index] = 2;
                                                }

                                                if (initial_succ_prev)
                                                {
                                                    // 如果上一帧已VI初始化，则将那些一直以来都是静态点，且已连续被观测到足够多帧的点观测加入地图
                                                    // 将该跟踪点最新帧观察加入地图后，即使当前滑窗后续marg掉次新帧的观测（该点一定参与了LBA和次新帧的marg），该点在下一滑窗中不论下一帧有没有被跟踪到都至少还有足够的帧观测数量（可以参与LBA和marg）
                                                    // 注意，即使该静态点实际被观测帧数是这么多，不意味着这里将其加入地图后 其在地图中的观测帧数会等于这个数（因为物体点不一定从一开始就加入地图）。但是有可能形成足够多帧的地图点，后续如果不足，则删除即可
                                                    if(track_cnt_sift[index] >= (TH_NUM_FRAME_FOR_LBA+1) && state_prev_obj == 1 && prev_sift_dep[index] <= 5)
                                                    {
                                                        new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                        ++num_inliers_PnP;
                                                    }
                                                }
                                            }
                                            continue;
                                        }
                                        // 如果是运动估计外点，则将深度可靠的点改为背景中的新点
                                        // 如果当前帧不需要保留背景中的新点
                                        if(add_new_sift_in_next_frame) continue;

                                        if (id_sift_no_depth.find(index) != id_sift_no_depth.end()) continue;
                                        
                                        if(cur_sift_dep[index] > 0)
                                        {
                                            if(featureTracker.TrackObjFeaFrame[0][lost_pt_in_bg].size() == 2)
                                                status_sift[index] = 1;
                                            else
                                                continue;
                                        }
                                        else
                                        {
                                            // 当前帧没有立体匹配的新点的status为3。是否需要保留这个没有立体匹配的点？感觉不需要，因为它也可能在下一帧不被跟踪
                                            // status_sift[index] = 3;
                                            continue;
                                        }

                                        obj_cls_id_sift[index].first = 0;
                                        ids_sift[index] = pt_id++;
                                        track_cnt_sift[index] = 1;
                                        prev_sift_global_obj_id[index] = 0;
                                    }
                                    else
                                    {
                                        index = index - 1;
                                        // 如果是运动估计内点
                                        if(status_FAST[index] != 0)
                                        {
                                            if ((frame_count < WINDOW_SIZE || marg_old))
                                            {
                                                // 当前帧PnP得到的静态跟踪内点是否足够多
                                                if(num_inliers_PnP < Thres_num_track_cur)
                                                {
                                                    float dep = prev_FAST_dep[index];
                                                    // 如果marg最老帧，则说明次新帧与次次新帧之间的视差较大（或者跟踪点较少），总之就是之后有跟踪丢失的风险，则考虑把当前帧所有的跟踪点都加入到地图（后续加入多少，根据长距离点和跟踪点的数量来决定）
                                                    if(dep > 0 && dep <= 7) 
                                                    {
                                                        new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                        ++num_inliers_PnP;
                                                    }
                                                }
                                                
                                                if (state_prev_obj == 0) track_cnt_FAST[index] = 2;
                                            }
                                            else
                                            {
                                                // 如果该点是上一帧的物体新点，当前帧确认为静态点，但是又刚好要marg次新帧，则应该及时将上一帧的物体点加入地图！
                                                // 不需要，理由同上！
                                                // if(track_cnt_FAST[i] == 2) 
                                                // {
                                                //     new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                // }
                                                
                                                // track_cnt_FAST[i] = track_cnt_FAST[i] - 1;
                                                
                                                if (state_prev_obj == 0) 
                                                {
                                                    // track_cnt_FAST[index] = 1;
                                                    track_cnt_FAST[index] = 2;
                                                }

                                                if (initial_succ_prev)
                                                {
                                                    if(track_cnt_FAST[index] >= (TH_NUM_FRAME_FOR_LBA+1) && state_prev_obj == 1)
                                                    {
                                                        float dep = prev_FAST_dep[index];
                                                        if(dep > 0 && dep <= 5)
                                                        {
                                                            new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                            ++num_inliers_PnP;
                                                        }
                                                    }
                                                }
                                            }
                                            continue;
                                        }
                                        // FAST外点 或 关联异常点 则直接抛弃
                                    }
                                }
                            }
                            // 当前帧匹配到物体
                            else
                            {
                                // FeaFrame::iterator end = featureTracker.FinalTrackObjFea[prev_id].end();

                                for(const auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                                {
                                    int index = gl_id_index_map[pt.first];
                                    
                                    auto iter = featureTracker.FinalTrackObjFea[prev_id].find(pt.first);
                                    // 物体上非关联的特征点或异常匹配点如果没有立体匹配则直接去除，不恢复为新点
                                    if (iter == featureTracker.FinalTrackObjFea[prev_id].end())
                                    {
                                        if (index > 0)
                                        {
                                            index -= 1;
                                            if(status_FAST[index] != 1)
                                                status_FAST[index] = 0;
                                            else
                                            {
                                                // 保留有立体匹配的点
                                                track_cnt_FAST[index] = 1;
                                                // 对于匹配外点，此处必须要修改cls。静态物体的第二个值为0
                                                obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_prev,0);
                                                ids_FAST[index] = pt_id++;
                                                prev_FAST_global_obj_id[index] = prev_id;
                                            }
                                        }
                                        else
                                        {
                                            index = -1 * index;
                                            // 非关联点，或者正确关联但是被视为异常的点，指的是上一帧漏检部分的异常匹配
                                            if(status_sift[index] != 1) 
                                            {
                                                status_sift[index] = 0;
                                                continue;
                                            }
                                            
                                            // 新特征点
                                            track_cnt_sift[index] = 1;
                                            //old_pt_id = ids_sift[index];
                                            // 对于匹配外点，此处必须要修改cls
                                            obj_cls_id_sift[index] = std::pair<uchar,int>(cls_prev,0);
                                            ids_sift[index] = pt_id++;
                                            prev_sift_global_obj_id[index] = prev_id;
                                            // new_stat_obj_sift.emplace_back(old_pt_id,pt_id-1);
                                            continue;
                                        }
                                    }
                                    // 匹配的跟踪点，包含运动估计内点和外点
                                    else
                                    {
                                        // 完成关联的物体，其在上一帧 和 当前帧 中可能同时出现部分漏检；其中，当前帧漏检的点的cls label已经与上一帧的全局cls label对齐；而上一帧漏检的点在当前帧使用局部的检测类别
                                        // 为了方便，直接对当前帧的所有匹配点的cls label都重新赋值为上一帧该物体的全局类别
                                        if (index > 0)
                                        {
                                            index = index - 1;
                                            // 静态物体的第二个变量与背景一样，都是0；动态物体则是global obj id
                                            obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_prev,0);
                                            // 如果上一帧部分漏检，则当前帧物体点可能匹配到上一帧的这些背景点，则直接统一修改此变量
                                            // if (par_lost_prev) 
                                            {
                                                prev_FAST_global_obj_id[index] = prev_id;
                                            }

                                            // 如果该特征点匹配在位姿估计中是外点，则已经被置为待删除，这里也不再把它当作物体的新特征点了(这个FAST点本来就是光流跟踪来的，本身不一定具有好的特征)
                                            if (status_FAST[index] == 0) 
                                            {
                                                if (cur_FAST_dep[index] <= 0 || id_FAST_no_depth.find(index)!= id_FAST_no_depth.end()) continue;

                                                // 新特征点必须有立体匹配
                                                status_FAST[index] = 1;
                                                
                                                track_cnt_FAST[index] = 1;

                                                ids_FAST[index] = pt_id++;
                                                continue;
                                            }
                                            
                                            int id_pt = ids_FAST[index];
                                            if ((frame_count < WINDOW_SIZE || marg_old))
                                            {
                                                if(num_inliers_PnP < Thres_num_track_cur)
                                                {
                                                    float dep = prev_FAST_dep[index];
                                                    // 如果marg最老帧，则说明次新帧与次次新帧之间的视差较大（或者跟踪点较少），总之就是之后有跟踪丢失的风险，则考虑把当前帧所有的跟踪点都加入到地图（后续加入多少，根据长距离点和跟踪点的数量来决定）
                                                    if(dep > 0 && dep <= 7) 
                                                    {
                                                        new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                                        ++num_inliers_PnP;
                                                    }
                                                }
                                                
                                                if (state_prev_obj == 0) track_cnt_FAST[index] = 2;
                                            }
                                            else
                                            {
                                                // track_cnt_FAST[i] = track_cnt_FAST[i] - 1;

                                                if (state_prev_obj == 0) 
                                                {
                                                    // track_cnt_FAST[index] = 1;
                                                    track_cnt_FAST[index] = 2;
                                                }

                                                if (initial_succ_prev)
                                                {
                                                    if(track_cnt_FAST[index] >= (TH_NUM_FRAME_FOR_LBA+1) && state_prev_obj == 1)
                                                    {
                                                        float dep = prev_FAST_dep[index];
                                                        if(dep > 0 && dep <= 5)
                                                        {
                                                            new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                                            ++num_inliers_PnP;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            index = -1 * index;
                                            obj_cls_id_sift[index] = std::pair<uchar,int>(cls_prev,0);
                                            // if (par_lost_prev) 
                                            {
                                                prev_sift_global_obj_id[index] = prev_id;
                                            }
                                            
                                            // 物体上(非漏检部分）的sift运动估计外点可以保留，作为该静态物体上的新特征点
                                            if (status_sift[index] == 0) 
                                            {
                                                // 该点在当前帧必须有立体匹配
                                                if (cur_sift_dep[index] <= 0 || id_sift_no_depth.find(index)!= id_sift_no_depth.end()) continue;
                                                
                                                status_sift[index] = 1;
                                                
                                                track_cnt_sift[index] = 1;
                                                
                                                ids_sift[index] = pt_id++;
                                               
                                                continue;
                                            }

                                            // 物体上的运动估计内点
                                            int id_pt = ids_sift[index];
                                            if ((frame_count < WINDOW_SIZE || marg_old))
                                            {
                                                if(num_inliers_PnP < Thres_num_track_cur)
                                                {
                                                    float dep = prev_sift_dep[index];
                                                    // 如果marg最老帧，则说明次新帧与次次新帧之间的视差较大（或者跟踪点较少），总之就是之后有跟踪丢失的风险，则考虑把当前帧所有的跟踪点都加入到地图（后续加入多少，根据长距离点和跟踪点的数量来决定）
                                                    if(dep > 0 && dep <= 7) 
                                                    {
                                                        new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                                        ++num_inliers_PnP;
                                                    }
                                                }

                                                if (state_prev_obj == 0) track_cnt_sift[index] = 2;
                                            }
                                            // 如果要marg次新帧，且该点观测帧数足够其下一帧参与LBA，则把当前帧的观测加入
                                            else
                                            { 
                                                // track_cnt_sift[i] = track_cnt_sift[i] - 1;

                                                if (state_prev_obj == 0) 
                                                {
                                                    // track_cnt_sift[index] = 1;
                                                    track_cnt_sift[index] = 2;
                                                }

                                                if (initial_succ_prev)
                                                {
                                                    if(track_cnt_sift[index] >= (TH_NUM_FRAME_FOR_LBA+1) && state_prev_obj == 1)
                                                    {
                                                        float dep = prev_sift_dep[index];
                                                        if(dep > 0 && dep <= 5)
                                                        {
                                                            new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // 记录当前帧临时物体的最终匹配，已便后续统一修改各个临时物体上的新检测点
                                FinalTrackCurObj[cur_id] = prev_id;
                                FinalTrackCurObjCls[cur_id] = cls_prev;
                            }
                            // 记录当前帧该物体的全局id和全局cls label
                            featureTracker.glob_obj_id_prev.push_back(prev_id);
                            featureTracker.obj_cls_prev.push_back(cls_prev);

                            // 记录跟踪到的全局静态物体
                            featureTracker.id_gl_sta_obj.insert(prev_id);

                            // 静态物体为1
                            featureTracker.status_objs_prev[prev_id] = 1;
                            // 将当前帧的该静态物体（可能只是整个物体的一部分被检测为一个单独的物体）的匹配内点和新检测点都加入静态地图。
                            // 如果该物体在上一帧就是静态物体，则在静态地图中就已经有上一帧该物体的所有特征点了（如果上一帧该物体有部分漏检，则这些特征点也是在背景中，会被加入静态地图）
                            if (new_tracked_stat_fea.size() > 0)
                            {
                                vec_new_tracked_stat_fea.push_back(new_tracked_stat_fea);
                                vec_cur_prev_obj_id.emplace_back(cur_id,prev_id);
                                vec_g_cls.push_back(cls_prev);
                            }
                            // 新的物体点暂时不放入地图！
                            // else if(new_stat_obj_sift.size() > 0)
                            // {
                            //     vec_new_tracked_stat_fea.push_back(vector<pair<int,int>>());
                            //     vec_cur_prev_obj_id.emplace_back(cur_id,prev_id);
                            //     vec_g_cls.push_back(cls_prev);
                            // }
                            
                            // if (new_stat_obj_sift.size() > 0) 
                            // {
                            //     vec_new_stat_obj_sift.push_back(new_stat_obj_sift);
                            // }
                            // // 否则插入一个空的vector，这是与vec_new_tracked_stat_fea进行元素对齐
                            // else if (new_tracked_stat_fea.size() > 0)
                            //     vec_new_stat_obj_sift.push_back(vector<pair<int,int>>());
                        }
                        // 如果上一帧该物体有部分漏检，则应该检查这些点的全局跟踪次数是否等于2，如果大于2，则说明其上上帧也是背景点，则它当前帧不应该变成物体点。这种情况在特征点跟踪时就排除了
                    }
                    // 对于上一帧中完全漏检的物体，其在当前帧中只会与一个物体相关联，则该物体将作为新静态物体
                    else
                    {
                        if (!new_tracked_stat_fea.empty()) new_tracked_stat_fea.clear();
                        if (!new_stat_obj_sift.empty()) new_stat_obj_sift.clear();
                        
                        int cur_id = id_lost[i-num_gl_dyn_objs];
                        // FeaFrame::iterator end = featureTracker.TotalLostObjPrevBg[cur_id].end();
                        // 随机取一个匹配点的像素坐标
                        float x = featureTracker.TotalLostObjPrevBg[cur_id].begin()->second[0].second(3);
                        float y = featureTracker.TotalLostObjPrevBg[cur_id].begin()->second[0].second(4);
                        uchar cls_cur = cls_map.at<uchar>(y,x);
                        
                        for(const auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                        {
                            int index = gl_id_index_map[pt.first];
                            // 如果不是匹配物体之间的关联点（匹配错物体，或者异常的匹配），则去除FAST点，保留sift点作为新特征点
                            if (featureTracker.TotalLostObjPrevBg[cur_id].find(pt.first) == featureTracker.TotalLostObjPrevBg[cur_id].end())
                            {
                                if(index > 0)
                                {
                                    index -= 1;
                                    // 该外点需要在当前帧有立体匹配
                                    if(status_FAST[index] != 1)
                                        status_FAST[index] = 0;
                                    else
                                    {
                                        status_FAST[index] = 1;
                                        track_cnt_FAST[index] = 1;
                                        // 对于上一帧完全漏检的物体，则使用当前帧的检测类别来作为全局类别。
                                        // 注意，该物体的这些点由于不是和上一帧的背景点关联，而是和别的物体相关联，因此其obj_cls_id_sift的first是与其关联物体的全局cls相关联了。这里改为其当前帧检测类别
                                        obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_cur,0);
                                        ids_FAST[index] = pt_id++;
                                        // 新物体
                                        prev_FAST_global_obj_id[index] = n_obj_id;
                                    }
                                }
                                else
                                {
                                    index = -1 * index;
                                    // 该外点需要在当前帧有立体匹配
                                    // 错误关联点或异常关联点
                                    if(status_sift[index] != 1) 
                                    {
                                        status_sift[index] = 0;
                                        continue;
                                    }

                                    status_sift[index] = 1;
                                    track_cnt_sift[index] = 1;
                                    obj_cls_id_sift[index] = std::pair<uchar,int>(cls_cur,0);
                                    ids_sift[index] = pt_id++;
                                    // 新物体
                                    prev_sift_global_obj_id[index] = n_obj_id;
                                    continue;
                                }
                            }
                            // 如果是正确的物体匹配点
                            else
                            {
                                if (index > 0)
                                {
                                    index = index - 1;
                                    // 运动估计外点
                                    obj_cls_id_FAST[index].second = 0;
                                    prev_FAST_global_obj_id[index] = n_obj_id;

                                    if (status_FAST[index] == 0) 
                                    {
                                        // 该外点需要在当前帧有立体匹配
                                        if (cur_FAST_dep[index] <= 0 || id_FAST_no_depth.find(index) != id_FAST_no_depth.end()) continue;

                                        // 当前帧的物体点都应该是有立体匹配的
                                        status_FAST[index] = 1;
                                        // 新特征点
                                        track_cnt_FAST[index] = 1;
                                        
                                        ids_FAST[index] = pt_id++;
                                        continue;
                                    }
                                    
                                    int id = ids_FAST[index];
                                    float dep = prev_FAST_dep[index];
                                    // obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_cur,0);                           
                                    if ((frame_count < WINDOW_SIZE || marg_old))
                                    {
                                        if(num_inliers_PnP < Thres_num_track_cur)
                                        {
                                            if(dep > 0 && dep <= 7)
                                            {
                                                new_tracked_stat_fea.emplace_back(id,id);
                                                ++num_inliers_PnP;
                                            }
                                        }
                                    }
                                    // 上一帧完全漏检的物体点最多只有2帧观测
                                    // else
                                    // {
                                    //     // track_cnt_FAST[index] = track_cnt_FAST[index] - 1;
                                    //     if(initial_succ_prev)
                                    //     {
                                    //         // 所匹配的上一帧背景点可能已经在背景中被跟踪了很多帧了！则都把这些观测都作为该物体在各帧中的漏检点！
                                    //         if(track_cnt_FAST[index] >= (TH_NUM_FRAME_FOR_LBA+1))
                                    //         {
                                    //             if(dep > 0 && dep <= 5)
                                    //             new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                    //         }
                                    //     }
                                    // }
                                }
                                else
                                {
                                    index = -1 * index;
                                    obj_cls_id_sift[index].second = 0;
                                    // obj_cls_id_sift[index].first = cls_cur;
                                    prev_sift_global_obj_id[index] = n_obj_id;

                                    if (status_sift[index] == 0) 
                                    {
                                        // 该外点需要在当前帧有立体匹配
                                        if (cur_sift_dep[index] <= 0 || id_sift_no_depth.find(index)!= id_sift_no_depth.end()) continue;
                                        
                                        // 当前帧的物体点都应该是有立体匹配的
                                        status_sift[index] = 1;
                                        // 新特征点
                                        track_cnt_sift[index] = 1;
                                        
                                        ids_sift[index] = pt_id++;
                                        continue;
                                    }

                                    // 如果是运动估计的内点
                                    track_cnt_sift[index] = 2;
                                    int id = ids_sift[index];
                                    
                                    // 如果不是marg当前次新帧
                                    if ((frame_count < WINDOW_SIZE || marg_old))
                                    {
                                        if(num_inliers_PnP < Thres_num_track_cur)
                                        {
                                            float dep = prev_sift_dep[index];
                                            if(dep > 0 && dep <= 7)
                                            {
                                                new_tracked_stat_fea.emplace_back(id,id);
                                                ++num_inliers_PnP;
                                            }
                                        }
                                    }
                                    // 上一帧完全漏检的物体点最多只有2帧观测
                                    // else
                                    // {
                                    //     // track_cnt_sift[index] = track_cnt_sift[index] - 1;
                                    //     if(initial_succ_prev)
                                    //     {
                                    //         if(track_cnt_sift[index] >= (TH_NUM_FRAME_FOR_LBA+1))
                                    //         {
                                    //             if(dep > 0 && dep <= 5)
                                    //                 new_tracked_stat_fea.emplace_back(id_pt,id_pt);
                                    //         }
                                    //     }
                                    // }
                                }
                            }
                        }
                        // 记录当前帧临时物体的最终匹配，已便后续统一修改各个临时物体上的新检测点
                        FinalTrackCurObj[cur_id] = n_obj_id;
                        FinalTrackCurObjCls[cur_id] = cls_cur;
                        // 记录当前帧新物体的全局id和全局cls label
                        featureTracker.glob_obj_id_prev.push_back(n_obj_id);
                        featureTracker.obj_cls_prev.push_back(cls_cur);
                        featureTracker.status_objs_prev[n_obj_id] = 1;

                        // 记录跟踪到的全局静态物体
                        featureTracker.id_gl_sta_obj.insert(n_obj_id);
                        
                        ++n_obj_id;
                        // 将当前帧该静态物体的匹配内点和新检测点都加入静态地图。
                        if (new_tracked_stat_fea.size() > 0)
                        {
                            vec_new_tracked_stat_fea.push_back(new_tracked_stat_fea);
                            // 上一帧该物体完全漏检，则其prev_id为0
                            vec_cur_prev_obj_id.emplace_back(cur_id,0);
                            vec_g_cls.push_back(cls_cur);
                        }
                        // else if(new_stat_obj_sift.size() > 0)
                        // {
                        //     vec_new_tracked_stat_fea.push_back(vector<pair<int,int>>());
                        //     vec_cur_prev_obj_id.emplace_back(cur_id,0);
                        //     vec_g_cls.push_back(cls_cur);
                        // }
                        
                        // if (new_stat_obj_sift.size() > 0) 
                        //     vec_new_stat_obj_sift.push_back(new_stat_obj_sift);
                        // // 否则插入一个空的vector，这是与vec_new_tracked_stat_fea进行元素对齐
                        // else if (new_tracked_stat_fea.size() > 0)
                        //     vec_new_stat_obj_sift.push_back(vector<pair<int,int>>());
                    }
                }
            }
        }
        // 如果当前帧没有任何可能的动态物体，则当前帧动态物体的运动模型个数为0
        else
        {
            featureTracker.RP_objs_pred.clear();
        }
        
        cout << "Num of dynamic object in current frame: " << num_dyn << endl;
        // 至此，完成了上一帧与当前帧的潜在动态物体的运动估计和跟踪点处理

        // 下面处理当前帧的静态物体的跟踪点信息
        // 遍历所有在物体匹配阶段就确定为静态物体的特征点，把其中错误匹配(到上一帧其他物体）的sift点作为物体的新特征点
        // 注意，另外在物体关联阶段，有些静态物体的特征点关联可能由于被认为是匹配异常点而被剔除，这些点在先前就已经被修改status为0了！
        // 另外，在物体关联就已经确定的静态物体，其并非所有的FAST跟踪都会被加入到静态地图中！
        // int &num_track_fea_stat = featureTracker.num_track_fea_stat;
        
        // 那些还未加入地图的静态点（仅观测2帧）如何更新和检查它们的深度值呢？
        // 就在这里进行更新，使用PnP估计得到的相机运动
        // int num_valid_track_stat = 0;
        
        for(auto &obj: featureTracker.cur_stat_objs)
        {
            // if (!new_stat_obj_sift.empty()) new_stat_obj_sift.clear();

            int prev_id = obj.first;

            // 平均深度大于物体深度阈值的静态物体，之后直接删除所有点，全局物体直接置为跟丢。上面那些通过运动估计才确认为丢失的物体，就暂时不这么操作了，下一帧再说
            if(!featureTracker.invalid_stat_objs.empty())
            {
                if(featureTracker.invalid_stat_objs.find(prev_id) != featureTracker.invalid_stat_objs.end()) 
                    continue;
            }
            ++num_static_obj;
            // 整个系统中跟踪到的静态物体的次数
            // ++featureTracker.num_sta_objs_found;
            // 在物体关联阶段就确定为静态的物体，其上一帧与上上帧之间一定也是静态的
            // uchar state_prev_obj = status_objs_prev[prev_id];
            // 该物体是否在上一帧有部分漏检
            bool par_lost_prev = (featureTracker.ParLostObjPrevBg.find(prev_id) != featureTracker.ParLostObjPrevBg.end());

            auto id_iter = std::find(glob_obj_id_prev.begin(),glob_obj_id_prev.end(),prev_id);
            int cls_index = std::distance(glob_obj_id_prev.begin(),id_iter);
            // int glo_obj_id = featureTracker.glob_obj_id_prev[cls_index];
            uchar cls_prev = obj_cls_prev[cls_index];

            if(prev_id == 0)
                assert(false && "Static object during object-matching should not be totally lost-detected object in the last frame!");
            
            // FeaFrame::iterator iter_stat_pt_end = featureTracker.TrackBgFea.end();
            vector<pair<int,vector<int>>>::iterator iter = std::find_if(featureTracker.FinalTrackObj.begin(),featureTracker.FinalTrackObj.end(),[&prev_id](const std::pair<int,vector<int>> &a){
                                                                        return a.first == prev_id;});
            
            if(iter == featureTracker.FinalTrackObj.end())
                assert(false && "Why static object during objs matching is not in this obj-matching variable!");
            
            Vector3d pt_prev, pt_cur;
            float cur_z;
            
            for (int i = 0; i < (*iter).second.size(); ++i)
            {
                if(!new_tracked_stat_fea.empty()) new_tracked_stat_fea.clear();

                int cur_id = (*iter).second[i];
                if (cur_id == 0)
                {
                    // 该全局静态物体在当前帧中有漏检
                    auto iter_lose = std::find(featureTracker.lose_objs_cur_bg.begin(),featureTracker.lose_objs_cur_bg.end(),prev_id);
                    int id_in_lost = std::distance(featureTracker.lose_objs_cur_bg.begin(), iter_lose);
                    // 会加入到fea_cur_lose_objs的点都是objs-matching筛选出来的有效匹配点。无效匹配点在objs-matching已经处理了
                    for(const auto lost_pt_in_bg: featureTracker.fea_cur_lose_objs[id_in_lost])
                    {
                        // 对于当前帧背景中的漏检部分跟踪点而言，其与上一帧匹配点的cls是对齐的，因此不需要修改cls
                        int index = gl_id_index_map[lost_pt_in_bg];
                        // sift点只是需要修改那些异常匹配点为背景点
                        if(index <= 0)
                        {
                            index = -1*index;

                            // 当前帧跟踪点为背景点，则其信息不需要修改

                            // 关联阶段的异常匹配sift点，则作为背景的新点。
                            // 对于当前帧的漏检部分，在物体匹配期间肯定是用特征点验证静态性，如果存在异常点，则这部分点在物体关联阶段就已经被修正了！！！且这些异常点不会加入静态物体的最终匹配点集！
                            // 因此这里的条件应该是不会成立的
                            if(status_sift[index] == 0)
                            {
                                if(add_new_sift_in_next_frame) continue;
                                // 深度估计不可靠的点则放弃
                                if (id_sift_no_depth.find(index)!= id_sift_no_depth.end() || cur_sift_dep[index] <= 0) 
                                {
                                    continue;
                                }
                                // obj_cls_id_sift[index].second = 0;
                                obj_cls_id_sift[index].first = 0;
                                if(cur_sift_dep[index] > 0)
                                {
                                    if(featureTracker.TrackObjFeaFrame[0][lost_pt_in_bg].size() == 2)
                                        status_sift[index] = 1;
                                    else
                                        continue;
                                }

                                track_cnt_sift[index] = 1;
                                ids_sift[index] = pt_id++;
                                prev_sift_global_obj_id[index] = 0;
                                // 新的背景点暂时不加入静态地图
                                // new_bg_sift.emplace_back(old_pt_id,pt_id-1);
                                continue;
                            }
                            
                            // 如果是非异常sift匹配点
                            // 更新匹配内点的深度值。
                            // 这里只负责更新那些静态物体上当前帧观测还未加入地图的跟踪点的深度
                            // 所有超过2帧观测的跟踪点（深度要小于阈值，可以参与LBA） 和 一部分只有2帧观测的点（需要上一帧有立体匹配且深度值较小） 已经在地图中了（这些在相机位姿估计线程中进行更新）
                            if(sta_obj_fea_in_map.find(lost_pt_in_bg) == sta_obj_fea_in_map.end())
                            {
                                float dep = prev_sift_dep[index];
                                if(dep <= 0) 
                                {
                                    status_sift[index] = 0;
                                    continue;
                                }
                                
                                // 只更新那些没有立体匹配的物体点
                                if(id_sift_no_depth.find(index) != id_sift_no_depth.end()) 
                                {
                                    pt_prev(2) = dep;
                                    pt_prev(0) = prev_un_Fea_map[lost_pt_in_bg](0) * dep;
                                    pt_prev(1) = prev_un_Fea_map[lost_pt_in_bg](1) * dep;
                                    pt_cur = Cam_R_Trans * pt_prev + Cam_P_Trans;
                                    cur_z = pt_cur(2);
                                    // if (cur_z < mMinDepthPt || cur_z > mThDepthObj)
                                    if (cur_z < 1.2 || cur_z > (1.2 * mThDepthObj))
                                    {
                                        status_sift[index] = 0;
                                        continue;
                                    }
                                    else
                                        cur_sift_dep[index] = cur_z;

                                    // 这些静态物体点之所以前面没有加入地图，就是因为它们的深度不满足要求（上一帧没有立体匹配，或者深度值大于阈值）
                                    // 这里是否还有必要将它们加入地图？如果其在上一帧没有立体匹配，则不加入
                                }
                                
                                
                                // 之前没加入不一定是这些点不满足要求吧？看tracker中添加静态物体点的条件

                                // 当前帧不是marg次新帧，则看PnP后的内点和上面已经添加的点是否已经足够
                                // 如果marg最老帧，则说明次新帧与次次新帧之间的视差较大（或者跟踪点较少），总之就是之后有跟踪丢失的风险，则考虑把当前帧所有的跟踪点都加入到地图（后续加入多少，根据长距离点和跟踪点的数量来决定）
                                if ((frame_count < WINDOW_SIZE || marg_old))
                                {
                                    if(num_inliers_PnP < Thres_num_track_cur)
                                    {
                                        // 需要该点在上一帧有立体匹配
                                        if(track_cnt_sift[index] == 2)
                                        {
                                            if(prevRightFeaMap.find(lost_pt_in_bg) != prevRightFeaMap.end())
                                            {
                                                if(dep <= 7) 
                                                {
                                                    new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                    ++num_inliers_PnP;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            // 增加后续参与LBA的点
                                            if(dep <= 5) 
                                            {
                                                new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                ++num_inliers_PnP;
                                            }
                                        }
                                    }
                                }
                                // 如果要marg次新帧，且该点观测帧数足够其下一帧参与LBA，则把当前帧的观测加入
                                else
                                { 
                                    // track_cnt_sift[i] = track_cnt_sift[i] - 1;

                                    if (initial_succ_prev)
                                    {
                                        if(track_cnt_sift[index] >= (TH_NUM_FRAME_FOR_LBA+1))
                                        {
                                            if(dep <= 5)
                                            {
                                                new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            index -= 1;
                            // 此条件同样不会成立。因为关联物体的漏检部分不会加入这些异常匹配点
                            if(status_FAST[index] == 0) continue;

                            if(sta_obj_fea_in_map.find(lost_pt_in_bg) == sta_obj_fea_in_map.end())
                            {
                                float dep = prev_FAST_dep[index];
                                if(dep <= 0) 
                                {
                                    status_FAST[index] = 0;
                                    continue;
                                }

                                // 只更新那些没有立体匹配的物体点
                                if(id_FAST_no_depth.find(index) != id_FAST_no_depth.end())
                                {
                                    pt_prev(2) = dep;
                                    pt_prev(0) = prev_un_Fea_map[lost_pt_in_bg](0) * dep;
                                    pt_prev(1) = prev_un_Fea_map[lost_pt_in_bg](1) * dep;
                                    pt_cur = Cam_R_Trans * pt_prev + Cam_P_Trans;
                                    cur_z = pt_cur(2);
                                    // if (cur_z < mMinDepthPt || cur_z > mThDepthObj)
                                    if (cur_z < 1.2 || cur_z > 1.2 * mThDepthObj)
                                    {
                                        status_FAST[index] = 0;
                                        continue;
                                    }
                                    else
                                        cur_FAST_dep[index] = cur_z;
                                }
                            
                                // 是否将该点加入地图
                                if ((frame_count < WINDOW_SIZE || marg_old))
                                {
                                    if(num_inliers_PnP < Thres_num_track_cur)
                                    {
                                        // 如果仅有2帧观测，则需要该点在上一帧有立体匹配
                                        if(track_cnt_FAST[index] == 2)
                                        {
                                            if(prevRightFeaMap.find(lost_pt_in_bg) != prevRightFeaMap.end())
                                            {
                                                if(dep <= 7) 
                                                {
                                                    new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                    ++num_inliers_PnP;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            // 增加后续参与LBA的点
                                            if(dep <= 5) 
                                            {
                                                new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                ++num_inliers_PnP;
                                            }
                                        }
                                    }
                                }
                                // 如果要marg次新帧，且该点观测帧数足够其下一帧参与LBA，则把当前帧的观测加入
                                else
                                { 
                                    // track_cnt_FAST[i] = track_cnt_FAST[i] - 1;

                                    if (initial_succ_prev)
                                    {
                                        if(track_cnt_FAST[index] >= (TH_NUM_FRAME_FOR_LBA+1))
                                        {
                                            if(dep <= 5)
                                            {
                                                new_tracked_stat_fea.emplace_back(lost_pt_in_bg,lost_pt_in_bg);
                                                ++num_inliers_PnP;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // continue;
                }
                // 当前帧匹配的是物体
                else
                {
                    // FeaFrame::iterator iter_pt_end = featureTracker.FinalTrackObjFea[prev_id].end();
                    if(!featureTracker.TrackObjFeaFrame[cur_id].empty())
                    {
                        // 修改所有跟踪点的信息
                        for(auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                        {
                            int cur_pt_id = pt.first;
                            // 该静态物体可能是完全由像素点完成匹配的，并没有有效的跟踪特征点
                            if(cur_pt_id < 0) continue;
                            int index = gl_id_index_map[cur_pt_id];
                            // 当前帧在匹配阶段就已确定的静态物体的跟踪FAST内点不一定都在静态地图中（即入选相机运动估计的静态匹配点集TrackBgFea，取决于sift点数以及观测帧数大于2的FAST点是否充足），但是都在跟踪线程中被保留下来，以备下一帧跟踪该物体使用
                            // if (featureTracker.FinalTrackObjFea[prev_id].find(cur_pt_id) != iter_pt_end) continue;
                            if (featureTracker.FinalTrackObjFea[prev_id].find(cur_pt_id) != featureTracker.FinalTrackObjFea[prev_id].end())
                            {
                                float dep;
                                uchar status_pt;
                                // 先修改当前帧该点的信息
                                if (index > 0)
                                {
                                    status_pt = status_FAST[(index-1)];
                                    // 早已确定为静态的物体在FinalTrackObjFea中的点应该都是有效跟踪点
                                    if(status_pt == 0)
                                    {
                                        assert(false);
                                    }

                                    // 点的cls都是对齐自上一帧的，除非上一帧的匹配点是背景点
                                    // 静态物体
                                    obj_cls_id_FAST[(index-1)].second = 0;
                                    // if (par_lost_prev)
                                    {
                                        obj_cls_id_FAST[(index-1)].first = cls_prev;
                                        prev_FAST_global_obj_id[(index-1)] = prev_id;
                                    }

                                    dep = prev_FAST_dep[(index-1)];
                                }
                                else
                                {
                                    status_pt = status_sift[(-index)];
                                    if(status_pt == 0)
                                    {
                                        assert(false);
                                    }

                                    obj_cls_id_sift[(-index)].second = 0;
                                    // if (par_lost_prev)
                                    {
                                        obj_cls_id_sift[(-index)].first = cls_prev;
                                        prev_sift_global_obj_id[(-index)] = prev_id;
                                    }

                                    dep = prev_sift_dep[(-index)];
                                }

                                int cnt_track = 0;
                                // 这里只更新那些没加入地图的静态物体点的深度。那些加入地图的点则在其他函数中更新深度
                                if(sta_obj_fea_in_map.find(cur_pt_id) == sta_obj_fea_in_map.end())
                                {
                                    // 只更新那些没有立体匹配的点的深度
                                    if (status_pt == 1) 
                                        continue;
                                    
                                    if(dep <= 0)
                                    {
                                        if(index > 0)
                                            status_FAST[index-1] = 0;
                                        else
                                            status_sift[-index] = 0;
                                        
                                        continue;
                                    }
                                    
                                    pt_prev(2) = dep;
                                    pt_prev(0) = prev_un_Fea_map[cur_pt_id](0) * dep;
                                    pt_prev(1) = prev_un_Fea_map[cur_pt_id](1) * dep;
                                    pt_cur = Cam_R_Trans * pt_prev + Cam_P_Trans;
                                    cur_z = pt_cur(2);
                                    // if (cur_z < mMinDepthPt || cur_z > mThDepthObj)
                                    if (cur_z < 1.2 || cur_z > 1.2 * mThDepthObj)
                                    {
                                        if(index > 0)
                                            status_FAST[(index-1)] = 0;
                                        else
                                            status_sift[(-index)] = 0;
                                        continue;
                                    }
                                    else
                                    {
                                        if(index > 0)
                                        {
                                            cur_FAST_dep[(index-1)] = cur_z;
                                            cnt_track = track_cnt_FAST[(index-1)];
                                        }
                                        else
                                        {
                                            cur_sift_dep[(-index)] = cur_z;
                                            cnt_track = track_cnt_sift[(-index)];
                                        }
                                    }

                                    if ((frame_count < WINDOW_SIZE || marg_old))
                                    {
                                        if(num_inliers_PnP < Thres_num_track_cur)
                                        {
                                            // 如果仅有2帧观测，则需要该点在上一帧有立体匹配
                                            if(cnt_track == 2)
                                            {
                                                if(prevRightFeaMap.find(cur_pt_id) != prevRightFeaMap.end())
                                                {
                                                    if(dep <= 7) 
                                                    {
                                                        new_tracked_stat_fea.emplace_back(cur_pt_id,cur_pt_id);
                                                        ++num_inliers_PnP;
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                // 增加后续参与LBA的点。无论其在上一帧中是否有立体匹配
                                                if(dep <= 5) 
                                                {
                                                    new_tracked_stat_fea.emplace_back(cur_pt_id,cur_pt_id);
                                                    ++num_inliers_PnP;
                                                }
                                            }
                                        }
                                    }
                                    // 如果要marg次新帧，且该点观测帧数足够其下一帧参与LBA，则把当前帧的观测加入
                                    else
                                    { 
                                        // track_cnt_FAST[i] = track_cnt_FAST[i] - 1;

                                        if (initial_succ_prev)
                                        {
                                            if(cnt_track >= (TH_NUM_FRAME_FOR_LBA+1))
                                            {
                                                if(dep <= 5)
                                                {
                                                    new_tracked_stat_fea.emplace_back(cur_pt_id,cur_pt_id);
                                                    ++num_inliers_PnP;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // 如果是非关联点 或者 关联点中的异常匹配点
                            else
                            {
                                if (index > 0)
                                {
                                    index -= 1;
                                    // 仅保留当前帧有立体匹配的跟踪点作为新点
                                    if(status_FAST[index] != 1)
                                        status_FAST[index] = 0;
                                    else
                                    {
                                        // 保留有立体匹配的点
                                        track_cnt_FAST[index] = 1;
                                        // 对于匹配外点，此处必须要修改cls。静态物体的第二个值为0
                                        obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_prev,0);
                                        ids_FAST[index] = pt_id++;
                                        prev_FAST_global_obj_id[index] = prev_id;
                                    }
                                }
                                else
                                {
                                    index = -1 * index;
                                    if(status_sift[index] != 1) 
                                    {
                                        status_sift[index] = 0;
                                        continue;
                                    }
                                    
                                    // 新特征点
                                    track_cnt_sift[index] = 1;
                                    //old_pt_id = ids_sift[index];
                                    // 对于匹配外点，此处必须要修改cls
                                    obj_cls_id_sift[index] = std::pair<uchar,int>(cls_prev,0);
                                    ids_sift[index] = pt_id++;
                                    prev_sift_global_obj_id[index] = prev_id;
                                    // new_stat_obj_sift.emplace_back(old_pt_id,pt_id-1);
                                    continue;
                                }
                            }
                        }
                    }
                    // 成功匹配的静态物体，其id匹配不需要修改
                    FinalTrackCurObj[cur_id] = prev_id;
                    FinalTrackCurObjCls[cur_id] = cls_prev;
                }

                // 将当前帧该静态物体的匹配内点加入静态地图
                if (new_tracked_stat_fea.size() > 0)
                {
                    vec_new_tracked_stat_fea.push_back(new_tracked_stat_fea);
                    vec_cur_prev_obj_id.emplace_back(cur_id,prev_id);
                    vec_g_cls.push_back(cls_prev);
                }
                // 对于当前帧静态物体的新特征点和只有两帧观测的跟踪点(且没被放进静态点集），暂时不把它们加入静态地图，因为并不确定它们在下一帧中会被跟踪到。如果下一帧跟踪到了，则等到下一帧时再两帧观测一起添加；反之如果没跟踪到，则省去了后续再从静态地图中删除该点的需要！
                // f_manager.addStaticFeature(frame_count, td, FeatureTracker, new_stat_obj_sift, cur_id, true, false, glo_cls);
            }
            
            // 记录当前帧旧物体的全局id和全局cls label
            featureTracker.glob_obj_id_prev.push_back(prev_id);
            featureTracker.obj_cls_prev.push_back(cls_prev);
            featureTracker.status_objs_prev[prev_id] = 1;
            // 记录跟踪到的全局静态物体
            featureTracker.id_gl_sta_obj.insert(prev_id);

            // 物体关联阶段已经确定的静态物体中，存在有在当前帧完全漏检的,则同样需要获取其在当前帧中的采样像素点
            if(detect_lost_objs_cur.size() > num_lost)
            {
                // vector<int>::iterator iter_lost;
                auto iter_lost = std::find(detect_lost_objs_cur.begin(),detect_lost_objs_cur.end(),prev_id);
                if(iter_lost == detect_lost_objs_cur.end()) 
                    continue;
                else
                {
                    int id_lost = std::distance(detect_lost_objs_cur.begin(), iter_lost);

                    if(featureTracker.pixel_objs_prev.find(prev_id) == featureTracker.pixel_objs_prev.end())
                    {
                        cout << "Weired!" << endl;
                        abort();
                    }

                    float* ptr_pix_prev = featureTracker.pixel_objs_prev[prev_id];

                    // +号前为float*，代表的是某一行的一维数组的首元素指针，+2后指该一行一维数组的第3个元素的指针
                    float* ptr_pix_cur = featureTracker.sampled_pixel[(valid_objs+id_lost)] + 2;

                    int num_pixel_cur = 0;
                    // 根据相机运动估计结果来传播该物体的像素点
                    Vector3f ave_3d_pts = sample_pixel_for_lost_obj(ptr_pix_prev, ptr_pix_cur, num_pixel_cur, Cam_P_Trans, Cam_R_Trans, true);
                    // 物体的全局cls继承自上一帧的匹配物体，在sample_pixel_for_lost_obj函数内已获取
                    // ptr_pix_cur[-2] = cls_prev;
                    ptr_pix_cur[-1] = num_pixel_cur;
                    // 静态物体在当前帧的平均3D点。物体关联阶段暂时只计算了该物体的3D点的平均深度，且不一定是特征点的（也可能是使用像素采样点完成的关联验证）。
                    // TODO：这里暂时使用此静态物体的采样像素点的3D值的平均。如果为了提高精度，可以使用该静态物体的3D特征点的平均？
                    ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3d_pts(0);
                    ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3d_pts(1);
                    ptr_pix_cur[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3d_pts(2);
                    // 保存采样像素点的内存指针
                    sampled_pixel_lost_obj[id_lost] = ptr_pix_cur;

                    ++num_lost;
                }
            }
        }
        cout << "Num of static object in current frame: " << num_static_obj << endl;
    } 

    // 下面的这些即使是系统首帧也可以执行的代码
     
    // 已经处理完当前帧完全漏检物体的采样像素点的传递，通知子线程可以将GPU上采样的当前帧物体像素点复制到数组中
    featureTracker.check_total_lost_cur_objs = true;

    vector<int> &valid_detect_obj = featureTracker.valid_detect_obj;
    vector<uchar> correct_obj(_valid_objs,0);

    float depth_ave;
    set<int> invalid_new_objs;

    while(!done_sample)
    {
        usleep(300);
    }
    done_sample = false;
    int num_new = 0;
    // 遍历new objs，改变其track fea为new fea，添加新物体。新物体的点无需加入静态地图（因为不知道其是否为静态）
    for (const auto &obj_id: featureTracker.new_objs_cur)
    {
        // auto iter = std::find(valid_detect_obj.begin(),valid_detect_obj.end(),obj_id);
        // int id = distance(valid_detect_obj.begin(),iter);
        // float* pixel_ptr = featureTracker.sampled_pixel[id] + 2;
        float* pixel_ptr = &(featureTracker.sampled_pixel[0][0]) + (obj_id-1)*(NUM_SAMPLED_PIXEL_OBJ*3+2+7);
        // 该物体像素点在相机坐标系下的平均深度
        int shift = NUM_SAMPLED_PIXEL_OBJ*3+2+6;
        float depth_ave = pixel_ptr[shift];
        // cout << "ave depth of new obj: " << depth_ave << endl;
        // abort();

        // 像素点的深度来自于depth_map，而立体匹配网络的最大有效视差为192，换算下来最小深度值为2.0
        // if(depth_ave < mMinDepthPt || depth_ave > mThDepthObj)
        if(depth_ave < 2.0 || depth_ave > mThDepthObj)
        {
            invalid_new_objs.insert(obj_id);

            continue;
        }

        int num_fea = 0;
        if(featureTracker.NewObjFeaFrame.find(obj_id) != featureTracker.NewObjFeaFrame.end())
            num_fea += featureTracker.NewObjFeaFrame[obj_id].size();
        uchar cls_cur = bbox_mask[obj_id].class_label;

        if(featureTracker.TrackObjFeaFrame.find(obj_id) != featureTracker.TrackObjFeaFrame.end())
        {
            // float x = featureTracker.TrackObjFeaFrame[obj_id].begin()->second[0].second(3);
            // float y = featureTracker.TrackObjFeaFrame[obj_id].begin()->second[0].second(4);
            // cls_cur = cls_map.at<uchar>(y,x);
            
            // 如果检测新点的数量太少
            // bool invalid_obj = (num_fea < 4);
            // vector<int> pts_index;

            // assert(obj_id == 0 && "Camera (background) can't be recognized as static objects!");
            for (const auto &pt:featureTracker.TrackObjFeaFrame[obj_id])
            {
                if(pt.first < 0) continue;
                int index = gl_id_index_map[pt.first];
                // 要保存这些新物体上的跟踪FAST点吗？（正常来说这些点数应该不会很多吧？）
                if (index > 0)
                {
                    index -= 1;
                    if (cur_FAST_dep[index] <= 0 || id_FAST_no_depth.find(index)!= id_FAST_no_depth.end()) 
                    {
                        status_FAST[index] = 0;
                        continue;
                    }
                    
                    status_FAST[index] = 1;
                    track_cnt_FAST[index] = 1;
                    obj_cls_id_FAST[index] = std::pair<uchar,int>(cls_cur,n_obj_id);
                    ids_FAST[index] = pt_id++;
                    prev_FAST_global_obj_id[index] = n_obj_id;
                    // if (invalid_obj) pts_index.push_back(index);
                }
                else
                {
                    index = -1 * index;
                    // 在物体匹配阶段被认为是异常匹配点的sift，在之前就已经被处理，即直接删除
                    // 可以保留其中有立体匹配的点
                    // if(status_sift[index] == 0) continue;

                    // 深度不可靠的点放弃
                    if (cur_sift_dep[index] <= 0 || id_sift_no_depth.find(index)!= id_sift_no_depth.end()) 
                    {
                        status_sift[index] = 0;
                        continue;
                    }
                    status_sift[index] = 1;
                    track_cnt_sift[index] = 1;
                    obj_cls_id_sift[index] = std::pair<uchar,int>(cls_cur,n_obj_id);
                    ids_sift[index] = pt_id++;
                    prev_sift_global_obj_id[index] = n_obj_id;
                    
                    // if (invalid_obj) pts_index.push_back(index);
                }
                ++num_fea;
            }

            // todo: 如果原先的跟踪点+新点 总数仍很小，是否放弃该新物体？
            // if (num_fea < 4)
            // {
            //     // for(auto &index: pts_index)
            //     // {
            //     //     status_sift[index] = 0;
            //     // }
            //     // 无效物体
            //     FinalTrackCurObj[obj_id] = 0;
            //     continue;
            // }
            // 特征点数满足最小阈值时才保留该物体
            // else
            // {

            // }
        }

        if(featureTracker.NewObjFeaFrame.find(obj_id) != featureTracker.NewObjFeaFrame.end())
        {
            // if(cls_cur == 0)
            // {
            //     float x = featureTracker.NewObjFeaFrame[obj_id].begin()->second[0].second(3);
            //     float y = featureTracker.NewObjFeaFrame[obj_id].begin()->second[0].second(4);
            //     cls_cur = cls_map.at<uchar>(y,x);
            // }

            // 更改该物体的新检测点的全局物体id
            for(const auto &pt:featureTracker.NewObjFeaFrame[obj_id])
            {
                int index = gl_id_index_map[pt.first];
                if (index > 0)
                {
                    index = index - 1;
                    // 新检测点的cls label不需要改变
                    // 创建新的全局物体。注意，n_obj_id的初始值是1，而0是属于背景的id
                    obj_cls_id_FAST[index].second = n_obj_id;
                    prev_FAST_global_obj_id[index] = n_obj_id;
                }
                else
                {
                    index = -1 * index;
                    // 点的cls label不需要改变
                    obj_cls_id_sift[index].second = n_obj_id;
                    prev_sift_global_obj_id[index] = n_obj_id;
                }
            }
        }
        
        // 该物体必须有新特征点，否则就无法成为一个新物体（因为下一帧不会再在当前帧图像中检测新的物体点），即每一帧中的物体点必须在该帧处理中被检测和保留
        if(num_fea > 0)
        {
            // 记录当前帧各个物体的全局id和全局cls label
            featureTracker.glob_obj_id_prev.push_back(n_obj_id);
            featureTracker.obj_cls_prev.push_back(cls_cur);
            // 新物体
            featureTracker.status_objs_prev[n_obj_id] = 2;
            // 用负值代表该临时物体的新特征点已经被处理过了
            FinalTrackCurObj[obj_id] = -1 * n_obj_id;
            FinalTrackCurObjCls[obj_id] = cls_cur;
            ++n_obj_id;
            ++num_new;
        }
        else
        {
            invalid_new_objs.insert(obj_id);
            // cout << "num of fea in invalid obj: " << num_fea << endl;
            // continue;
        }
    }
    cout << "Num of new object: " << num_new << endl;

    // 删除无效新物体的所有点
    if(!invalid_new_objs.empty())
    {
        cout << "Num of invalid new object: " << invalid_new_objs.size() << endl;
        // cout << "Num of invalid new object: " << invalid_new_objs.size() << endl;
        for(const auto &obj_id: invalid_new_objs)
        {
            if(featureTracker.TrackObjFeaFrame.find(obj_id) != featureTracker.TrackObjFeaFrame.end())
            {
                for (const auto &pt:featureTracker.TrackObjFeaFrame[obj_id])
                {
                    if(pt.first < 0) continue;

                    int index = gl_id_index_map[pt.first];
                    // 要保存这些新物体上的跟踪FAST点吗？（正常来说这些点数应该不会很多吧？）
                    if (index > 0)
                    {
                        // FAST跟踪点没必要保留！因为这些点并不是多鲁棒
                        status_FAST[index-1] = 0;
                    }
                    else
                    {
                        status_sift[(-1*index)] = 0;
                    }
                }
            }

            // 新特征点放在下面统一处理
            // if(featureTracker.NewObjFeaFrame.find(obj_id) != featureTracker.NewObjFeaFrame.end())
            // {
            //     for(const auto &pt:featureTracker.NewObjFeaFrame[obj_id])
            //     {
            //         int index = gl_id_index_map[pt.first];
            //         if (index > 0)
            //         {
            //             status_FAST[index-1] = 0;
            //         }
            //         else
            //         {
            //             status_sift[(-1*index)] = 0;
            //         }
            //     }
            // }

            FinalTrackCurObj[obj_id] = 0;
            FinalTrackCurObjCls[obj_id] = 0;
        }
    }
    invalid_new_objs.clear();

    // for(auto iter: featureTracker.status_objs_prev)
    // {
    //     cout << "status_objs_prev: " << iter.first << endl;
    // }

    // 把无效的静态物体的点均删除。将该全局物体置为跟丢。
    // 这里出现的无效静态物体指的是该静态物体的平均深度超出阈值区间！！
    // 因为在特征点跟踪和物体匹配过程中（主要是针对当前帧或上一帧的漏检部分）进行了检查和排除！
    if(!featureTracker.invalid_stat_objs.empty())
    {
        cout << "Num of invalid static object: " << featureTracker.invalid_stat_objs.size() << endl;
        for(auto &obj_id: featureTracker.invalid_stat_objs)
        {
            //FeaFrame::iterator iter_pt_end = featureTracker.FinalTrackObjFea[obj_id].end();
            // FeaFrame::iterator iter_stat_pt_end = featureTracker.TrackBgFea.end();
            vector<pair<int,vector<int>>>::iterator iter = std::find_if(featureTracker.FinalTrackObj.begin(),featureTracker.FinalTrackObj.end(),[&obj_id](const std::pair<int,vector<int>> &a){
                                                                        return a.first == obj_id;});
            
            if(iter == featureTracker.FinalTrackObj.end())
            {
                assert(false);
            }

            for (int i = 0; i < (*iter).second.size(); ++i)
            {
                int cur_id = (*iter).second[i];
                if (cur_id == 0)
                {
                    auto iter_lose = std::find(featureTracker.lose_objs_cur_bg.begin(),featureTracker.lose_objs_cur_bg.end(),obj_id);
                    if(iter_lose == featureTracker.lose_objs_cur_bg.end())
                        assert(false);

                    int id_in_lost = std::distance(featureTracker.lose_objs_cur_bg.begin(), iter_lose);

                    for(const auto lost_pt_in_bg: featureTracker.fea_cur_lose_objs[id_in_lost])
                    {
                        int index = gl_id_index_map[lost_pt_in_bg];
                        if(index > 0)
                            status_FAST[index-1] = 0;
                        else
                            status_sift[-1*index] = 0;
                    }
                    continue;
                }
                else
                {
                    if(featureTracker.TrackObjFeaFrame.find(cur_id) != featureTracker.TrackObjFeaFrame.end())
                    {
                        for(auto &pt: featureTracker.TrackObjFeaFrame[cur_id])
                        {
                            int cur_pt_id = pt.first;
                            if(cur_pt_id < 0) continue;
                            // 当前帧在匹配阶段就已确定的静态物体的跟踪FAST内点不一定都在静态地图中（即入选相机运动估计的静态匹配点集TrackBgFea，取决于sift点数以及观测帧数大于2的FAST点是否充足），但是都在跟踪线程中被保留下来，以备下一帧跟踪该物体使用
                            // if (featureTracker.FinalTrackObjFea[prev_id].find(cur_pt_id) != iter_pt_end) continue;
                            int index = gl_id_index_map[cur_pt_id];
                            if(index > 0)
                                status_FAST[index-1] = 0;
                            else
                                status_sift[-1*index] = 0;
                        }
                    }

                    // 当前帧物体的新点同样在下面统一删除

                    // 0表示此当前帧检测的物体上所有点都要被删除，包括新检测的点
                    FinalTrackCurObj[cur_id] = 0;
                    FinalTrackCurObjCls[cur_id] = 0;
                }
            }

            (*iter).second.clear();
            featureTracker.FinalLostObjPrev.push_back(obj_id);
        }
    }
    
    assert(FinalTrackCurObjCls.size() == FinalTrackCurObj.size());

    // 剩下所有非新物体中的新特征点修改信息
    for(int j = 0; j < _valid_objs; ++j)
    {
        int i = featureTracker.valid_detect_obj[j];
        // 是否存在这种情况？
        if (FinalTrackCurObj.find(i) == FinalTrackCurObj.end()) continue;
        // 已经处理过的新物体，跳过
        if (FinalTrackCurObj[i] < 0) 
        {
            FinalTrackCurObj[i] = -1 * FinalTrackCurObj[i];
            continue;
        }
        // int num_fea = 0;
        // num_fea += featureTracker.NewObjFeaFrame[i+1].size();

        // 平均深度超过阈值的无效物体，其上所有的点都删除
        if(FinalTrackCurObjCls[i] == 0)
        {
            if(featureTracker.NewObjFeaFrame.find(i) != featureTracker.NewObjFeaFrame.end())
            {
                for(const auto &pt:featureTracker.NewObjFeaFrame[i])
                {
                    int index = gl_id_index_map[pt.first];
                    if (index > 0)
                        status_FAST[index-1] = 0;
                    else
                        status_sift[-1*index] = 0;
                }
            }
        }
        // 还未处理新特征点的有效物体，即进行了运动估计并且为有效的物体
        else
        {
            int gl_obj_id = FinalTrackCurObj[i];
            int gl_cls = FinalTrackCurObjCls[i];
            std:: pair<uchar,int> new_info(gl_cls,gl_obj_id);
            if(featureTracker.NewObjFeaFrame.find(i) != featureTracker.NewObjFeaFrame.end())
            {
                for(const auto &pt:featureTracker.NewObjFeaFrame[i])
                {
                    int index = gl_id_index_map[pt.first];
                    if (index > 0)
                    {
                        index = index - 1;
                        prev_FAST_global_obj_id[index] = gl_obj_id;
                        obj_cls_id_FAST[index] = new_info;
                        // 如果该物体为静态
                        if(featureTracker.status_objs_prev[gl_obj_id] == 1)
                        obj_cls_id_FAST[index].second = 0;
                    }
                    else
                    {
                        index = -1 * index;
                        prev_sift_global_obj_id[index] = gl_obj_id;
                        obj_cls_id_sift[index] = new_info;

                        if(featureTracker.status_objs_prev[gl_obj_id] == 1)
                            obj_cls_id_sift[index].second = 0;
                    }
                }
            }
        }
    }
    
    // 至此，当前帧所有检测物体的点的信息均已更新。其中还包括了背景中的漏检物体。

    // 等待主线程中完成LBA和对已有静态地图点的更新，然后再往静态地图中添加新的地图点观测
    // 在LBA中会优化估计出下一帧的td值，因此此函数需要给定当前帧的td值cur_td_old
    while(!map_fea_optimized)
    {
        usleep(300);
    }

    vector<int> reserve_fea;
    int &num_new_sift_bg = featureTracker.num_new_sift_bg;
    int &num_bg_sift_with_dep = featureTracker.num_bg_sift_with_dep;

    // 相机运动估计（此时VI还未初始化）和LBA 后是否有当前帧跟踪的sift点被作为外点，如果有，则将其保留为当前帧该物体的新点。
    // 这些保留点的要求是在当前帧中必须要有深度估计（即立体匹配）
    if (reserve_new_sift.size() > 0)
    {
        vector<Point2f> &cur_sift = featureTracker.cur_sift;
        for(auto &pts: reserve_new_sift)
        {
            int index = -1 * gl_id_index_map[pts];
            // 如果该静态点在当前帧已经决定被删除(例如为运动估计外点）。这是针对那些在运动估计后才确认为静态的物体，下同
            // 如果当前帧该跟踪点也没有立体匹配，则放弃转为新点。
            if(status_sift[index] == 0 || status_sift[index] == 2) continue;
            // 如果该点在当前帧已经修改为动态点。这指的是那些之前就已经加入地图的静态物体点，但在当前帧没有加入，而在清理地图中外点时被作为了外点，且尝试保留为新的物体点
            if(obj_cls_id_sift[index].second > 0) continue;

            Vec2b &pt_info = full_seg_map.at<Vec2b>(cur_sift[index].y, cur_sift[index].x);
            
            int det_obj_id = pt_info(1);
            uchar gl_cls = 0;

            if(det_obj_id != 0)
            {
                // 当前帧的特征点应该不会是无效的非刚体或无效类别的点
                assert(FinalTrackCurObjCls.find(det_obj_id) != FinalTrackCurObjCls.end());
                gl_cls = FinalTrackCurObjCls[det_obj_id];
            }
            
            // 如果该临时物体已经成为了无效物体
            if(det_obj_id != 0 && gl_cls == 0)
            {
                status_sift[index] = 0;
                continue;
            }

            uchar det_cls = pt_info(0);

            // 如果当前帧检测为背景点，则这里将其修改为背景的新点
            // 如果该点在当前帧的检测类别是物体，则此时需要检查该局部物体是否有完成全局物体关联？该点所属物体在物体关联阶段就已经确认关联且为静态，因此肯定有全局关联
            // !另外，我们默认即使其上有个别跟踪点成为运动估计外点，也不影响该物体整体的全局关联结果！
            // 但是，是否需要检查一下该点所属的局部物体在当前帧中是否还有足够特征点（跟踪内点和新特征点）使得它还能称为一个物体并与某全局物体相关联？无所谓，即使它特征点数很少，大不了就是下一帧跟踪该物体失败！
            if(det_cls == 0)
            {
                if(add_new_sift_in_next_frame)
                {
                    status_sift[index] = 0;
                    continue;
                }

                obj_cls_id_sift[index].first = 0;
                prev_sift_global_obj_id[index] = 0;

                // 当前帧背景中的新sift点数增加
                ++num_new_sift_bg;
                // 如果该原跟踪点在当前帧中是有立体匹配的
                if(status_sift[index] == 1) ++num_bg_sift_with_dep;
            }
            else
            {
                int gl_obj_id = FinalTrackCurObj[det_obj_id];
                obj_cls_id_sift[index].first = gl_cls;

                assert(featureTracker.status_objs_prev.find(gl_obj_id) != featureTracker.status_objs_prev.end());
                // 如果最终该物体为静态物体
                if(featureTracker.status_objs_prev[gl_obj_id] == 1)
                    obj_cls_id_sift[index].second = 0;
                else
                    obj_cls_id_sift[index].second = gl_obj_id;
                
                prev_sift_global_obj_id[index] = gl_obj_id;
            }

            reserve_fea.push_back(pts);
            ids_sift[index] = pt_id++;
            track_cnt_sift[index] = 1;
        }
    }

    if (reserve_new_FAST.size() > 0)
    {
        vector<Point2f> &cur_FAST = featureTracker.cur_FAST;
        for(auto &pts: reserve_new_FAST)
        {
            int index = gl_id_index_map[pts] -1;
            
            // 如果该静态点在当前帧已经决定被删除(例如为运动估计外点）。这是针对那些在运动估计后才确认为静态的物体，下同
            if(status_FAST[index] == 0 || status_FAST[index] == 2) continue;
            // 如果该点在当前帧已经修改为动态点
            if(obj_cls_id_FAST[index].second > 0) continue;

            Vec2b &pt_info = full_seg_map.at<Vec2b>(cur_FAST[index].y, cur_FAST[index].x);

            int det_obj_id = pt_info(1);
            
            uchar gl_cls = 0;
            if(det_obj_id != 0)
            {
                // 当前帧的特征点应该不会是无效的非刚体或无效类别的点
                assert(FinalTrackCurObjCls.find(det_obj_id) != FinalTrackCurObjCls.end());
                gl_cls = FinalTrackCurObjCls[det_obj_id];
            }

            // 如果该临时物体已经成为了无效物体
            if(det_obj_id != 0 && gl_cls == 0)
            {
                status_FAST[index] = 0;
                continue;
            }

            uchar det_cls = pt_info(0);

            // 实际上背景中的FAST跟踪外点是不会保留为新点的，因为认为其匹配精度不够高。
            if(det_cls == 0)
            {
                if(add_new_sift_in_next_frame)
                {
                    status_FAST[index] = 0;
                    continue;
                }

                obj_cls_id_FAST[index].first = 0;
                prev_FAST_global_obj_id[index] = 0;
                
                // 当前帧背景中的新FAST点数增加。这里
                ++featureTracker.num_new_FAST_bg;
                // 如果该原跟踪点在当前帧中是有立体匹配的。并没有定义对应的num_bg_FAST_with_dep变量。
                // if(status_FAST[index] == 1) ++num_bg_sift_with_dep;
            }
            else
            {
                int gl_obj_id = FinalTrackCurObj[det_obj_id];
                obj_cls_id_FAST[index].first = gl_cls;

                assert(featureTracker.status_objs_prev.find(gl_obj_id) != featureTracker.status_objs_prev.end());
                // 如果最终该物体为静态物体
                if(featureTracker.status_objs_prev[gl_obj_id] == 1)
                    obj_cls_id_FAST[index].second = 0;
                else
                    obj_cls_id_FAST[index].second = gl_obj_id;
                
                prev_FAST_global_obj_id[index] = gl_obj_id;
            }

            reserve_fea.push_back(pts);
            ids_FAST[index] = pt_id++;
            track_cnt_FAST[index] = 1;
        }
    }

    // 这是针对那些 在LBA之后被认定为外点的当前帧跟踪点（包括背景跟踪点，以及部分静态物体点（还未加入地图）），则这里需要将它们全部删除。还要防止它们被作为已有静态点的新观测加入地图
    if(direct_erase_fea.size() > 0)
    {
        for(auto &pts:direct_erase_fea)
        {
            int index = gl_id_index_map[pts];
            if(index > 0)
            {
                index -= 1;
                // 如果该物体在当前帧变成动态的，则其点id已改变，且不会加入地图。
                if (obj_cls_id_FAST[index].second > 0) continue;

                // TODO: 是否有可能是那些在运动估计后才确定为静态物体的点呢？即它们在地图中的先前观测被LBA认为是外点，才使得它们被认为应该删除。那么这些跟踪点是否应该保留？？或者保留为新点？
                // 查看这些点所属的物体最终是什么状态，如果是静态物体，且如果它们没有在当前帧提前加入地图，则这里可以保留这些跟踪点，或者另外保留为新点（如果它们加入了地图）

                status_FAST[index] = 0;
            }
            else
            {
                index = -1 * index;
                if (obj_cls_id_sift[index].second > 0) continue;
                // todo: 通上

                status_sift[index] = 0;
            }  
        }
    }

    // 这里选择暂时不再当前帧把背景或者任何静态物体的新特征点加入静态地图，而是等到下一帧跟踪到该点并且仍然为静态物体时才将两帧观测一起加入！
    
    // 添加当前帧背景的新sift点到静态地图中。但是不添加
    // if (!new_bg_sift.empty())
    // {
    //     f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, FeatureTracker, new_bg_sift, 0, true, false);
    // }

    // 添加静态物体上的跟踪点观测。这里添加的跟踪点，有一部分是在运动估计后才确认为静态的物体上的点（其中有些是观测帧数大于4，因此无论当前帧marg哪一帧，都要添加该点），有一部分可能是所有静态物体的仅有2帧跟踪的点（即上一滑窗已完成初始化)
    if (vec_new_tracked_stat_fea.size() > 0)
    {
        for(int i = 0; i < vec_new_tracked_stat_fea.size(); ++i)
        {
            int prev_id = vec_cur_prev_obj_id[i].second;
            uchar glo_cls = vec_g_cls[i];
            if (prev_id > 0)
            {
                // 上一帧非完全漏检的物体，其在当前帧的匹配点中还可能会有背景漏检点。另外，上一帧的物体还可能部分漏检！
                if(vec_new_tracked_stat_fea[i].size() == 0)
                    assert(false && "It is impossible that the num of track point for two matched objects is zero!");
                
                // !注意，这里status_objs_prev和featureTracker.status_objs_prev已经是不同的变量！前者是当前函数内的临时变量，后者是已经被clear，用于存放新一帧的物体的状态信息！
                if(status_objs_prev[prev_id] == 1)
                    f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, featureTracker, vec_new_tracked_stat_fea[i], vec_cur_prev_obj_id[i].first, reserve_fea, direct_erase_fea);
                // 如果上一帧的该物体为动态或者新物体，则其特征点肯定还没有在地图中（有没有可能是先静态，后动态，当前帧又静态？这种情况下跟踪已不连续，当作是两个不同的静态物体了）
                else
                    f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, featureTracker, vec_new_tracked_stat_fea[i], vec_cur_prev_obj_id[i].first, reserve_fea, direct_erase_fea);  
                
                // 这里可以保存当前帧静态物体的信息？

                // 物体上的新点这里仅指的是关联外点sift、位姿估计外点sift）
                // 暂时不添加新特征点
                if(0)
                {
                    if (vec_new_stat_obj_sift[i].size() > 0)
                        f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, featureTracker, vec_new_stat_obj_sift[i], vec_cur_prev_obj_id[i].first, reserve_fea, direct_erase_fea, true); 
                }
            }
            // 如果上一帧该物体完全漏检，则该物体必然是新的静态物体
            else
            {
                if(vec_new_tracked_stat_fea[i].size() > 0)
                    // 因为该物体上一帧全漏检，则当前帧所匹配的上一帧的点都是背景点。
                    // 该特征点正常来说还不在静态地图中，因为上一帧为背景点，而当前帧为物体点，则该点必须为上一帧背景的新点而没加入地图（如果该点上一帧被发现为物体点，则id肯定已校正；如果该点上一帧和上上帧背景点关联，则不太可能当前帧突然变为物体点）
                    f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, featureTracker, vec_new_tracked_stat_fea[i], vec_cur_prev_obj_id[i].first, reserve_fea, direct_erase_fea);
                if(0)
                {
                    if (vec_new_stat_obj_sift[i].size() > 0) 
                        f_manager.addStaticFeature(frame_count, prev_td, cur_td_old, featureTracker, vec_new_stat_obj_sift[i], vec_cur_prev_obj_id[i].first, reserve_fea, direct_erase_fea, true); 
                }
            }
        }
    }
    reserve_new_sift.clear();
    reserve_new_FAST.clear();
    direct_erase_fea.clear();
    // 至此，当前帧所有需要添加到地图的静态点观测记录已完成！

    // 通知主线程，子线程中已经完成了对地图的写操作
    map_fea_optimized = false;
    map_fea_writen = true;
    
    // 到这里就可以清空上一帧的物体像素数组的指针信息，下面不会再用到上一帧的该信息了
    featureTracker.pixel_objs_prev.clear();
    int gl_obj, local_obj;
    
    // 对于匹配物体集中，当前帧被一分为二的错误检测的物体，将它们的采样点混合
    if(!FinalTrackObj.empty())
    {
        for(auto &iter:FinalTrackObj)
        {
            int prev_id = iter.first;
            // 可能需要融合的不止是参与运动估计的物体，在物体匹配阶段就确定为静态的物体也可能需要。
            // 这里取决于需不需要对静态物体也进行像素集合的融合？毕竟这些静态物体在本实验中不需要参与评估，不需要与真值进行匹配。但是下一帧可能还要用到像素点集来进行配对
            if(1)
            {
                // 部分完成关联的物体，其深度值不满足要求，则上一帧该物体会当作丢失，而所有关联的当前帧物体会被删除。此时该全局物体的匹配物体集已被清空
                if (iter.second.size() == 0) continue;
                // 单个物体匹配关系的留到后面一起处理
                if (iter.second.size() == 1)
                {
                    continue;
                }
                // 当前帧物体被分为两部分或三部分（一部分漏检）
                else
                {
                    int obj_1 = iter.second[0];
                    int obj_2 = iter.second[1];
                    // 如果分为2部分，且其中一部分是漏检
                    if(iter.second.size() == 2 && (obj_1 == 0 || obj_2 == 0))
                    {
                        int id;
                        if (obj_1 == 0)
                        {
                            local_obj = obj_2;
                        }
                        else
                        {
                            local_obj = obj_1;
                        }  
                        auto iter_ = std::find(valid_detect_obj.begin(),valid_detect_obj.end(),local_obj);
                        if(iter_ == valid_detect_obj.end())
                            assert(false && "Something weird happened!");
                        
                        id = distance(valid_detect_obj.begin(),iter_);
                        correct_obj[id] = 1;
                        float* ptr_obj = featureTracker.sampled_pixel[id] + 2;
                        int num_pixel = (int)ptr_obj[-1];
                        
                        // 如果未漏检部分的采样像素点数量未达到最大值，则从漏检部分的特征点中选择点加入像素点集合中
                        if (num_pixel < NUM_SAMPLED_PIXEL_OBJ)
                        {
                            float x = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] * num_pixel;
                            float y = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] * num_pixel;
                            float z = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] * num_pixel;
                            float left   = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ)];
                            float right  = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+1)];
                            float top    = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+2)];
                            float bottom = ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+3)];
                            for(auto &pt: featureTracker.FinalTrackObjFea[prev_id])
                            {
                                if(num_pixel == NUM_SAMPLED_PIXEL_OBJ) break;
                                // 可能是添加的像素跟踪点？
                                if(pt.first < 0) continue;
                                float u = pt.second[0].second(3);
                                float v = pt.second[0].second(4);
                                if(cls_map.at<uchar>(v,u) == 0)
                                {
                                    uchar status;
                                    int index = gl_id_index_map[pt.first];
                                    if(index > 0)
                                    {
                                        index -= 1;
                                        status = status_FAST[index];
                                        // 如果当前帧的跟踪点没有立体匹配，则放弃该点，因为其深度值不可靠
                                        if (status == 0 || status == 2) continue;
                                        // 漏检部分的运动估计外点还可能成为背景新点，因此校验该点的全局obj id
                                        if(prev_FAST_global_obj_id[index] != prev_id) continue;
                                    }
                                    else
                                    {
                                        index = -1 *index;
                                        status = status_sift[index];
                                        if (status == 0 || status == 2) continue;
                                        // 漏检部分的运动估计外点还可能成为背景新点，因此校验该点的全局obj id
                                        if(prev_sift_global_obj_id[index] != prev_id) continue;
                                    }
                                    
                                    float dep = pt.second[0].second(2);
                                    if(dep <= 0) continue;
                                    ptr_obj[(3*num_pixel)] = u;
                                    ptr_obj[(3*num_pixel+1)] = v;
                                    ptr_obj[(3*num_pixel+2)] = dep;
                                    
                                    x += pt.second[0].second(0) * dep;
                                    y += pt.second[0].second(1) * dep;
                                    z += dep;
                                    // 更新2D bbox的范围
                                    if(u < left || left == (float)0.0)
                                        left = u;
                                    
                                    if(u > right || right == (float)0.0)
                                        right = u;
                                    
                                    if (v < top || top == (float)0.0)
                                        top = v;
                                    
                                    if(v > bottom || bottom == (float)0.0)
                                        bottom = v;
                                    
                                    ++num_pixel;
                                }
                            }
                            // 如果该物体是动态物体，且内点中特征点的数量足够多（至少3个），则使用3D特征点的均值来代替像素3D点均值
                            // if(ave_3d_pts_objs.find(prev_id) != ave_3d_pts_objs.end())
                            // {
                            //     if(ave_3d_pts_objs[prev_id](2) != 0.0)
                            //     {
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3d_pts_objs[prev_id](0);
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3d_pts_objs[prev_id](1);
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3d_pts_objs[prev_id](2);
                            //     }
                            //     else
                            //     {
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = x/num_pixel;
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = y/num_pixel;
                            //         ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = z/num_pixel;
                            //     }
                            // }
                            // else
                            {
                                if(num_pixel > 0)
                                {
                                    ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = x/num_pixel;
                                    ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = y/num_pixel;
                                    ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = z/num_pixel;
                                }
                            }
                            
                            ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ)]   = left;
                            ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+1)] = right;
                            ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+2)] = top;
                            ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+3)] = bottom;
                            ptr_obj[-1] = num_pixel;

                            
                        }

                        if(FinalTrackCurObjCls.find(local_obj) == FinalTrackCurObjCls.end() || FinalTrackCurObjCls[local_obj] == 0)
                            assert(false);
                        
                        ptr_obj[-2] = (float)FinalTrackCurObjCls[local_obj];
                        featureTracker.pixel_objs_prev[prev_id] = ptr_obj;
                    }
                    // 如果物体被分割为两个被检测成独立的非背景物体的部分，则均匀合并。
                    else
                    {
                        // 还可能存在被分割第三部分且漏检，则漏检这部分不再放入像素点集了
                        if (iter.second.size() == 3)
                        {
                            int temp = iter.second[2];
                            if(temp != 0)
                            {
                                if(obj_1 == 0)
                                    obj_1 = temp;
                                else
                                    obj_2 = temp;
                            }
                        }
                        local_obj = obj_1;
                        // vector<int>::iterator iter_1, iter_2;
                        auto iter_1 = std::find(valid_detect_obj.begin(),valid_detect_obj.end(),obj_1);
                        auto iter_2 = std::find(valid_detect_obj.begin(),valid_detect_obj.end(),obj_2);
                        int id_1 = distance(valid_detect_obj.begin(),iter_1);
                        int id_2 = distance(valid_detect_obj.begin(),iter_2);
                        correct_obj[id_1] = 1;
                        correct_obj[id_2] = 1;
                        float* ptr_obj_1 = featureTracker.sampled_pixel[id_1] + 2;
                        float* ptr_obj_2 = featureTracker.sampled_pixel[id_2] + 2;
                        int num_pixel_1 = (int)ptr_obj_1[-1];
                        int num_pixel_2 = (int)ptr_obj_2[-1];
                        // 注意，一个语句中定义多个指针变量时，需要在每个变量名前面都加上"*"!!! 而"float* a, b, c;"中，只有a是float型指针，b和c均为float型变量！！！
                        float *ptr_put, *ptr_get, *final_ptr;
                        int num_add;
                        
                        float x, y, z, left, right, top, bottom;
                        float u, v, dep;
                        if(num_pixel_1 + num_pixel_2 <= NUM_SAMPLED_PIXEL_OBJ)
                        {
                            ptr_put = ptr_obj_1 + 3 * num_pixel_1;
                            ptr_get = ptr_obj_2;
                            num_add = num_pixel_2;
                        }
                        else
                        {
                            int min_num = num_pixel_1;
                            ptr_put = ptr_obj_2;
                            ptr_get = ptr_obj_1;
                            final_ptr = ptr_obj_2;
                            if (num_pixel_2 < min_num)
                            {
                                min_num = num_pixel_2;
                                ptr_put = ptr_obj_1;
                                ptr_get = ptr_obj_2;
                                final_ptr = ptr_obj_1;
                            }

                            if(min_num <= NUM_SAMPLED_PIXEL_OBJ/2)
                            {
                                num_add = min_num;
                                ptr_put = ptr_put + 3 * (NUM_SAMPLED_PIXEL_OBJ - num_add);
                            }
                            else
                            {
                                num_add = NUM_SAMPLED_PIXEL_OBJ/2;
                                ptr_put = ptr_put + 3 * (NUM_SAMPLED_PIXEL_OBJ - num_add);
                            }
                        }
                        
                        for(int k = 0; k < num_add; ++k)
                        {
                            ptr_put[(3*k)]   = ptr_get[(3*k)];
                            ptr_put[(3*k+1)] = ptr_get[(3*k+1)];
                            ptr_put[(3*k+2)] = ptr_get[(3*k+2)];
                        }
                        // 这里就不把要覆盖的部分3D点从平均3D点中剔除，直接两部分点的3D点全部相加再取均值即可
                        int final_num = num_pixel_1 + num_pixel_2;
                        // 条件1如果不成立，则条件2不会执行和判断
                        // if(ave_3d_pts_objs.find(prev_id) != ave_3d_pts_objs.end() && ave_3d_pts_objs[prev_id](2) != 0.0)
                        // {
                        //         final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3d_pts_objs[prev_id](0);
                        //         final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3d_pts_objs[prev_id](1);
                        //         final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3d_pts_objs[prev_id](2);
                        // }
                        // else
                        {
                            x = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+4)] * num_pixel_1 + ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+4)] * num_pixel_2;
                            y = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+5)] * num_pixel_1 + ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+5)] * num_pixel_2;
                            z = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+6)] * num_pixel_1 + ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+6)] * num_pixel_2;
                            final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = x/final_num;
                            final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = y/final_num;
                            final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = z/final_num;
                        }

                        // bbox left border
                        float l_min_1 = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ)];
                        float l_min_2 = ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ)];
                        if(l_min_1 == 0 && l_min_2 == 0)
                        {
                            cout << "Weired!" << endl;
                            abort();
                        }
                        else if(l_min_1 == 0)
                            left = l_min_2;
                        else if(l_min_2 == 0)
                            left = l_min_1;
                        else
                            left = std::min(l_min_1, l_min_2);
                        
                        // bbox right border
                        float r_max_1 = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+1)];
                        float r_max_2 = ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+1)];
                        if(r_max_1 == 0 && r_max_2 == 0)
                        {
                            cout << "Weired!" << endl;
                            abort();
                        }
                        else
                            right = std::max(r_max_1, r_max_2);

                        // bbox top border
                        float t_min_1 = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+2)];
                        float t_min_2 = ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+2)];
                        if(t_min_1 == 0 && t_min_2 == 0)
                        {
                            cout << "Weired!" << endl;
                            abort();
                        }
                        else if(t_min_1 == 0)
                            top = t_min_2;
                        else if(t_min_2 == 0)
                            top = t_min_1;
                        else
                            top = std::min(t_min_1, t_min_2);

                        // bbox right border
                        float b_max_1 = ptr_obj_1[(3*NUM_SAMPLED_PIXEL_OBJ+3)];
                        float b_max_2 = ptr_obj_2[(3*NUM_SAMPLED_PIXEL_OBJ+3)];
                        if(b_max_1 == 0 && b_max_2 == 0)
                        {
                            cout << "Weired!" << endl;
                            abort();
                        }
                        else
                            bottom = std::max(b_max_1, b_max_2);

                        final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ)]   = left;
                        final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+1)] = right;
                        final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+2)] = top;
                        final_ptr[(3*NUM_SAMPLED_PIXEL_OBJ+3)] = bottom;

                        if (final_num < NUM_SAMPLED_PIXEL_OBJ)
                            final_ptr[-1] = final_num;
                        else
                            final_ptr[-1] = NUM_SAMPLED_PIXEL_OBJ;

                        if(FinalTrackCurObjCls.find(local_obj) == FinalTrackCurObjCls.end() || FinalTrackCurObjCls[local_obj] == 0)
                            assert(false);
                        final_ptr[-2] = (float)FinalTrackCurObjCls[local_obj];
                        featureTracker.pixel_objs_prev[prev_id] = final_ptr;
                    }
                }
            }
        }
    }
    
    // 所有有效的检测物体的bbox信息和(采样像素点的）平均3D点在GPU中已经获取。
    // 处理剩下的当前帧检测物体的采样像素点，剩下的都是新物体或与上一帧全局物体一对一匹配的
    // cout << "num of detected obj: " << _valid_objs << endl;
    for(int i = 0; i < _valid_objs; ++i)
    {
        // 已经在上面处理过的物体
        if (correct_obj[i] == 1) 
        {
            continue;
        }

        float* ptr_obj = featureTracker.sampled_pixel[i] + 2;
        // GPU采样像素点时此处保存的是该检测物体的临时id
        local_obj = (int)ptr_obj[-2];

        // 当前帧的检测物体是否一定在FinalTrackCurObj中，不论其是否有匹配物体？
        // 一开始只有在objs-matching中找到匹配 的 当前帧检测物体会在FinalTrackCurObj中
        // 但是 上面把 所有没有得到匹配的新物体 也放入其中了
        if(FinalTrackCurObj.find(local_obj) == FinalTrackCurObj.end())
        {
            assert(false);
            continue;
        }
        
        gl_obj = FinalTrackCurObj[local_obj];
        // cout << "final tracked gl obj: " << gl_obj << endl;
        // 当前该物体不是被删除的无效物体（其值为0）
        if(gl_obj > 0)
        {
            if(FinalTrackCurObjCls.find(local_obj) == FinalTrackCurObjCls.end() || FinalTrackCurObjCls[local_obj] == 0)
                assert(false);
            
            // 保存全局cls，uchar强制变为float
            ptr_obj[-2] = (float)FinalTrackCurObjCls[local_obj];
            // cout << "Num of pixel of obj " << (i+1) << "is " << (int)ptr_obj[-1] << endl;
            featureTracker.pixel_objs_prev[gl_obj] = ptr_obj;
            // cout << "num of sampled pixel: " << (int)ptr_obj[-1] << endl;

            // 与上一帧为一对一物体配对，且完成了运动估计。用特征点的平均深度来替换这些物体的像素平均深度
            // if(ave_3d_pts_objs.find(gl_obj) != ave_3d_pts_objs.end() && ave_3d_pts_objs[gl_obj](2) != 0.0)
            // {
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3d_pts_objs[gl_obj](0);
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3d_pts_objs[gl_obj](1);
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3d_pts_objs[gl_obj](2);
            // }
            // // 上一帧完全漏检的物体且在当前帧完成了运动估计
            // else if(ave_3d_pts_objs.find((-1*local_obj)) != ave_3d_pts_objs.end())
            // {
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+4)] = ave_3d_pts_objs[(-1*local_obj)](0);
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+5)] = ave_3d_pts_objs[(-1*local_obj)](1);
            //     ptr_obj[(3*NUM_SAMPLED_PIXEL_OBJ+6)] = ave_3d_pts_objs[(-1*local_obj)](2);
            // }
        }
    }

    // 所有当前帧完全漏检物体的采样像素点的内存位置。注意，在sampled_pixel数组中，当前帧的所有有效物体可能不会完全紧密排列，检测物体和漏检物体中可能有部分是上一帧的遗留的物体信息，这部分没有被覆盖
    for(int j = 0; j < sampled_pixel_lost_obj.size(); ++j)
    {
        cout << "get a detected lost objs in cur frame!" << endl;
        // 无效物体
        if (sampled_pixel_lost_obj[j] == nullptr)
            continue;
        else
        {
            int gl_id = detect_lost_objs_cur[j];
            // cout << "id of tracked gl obj that is not detected in cur frame: " << gl_id << endl;
            // 这些当前帧完全漏检的物体的像素点数组的所有信息均已修改，包括3D特征点的均值
            featureTracker.pixel_objs_prev[gl_id] = sampled_pixel_lost_obj[j];
            sampled_pixel_lost_obj[j] = nullptr;
        }
    }
    
    // 等待主线程中完成根据相机位姿估计结果设置status_sift等变量，并完成静态跟踪点的三角化测量
    while(!tracker_pts_updated)
    {
        usleep(300);
    }
    // 根据status_sift和statusLeftRIght删除和修改featureTracker中的跟踪点
    featureTracker.RemoveOutliers();
    tracker_pts_updated = false;
    
    printf("Finish motion estimation and information correction of objects! It costs %fms \n", t_obj_motion_esti.toc());
}

// VI初始化成功之后，对滑窗内所有帧进行LBA优化（对于双目-IMU而言，首个滑窗VI初始化失败的情况还没考虑，需要完善），并marg某一帧，为下一个滑窗的LBA提供部分变量的残差先验信息
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    // 将vector变量转为ceres需要的C语言数组格式
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        // 局部参数化，即把7维的位姿（3维位置加4维四元数）用更低维度的参数化来求解该优化问题（方便线性化）
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // 位姿用位移+四元数（w在最后）共7维数组来表示每帧IMU坐标系的全局位姿
        // AddParameterBlock用于添加需要被优化估计的变量
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
        {
            // 将IMU的两种bias和机体速度放在一起，9维
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
        }
    }
    // 如果是纯双目，则会固定首帧的位姿（为上一个滑窗优化之后的值），不进行优化
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);
    else
    {
        // added by CRS
        // 这里是直接固定某些视觉跟踪点不足的帧；其实还可以更改H矩阵中对应的部分的值，使其变得很大，这样就可以减小该位姿值的变化
        // TODO：这里是固定该帧的绝对位姿值，能否固定2帧之间相对位姿值的方法？可以将状态变量变为前后2帧的运动值，而不是每一帧的位姿值
        if(!id_frame_const_pose.empty())
        {
            for(const auto iter: id_frame_const_pose)
            {
                problem.SetParameterBlockConstant(para_Pose[iter]);
            }
        }
    }

    // 估计各个相机和IMU之间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        // 认为还需要优化外参，且当前滑窗帧数已满，且系统首帧相机时刻的IMU速度的模长大于0.2？系统初始时刻的位置和速度不是直接设为0吗？
        // 因此这里外参加入LBA不是在系统首个滑窗时进行的？？？（即得确保滑窗内的帧有一定的线速度才能优化）
        // openExEstimation这个变量只在系统重置初始化时被设置过，且=0
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            // 如果要固定某个优化变量的值，则需要在将调用AddParameterBlock将它加入优化变量数组之后就立即执行，应该时默认固定住最新的指定维度的变量
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    // 为什么滑窗首帧IMU的速度小于阈值，就不需要优化时间戳位移了？？
    // 应该说是当当前滑窗内相机的位移比较小时，意味着估计Td的方法就不太管用了。
    // 因为需要根据Td把相机时间戳时刻的图像像素点根据像素点的速度值给线性传播到IMU时刻的假设图像上，然而当物体没有位移或者旋转占主要运动时，则这种用光流线性传播的方法就会误差极大，因此此时不如暂时不优化Td，而是用旧值
    // 因此，这要求IMU的频率必须比较高（最好是大于200hz），这样像素点的时间插值的误差就会比较小！
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    // 添加上一滑窗marg后的先验信息作为部分变量的先验约束。
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        // 从上一帧的先验信息last_marginalization_info中marg后的H和b矩阵，可以计算与相关变量（参与了上一滑窗marg且保留的变量）相关的J_p和r_p，它们直接参与构建当前滑窗的H*deltaX=b的构建
        // 先验信息中的J_p是随着当前滑窗LBA中相关变量的更新而更新，因为形成此J的一部分观测已经被抛弃了，无法知道该部分雅可比的计算公式；
        // 而先验信息中的线性化残差r_p则会跟随变量的更新（设为X）而更新，具体地，保持先验信息中先验残差的线性化点不变，即上一滑窗中优化后参与marg的相关剩余变量的估计值（假设为X'），则新的线性化残差r_p'就是J_p'*（X-X‘）
        // 上面说保持先验残差信息中关于marg的剩余变量的线性化点不变，而这些变量中的某些在当前滑窗中很可能还参与了其他测量残差，则其线性化点会随着优化迭代而更新；
        // 同一个变量在一个系统中的两个部分采用了不同的线性化点，这很可能改变此线性化系统原本的可观性！这可以采用所谓的FEJ算法来解决此问题（即当前滑窗中有关上一滑窗marg剩余变量的测量残差的线性化点均与先验信息中的相同）
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // 往残差数组（每个残差其实就相当于一份观测，先验残差就是广义的观测）中添加成员，首先是上一回marg之后得到的先验残差。与此残差相关的优化变量是由上一帧marg后来确定的
        // 残差项的维度（列数）与相关优化参数的维度要一致
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    if(USE_IMU)
    {
        // 添加滑窗内每两帧之间的预积分残差
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            // TODO：如果某一个IMU的预积分的时长大于10s？就算出现了连续多帧图像掉帧，也不应该持续这么长时间吧？这个阈值会不会太大了？
            // 因为此处优化的是滑窗内的帧，假设某一时刻滑窗已满，且从之前一帧开始物体就在接下去的10s时间内几乎无运动，那么系统根据平均视差就会持续地marg掉次新帧（实际上就是持续地marg最新帧）
            // 最终这10s内的视觉信息在滑窗中就只被归成两个关键帧（即倒数第二帧和当前帧），那么由于运动很小，用纯视觉应该就可以一直得到较好的位姿估计
            // 反而由于时间间隔太大，此两帧之间的预积分就再不值得作为观测参与优化了（虽然这10s内每帧都会调用此函数进行VI的联合BA，但是总归IMU的发散性还是会拉低整体精度）？
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            // 预积分残差的相关优化变量是首尾两帧IMU的位姿、速度、bias
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    // 总的特征点观测数
    int f_m_cnt = 0;
    int feature_index = -1;
    // 通过视觉观测建立特征地图点世界坐标与其观测帧相机（IMU）位姿之间的残差约束
    // 只选择那些至少被4帧相机观测到的图像
    for (auto &it_per_id : f_manager.feature)
    {
        // 需要该点在当前滑窗内至少被连续4帧相机观察到，即连续跟踪3次。但是不要求该点在当前帧仍被跟踪到！
        // 但是，纯双目IMU在滑窗未满时从第2帧起每一帧都进行LBA，那这里前3帧不就都不可能有点参与LBA吗?！是的，没有观测约束时，就不会有因子图中的edge，也就不会对变量进行优化！那下面不是在做无用功吗...加个if那么难吗...
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < TH_NUM_FRAME_FOR_LBA || it_per_id.estimated_depth <= 0)
            continue;

        // 记录提供约束参与BA优化的特征点的序号，用于在para_Feature结果变量中索引
        ++feature_index;
        // 记录该点参与了LBA
        it_per_id.has_LBA = true;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 该地图点在被观测首帧中的归一化平面坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 该点的各帧观测（包括右图像中的观测） 与 首观测帧左图像之间的 约束，之所以必须与首帧左图像建立约束，是因为地图点的深度表达在首帧左相机坐标系下！
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                // 利用各帧计算出来的归一化平面上的特征点的速度（后一帧的该点坐标-前一帧的该点坐标），来线性推断与其绑定的IMU时间戳对齐的真实图像中的该特征点的坐标（此为真实观测）
                // 然后使用待估计的该点逆深度（在第i帧下的）、相机和IMU的外参、第i帧相机的全局位姿、第j帧的相机的全局位姿以及相机和IMU的time shift 来得到该地图点在第j帧相机中的归一化坐标（此为理论观测）
                // 根据估计变量的当前估计值不断更新 观测残差，更新残差对各变量的雅可比
                // TwoFrame指两帧左图像，OneCam指都是观测都在左相机
                // velocity指每一帧的中的特征观测点的速度（ （当前帧该特征点（归一化平面）坐标 - 上一帧中的匹配特征点的坐标）/两镇之间的时间差dt），此速度值用于推断两帧附近的插值图像中对应匹配点的坐标
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    // TwoFrame指 观测匹配在 首帧i的左图像和 第j帧的右图像，TwoCam指用到了左相机和右相机。创建测量因子时需要给定观测量 以及 一些待估计参数的初始值（这不是必须的，取决于残差和雅可比计算的需要）
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    // 观测首帧的左图像和右图像之间的约束，OneFrame指在同一相机帧（时刻）
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }
    // 参与LBA的点一共提供了多少视觉约束
    printf("visual measurement count: %d\n", f_m_cnt);
    // ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    // 为什么marg掉老的关键帧时会减小LBA优化时间?
    if (marginalization_flag == MARGIN_OLD)
        // config文件中SOLVER_TIME给定为0.08，即80ms
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    // 根据设置options求解此最小二乘问题problem，求解过程的总结存放在summary中
    // 其内部其实只是定义了求解最小二乘问题的方法（一般是高斯牛顿法或LM），最后其实都是进行非线性残差的线性化和线性残差方程AX=B的迭代更新
    // 而测量残差的计算方式、用于线性化的雅可比矩阵、线性化残差的更新方式都是在我们自己定义的各种残差类（Ceres::CostFunction的子类）中所规定的，一般都是其中的Evaluate()成员函数
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    printf("Iterations : %d\n", static_cast<int>(summary.iterations.size()));
    // ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    // 将LBA优化后得到的变量结果（数组形式）赋值给当前帧滑窗中的Rs、Ps、td等变量集
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    // 如果当前的LBA还不是对于整个滑窗的优化，则到此已完成任务。这只会发生在纯双目系统在首个滑窗还未满帧之前，即每完成一帧的跟踪，就与先前帧一起进行BA
    if(frame_count < WINDOW_SIZE)
        return;
    
    // 下面是针对当前滑窗满帧的情况，则需要marg操作来去除某一帧，并且得到剩下某些变量的先验约束，用于下一滑窗内这些变量的LBA！
    // 注意，当前滑窗marg后形成的先验信息，不会对当前帧的变量的估值再产生影响（因为下面的代码中没有再调用double2vector()函数来更新Rs等变量集），而是直接成为下一滑窗相关变量的优化约束！
    
    // marg掉滑窗首帧的变量以及与这些变量相关的所有测量
    // 因为只选择了那些跟被marg变量相关的测量，所以marg后得到的先验信息H、b是与其他测量没有关系的。下一帧中这些无关的测量按照正常的方式组建H、b和更新相关的变量、J和r。
    if (marginalization_flag == MARGIN_OLD)
    {
        // 是否要进行marg
        if(use_Marg)
        {
            TicToc t_whole_marginalization;

            vector2double();

            // 创建当前滑窗marg之后形成的先验信息。创建此对象时默认其成员变量valid=true
            // 先验信息类中包含了 HX=b中的H和b，由于H=J‘J。b=-J'r，可以求解出与先验信息相关的雅可比J和残差r，这部分需要与下一帧的新J和r累积（新的J和r来自于新的帧与上一帧剩余变量之间的约束）
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            
            // 上一帧是否有先验残差保留下来
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                // 需要被marg的变量在该residual_block的所有相关变量中的序号
                vector<int> drop_set;
                // 取出上一帧marg后剩余的信息中属于当前要被marg帧的部分（即当前帧滑窗的首帧，其与上一个滑窗被marg的帧之间可能有关联，则其会出现在上一滑窗marg后的相关信息中）
                // 被marg的变量会通过与其他帧的相关变量的联系（视觉共视观测或者IMU）来形成新的先验，同时继承自上一帧的先验残差块也会更新（因为与其相关的变量要被marg了，即先验本身也是一种观测形成的！）！
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                        last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 从上一个滑窗继承的先验信息因子，该类继承自ceres::CostFunction，其中会记录残差的维度，优化变量个数以及各个变量的维度大小
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                // 从上一个滑窗继承的先验信息所形成的残差块，给定相关的优化变量信息，以及其中需要marg的变量
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                            last_marginalization_parameter_blocks,
                                                                            drop_set);
                // 将上一帧的先验残差放进当前帧即将形成的先验信息中
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            if(USE_IMU)
            {
                // 这里是与上面进行LBA时的设置一致，即如果某两帧之间的IMU积分时间过长，则不把期间的IMU观测作为约束加入到LBA中。如果首2帧之间的IMU观测没有参与优化，则这里自然也不会加入首帧的marg操作来形成先验信息
                // 大于10s的情况应该就是上面所说的物体长时间不运动（或者运动非常小），此时选择抛弃此段预积分
                if (pre_integrations[1]->sum_dt < 10.0)
                {
                    // IMUFactor也是个CostFucntion类
                    IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                    // 当前滑窗首帧和第二帧之间的IMU预积分因子，同样给定了残差相关的变量和待marg的变量。先验信息就是由观测提供的，它是观测附加在相关变量当前估计值上的约束
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                            vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                            vector<int>{0, 1});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
            
            // 寻找与被marg帧相关的特征点
            {
                int feature_index = -1;
                for (auto &it_per_id : f_manager.feature)
                {
                    // 必须与上面进行LBA时一致，只取观测帧数至少4帧的特征点
                    it_per_id.used_num = it_per_id.feature_per_frame.size();
                    // todo: 如果某个点参与了LBA，但是优化后深度值变为了负值，那么还要marg该点吗？按照这里的条件，这样的点就不参与marg了，否则会形成负面的先验信息
                    // 这样的地图点后续是直接删除
                    if (it_per_id.used_num < TH_NUM_FRAME_FOR_LBA || it_per_id.estimated_depth <= 0)
                        continue;
                    // 这里的feature_index是对参与当前滑窗LBA的所有特征点的索引序号，所以每次寻找到一个都必须+1
                    ++feature_index;

                    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                    // 只选择那些观测初始帧是滑窗首帧的点，因为首帧的变量要被marg掉，与其相关的测量约束也要marg。
                    if (imu_i != 0)
                        continue;

                    // 特征在首帧左相机的归一化平面坐标
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    for (auto &it_per_frame : it_per_id.feature_per_frame)
                    {
                        imu_j++;
                        // 非首帧
                        if(imu_i != imu_j)
                        {
                            Vector3d pts_j = it_per_frame.point;
                            // 与此重投影残差相关的变量是两帧的位姿，左相机与IMU的外参（即名字中的OneCam)，另外还有td，这与残差类型的名字相对应
                            // marg阶段的视觉观测残差仍然需要td的参与，因为与优化的IMU的时刻对应的相机观测需要通过真实视觉观测和td来推断
                            // 因此marg得到的先验信息中是包含了对 td的先验约束的！！
                            // 如果相机与IMU之间的外参不参与优化，则这里marg时还需要形成对它的先验信息吗（如果形成了，不就说明它还是一个随机变量吗）？
                            ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                            vector<int>{0, 3});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        // 右图像的观测的约束
                        if(STEREO && it_per_frame.is_stereo)
                        {
                            Vector3d pts_j_right = it_per_frame.pointRight;
                            if(imu_i != imu_j)
                            {
                                ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                            vector<int>{0, 4});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                            // 首帧的左右相机之间的特征点重投影约束，这与两个相机的全局位姿均无关，只与左右相机之间的外参（需要通过它们各自与IMU的外参来计算）相关
                            else
                            {
                                ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                            vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                            vector<int>{2});
                                marginalization_info->addResidualBlockInfo(residual_block_info);
                            }
                        }
                    }
                }
            }

            TicToc t_pre_margin;
            marginalization_info->preMarginalize();
            printf("pre marginalization %fms\n", t_pre_margin.toc());
            // ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
            
            TicToc t_margin;
            marginalization_info->marginalize();
            printf("marginalization %fms\n", t_margin.toc());
            // ROS_DEBUG("marginalization %f ms", t_margin.toc());

            // 每个变量以数组的形式存储，这些变量当前帧的全局地址（即数组地址）被用来作为这些变量唯一的识别码（即key），而其真实存储地址则是由double*指出（所有没有被marg的变量在其全局数组中都往前移动）
            std::unordered_map<long, double *> addr_shift;
            // 当前滑窗marg首帧的变量，则剩余变量的真实存储地址会往前移动首帧的变量的总size长度；
            // 而在当前滑窗marg后的先验信息中，剩余变量的唯一标识码(key)没必要改变（因为这些地址虽然被别的变量占据着，但是它们在下一帧的先验信息中不需要被索引），我们只需要把该各剩余变量移动后的真实地址赋给它所对应的value即可！在下一滑窗进行marg时，我们会对新的参与marg的变量更新其key（即其真实的存储地址）
            // TODO：有没有必要把所有的变量都放进这个临时map中？可不可以在getParameterBlocks函数中根据参与marg的剩余变量在数组中的id，其新的真实地址不就是当前的地址直接减去1个变量的长度？甚至外参和Td的内存地址都是不变的！
            for (int i = 1; i <= WINDOW_SIZE; i++)
            {
                addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                if(USE_IMU)
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            // 当前滑窗marg后剩余变量（每个变量是一个数组，用double*来指明它的存储地址）的索引key和新的存储地址（即在下一个滑窗中该变量所在的地址）
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            
            // 清除上一个滑窗的先验信息的占用内存
            if (last_marginalization_info)
                delete last_marginalization_info;
            // 将当前滑窗的先验信息留给下一个滑窗进行操作
            last_marginalization_info = marginalization_info;
            // 当前滑窗的先验信息所对应的（剩余）变量在下一滑窗中的存储地址
            last_marginalization_parameter_blocks = parameter_blocks;

            printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
        }
        else
        {
            // 每个变量以数组的形式存储，这些变量当前帧的全局地址（即数组地址）被用来作为这些变量唯一的识别码（即key），而其真实存储地址则是由double*指出（所有没有被marg的变量在其全局数组中都往前移动）
            std::unordered_map<long, double *> addr_shift;
            // 当前滑窗marg首帧的变量，则剩余变量的真实存储地址会往前移动首帧的变量的总size长度；
            // 而在当前滑窗marg后的先验信息中，剩余变量的唯一标识码(key)没必要改变（因为这些地址虽然被别的变量占据着，但是它们在下一帧的先验信息中不需要被索引），我们只需要把该各剩余变量移动后的真实地址赋给它所对应的value即可！在下一滑窗进行marg时，我们会对新的参与marg的变量更新其key（即其真实的存储地址）
            // TODO：有没有必要把所有的变量都放进这个临时map中？可不可以在getParameterBlocks函数中根据参与marg的剩余变量在数组中的id，其新的真实地址不就是当前的地址直接减去1个变量的长度？甚至外参和Td的内存地址都是不变的！
            for (int i = 1; i <= WINDOW_SIZE; i++)
            {
                addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                if(USE_IMU)
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }

            // 2维数组，第一个索引[]是指向第几行的一维数组指针
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
    }
    else
    {
        if(use_Marg)
        {
            // 对于marg掉次新帧的情况，这里默认直接去掉次新帧中的所有变量及其相关视觉测量，而不增加新的关于视觉残差marg的先验信息（认为次新帧的观测所提供的信息不重要？）；
            // 而与次新帧相关的前后两个IMU预积分则会直接合并为一个；
            // 最后如果上一个滑窗的先验信息中与当前次新帧（即上一个滑窗的最新帧/尾帧）变量相关（即上一个滑窗marg了首帧（如果上一滑窗也marg次新帧，根据下面的操作，省略掉视觉测量的marg，则它的先验信息不可能与其最新帧有关），且其首帧和最后一帧建立了视觉联系（不可能有其他类型测量能让滑窗首帧和尾帧关联起来！））
            if (last_marginalization_info &&
                std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
            {
                TicToc t_whole_marginalization;
                MarginalizationInfo *marginalization_info = new MarginalizationInfo();
                vector2double();
                if (last_marginalization_info && last_marginalization_info->valid)
                {
                    vector<int> drop_set;
                    for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                    {
                        assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                        // ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                        if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                            drop_set.push_back(i);
                    }
                    // construct new marginlization_factor
                    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                                last_marginalization_parameter_blocks,
                                                                                drop_set);

                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }

                TicToc t_pre_margin;
                printf("begin marginalization\n");
                // ROS_DEBUG("begin marginalization");
                marginalization_info->preMarginalize();
                printf("end pre marginalization, %fms\n", t_pre_margin.toc());
                // ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

                TicToc t_margin;
                printf("begin marginalization\n");
                // ROS_DEBUG("begin marginalization");
                marginalization_info->marginalize();
                printf("end marginalization, %fms\n", t_margin.toc());
                // ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
                
                std::unordered_map<long, double *> addr_shift;
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    if (i == WINDOW_SIZE - 1)
                        continue;
                    else if (i == WINDOW_SIZE)
                    {   
                        // 当前滑窗最新一帧的pose变量的存储地址往前移动，下面对IMU的其他参数同
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                        if(USE_IMU)
                            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                        if(USE_IMU)
                            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                    }
                }
                for (int i = 0; i < NUM_OF_CAM; i++)
                    addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

                
                vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
                if (last_marginalization_info)
                    delete last_marginalization_info;
                last_marginalization_info = marginalization_info;
                last_marginalization_parameter_blocks = parameter_blocks;
                
                printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
            }
        }
        else
        {
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {   
                    // 当前滑窗最新一帧的pose变量的存储地址往前移动，下面对IMU的其他参数同
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }

            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
    }

    cout << "Optimization finished!" << endl;
    
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        // 当前滑窗首帧的时间戳
        double t_0 = Headers[0];
        // 保留当前滑窗首帧IMU的估计位姿
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // 第一个滑窗满了才会持续地滑帧
        if (frame_count == WINDOW_SIZE)
        {
            // 数组中的所有元素的存储位置往前移一位
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                // swap和直接copy赋值的区别是什么？Eigen的交换是只()中的赋给括号外的，还是两者真的会交换？
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    // 交换数组中两个元素的值。交换时需要一个临时的同类型对象的内存空间，用于临时保存其中一个元素的值
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    // 保存的是滑窗内各个帧（11帧）与其上一帧之间的IMU测量，因此系统首个滑窗的首帧的对应元素应该是空的vecctor<Vector3d>
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            // 下一帧的这些量的初始帧为当前滑窗最后一帧的值
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                // 创建下一帧的预积分对象，其首个IMU测量相对的坐标系是当前滑窗的最后一帧IMU坐标系，预积分的左端点测量数据就是当前滑窗读入的最后一个
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
                // 这里的clear是清除指定内存位置上的数据。如果该位置上是指针对象，则只会清除该指针，而不会清理指针所指向的对象的内存！
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 一定要执行？第二个条件的意义？
            // 第二个条件此时没有意义。
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                // all_image_frame保存了当前滑窗内各帧的内容，包括该帧的预积分、特征观测（包括和上一帧的匹配，当前帧的新检测）、图像时间戳等
                it_0 = all_image_frame.find(t_0);
                // 必须delete掉ImageFrame对象中new出来的预积分对象！
                if(it_0->second.pre_integration != nullptr)
                {
                    delete it_0->second.pre_integration;
                    it_0->second.pre_integration = nullptr;
                }
                // 清理map中指定的[begin, last)之间的所有元素，因为同样不会delete对象中的指针成员所指的对象，因此需要先人为清理，避免内存泄漏！
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }

            // 阻塞，等待物体更新中将新的地图点的观测加入地图
            while(!map_fea_writen)
            {
                usleep(300);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                // 将当前滑窗倒数第2个和最后一个预积分（即次新帧前后的两个预积分）融合成一个
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }
                
                Vs[frame_count - 1] = Vs[frame_count];
                // bias为什么直接取当前滑窗最后一帧处的bias？这个其实是作为下一帧预积分的bias的初始值的，如下面的预积分对象创建所示
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                // 清空接下来新一帧的IMU测量序列
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 阻塞，等待物体更新中将新的地图点的观测加入地图
            while(!map_fea_writen)
            {
                usleep(300);
            }
            slideWindowNew();
        }
    }
}

// 需要marg掉当前滑窗中的次新帧，前面已经做好了跟Frame相关的变量中元素的地址迁移，以及最后两个预积分的融合。这里只需要再处理将受到marg帧影响的地图点进行处理
void Estimator::slideWindowNew()
{
    // 这个量和下面的sum_of_back一样没被用到
    sum_of_front++;
    // 将最新帧中的新检测的特征地图点的首观测帧id减1；对于被次新帧观测到的点，去除该观测记录，如果去除后就没有了观测帧（或者只剩一帧且不是最新帧？）则删除掉该地图点
    f_manager.removeFront(frame_count);
}

// 需要marg掉当前滑窗中的首帧，前面已经做好了跟Frame相关的变量中元素的地址迁移，这里只需要再处理将受到marg帧影响的地图点进行处理
void Estimator::slideWindowOld()
{
    // 记录从首个滑窗满帧开始，一共进行了多少次marg最旧帧。当sum_of_back = WINDOW_SIZE+1时，则首个滑窗的所有帧就都已经全部（要）被marg掉了！
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    // VI初始化之后，所有地图点的深度都调整好了。此时如果要marg掉最老的帧，则需要改变那些初始观测帧在最老帧的点的深度
    // 将这些点的的初始观测帧改为当前滑窗的第2帧，计算其在第二帧左图像中的深度作为其深度估计
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // 当前窗口首帧和第2帧的相机坐标系到世界坐标系的变换
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        // 去除初始观测帧在滑窗首帧的点在首帧的观测记录
        // 原始的VINS代码中意味着：如果该点的观测帧至少还有两帧，则计算它在剩下的首观测帧（这里默认就是当前滑窗第2帧，因为VINS不存在某地图点有跟踪期间暂时中断的情况）下的深度；
        // 否则，将该地图点直接从地图集合中去除。对于初始观测帧不是滑窗首帧的地图点，将初始观测帧的滑窗内全局id减1即可。
        // VINS只有当滑窗满了时才会调用这里的slidewindow函数，而如果marg最旧帧后就只剩下不超过4帧观测，则意味着该点永远不可能参与LBA了（至少需要连续4帧观测）和marg了，那还保留它干吗？
        // 注意，一个点是否参与LBA与其是否参与marg是等同的！因此一旦某个点不会参与LBA，则可以直接将其删除了！
        // 因此一个点一旦被marg过，则它所有的信息就都转化到其他帧相关变量（位姿和IMU参数）的先验约束上！这个点就应该彻底从点云中被删除了！！因为它无法再提供别的信息了！
        // 那如果某点在当前首帧被marg后剩下的观测帧数还大于等于4帧观测呢？如果还保留在地图中，则下一滑窗岂不是还继续参与LBA和marg？是的，marg掉某一帧只是marg该帧位姿和一些特征点在该帧的观测，但这些点仍然是可以在地图中并被剩余的帧所观测的！
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);

        // f_manager.removeBackShiftDepth();
    }
    else
        // 如果还没有VI初始化成功，则认为地图点的深度还没有计算和校正，与上面的函数操作类似，只是不需要计算点在剩余观测首帧中的深度
        f_manager.removeBack();
}

// 得到最新帧的IMU坐标系在参考世界坐标系中的位姿
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T, bool pose_cam)
{
    T = Eigen::Matrix4d::Identity();
    if (!pose_cam)
    {
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
    }
    else
    {
        T.block<3, 3>(0, 0) = Rs[frame_count] * ric[0];
        T.block<3, 1>(0, 3) = Rs[frame_count] * tic[0] + Ps[frame_count];
    }
}

// 得到指定id的帧的IMU坐标系在参考世界坐标系中的位姿
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T, bool pose_cam)
{
    T = Eigen::Matrix4d::Identity();
    if (!pose_cam)
    {
        T.block<3, 3>(0, 0) = Rs[index];
        T.block<3, 1>(0, 3) = Ps[index];
    }
    else
    {
        T.block<3, 3>(0, 0) = Rs[index] * ric[0];
        T.block<3, 1>(0, 3) = Rs[index] * tic[0] + Ps[index];
    }
}

// 预测当前帧中的特征地图点在下一帧中的可能坐标
// 可以跟ORB-SLAM3一样，在VI初始化之后，使用IMU的测量数据来推断下一帧的位姿和速度
// ！！此函数不应该在当前帧被调用，而是应该在下一帧处理的一开始，那样才能获得下一帧和当前帧之间的时间间隔dt！
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    // 当前最新帧至少需要是滑窗的第三帧。实际上对于有IMU的系统，必须要在VI初始化成功之后才会调用此函数。并且此函数是在当前滑窗完成LBA之后才会调用！
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion (and constant time interval)
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    // 采用恒速模型，但是此处还假设了每两帧图像之间的时间间隔是相同的！IMU的速度就是相机的速度
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            // 只选那些被连续跟踪（至少被跟踪一次）到了当前最新帧的地图点。
            // 当前帧中新检测且有深度估计（通过左右图像的匹配和三角化）的特征地图点后续是直接使用当前帧的位置作为预测值。但是为什么不能通过恒速模型来预测其下一帧的位置？
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                // 预测的是在下一帧（还得时间差与当前帧的相同）左相机坐标系下的3D点坐标
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

// 计算第i帧的特征点重投影到第j帧之后的重投影误差，depth是该地图点在第i帧相机中的深度估计值，rici是第i帧的相机和其IMU之间的外参，uvi是观测点在第i帧相机的归一化平面上的坐标
// 当计算的是在某一帧的右图像上的重投影误差时，rici/tici和ricj/ticj就会不同，因为代表的是左右相机各自与IMU的外参
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

// 根据参与LBA优化后的各帧位姿和各点（在其首观测帧下的）深度，计算各点在其首观测帧与其他观测帧之间的重投影误差之和，如果大于阈值，则作为外点去除
// 对于其中在当前帧被跟踪到的sift点，可以选择将其保留为新的sift检测点（但是仍将该点从地图中删除，至于是否将该点再加入地图，取决于它在下一帧是否被跟踪到）
void Estimator::outliersRejection(set<int> &removeIndex)
{
    // 首先对当前帧的所有静态内点的深度进行更新！把更新得到的深度不符合要求的点全部加入删除行列！
    // 注意，计算的是当前帧图像时间戳时刻的点的深度，因此如果是VI初始化之后的LBA计算出的当前帧位姿，则需要先插值得到图像时间戳对应时刻的相机位姿！
    if(USE_IMU && td != 0) intepolate_pose(frame_count, prevTime, curTime, curTime-prev_td+td, cur_cam_R, cur_cam_P); 
    
    Matrix3d motion_R = cur_cam_R.transpose() * prev_cam_R;
    Vector3d motion_P = cur_cam_R.transpose() * (prev_cam_P - cur_cam_P);

    vector<int> &id_sift_no_depth = featureTracker.id_sift_no_depth;    
    vector<int> &id_FAST_no_depth = featureTracker.id_FAST_no_depth;
    vector<int>::iterator sift_no_dep_iter_end = id_sift_no_depth.end();
    vector<int>::iterator FAST_no_dep_iter_end = id_FAST_no_depth.end();
    vector<int> &sift_no_stereo_bg = featureTracker.sift_no_stereo_bg;
    vector<int> &FAST_no_stereo_bg = featureTracker.FAST_no_stereo_bg;
    vector<int>::iterator sift_no_stereo_iter_end = sift_no_stereo_bg.end();
    vector<int>::iterator FAST_no_stereo_iter_end = FAST_no_stereo_bg.end();

    map<int,int> &gl_id_index_map = featureTracker.gl_id_index_map;
    map<int, Vec4f> &prev_un_Fea_map = featureTracker.prev_un_Fea_map;
    vector<float> &cur_dep_FAST = featureTracker.cur_FAST_dep;
    vector<float> &cur_dep_sift = featureTracker.cur_sift_dep;
    vector<float> &prev_dep_FAST = featureTracker.prev_FAST_dep;
    vector<float> &prev_dep_sift = featureTracker.prev_sift_dep;
    vector<uchar> &statusLeftRIght = featureTracker.statusLeftRIght;
    vector<uchar> &status_sift = featureTracker.status_sift;
    vector<pair<uchar,int>> &obj_cls_id_FAST = featureTracker.obj_cls_id_FAST;
    vector<pair<uchar,int>> &obj_cls_id_sift = featureTracker.obj_cls_id_sift;

    int pt_id, index;
    Vector3d pt_prev, pt_cur;
    float cur_z, max_depth;

    int feature_index = -1;
    bool valid;
    double err, ave_err;
    int errCnt;
    uchar cls;

    Matrix3d motion_R_update, first_frame_R;
    Vector3d motion_P_update, first_frame_P;
    float prev_dep;
    int frame_first;
    bool use_prev_dep;
    
    // 如果当前帧的sift新点是根据下一帧的跟踪情况添加，则当前帧背景中的sift跟踪外点不再保留为新点
    bool add_new_sift_in_next_frame = featureTracker.add_new_sift_in_next_frame;

    vector<int> erase_map_pt;

    //int &num_track_fea_stat = featureTracker.num_track_fea_stat;
    for (auto &it_per_id : f_manager.feature)
    {
        valid = true;
        pt_id = it_per_id.feature_id;
        
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 首先计算长跟踪点的平均重投影误差。只考察那些在当前滑窗内至少被连续观测4帧的地图点！这与LBA中的标准是一样的，即只有这些点参与了LBA
        // 参与LBA的点还必须是有深度估计的,即首帧下的点深度三角化成功了
        if (it_per_id.used_num >= TH_NUM_FRAME_FOR_LBA && it_per_id.estimated_depth > 0)
        {
            err = 0;
            errCnt = 0;
            // 当前滑窗内保留的地图点计数是从0开始的
            ++feature_index;
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            double depth = it_per_id.estimated_depth;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                ++imu_j;
                // 计算该地图点在其观测首帧和其他各观测帧之间的重投影误差
                if (imu_i != imu_j)
                {
                    // 归一化平面坐标
                    Vector3d pts_j = it_per_frame.point;             
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                        depth, pts_i, pts_j);
                    err += tmp_error;
                    ++errCnt;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                // need to rewrite projecton factor.........这个在optimization()函数中实现了，即每个地图点的观测首帧的左图像和其他各帧左图像以及所有观测帧的右图像的观测进行重投影误差约束！
                // STEREO是多余的吧？单目-IMU系统每一帧的is_stereo也不会是true呀？
                if(STEREO && it_per_frame.is_stereo)
                {
                    Vector3d pts_j_right = it_per_frame.pointRight;
                    if(imu_i != imu_j)
                    {   
                        // 需要给定IMU和右相机之间的外参ric[1], tic[1]
                        // 不计算同一帧下左右匹配的重投影误差吗？         
                        double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                            Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                            depth, pts_i, pts_j_right);
                        err += tmp_error;
                        ++errCnt;
                        //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                    }
                    // 没必要把将观测首帧的左右图像重投影的情况单独列出来吧?
                    // else
                    // {
                    //     double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                    //                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                    //                                         depth, pts_i, pts_j_right);
                    //     err += tmp_error;
                    //     ++errCnt;
                    //     //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                    // }       
                }
            }

            ave_err = err / errCnt;
            
            
            // 为什么要乘以相机的焦距？ave_err是在归一化平面上的距离误差，乘以焦距以后就是在图像上的像素误差！
            if(ave_err * FOCAL_LENGTH_X > 3)
            {
                valid = false;
                
                // 如果该点是物体点或者背景的sift点，且在当前帧被跟踪到，则可以保留当该点在当前帧的观测作为新点。修改该点的信息留到“物体运动估计”的子线程中进行
                
                // 如果当前帧观测已经在地图中
                if (it_per_id.endFrame() == WINDOW_SIZE)
                {
                    // 加入removeIndex的点，其已加入地图中的现有观测都会被删除
                    removeIndex.insert(pt_id);

                    index = gl_id_index_map[pt_id];
                    if(index <= 0)
                    {
                        index = -1 * index;
                        // 深度不可靠的静态物体点跳过.
                        // 背景点在每一帧中不一定会有stero match（这部分如果被保留为新背景点，那么就不会有深度估计值，如果它下一帧能被跟踪到，则需要使用三角化来得到其首帧下的深度值）
                        if(obj_cls_id_sift[index].second > 0)
                        {
                            // 物体点只保留有立体匹配的点
                            // if(std::find(id_sift_no_depth.begin(), id_sift_no_depth.end(),index) == sift_no_dep_iter_end)
                            if(status_sift[index] == 1)
                                reserve_new_sift.push_back(pt_id);
                            else
                                direct_erase_fea.push_back(pt_id);
                        }
                        else if(!add_new_sift_in_next_frame)
                        {
                            // 如果要保留这些背景点为新点，则也只选择其中有立体匹配的
                            if(status_sift[index] == 1) 
                                reserve_new_sift.push_back(pt_id);
                            else
                                direct_erase_fea.push_back(pt_id);
                        }
                    }
                    else
                    {
                        index -= 1;
                        // 对于FAST点，只保留其中静态物体的有深度值的FAST点为新点？算了，就直接放弃这些点吧
                        if(obj_cls_id_FAST[index].second > 0)
                        {
                            // if(std::find(id_FAST_no_depth.begin(), id_FAST_no_depth.end(),(index)) == FAST_no_dep_iter_end)
                            if(statusLeftRIght[index] == 1)
                                reserve_new_FAST.push_back(pt_id);
                            else
                                direct_erase_fea.push_back(pt_id);
                        }
                        else
                            direct_erase_fea.push_back(pt_id);
                    }
                    // 去除当前帧该点的观测？这不仅仅该点当前帧观测的问题，而是该点应该被完全删除？
                    // it_per_id.feature_per_frame.pop_back();
                    //--num_track_fea_stat;
                }
                // 如果上一帧观测还未加入地图，则一定是物体点
                // 如果该物体静态点在当前帧仍被跟踪到，但是最新观测还没加入地图。则这种情况是该静态物体在当前帧需要后续运动估计才能确认是否仍为静态。这里需要记录其可能被删除的观测点记录，以便后续不往地图中添加观测记录，而是作为当前帧的新特征点
                // 对于这些最新帧观测还未加入地图的点,使用direct_erase_fea和reserve_new_sift来决定其是否要作为新点加入地图
                // else if(it_per_id.endFrame() == WINDOW_SIZE-1 || it_per_id.endFrame() == WINDOW_SIZE-2)
                else if(it_per_id.endFrame() == WINDOW_SIZE-1)
                {   
                    // 如果当前帧没跟踪到该点，则直接从地图中删除该点
                    if (gl_id_index_map.find(pt_id) == gl_id_index_map.end()) 
                    {
                        removeIndex.insert(pt_id);
                        continue;
                    }

                    // 如果是当前帧观测还未加入的（之前静态点），则直接删除该点在地图中的记录。在物体运动估计函数中如果当前帧该点属于运动估计内点，则将其作为新的跟踪点加入地图（在地图中找不到该点之前的记录，则会创建新的地图点）
                    
                    // index = gl_id_index_map[pt_id];
                    // if(index <= 0)
                    // {
                    //     index = -1 * index;
                        
                    //     // 深度不可靠的点在当前帧的跟踪点后续也直接删除（这种情况下就不考虑上一帧未加入地图的该点观测的深度是否可靠了，直接从当前帧观测考虑是否改为新点）
                    //     // 没有立体匹配的背景跟踪点 或 物体跟踪点（这里指的是用sift匹配或者CPU光流得到的右图像匹配点，而不是用flow map得到的近似匹配点）
                    //     if (obj_cls_id_sift[index].second > 0)
                    //     {
                    //         auto iter1 = std::find(id_sift_no_depth.begin(), id_sift_no_depth.end(), index);
                            
                    //         if(iter1 == sift_no_dep_iter_end)
                    //             reserve_new_sift.push_back(pt_id);
                    //         else
                    //             direct_erase_fea.push_back(pt_id);
                    //     }
                    //     else if(!add_new_sift_in_next_frame)
                    //     {
                    //         auto iter2 = std::find(sift_no_stereo_bg.begin(),sift_no_stereo_bg.end(),index);
                    //         if(iter2 == sift_no_stereo_bg.end())
                    //             reserve_new_sift.push_back(pt_id);
                    //         else
                    //             direct_erase_fea.push_back(pt_id);
                    //     }
                    // }
                    // else
                    // {
                    //     index -= 1;
                    //     if (obj_cls_id_FAST[index].first > 0)
                    //     {
                    //         auto iter1 = std::find(id_FAST_no_depth.begin(), id_FAST_no_depth.end(), index);
                            
                    //         if(iter1 == FAST_no_dep_iter_end)
                    //             reserve_new_FAST.push_back(pt_id);
                    //         else
                    //             direct_erase_fea.push_back(pt_id);
                    //     }
                    //     else
                    //     {
                    //         direct_erase_fea.push_back(pt_id);
                    //     }
                    // }

                    // 如果该点在当前帧被跟踪到，但是又还未加入地图，则在此处直接将该点从地图中删除。后续该点在当前帧的跟踪是否有效取决于 物体运动估计函数中的判断
                    erase_map_pt.push_back(pt_id);

                    continue;

                    //--num_track_fea_stat;
                }
                else
                {
                    // 加入removeIndex的点，其已加入地图中的现有观测都会被删除
                    removeIndex.insert(pt_id);
                }
            }
        }
        else
        {
            valid = false;
            // 如果该点在参与LBA之后，其首帧下的深度值变为负的，则删除该地图点
            if(it_per_id.used_num >= TH_NUM_FRAME_FOR_LBA && it_per_id.has_LBA)
            {
                if(it_per_id.endFrame() == frame_count)
                    removeIndex.insert(pt_id);
                else
                {
                    // 删除该地图点的操作是否要在这里进行？
                }
            }
        }
        
        // 如果不需要被（从地图中）完全删除或者修改为新点，且当前帧跟踪到该点且加入了地图，则更新其中的静态物体点在当前相机坐标系的深度
        // 这里是只更新那些参与了LBA，且在当前帧仍然被跟踪到的点的深度
        if(valid)
        {
            // VI初始化之后，当前帧在物体关联阶段就确定为静态的物体的特征点，只有那些观测帧数大于等于3帧的点才会被加入地图；
            // VI初始化之前，所有静态物体的跟踪点都会被加入地图，用于进行每一帧的相机位姿的估计（以便PnP的点数足够多）
            // 那些当前帧还没加入地图的静态点如何更新深度值？一方面是那些物体关联阶段未确定为静态的物体，另一方面是那些VI初始化后仅有2帧观测的静态物体点。这些统一都在物体运动估计线程中去进行深度更新
            // 这里对那些参与了LBA的静态物体点也进行深度更新
            if(it_per_id.endFrame() == WINDOW_SIZE)
            {
                index = gl_id_index_map[pt_id];
                if(index > 0)
                {
                    index -= 1;
                    cls = obj_cls_id_FAST[index].first;
                    // 对于静态物体点，只更新那些用depth_map获取当前帧深度估计的点的深度。最后决定对当前帧的点都更新深度值，因为参与LBA的点在三角化测量函数中没有更新当前帧的深度值
                    // if(cls > 0 && std::find(id_FAST_no_depth.begin(), FAST_iter_end, index) == FAST_iter_end) continue;
                    // 对于背景点，更新那些在当前帧没有stereo match的点的深度（方便为该点在下一帧的位置提供预测值）
                    // if(cls == 0 && statusLeftRIght[index] != 2) continue; 

                    // prev_dep = prev_dep_FAST[index];
                    // if(cls > 0 && prev_dep <= 0)
                    //     assert(false && "Something wrong with obj fea depth in prev_dep_FAST!");
                    
                    // 如果该点为静态物体点， 或者为背景跟踪点但 还未完成三角化测量（即得到首观测帧下的深度）或者完成了三角化且当前帧为其第二个观测帧，则使用上一帧该点的深度值和当前帧相机位姿来更新当前帧该点的深度值
                    // use_prev_dep = (prev_dep > 0) && ((cls > 0) || (cls == 0 && (it_per_id.estimated_depth <= 0 || it_per_id.feature_per_frame.size() == 2)));
                    // 需要使用上一帧点的深度值来更新当前帧深度值的点。
                    // 用上一帧的深度值来更新当前帧深度，这是在三角化测量函数中进行的，因为这些点不参与LBA，这里不需要再次更新
                    // if(use_prev_dep)
                    // {
                    //     pt_prev(2) = prev_dep;
                    //     pt_prev(0) = prev_un_Fea_map[pt_id](0) * prev_dep;
                    //     pt_prev(1) = prev_un_Fea_map[pt_id](1) * prev_dep;
                    //     motion_R_update = motion_R;
                    //     motion_P_update = motion_P;
                    // }
                    // else
                    {
                        prev_dep = it_per_id.estimated_depth;
                        // 如果该点仍然未完成三角化测量，且上一帧该点也没有立体匹配，则当前帧该点的深度值仍保留为-1.0，等待其继续被跟踪并尝试三角化测量
                        if(prev_dep <= 0)
                        {
                            removeIndex.insert(pt_id);
                            continue;
                        }
                        else
                        {   
                            pt_prev(2) = prev_dep;
                            pt_prev(0) = it_per_id.feature_per_frame[0].point.x() * prev_dep;
                            pt_prev(1) = it_per_id.feature_per_frame[0].point.y() * prev_dep;
                            // 对于需要从观测首帧的深度来推测当前帧深度的背景点，其观测首帧的相机位姿就不进行插值了
                            frame_first = it_per_id.start_frame;
                            first_frame_R = Rs[frame_first] * ric[0];
                            first_frame_P = Ps[frame_first] + Rs[frame_first] * tic[0];
                            motion_R_update = cur_cam_R.transpose() * first_frame_R;
                            motion_P_update = cur_cam_R.transpose() * (first_frame_P - cur_cam_P);
                        }
                    }
                    
                    cur_z = (motion_R_update * pt_prev + motion_P_update)(2);
                    if(cls == 0)
                        max_depth = mThDepthBg;
                    else
                        max_depth = mThDepthObj;
                    
                    // if (cur_z < max_depth && cur_z > mMinDepthPt) 
                    if (cur_z < 1.2 * max_depth && cur_z > 1.2) 
                        cur_dep_FAST[index] = cur_z;
                    // 如果深度超过了阈值范围，则下一帧不再跟踪该点了。
                    // 但是仍然保留该点的当前帧观测记录在地图中（如果该点的观测帧数小于4，则后续会直接被从地图删除；如果观测帧数已经大于4，则说明在上面的重投影误差检验中通过了，可以等待参与后续滑窗的marg）
                    else
                    {
                        //--num_track_fea_stat;
                        statusLeftRIght[index] = 0;
                        // 如果深度值为负（会有这种情况发生吗？），则删除该点当前帧观测在地图中的记录（或者直接将该点全部记录从地图中删除？）
                        if(cur_z <= 0.8 || cur_z >= 1.4*max_depth)
                        {
                            removeIndex.insert(pt_id);
                            // it_per_id.feature_per_frame.pop_back();
                            
                        }
                    }
                }
                else
                {
                    index = -1 * index;
                    cls = obj_cls_id_sift[index].first;

                    // if(cls > 0 && std::find(id_sift_no_depth.begin(), sift_no_dep_iter_end, index) == sift_no_dep_iter_end) continue;

                    // if(cls == 0 && status_sift[index] != 2) continue;

                    // prev_dep = prev_dep_sift[index];
                    // 其实还有可能是上一帧为背景中的漏检点（且没有深度）！
                    // if(cls > 0 && prev_dep <= 0)
                    //     assert(false && "Something wrong with obj fea depth in prev_dep_sift!");
                    
                    // use_prev_dep = (prev_dep > 0) && ((cls > 0) || (cls == 0 && (it_per_id.estimated_depth <= 0 || it_per_id.feature_per_frame.size() == 2)));
                    // if(use_prev_dep)
                    // {
                    //     pt_prev(2) = prev_dep;
                    //     pt_prev(0) = prev_un_Fea_map[pt_id](0) * prev_dep;
                    //     pt_prev(1) = prev_un_Fea_map[pt_id](1) * prev_dep;
                    //     motion_R_update = motion_R;
                    //     motion_P_update = motion_P;
                    // }
                    // else
                    {
                        prev_dep = it_per_id.estimated_depth;
                        if(prev_dep <= 0)
                        {
                            removeIndex.insert(pt_id);
                            continue;
                        }
                        else
                        {
                            pt_prev(2) = prev_dep;
                            pt_prev(0) = it_per_id.feature_per_frame[0].point.x() * prev_dep;
                            pt_prev(1) = it_per_id.feature_per_frame[0].point.y() * prev_dep;
                            // 对于需要从观测首帧的深度来推测当前帧深度的背景点，其观测首帧的相机位姿就不进行插值了
                            frame_first = it_per_id.start_frame;
                            first_frame_R = Rs[frame_first] * ric[0];
                            first_frame_P = Ps[frame_first] + Rs[frame_first] * tic[0];
                            motion_R_update = cur_cam_R.transpose() * first_frame_R;
                            motion_P_update = cur_cam_R.transpose() * (first_frame_P - cur_cam_P);
                        }
                    }
                    
                    cur_z = (motion_R_update * pt_prev + motion_P_update)(2);

                    if(cls == 0)
                        max_depth = mThDepthBg;
                    else
                        max_depth = mThDepthObj;
                    
                    // if (cur_z < max_depth && cur_z > mMinDepthPt)
                    if (cur_z < 1.2 * max_depth && cur_z > 1.2)
                        cur_dep_sift[index] = cur_z;
                    else
                    {
                        //--num_track_fea_stat;
                        status_sift[index] = 0;
                        if(cur_z <= 0.8 || cur_z >= 1.4*max_depth)
                        {
                            removeIndex.insert(pt_id);
                            // it_per_id.feature_per_frame.pop_back();
                        }
                    }
                }
            }
        }
    }

    if(!erase_map_pt.empty())
    {
        for(auto pt_id: erase_map_pt)
        {
            auto iter = find_if(f_manager.feature.begin(),f_manager.feature.end(),[pt_id](const FeaturePerId &it)
                                {
                                    return it.feature_id == pt_id;
                                });
            
            if(iter != f_manager.feature.end())
            {
                f_manager.feature.erase(iter);
            }
        }
    }
}

// VI初始化成功之后，当有新的IMU测量被插入消息队列时，用IMU测量来估计此时刻的IMU坐标下的在当前全局参考坐标系下的位姿和速度。
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    // 得到此IMU的加速度测量在全局坐标系下的表达
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

// 取IMU的测量queue中的所有新数据（大于或等与当前图像时间戳+td的时刻的IMU数据），重新持续积分得到系统可得到的最新（取决于最新的IMU测量的时间戳）的系统全局状态量的预测
// 之所以说“重新积分”，是因为在InputIMU()函数中，当系统完成VI初始化之后，每当有新的IMU测量被push进buf中，就会积分更新latest_P等变量，并且发布出去用于视觉化展示！
// 这里重新积分会使得 用于展示的 预测的最新状态（latest_Q等） 是基于当前滑窗的最新一帧的优化后的状态，使用之后的所有新IMU测量来预测的。这样得到的最新状态的视觉展示与前一IMU时刻的相比，会不会有突变？
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    // Headers中保存的是相机的时间戳，latest_time保存与当前最新帧图像的真实时刻对应的IMU的时间戳
    // TODO:问题：此处td是否应该使用当前滑窗LBA优化之前的td？我们取IMU测量数据的时候是按照优化前的td来取的吧？
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    // 此时IMU的queue中的数据应该都是 时间戳>=当前帧相机时间戳+td（旧）的测量数据，取这些数据进行积分的目的是什么？
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}

// 已知两个时间点的位姿，计算在第二个时间点附近的位姿（可以是内插，也可以是外推）
void Estimator::intepolate_pose(int index_frame, double t1, double t2, double t_new, Matrix3d &R_new, Vector3d &P_new)
{
    if(index_frame < 1) return;
    if (t1 >= t2 || t1 >= t_new) return;
    
    Eigen::Matrix3d R1 = Rs[index_frame-1] * ric[0];
    Eigen::Vector3d P1 = Rs[index_frame-1] * tic[0] + Ps[index_frame-1] ;
    Eigen::Matrix3d R2 = Rs[index_frame] * ric[0];
    Eigen::Vector3d P2 = Rs[index_frame] * tic[0] + Ps[index_frame];

    // 如果初始td与优化后的TD相差不到0.5ms，则直接认为当前帧IMU的位姿估计的时刻就是相机的时刻
    if (abs(t_new - t2) <= 0.0005)
    {
        R_new = R2;
        P_new = P2;
        return;
    }

    Eigen::Matrix3d R = R2 * R1.transpose();
    Eigen::AngleAxisd angle_axis(R);

    P_new = (- R * P1 + P2) / (t2 - t1) * (t_new - t1);

    // 重复使用P1
    P1 = angle_axis.axis() * angle_axis.angle() / (t2 - t1);
    double angle = P1.norm() * (t_new - t1);
    Eigen::Vector3d axis = P1 / P1.norm();
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    // 重复使用R2
    // 罗德里格斯公式的两种表达方式
    R2 = Eigen::Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * K * K;
    // R2 = cos(angle) * Eigen::Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * axis * axis.transpose();
    R_new = R2 * R1;
}

// 根据前后两帧的物体的全局位姿，计算出该物体的全局位姿变换，从而计算出瞬时的全局的角速度和线速度
// 统一乘以局部变换运动G！相机的这个值为I
void Estimator::velocity_from_poses(const Matrix3d &R1, const Vector3d &p1, const Matrix3d &R2, const Vector3d &p2, const Matrix3d &gl_trans_R, 
                                    const Vector3d &gl_trans_P, double &t, Eigen::Vector3d &l_vel, Eigen::Vector3d &ang_vel)
{
    // Compute rotation from pose1 to pose2, expressed in global inference
    // 这种计算运动速度的方式与ORB-SLAM2中的相同，即表达在世界坐标系下的运动变换！如果是R2‘*R1，则是表达在R2局部坐标系下的运动变换
    Eigen::Matrix3d R = R2 * gl_trans_R * R1.transpose();
    
    // Convert to axis-angle representation
    Eigen::AngleAxisd angle_axis(R);
    
    // Compute angular velocity
    ang_vel = angle_axis.axis() * angle_axis.angle() / t;

    // Compute linear velocity
    l_vel = (- R * p1 + R2 * gl_trans_P + p2) / t;
    // l_vel = (p2 - p1) / t;
    
    cout << "p1: " << p1.transpose() << endl;
    cout << "p2: " << p2.transpose() << endl;
    cout << "linear velocity: " << l_vel.transpose() << endl;
    cout << "prev delta_t: " << t << endl;
}

bool Estimator::pred_pose_with_vel(const Eigen::Matrix3d &R_1, const Eigen::Vector3d &p_1, const Eigen::Vector3d &ang_vel, const Eigen::Vector3d &l_vel, double t, Eigen::Matrix3d &R_2, Eigen::Vector3d &p_2)
{
    double theta = ang_vel.norm() * t;
    Eigen::Vector3d delta_p = l_vel * t;
    // false表示两帧的位姿没变化
    if (theta == 0 && delta_p.norm() == 0) 
    {
        return false;
    }
    Eigen::Vector3d axis = ang_vel / ang_vel.norm();
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    // 罗德里格斯公式的两种表达方式
    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity() + sin(theta) * K + (1 - cos(theta)) * K * K;
    // Eigen::Matrix3d delta_R = cos(theta) * Eigen::Matrix3d::Identity() + sin(theta) * K + (1 - cos(theta)) * axis * axis.transpose();
    
    R_2 = delta_R * R_1;
    p_2 = delta_R * p_1 + delta_p;
    // p_2 = p_1 + delta_p;
    
    return true;
}

// 将当前帧（从系统第2帧开始）的动态物体的估计结果写进文件中
void Estimator::write_result_objs(FILE* outFile_cam, FILE* outFile_objs)
{
    if(outFile_cam == NULL) return;
    //featureTracker.sampled_pixel
    if(outFile_objs == NULL) return;

    cout << "Start write results!" << endl;

    // 不论VIO是否完成初始化，两帧间的物体运动估计都可以输出（因为计算的是后一帧相机坐标系下的物体位姿变换，这与全局参考坐标系无关）
    if(!featureTracker.RP_objs_pred.empty())
    {
        int obj_id;
        // 保存各个运动物体的运动估计 以及 物体的bbox、平均深度、cls_label、id等信息
        float* bbox_info;
        int num_size = NUM_SAMPLED_PIXEL_OBJ*3;
        for(auto &obj: featureTracker.RP_objs_pred)
        {
            obj_id = obj.first;
            bbox_info = featureTracker.pixel_objs_prev[obj_id];
            if (bbox_info[-1] == 0) continue;
            Matrix3d &obj_R = obj.second.first;
            Vector3d &obj_P = obj.second.second;
            // gl_obj_id， gl_cls_label, [R|t], left_bbox, right_bbox, top_bbox, bottom_bbox, ave_3D_x, ave_3D_y, ave_3D_z
            fprintf(outFile_objs, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", obj_id, (int)bbox_info[-2], 
                                                                                                obj_R(0,0), obj_R(0,1), obj_R(0,2), obj_P(0),
                                                                                                obj_R(1,0), obj_R(1,1), obj_R(1,2), obj_P(1),
                                                                                                obj_R(2,0), obj_R(2,1), obj_R(2,2), obj_P(2),
                                                                                                bbox_info[num_size], bbox_info[num_size+1],
                                                                                                bbox_info[num_size+2], bbox_info[num_size+3],
                                                                                                bbox_info[num_size+4], bbox_info[num_size+5], bbox_info[num_size+6]);                                                                              
        }
    }
    cout << "Succeed write result of dynamic objects!" << endl;
    // 相机的位姿估计则需要等到VIO初始化并校正全局位姿后再一起输出
    if(USE_IMU && solver_flag == INITIAL) return;

    // 如果当前帧正好完成了初始化，则直接一次性写入已有所有帧的相机位姿结果，因此此时相机位姿已经对齐到新的全局坐标系
    if(initial_succ_first_win)
    {
        if(all_cam_R_before_init.size() != 0)
        {
            for(int i = 0; i < all_cam_R_before_init.size(); ++i)
            {
                Matrix3d &corr_cam_R = all_cam_R_before_init[i];
                Vector3d &corr_cam_P = all_cam_P_before_init[i];
                fprintf(outFile_cam, "%f %f %f %f %f %f %f %f %f %f %f %f\n", corr_cam_R(0,0), corr_cam_R(0,1), corr_cam_R(0,2), corr_cam_P(0),
                                                                              corr_cam_R(1,0), corr_cam_R(1,1), corr_cam_R(1,2), corr_cam_P(1), 
                                                                              corr_cam_R(2,0), corr_cam_R(2,1), corr_cam_R(2,2), corr_cam_P(2));  
            }
        }
        all_cam_R_before_init.clear();
        all_cam_P_before_init.clear();
        all_cam_R_before_init.shrink_to_fit();
        all_cam_P_before_init.shrink_to_fit();

        initial_succ_first_win = false;
    }
    else
    {
        Matrix3d cur_R;
        Vector3d cur_P;
        if(USE_IMU)
        {
            cur_R = last_R;
            cur_P = last_P;
        }
        else
        {
            cur_R = cur_cam_R;
            cur_P = cur_cam_P;
        }
        fprintf(outFile_cam, "%f %f %f %f %f %f %f %f %f %f %f %f\n", cur_R(0,0), cur_R(0,1), cur_R(0,2), cur_P(0),
                                                                      cur_R(1,0), cur_R(1,1), cur_R(1,2), cur_P(1), 
                                                                      cur_R(2,0), cur_R(2,1), cur_R(2,2), cur_P(2));          
    }
    cout << "Succeed write result of camera!" << endl;
}
