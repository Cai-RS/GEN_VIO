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

#include "feature_tracker.h"

int NUM_THREADS = 6;
int min_num_fea_track_obj  = 8;
int min_num_sift_track_obj = 4;
int min_num_fea_track_bg   = 20;
int min_num_sift_track_bg  = 10;
int thres_num_fea_lose_obj = 4;
// 对于当前帧的某个obj上的所有点，其所匹配的上一帧的物体中，匹配点最多的物体与第二多的物体间的匹配点数的比例，高于这个阈值则把当前帧物体匹配到最大匹配点数的上一帧物体
float thres_rel_ratio_match_1_2 = 2.5;


class Cmp_depth_stat_objs 
{
    public:
        // 重载 () 操作符
        // 参数必须加上const限定符
        // 按深度值升序排列；如果深度值相等，则id值小的排前面
        bool operator()(const pair<int, float>& a, const pair<int, float>& b) const 
        {
            return a.second == b.second ? (a.first < b.first) : (a.second < b.second);
        }
};

template<typename T> 
void cal_centre_and_dist_pts(const vector<T> &pts, const set<int> &outliers, vector<float> &dist_pts, float factor_, bool has_cent, T &cent_pt)
{
    int num = pts.size();
    if(num == 0) return;
    dist_pts.clear();

    float dist_pt;
    if(!has_cent)
    {
        // 注意，这里要先初始化该变量（不知道其之前被初始化了没有!）！
        cent_pt = 0.0;
        for(int i = 0; i < num; ++i)
        {
            if(outliers.find(i) != outliers.end()) continue;
            cent_pt = cent_pt + pts[i];
        }
        cent_pt = cent_pt * factor_ / num;
    }
    
    for (int i = 0; i < pts.size(); ++i)
    {
        dist_pt = norm(pts[i] * factor_- cent_pt);
        dist_pts.push_back(dist_pt);
    }
}

template<typename T> 
float cal_rubust_norm_stderr_pts_dist(vector<float> &orig_pts_dist, bool need_cal_dist, const vector<T> &pts, float factor_, set<int> &outliers_pts, bool has_cent, T &cent_pt, bool need_normalized, bool cal_std_dist)
{
    outliers_pts.clear();
    // 可以在一个模板函数内部再调用其它的模板函数！！
    if(need_cal_dist) cal_centre_and_dist_pts(pts, outliers_pts, orig_pts_dist, factor_, has_cent, cent_pt);

    // orig_pts_dist的点距离顺序是和pts对应的
    vector<float> pts_dist = orig_pts_dist;

    float dist_MAD = 0.0, dist_median = 0.0;
    vector<float> abs_dist;
    std::sort(pts_dist.begin(),pts_dist.end());
    int num = pts_dist.size();
    if (num % 2 == 0)
    {
        int second = num / 2;
        int first = second - 1;
        dist_median = (pts_dist[first] + pts_dist[second])/2; 
        for (int i = 0; i < num; ++i)
        {
            abs_dist.push_back(abs(pts_dist[i] - dist_median));
        }
        std::sort(abs_dist.begin(),abs_dist.end());
        // 1.4826是经验值，乘上它得到的dist_MAD是作为正态分布标准差的一种鲁棒的替代（即对极端异常值更为鲁棒，可以较大概率排除它）
        dist_MAD = 1.4826 * (abs_dist[first] + abs_dist[second])/2; 
    }
    else
    {
        int index = num / 2;
        dist_median = pts_dist[index];
        for (int i = 0; i < num; ++i)
        {
            abs_dist.push_back(abs(pts_dist[i] - dist_median));
        }
        std::sort(abs_dist.begin(),abs_dist.end());
        dist_MAD = 1.4826 * abs_dist[index];
    }

    float up_boundary = dist_median + 3 * dist_MAD;
    float low_boundary = dist_median - 3 * dist_MAD;
    
    for (int i = 0; i < num; ++i)
    {
        if (orig_pts_dist[i] > up_boundary || orig_pts_dist[i] < low_boundary)
            outliers_pts.insert(i);
    }

    if(!cal_std_dist) return 0.0;
    // 再次去除外点的影响
    if (!outliers_pts.empty())
        cal_centre_and_dist_pts(pts, outliers_pts, pts_dist, factor_, false, cent_pt);
    float ave_dist_pts = std::accumulate(pts_dist.begin(),pts_dist.end(),0)/pts_dist.size();
    float std_pts = 0.0;
    std::for_each(pts_dist.begin(),pts_dist.end(),[&](const float d){
        std_pts += (d-ave_dist_pts)*(d-ave_dist_pts);
    });
    if(need_normalized)
        return sqrt(std_pts)/ave_dist_pts;
    else
        return sqrt(std_pts);
}

// 函数内需要对pts进行修改，但是不允许对原变量有影响，因此用值传递
void cal_MAD_value(vector<float> pts, float &dist_MAD, float &dist_median)
{
    std::sort(pts.begin(),pts.end());
    int num = pts.size();
    if (num % 2 == 0)
    {
        int second = num / 2;
        int first = second - 1;
        dist_median = (pts[first] + pts[second])/2; 
        for (int i = 0; i < num; ++i)
        {
            pts[i] = abs(pts[i] - dist_median);
        }
        std::sort(pts.begin(),pts.end());
        // 1.4826是经验值，乘上它得到的dist_MAD是作为正态分布标准差的一种鲁棒的替代（即对极端异常值更为鲁棒，可以较大概率排除它）
        dist_MAD = 1.4826 * (pts[first] + pts[second])/2; 
    }
    else
    {
        int index = num / 2;
        dist_median = pts[index];
        for (int i = 0; i < num; ++i)
        {
            pts[i] = abs(pts[i] - dist_median);
        } 
        std::sort(pts.begin(),pts.end());
        dist_MAD = 1.4826 * pts[index];
    }

}

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 5; // 1
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 函数模板进行显式具体化的声明。也可以不在此声明实例化，而是在调用时直接进行显式实例化（即调用时指定类型）
template void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status_);
template void reduceVector(vector<int> &v, const vector<uchar> &status_);
template void reduceVector(vector<float> &v, const vector<uchar> &status_);
template void reduceVector(vector<pair<uchar, int>> &v, const vector<uchar> &status_);

// void reduceVector(vector<cv::Point2f> &v, const vector<uchar> &status_FAST)
// {
//     int j = 0;
//     for (int i = 0; i < int(v.size()); i++)
//         if (status_FAST[i])
//             v[j++] = v[i];
//     // resize会去除前j个元素之后的所有元素
//     v.resize(j);
// }

FeatureTracker::FeatureTracker()
{
    prev_prev_time = 0.0;
    prev_time = 0.0;
    cur_time = 0.0;

    stereo_cam = 0;
    n_id = 0;
    // 物体id从1开始计数，因为0始终是背景的id
    n_obj_id = 1;

    total_frame = 0;

    USE_TRIANGULATE_TWO_FRAME = true;
    
    has_motion_pred_first_two_frame = false;

    FAST_pred_motion = false;
    sift_pred_motion = false;
    // 从第3帧开始，是否需要为上一帧的sift特征点根据所属物体的运动模型计算出在当前帧的像素坐标预测？
    cal_pred_for_sift_track = false;
    // 一开始没有预测值
    hasPrediction = false;
    use_prev_fea = false;
    copy_mask_bg = true;
    done_select_sift = false;
    done_track_FAST = false;

    use_tria_stereo = true;
    no_add_new_sift = true;

    show_tracked_fea = true;
    
    reject_with_F = true;
    only_use_track_sift_for_F = false;
    // 当前帧是否要根据对极约束用前后帧的2d-2d sift匹配点估计F矩阵
    EASI_RANSAC_FH = true;
    // 如果要用sift匹配点估计F，是否估计成功
    cal_Mat_F_H = false;
    // F的计算还可以使用已知的2帧间的相机运动（如用IMU积分或者相机运动模型）。当前帧是否有有效的F矩阵
    has_valid_F = false;
    has_valid_H = false;
    fea_filtered = false;

    add_new_FAST_from_sift = false;
    add_new_sift_in_next_frame = true;

    ave_dep_bg_cur_frame = 0.0;
    num_bg_with_dep = 0;
    num_bg_sift_with_dep = 0;

    check_total_lost_cur_objs = false;

    last_id_track_fea_cur = 0;
    num_track_FAST = 0;
    num_track_sift = 0;
    num_track_FAST_bg = 0;
    num_track_sift_bg = 0;
    last_id_track_fea_prev = 0;
    num_track_FAST_static = 0;
    num_track_sift_static = 0;
    num_sta_FAST_long_track = 0;
    num_sta_sift_long_track = 0;
    num_long_track_fea_stat = 0;
    num_track_fea_static = 0;

    num_sta_objs_found = 0;

    num_sift_bg_prev = 0;
    num_sift_bg_cur = 0;

    has_lost_obj_prev = false;
    
    // 初始化openMP的锁
    omp_init_lock(&mylock);
}

FeatureTracker::~FeatureTracker()
{
    omp_destroy_lock(&mylock);
}

// 给上一帧图像中的物体
void FeatureTracker::draw_mask_fea_prev_obj(const double &cur_time_)
{
    use_prev_fea = false;
    cur_time = cur_time_;

    // copy_mask_bg = false;
    // 要复制的不是上一帧标记了物体和背景点的mask_bg，而是仅标记物体区域的mask_bg。因此复制放在了每一帧的mask_bg刚建立时的
    if(add_new_sift_in_next_frame)
    {
        // 上一帧的原始背景mask
        mask_bg_prev = mask_bg.clone();
        // 与其保留这些信息，不如在遍历每个有效跟踪点时就直接在mask_prev_fea_objs上画圈
        // vector<pair<uchar,int>> temp_obj_id_prev_FAST = obj_cls_id_FAST;
        // vector<pair<uchar,int>> temp_obj_id_prev_sift = obj_cls_id_sift;
    }
    copy_mask_bg = true;

    if (!mask_prev_fea_objs.data || row_img_prev != mask_prev_fea_objs.rows || col_img_prev != mask_prev_fea_objs.cols)
    {
        mask_prev_fea_objs.create(row_img_prev, col_img_prev, CV_8UC1);
    }

    mask_prev_fea_objs.setTo(255);

    // 如果待跟踪FAST点 需要在当前帧才能确定，则暂时先不标注mask_prev_fea_objs
    if(add_new_sift_in_next_frame) return;

    for(int i = 0; i < prev_FAST.size(); ++i)
    {
        // 非背景物体的特征
        if(obj_cls_id_FAST[i].first > 0)
        {
            Point2f &pt = prev_FAST[i];
            if (mask_prev_fea_objs.at<uchar>(pt.y,pt.x) == 255)
                cv::circle(mask_prev_fea_objs, pt, 3, 0, -1);
        }
    }
    for(int i = 0; i < prev_sift.size(); ++i)
    {
        // 非背景物体的特征
        if(obj_cls_id_sift[i].first > 0)
        {
            Point2f &pt = prev_sift[i];
            if (mask_prev_fea_objs.at<uchar>(pt.y,pt.x) == 255)
                cv::circle(mask_prev_fea_objs, pt, 3, 0, -1);
        }
    }
    use_prev_fea = true;
    cout << "Succeeded set mask for fea in prev frame!" << endl;
}

// 新版的mask设置函数，不考虑是否使用IMU或者是否已经完成初始化
// 首先，由于已经使用F矩阵或者相机运动预测值 来排除匹配异常点，则这里先添加 剩下的 上一帧有深度值 的跟踪点（背景跟踪点是先FAST再sift，物体跟踪点是先sift后FAST)；
// 然后，如果当前帧有使用F矩阵对2D-2D匹配进行筛选，则先添加剩下的FAST跟踪点（FAST点容易形成长期跟踪），然后再添加剩下的sift点跟踪（匹配较为准确，容易三角化成功），
// 如果当前帧没有有效的F（例如位移为0)，则2d-2d匹配没有经过筛选，则先添加剩下的sift跟踪点（匹配较为准确），再添加剩下的FAST跟踪点
// 最后，是否要在此处 添加当前帧新检测的sift点（新检测的sift点必须要有立体匹配，没有立体匹配的点由下一帧的cudasift中的前后帧匹配结果来获取）？
// 如果是纯视觉，或者使用IMU且还未初始化，则选择先添加sift新点标注（其实有深度估计的点也不太多）；否则，先添加FAST新点，再添加sift新点
void FeatureTracker::setMask(bool initial_succ, bool use_IMU)
{
    uchar status;
    int gl_id;
    // 先添加上一帧有立体匹配的背景跟踪FAST点.
    // 如果是跟踪后再选取sift点，则有另外一套标注逻辑。
    // 如果所有的背景点跟踪已经结束，且最新帧中不会添加新的背景点，则背景部分不需要标注了
    if(!add_new_sift_in_next_frame)
    {
        for (int i = 0; i < cur_FAST.size(); ++i)
        {
            // 先添加上一帧有深度值的跟踪点
            if (status_FAST[i] == 0) continue;

            // 如果当前帧匹配点为背景点，用obj_id来判断，而不是用cls label。这样做的好处是，可以把背景中漏检物体的点也标注起来（因为背景区域在mask_solid_objs全都被涂黑了，无法标注漏检物体的点）
            if (obj_cls_id_FAST[i].second == 0)
            {
                gl_id = ids_FAST[i];
                // 注意，由于use_tria_stereo的使用和当前帧临时添加的上一帧的新点，上一帧的点即使有立体匹配，此时其深度值也可能为-1.0。但是如果有立体匹配，则在prevRightFeaMap一定能找到该点的全局id
                // if (prev_FAST_dep[i] <= 0) continue;
                if(prev_FAST_dep[i] <= 0 && prevRightFeaMap.find(gl_id) == prevRightFeaMap.end()) continue;

                // 如果当前帧的FAST的匹配点跟已添加的FAST点过近，则放弃此FAST匹配
                if (mask_bg_cur.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 0)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                else
                {
                    cv::circle(mask_bg_cur, cur_FAST[i], MIN_DIST_BG, 0, -1);
                }
            }
        }
    }

    // 再添加上一帧有深度值的背景和物体跟踪sift点
    if(num_track_sift > 0)
    {
        for (int i = 0; i < num_track_sift; ++i)
        {   
            status = status_sift[i];
            // 如果不是有效跟踪点，则放弃标注该点。0为无效跟踪点，3只能为背景的新检测点
            if(status != 1 && status != 2) continue;

            gl_id = ids_sift[i];
            // 对于新增加的上一帧的跟踪点，如果在上一帧有立体匹配，此时其深度值也可能还是-1.0
            // if(prev_sift_dep[i] <= 0) continue;

            if(prev_sift_dep[i] <= 0 && prevRightFeaMap.find(gl_id) == prevRightFeaMap.end()) continue;

            Point2f &sift = cur_sift[i];
            if (obj_cls_id_sift[i].second == 0)
            {
                if(add_new_sift_in_next_frame) continue;
                if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                    cv::circle(mask_bg_cur, sift, MIN_DIST_BG, 0, -1);
                else
                    status_sift[i] = 0;
            }
            else
            {
                // MIN_DIST_OBJ的值要比较小，因为物体的区域一般很小
                if (mask_solid_objs.at<uchar>(sift.y,sift.x) == 255)
                    cv::circle(mask_solid_objs, sift, MIN_DIST_OBJ, 0, -1);
                else
                    status_sift[i] = 0;
            }
        }
    }
    
    // 再添加上一帧物体跟踪FAST点,物体跟踪点必须有深度值
    for (int i = 0; i < cur_FAST.size(); ++i)
    {
        if (status_FAST[i] == 0) continue;
        Point2f &FAST = cur_FAST[i];
        gl_id = ids_FAST[i];
        
        // 再添加上一帧有深度值的物体FAST跟踪点
        // 注意，由于use_tria_stereo的使用和当前帧临时添加的上一帧的新点，上一帧的点即使有立体匹配，此时其深度值也可能为-1.0
        // 但是对于物体点，其上一帧的点即使是新添加的，只要其有立体匹配，则会立即给出其有效深度值（即认为立体校正的精度还可以接受）
        if (prev_FAST_dep[i] > 0)
        {
            // 如果当前帧匹配点为物体点
            if (obj_cls_id_FAST[i].second > 0)
            {
                // 如果当前帧的FAST的匹配点跟已添加的FAST点过近，则放弃此FAST匹配
                if (mask_solid_objs.at<uchar>(FAST.y,FAST.x) == 0)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                else
                {
                    cv::circle(mask_solid_objs, FAST, MIN_DIST_OBJ, 0, -1);
                }
            }
        }
        // 同时添加上一帧没有立体匹配的背景FAST跟踪点
        else
        {
            if(add_new_sift_in_next_frame) continue;
            // 此处用cls来判断背景跟踪点
            // 上一帧没有深度的点，即使其是上一帧的漏检物体上的点，也放弃（因为后续没法参与物体的运动估计）
            if (obj_cls_id_FAST[i].first == 0)
            // if (obj_cls_id_FAST[i].second == 0)
            {
                if (mask_bg_cur.at<uchar>(FAST.y,FAST.x) == 0)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                else
                {
                    cv::circle(mask_bg_cur, FAST, MIN_DIST_BG, 0, -1);
                }
            }
        }
    }

    // 查看当前帧物体上的跟踪点
    // while(true)
    // {
    //     cv::imshow("mask for obj", mask_solid_objs);
    //     // 一直等待用户按下ESC键（ASCI码为27）
    //     if(waitKey(0) == 27)
    //     {
    //         break;
    //     }
    // }

    // 显示当前帧背景中的跟踪点
    // while(true)
    // {
    //     cv::imshow("mask for bg", mask_bg);
    //     // 一直等待用户按下ESC键（ASCI码为27）
    //     if(waitKey(0) == 27)
    //     {
    //         break;
    //     }
    // }
    
    // 添加剩下的sift跟踪点和部分新点
    // 最后选择优先FAST新点，因为当前帧的sift跟踪点和新点在下一帧中大部分都没法被跟踪
    if(0)
    {
        for (int i = 0; i < cur_sift.size(); ++i)
        {   
            status = status_sift[i];
            if(status == 0) continue;
            // 再添加上一帧没有深度值的背景sift跟踪点
            if(i < num_track_sift)
            {
                // 已经标注
                if(prev_sift_dep[i] > 0) continue;
                // 如果是跟踪点
                if(status == 1 || status == 2) 
                {
                    Point2f &sift = cur_sift[i];
                    if (obj_cls_id_sift[i].second == 0)
                    {
                        if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                            cv::circle(mask_bg_cur, sift, MIN_DIST_BG, 0, -1);
                        else
                            status_sift[i] = 0;
                    }
                }
            }
            // sift新点是否要在这里添加标注？
            // 最后的版本中在每一帧的背景集中不添加有立体匹配的新点，而是根据下一帧的跟踪结果再去寻找其中有立体匹配的点，这样可以避免消耗太多本用于FAST的点
            else
            {
                Point2f &sift = cur_sift[i];
                // 物体新点必须有立体匹配
                if(obj_cls_id_sift[i].second > 0)
                {
                    assert(status == 1 && "Why new sift of obj has no depth?");
                    if(mask_solid_objs.at<uchar>(sift.y,sift.x) == 0)
                    {
                        status_sift[i] = 0;
                        continue;
                    }
                    else
                    {
                        cv::circle(mask_solid_objs, sift, MIN_DIST_OBJ, 0, -1);
                    }
                }
                // 背景新点可以没有立体匹配
                else
                {
                    // 背景中有立体匹配的sift新点，在此直接添加
                    if(status == 1)
                    {
                        // 如果是纯视觉 或者 VI但还未初始化，则先添加有立体匹配的背景sift新点（这部分点其实比较少）
                        if(!use_IMU || (use_IMU && !initial_succ))
                        {
                            if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                                cv::circle(mask_bg_cur, sift, MIN_DIST_BG, 0, -1);
                            else
                                status_sift[i] = 0;
                        }
                    }
                    else
                    {
                       assert(status == 3 && "Why new sift of bg has other status besieds 1 and 3?");
                        
                        // if(!use_IMU || (use_IMU && !initial_succ))
                        // {
                        //     if (mask_bg.at<uchar>(sift.y,sift.x) == 255)
                        //         cv::circle(mask_bg, sift, MIN_DIST_BG, 0, -1);
                        //     else
                        //         status_sift[i] = 0;
                        // }
                        // else
                        //     continue;

                        // 没有深度的新点中，先添加FAST再添加sift。（最新版本中不保留没有立体匹配的sift新点，因为哪些sift点会被跟踪需要等到下一帧才知道，其中很多点根本不再当前帧的stereo匹配点集中）
                        continue;
                    }
                }
            }
        }
    }
    // 删除无效的sift点（包括跟踪点和新点）。如果sift点还包含没有立体匹配的新点，则先不再这里删除，而是等到FAST新点检测之后，先标注FAST新点，可能还有需要删除的新sift点。
    // reduce_invalid_fea(true);

    // 删除无效的FAST跟踪点。必须在这里删除FAST的跟踪外点，不然后面还要为这些无效跟踪点寻找立体匹配点
    reduce_invalid_fea(false);
}

// 老的setMask()函数
// 倾向于 1.先标注前后两帧的sift跟踪点（这些点对于VI初始化完成前的相机的PnP很重要，而且点数很少），对于物体点要求前后两帧的sift点都有深度估计，对于背景点则无要求
// 2.然后再标注FAST跟踪点（这很重要，需要有长期跟踪（观测数大于等于3帧或4帧）的FAST点用于LBA）
// 3.最后再标注没有深度估计的背景sift跟踪点或者新检测的sift点。
// 1和2的先后顺序可以改变，取决当前帧之前是否已经完成了VI初始化，如果还未初始化，则优先sift；否则VI已经初始化，对于物体，该是优先sift跟踪点，而对于背景则优先FAST跟踪点！
// ！最后为了保证参与LBA的点数足够多（能都连续观测三帧或以上的基本都是FAST，sift的跟踪方式太难有长跟踪点了！），在这里放弃3中的新检测sift点的标注！而是先检测新的FAST点，再排除跟新FAST重叠的新sift点！
void FeatureTracker::setMask(bool initial_succ)
{
    uchar status;
    // 如果还未VI初始化，则先1后2
    if(!initial_succ || !USE_IMU)
    {
        // 对于sift点，不在这里根据全局跟踪次数进行排序，如果需要，则这个操作会留到sift_assign函数中
        // if (cur_sift.size() > 0)
        // 优先标注sift跟踪点
        if(num_track_sift > 0)
        {
            for (int i = 0; i < num_track_sift; ++i)
            {   
                status = status_sift[i];
                // 如果不是有效跟踪点，则放弃标注该点。0为无效跟踪点，3只能为背景的新检测点
                if(status != 1 && status != 2) continue;
                Point2f &sift = cur_sift[i];
                // 在特征点位置画个黑色的小圆，内部填充（即变成了小黑点），这些黑点区域不进行FAST检测。
                // MIN_DIST_BG的值设为20，这是为FAST设置的，对于sift，我们只把sift点半径为14的范围内的点描黑
                // 采用临时的obj id来判断是否为背景点，这可以把当前帧漏检的物体跟踪点也标注起来
                if (obj_cls_id_sift[i].second == 0)
                {
                    if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                        cv::circle(mask_bg_cur, sift, MIN_DIST_BG*2.0/2, 0, -1);
                }
                else
                {
                    // MIN_DIST_OBJ的值设为12，这是为FAST设置的，对于sift，我们只把sift点半径为10的范围内的点描黑
                    if (mask_solid_objs.at<uchar>(sift.y,sift.x) == 255)
                        cv::circle(mask_solid_objs, sift, MIN_DIST_OBJ*3.0/3, 0, -1);
                }
            }
        }
        
        for (int i = 0; i < cur_FAST.size(); ++i)
        {
            // 没有跟踪到的点就不画了
            if (status_FAST[i] == 0) continue;
            // 如果当前帧匹配点为背景点，用obj_id来判断，而不是用cls label
            if (obj_cls_id_FAST[i].second == 0)
            {
                // 如果当前帧的FAST的匹配点跟sift点或以添加的FAST点重合（或过近），则放弃此FAST匹配
                if (mask_bg_cur.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 0)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                else
                {
                    cv::circle(mask_bg_cur, cur_FAST[i], MIN_DIST_BG*3.0/3, 0, -1);
                }
            }
            // 如果当前帧匹配点为刚体目标上的点。非刚性目标的点不会加入到cur_FAST中
            else
            {
                // 如果当前帧的FAST的匹配点跟sift点重合，则放弃此FAST匹配
                if (mask_solid_objs.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 0)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                else
                {
                    // 原始的VINS-FUSION此半径值设为30，但是为了获得较多物体上的FAST点，我们设置得密集些。
                    cv::circle(mask_solid_objs, cur_FAST[i], MIN_DIST_OBJ*3.0/3, 0, -1);
                }
            }
        }
    }
    // 如果是VI配置且已经初始化，则先2后1
    else if(USE_IMU)
    {
        for (int i = 0; i < cur_FAST.size(); ++i)
        {
            if (status_FAST[i] == 0) continue;
            // 严格的背景跟踪FAST点先标注,即如果是当前帧漏检的物体点(有深度值),则优先选择sift跟踪点
            if (obj_cls_id_FAST[i].first == 0)
            {
                if (mask_bg_cur.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 255)
                {
                    cv::circle(mask_bg_cur, cur_FAST[i], MIN_DIST_BG*3.0/3, 0, -1);
                }
            }
        }

        // 接着添加背景 和 物体上的 sift跟踪点
        for (int i = 0; i < num_track_sift; ++i)
        {   
            status = status_sift[i];
            // 如果不是跟踪点，则放弃标注该点。0为无效跟踪点，3为新检测点
            if(status != 1 && status != 2) continue;
            Point2f &sift = cur_sift[i];
            // 采用临时的obj id来判断是否为背景点，这可以把背景中漏检的物体点也标注起来
            if (obj_cls_id_sift[i].second == 0)
            {
                if(mask_bg_cur.at<uchar>(sift.y,sift.x) == 0)
                    status_sift[i] == 0;
                else
                    cv::circle(mask_bg_cur, sift, MIN_DIST_BG*2.0/2, 0, -1);
            }
            else
            {
                // MIN_DIST_OBJ的值设为12，这是为FAST设置的，对于sift，我们只把sift点半径为10的范围内的点描黑
                if (mask_solid_objs.at<uchar>(sift.y,sift.x) == 255)
                    cv::circle(mask_solid_objs, sift, MIN_DIST_OBJ*3.0/3, 0, -1);
            }
        }

        // 再添加物体上的FAST跟踪点
        for(int i = 0; i < cur_FAST.size(); ++i)
        {
            if (status_FAST[i] == 0) continue;
            if (obj_cls_id_FAST[i].first > 0)
            {
                // 当前帧中背景漏检物体的跟踪FAST点
                if(obj_cls_id_FAST[i].second == 0)
                {
                    if(mask_bg_cur.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 255)
                    {
                        cv::circle(mask_bg_cur, cur_FAST[i], MIN_DIST_OBJ*2.0/2, 0, -1);
                    }
                    else
                    {
                        status_FAST[i] = 0;
                    }
                    continue;
                }
                // 当前帧detected物体的跟踪FAST点
                // 如果当前帧的FAST的匹配点跟sift点重合，则放弃此FAST匹配
                if (mask_solid_objs.at<uchar>(cur_FAST[i].y,cur_FAST[i].x) == 0)
                {
                    status_FAST[i] = 0;
                }
                else
                {
                    // 原始的VINS-FUSION此半径值设为30，但是为了获得较多物体上的FAST点，我们设置得密集些。
                    cv::circle(mask_solid_objs, cur_FAST[i], MIN_DIST_OBJ*3.0/3, 0, -1);
                }
            }
        }
    }
    
    // 显示当前帧跟踪点
    // while(true)
    // {
    //     cv::imshow("mask for bg", mask_bg);
    //     // 一直等待用户按下ESC键（ASCI码为27）
    //     if(waitKey(0) == 27)
    //     {
    //         break;
    //     }
    // }

    // 执行步骤3,添加当前帧检测物体上的有立体匹配的新sift点
    uchar cls_pt;
    int num = cur_sift.size();
    for (int i = num_track_sift; i < num; ++i)
    {   
        status = status_sift[i];
        cls_pt = obj_cls_id_sift[i].first;
        // 如果不是物体的具有深度值的新sift点，则跳过暂时不标注
        // if(cls_pt == 0 || status != 1) continue;
        // 背景的新sift点需要有深度估计
        // if(cls_pt == 0 && status != 1) continue;
        // if(cls_pt > 0 && status != 1) continue;

        // 事实证明sift点的匹配质量比FAST要好很多
        Point2f &sift = cur_sift[i];
        
        // MIN_DIST_OBJ的值设为12，这是为FAST设置的，对于sift，我们只把sift点半径为10的范围内的点描黑
        if (cls_pt > 0)
        {
            if(mask_solid_objs.at<uchar>(sift.y,sift.x) == 255)
                cv::circle(mask_solid_objs, sift, MIN_DIST_OBJ*3.0/3, 0, -1);
            else
                status_sift[i] = 0;
        }
        else
        {
            if(mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                cv::circle(mask_bg_cur, sift, MIN_DIST_BG*3.0/3, 0, -1);
            else
                status_sift[i] = 0;
        }
    }

    // while(true)
    // {
    //     cv::imshow("mask for bg", mask_bg);
    //     // 一直等待用户按下ESC键（ASCI码为27）
    //     if(waitKey(0) == 27)
    //     {
    //         break;
    //     }
    // }

    // 排除不满足要求或者与sift点重复的FAST跟踪点！

    // 根据跟踪结果去除上一帧的特征地图点中，未能在当前帧被跟踪到的点。即在当前帧值只保留那些跟踪到的已有地图点。
    // VINS项目由于没有地图点合并（这需要地图的观测有描述子，如ORB特征点，才可以跨多帧进行地图点的投影和匹配！），所以如果某地图点有多个观测帧，则这些观测帧一定是连续的！！
    // 只估计两帧之间的相对位姿的话，不需要它们之间的跟踪点有其他帧的观测；但是两帧间的位姿估计和点深度估计并不准确，需要多帧进行BA，这时就需要地图点被多帧观测到以建立约束，且共视帧越多，约束越强。
    
    // ！！跟踪丢失点 和 异常匹配点 的删除移到reduce_invalid_fea中进行

    // reduceVector(cur_FAST, status_FAST);
    // // ids中保存的是当前帧特征点（包括从上一帧跟踪到的和当前帧中检测和新添的）所对应的(滑窗内的）特征地图点的全局id。当前帧的ids初始值是上一帧特征跟踪处理结束时的ids（即上一帧左图像中观测到的所有特征地图点）！
    // // track_cnt保存的是对应ids的特征地图点在滑窗内被观测了几帧（包括当前帧的观测）
    // // 对于特征点，只要该点被（连续）观测到的帧数大于2，那么就可以形成BA了！（3帧共视该点时，需要优化的变量是两个相对位姿和该点深度，3个帧互相之间的观测匹配可以形成3个约束）
    // reduceVector(ids_FAST, status_FAST);
    // reduceVector(track_cnt_FAST, status_FAST);
    // reduceVector(obj_cls_id_FAST, status_FAST);
    // reduceVector(prev_FAST, status_FAST);
    // reduceVector(prev_FAST_global_obj_id, status_FAST);
    // reduceVector(prev_FAST_dep, status_FAST);
    // reduceVector(predict_dep_FAST, status_FAST);
    
    // if (cur_FAST.size() == 0) return;

    bool sort_FAST = false; // true?是否有必要对跟踪点及其相关信息进行排序？
    if(sort_FAST)
    {
        // prefer to keep features that are tracked for long time 
        // 第一个int表示的是当前帧某特征点对应的地图点在（当前滑窗内）已经被跟踪的帧数
        // 按照连续跟踪帧数对FAST点进行排序，为了下一帧进行上面的画circle时，跟踪次数高的点先画，不会被跟踪次数少的点覆盖掉
        vector<tuple<int, cv::Point2f, int, pair<uchar, int>, int, float>> cnt_FAST_id_label;
        for (unsigned int i = 0; i < cur_FAST.size(); ++i){
            cnt_FAST_id_label.push_back(make_tuple(track_cnt_FAST[i], cur_FAST[i], ids_FAST[i], obj_cls_id_FAST[i], prev_FAST_global_obj_id[i], prev_FAST_dep[i]));
        }

        // 按照该特征地图点被跟踪次数来排序，次数多的排在前面。
        // TODO:但是下面貌似也没对跟踪次数少的点采取什么特别措施啊？
        sort(cnt_FAST_id_label.begin(), cnt_FAST_id_label.end(), [](const tuple<int, cv::Point2f, int, pair<uchar, int>, int, float> &a, const tuple<int, cv::Point2f, int, pair<uchar, int>, int, float> &b)
            {
                return std::get<0>(a) > std::get<0>(b);
            });

        cur_FAST.clear();
        ids_FAST.clear();
        track_cnt_FAST.clear();
        obj_cls_id_FAST.clear();
        prev_FAST_global_obj_id.clear();
        prev_FAST_dep.clear();

        for (auto &it : cnt_FAST_id_label)
        {
            // 这里不用再判断这些点是否还在mask图上被画圈
            // if (mask.at<uchar>(std::get<1>(it).y,std::get<1>(it).x) == 0)
            {
                cur_FAST.push_back(std::get<1>(it));
                ids_FAST.push_back(std::get<2>(it));
                track_cnt_FAST.push_back(std::get<0>(it));
                obj_cls_id_FAST.push_back(std::get<3>(it));
                prev_FAST_global_obj_id.push_back(std::get<4>(it));
                prev_FAST_dep.push_back(std::get<5>(it));
            }
        }
    }
}

// 删除跟踪点中的异常值（上一帧在当前帧没被跟踪的点，以及跟踪点中的外点）
void FeatureTracker::reduce_invalid_fea(bool for_sift)
{
    int j = 0;

    if(for_sift)
    {
        int prev_num = prev_sift.size();
        int valid_prev_num = prev_num;
        int cur_num = ids_sift.size();
        int num_cur_right = cur_right_sift.size();
        for (int i = 0; i < cur_num; ++i)
        {
            if (status_sift[i] > 0)
            {
                if (j < i)
                {
                    if(i < prev_num)
                    {
                        // 这些上一帧点的信息后面可能还需要用到
                        prev_sift[j] = prev_sift[i];
                        // 上一帧点的变量中只需要删减prev_x_dep即可，因为其他的信息都在相关map变量可以查询
                        prev_sift_dep[j] = prev_sift_dep[i];
                        //prev_un_sift[j] = prev_un_sift[i];
                    }
                    // 这个变量虽然写着prev，但是其长度和所有cur_变量是一直的，因为它是在继承自上一帧的同时添加了新的点
                    prev_sift_global_obj_id[j] = prev_sift_global_obj_id[i];
                    obj_cls_id_sift[j] = obj_cls_id_sift[i];
                    ids_sift[j] = ids_sift[i];
                    track_cnt_sift[j] = track_cnt_sift[i];
                    cur_sift_index[j] = cur_sift_index[i];
                    cur_sift[j] = cur_sift[i];
                    status_sift[j] = status_sift[i];
                    if(num_cur_right != 0) cur_right_sift[j] = cur_right_sift[i];
                }
                ++j;
            }
            else
            {
                if(i < prev_num) --valid_prev_num;
            }
        }

        if(j < cur_num)
        {
            // resize会去除前j个元素之后的所有元素
            if(valid_prev_num < prev_num)
            {
                prev_sift.resize(valid_prev_num);
                prev_sift_dep.resize(valid_prev_num);
            }
            prev_sift_global_obj_id.resize(j);
            obj_cls_id_sift.resize(j);
            ids_sift.resize(j);
            track_cnt_sift.resize(j);
            cur_sift_index.resize(j);
            cur_sift.resize(j);
            status_sift.resize(j);
            if(num_cur_right != 0) cur_right_sift.resize(j);
        }
    }
    else
    {
        int prev_num = prev_FAST.size();
        int valid_prev_num = prev_num;
        // 注意，FAST是否还有部分变量需要reduce？
        int cur_num = ids_FAST.size();
        int num_cur_right = cur_right_FAST.size();
        for (int i = 0; i < cur_num; ++i)
        {
            // 这里要用status_FAST吗？而不是用statusLeftRIght?
            // statusLeftRIght仅在每一帧处理结束后再使用，且不在此函数内使用
            if (status_FAST[i])
            {
                if (j < i)
                {
                    if(i < prev_num)
                    {
                        // 这些上一帧点的信息后面可能还需要用到
                        prev_FAST[j] = prev_FAST[i];
                        // 上一帧点的变量中只需要删减prev_x_dep即可，因为其他的信息都在相关map变量可以查询
                        prev_FAST_dep[j] = prev_FAST_dep[i];
                        // 这个值只会在新点还未添加时在这里进行删除某些点（即跟踪的无效点）
                        if(!predict_dep_FAST.empty()) predict_dep_FAST[j] = predict_dep_FAST[i];
                        //prev_un_FAST[j] = prev_un_FAST[i];
                    }
                    prev_FAST_global_obj_id[j] = prev_FAST_global_obj_id[i];
                    obj_cls_id_FAST[j] = obj_cls_id_FAST[i];
                    ids_FAST[j] = ids_FAST[i];
                    track_cnt_FAST[j] = track_cnt_FAST[i];
                    cur_FAST[j] = cur_FAST[i];
                    
                    status_FAST[j] = status_FAST[i];
                    if(num_cur_right != 0) cur_right_FAST[j] = cur_right_FAST[i];
                }
                ++j;
            }
            else
            {
                if(i < prev_num) --valid_prev_num;
            }
        }
        if(j < cur_num)
        {
            // resize会去除前j个元素之后的所有元素
            if(valid_prev_num < prev_num)
            {
                prev_FAST.resize(valid_prev_num);
                prev_FAST_dep.resize(valid_prev_num);
                if(!predict_dep_FAST.empty()) predict_dep_FAST.resize(valid_prev_num);
            }
            prev_FAST_global_obj_id.resize(j);
            obj_cls_id_FAST.resize(j);
            ids_FAST.resize(j);
            track_cnt_FAST.resize(j);
            cur_FAST.resize(j);
            status_FAST.resize(j);
            if(num_cur_right != 0) cur_right_FAST.resize(j);
        }
    }   
}

// 先标注新的FAST点，然后查看新sift是否跟已有点重叠，如果是，则放弃该新sift点
void FeatureTracker::set_new_fea_in_mask(bool initial_succ)
{
    int num = cur_FAST.size();
    uchar status;
    
    if(!add_new_sift_in_next_frame)
    {
        for (int i = num_track_FAST; i < num; ++i)
        {   
            status = statusLeftRIght[i];
            // FAST新点中status为0的点是物体点，是需要从depth_map中寻找深度值
            // 如果是无效的新FAST点。status为3只可能是背景新点
            if(status != 1 && status != 3) continue;
            
            // 在特征点位置画个黑色的小圆，内部填充（即变成了小黑点），这些黑点区域不进行FAST检测。
            // MIN_DIST_BG的值设为20，这是为FAST设置的，对于sift，我们只把sift点半径为14的范围内的点描黑
            if (obj_cls_id_FAST[i].first == 0)
            {
                Point2f &FAST = cur_FAST[i];
                // 新点的圈设得小一点，因为要为下一帧的sift跟踪点多争取一些空间（另外这些新点不一定会被下一帧跟踪到）
                if (mask_bg_cur.at<uchar>(FAST.y,FAST.x) == 255)
                    cv::circle(mask_bg_cur, FAST, MIN_DIST_BG*2.0/3, 0, -1);
                else
                    status_FAST[i] = 0;
            }
            // 物体点不用再标注了
            // else
            // {
            //     // MIN_DIST_OBJ的值设为12，这是为FAST设置的，对于sift，我们只把sift点半径为10的范围内的点描黑
            //     if (mask_solid_objs.at<uchar>(FAST.y,FAST.x) == 255)
            //         cv::circle(mask_solid_objs, FAST, MIN_DIST_OBJ*1.0/2, 0, -1);
            // }
        }
    }

    num = cur_sift.size();
    uchar cls_pt;
    
    // 跟踪点中那些在上一帧中没有深度值的
    if(!add_new_sift_in_next_frame)
    {
        if(1)
        {
            for(int i = 0; i < num_track_sift; ++i)
            {
                if(status_sift[i] == 0) continue;
                if(prev_sift_dep[i] > 0) continue;
                // 如果是跟踪点
                if(status == 1 || status == 2) 
                {
                    Point2f &sift = cur_sift[i];
                    if (obj_cls_id_sift[i].second == 0)
                    {
                        if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 255)
                            cv::circle(mask_bg_cur, sift, MIN_DIST_BG, 0, -1);
                        else
                            status_sift[i] = 0;
                    }
                }
            }
        }
    

        // 如果是VI且已经初始化，则所有的sift新点均没有添加(包括有立体匹配的新sift点)
        bool has_all = USE_IMU && initial_succ;
        for (int i = num_track_sift; i < num; ++i)
        {   
            status = status_sift[i];
            cls_pt = obj_cls_id_sift[i].first;
            // 如果是物体新点（之前已经标注了）或者无效的背景新sift点，则跳过。status为3只可能是背景新点（当前帧没有立体匹配）
            if(cls_pt > 0 || (status != 1 && status != 3)) continue;
            Point2f &sift = cur_sift[i];

            if(status == 1 && !has_all)
            {
                continue;
            }

            // 如果离已有背景点太近，则放弃该背景sift新点
            if (mask_bg_cur.at<uchar>(sift.y,sift.x) == 0)
                status_sift[i] = 0;
            else
                // 新点的圈设得小一点，因为要为下一帧的sift跟踪点多争取一些空间（另外这些新点不一定会被下一帧跟踪到）
                cv::circle(mask_bg_cur, sift, MIN_DIST_BG, 0, -1);
        }
    }
}

void FeatureTracker::sorted_sift()
{
    if (cur_sift.size() == 0) return;
    vector<pair<int,int>> cnt_sift_index;
    for (int i = 0; i < cur_sift.size(); ++i){
        cnt_sift_index.push_back(make_pair(track_cnt_sift[i], i));
    }

    sort(cnt_sift_index.begin(), cnt_sift_index.end(), [](const pair<int, int> &a, const pair<int, int> &b)
         {
            return a.first > b.first;
         });

    int num = cur_sift.size();

    int num_sift = cur_sift.size();
    if (num_sift != ids_sift.size() || num_sift != track_cnt_sift.size() || num_sift != cur_sift_index.size() || num_sift != obj_cls_id_sift.size() || num_sift != prev_sift_global_obj_id.size() || num_sift != prev_sift_dep.size())
    {
        fprintf(stderr, "Wrong number of ids_sift and other variables!");
        abort(); 
    }

    for (auto &it : cnt_sift_index)
    {
        int id = it.second;
        cur_sift.push_back(cur_sift[id]);
        ids_sift.push_back(ids_sift[id]);
        track_cnt_sift.push_back(track_cnt_sift[id]);
        cur_sift_index.push_back(cur_sift_index[id]);
        obj_cls_id_sift.push_back(obj_cls_id_sift[id]);
        prev_sift_global_obj_id.push_back(prev_sift_global_obj_id[id]);
        prev_sift_dep.push_back(prev_sift_dep[id]);
    }

    cur_sift.erase(cur_sift.begin(), cur_sift.begin() + num_sift);
    ids_sift.erase(ids_sift.begin(), ids_sift.begin() + num_sift);
    track_cnt_sift.erase(track_cnt_sift.begin(), track_cnt_sift.begin() + num_sift);
    cur_sift_index.erase(cur_sift_index.begin(), cur_sift_index.begin() + num_sift);
    obj_cls_id_sift.erase(obj_cls_id_sift.begin(), obj_cls_id_sift.begin() + num_sift);
    prev_sift_global_obj_id.erase(prev_sift_global_obj_id.begin(), prev_sift_global_obj_id.begin() + num_sift);
    prev_sift_dep.erase(prev_sift_dep.begin(), prev_sift_dep.begin() + num_sift);

}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 注意，pt1需要是当前帧的像素坐标，pts是上一帧的像素坐标
float FeatureTracker::cal_dist_img_center_to_flow_line(cv::Point2f &pt1, cv::Point2f &pt2)
{
    float dist = 10000;
    float x1 = pt1.x;
    float y1 = pt1.y;
    float x2 = pt2.x;
    float y2 = pt2.y;

    // 图像的中心点，应该要是整个图像的长宽的一半？标定的cx和cy给定了图像显示范围内的点坐标，而其起点就是相对于真正的投影中心。
    float c_x = SHIFT_X;
    float c_y = SHIFT_Y;
    // float c_x = COL/2.0;
    // float c_y = ROW/2.0;

    // x1和x2，y1和y2不会同时相等
    float proj_vec = (x1 - x2) * (c_x - x2) + (y1 - y2) * (c_y - y2);
    // >= 0? = 0意味着 pt1和pt2的连线 与 pt2和中点的连线 相垂直 或者 pt2和图像中点重合
    if (proj_vec <= 0)
    {
        if(x1 != x2)
        {
            // 直线方程 y = kx + b
            float k = (y1 - y2)/(x1 - x2);
            float b = y2 - k * x2;

            dist = abs(k*c_x - c_y + b)/sqrt(1 + k*k);
        }
        else
        {
            dist = abs(x1 - c_x);
        }
    }

    return dist;
}

// 更新相机和各个物体的位姿预测，并预测其特征点在当前帧中的像素位置
// 静态点的位置预测就用像素光流；动态点才采用运动模型估计！
void FeatureTracker::Ptspredict_motion(const int &frame_count, const Matrix3d &RCam_prev, const Vector3d &PCam_prev, const Matrix3d &RCam_cur_pred, const Vector3d &PCam_cur_pred, bool motion_pred_for_new_objs, bool use_motion)
{
    int num_pt = prev_FAST.size();
    // cur_FAST.clear();
    predict_dep_FAST.clear();
    cur_FAST.resize(num_pt,Point2f(0.0,0.0));
    
    // FAST_new_objs.clear();
    
    // 如果是系统的第2帧，则上一帧的所有点都是新物体的点，没有运动模型可以预测当前帧该点的位置，后续需要使用flow map提供预测值
    if(frame_count == 1 || !use_motion)
    {
        FAST_new_objs.resize(num_pt);
        std::iota(FAST_new_objs.begin(), FAST_new_objs.end(), 0);
        // 第二帧的点的深度预测要怎么给出？这里直接用上一帧该点的深度值是否合理。后面直接用了depth_map？
        // predict_dep_FAST = prev_FAST_dep;
        FAST_pred_motion = true;
        
        // 需要为sift光流和立体匹配 提供预测参考值吗？使用参考值也许可以筛选掉sift暴力匹配中的少数错误匹配？但如果flow_map误差大，那不是可能反而会浪费很多精确匹配的sift点？
        // 如果是系统第2帧，则跟FAST一样，需要flow_map来提供光流预测值
        // 这里暂时是不需要
        // if(cal_pred_for_sift_track)
        // {
        //     //todo
        // }

        return;
    }

    predict_dep_FAST.resize(num_pt,-1.0);
    int obj_id, pt_id;
    Vector3d pt1, pt2;
    cv::Point2f pixel;
    float prev_dep, depth;
    // 两帧之间相机的位姿变换，表达在相机坐标系下。只能用于静态点
    Matrix3d RCam_motion = RCam_cur_pred.transpose() * RCam_prev;
    Vector3d PCam_motion = RCam_cur_pred.transpose() * (PCam_prev - PCam_cur_pred);

    for(int i = 0; i < prev_FAST.size(); ++i)
    {
        prev_dep = prev_FAST_dep[i];
        // 如果该点上一帧没有深度估计，这里需要如何赋值?
        // 选择后续使用flow_map和depth_map获取
        if(prev_dep <= 0) 
        {
            // 可以选择直接用上一帧的点坐标，或者使用上一帧点的2维速度来推算预测坐标（但如果上一帧为新点，则无法使用）
            // predict_dep_FAST[i] = -1.0;
            // cur_FAST[i] = prev_FAST[i];

            FAST_new_objs.push_back(i);
            continue;
        }
        
        obj_id = prev_FAST_global_obj_id[i];
        
        // 物体点是否用其预测的运动模型来给定flow值和深度值？
        // 最好还是使用flow_map和depth_map设置当前帧跟踪点的位置和视差，因为其运动估计的精度很难保证（尤其用了较多像素匹配得到的估计），反而flow_map和depth_map的精度更可靠些（因为区域集中）
        if(obj_id != 0) continue;

        pt_id = ids_FAST[i];
        pt1(2) = prev_dep;
        pt1(0) = prev_un_Fea_map[pt_id](0) * prev_dep;
        pt1(1) = prev_un_Fea_map[pt_id](1) * prev_dep;

        // 对于动态物体
        if(obj_id != 0 && status_objs_prev[obj_id] == 0)
        {
            // 要提前更新上一帧物体的运动模型
            // 对于动态物体，就不考虑使用其观测首帧的3D点了，因为物体点的深度值没有经过优化，用哪一帧的点深度都差不多...
            Vector3d &motion_P = RP_objs_pred[obj_id].second;
            Matrix3d &motion_R = RP_objs_pred[obj_id].first;
            
            pt2 = motion_R * pt1 + motion_P;
            depth = pt2(2);
            if(depth <= 0)
            {
                cur_FAST[i] = prev_FAST[i];
                predict_dep_FAST[i] = prev_FAST_dep[i];
                continue;
            }
            spaceToPlane(pt2, pixel, m_camera[0]);
            if (!inBorder(pixel))
            {
                cur_FAST[i] = prev_FAST[i];
                predict_dep_FAST[i] = prev_FAST_dep[i];
                continue;
            }
            predict_dep_FAST[i] = depth;
            cur_FAST[i] = pixel;
        }
        // 如果上一帧是背景，静态物体或者新物体，统一使用相机运动模型来估计该点在当前帧的深度，像素坐标则可以有别的方式
        else
        {
            bool is_obj_fea = false;
            if(obj_id != 0 && status_objs_prev.find(obj_id) == status_objs_prev.end())
            {
                cout << "Weired!" << endl;
                abort();
            }

            // 如果该物体上一帧为新物体，且不使用相机运动模型来预测其跟踪点位置，则使用optical flow map
            if(obj_id != 0 && status_objs_prev[obj_id] == 2 && !motion_pred_for_new_objs)
            {
                // 如果需要使用光流网络的结果作为新物体的特征点像素坐标的预测值。我们设定只有系统第2帧才使用光流网络的结果提供预测值
                FAST_new_objs.push_back(i);
                assert(prev_FAST_dep[i] > 0 && "Obj fea should has valid depth estimation in every frame!");
                
                // predict_dep_FAST[i] = prev_FAST_dep[i];
            }
            // 如果上一帧是背景或者静态物体;或者上一帧为新物体，但是允许使用相机的运动来预测该物体点
            else
            {
                pt2 = RCam_motion * pt1 + PCam_motion;
                depth = pt2(2);
                // 估计的深度值无效，则后续使用flow_map和depth_map来给定
                if(depth <= 0)
                {
                    // cur_FAST[i] = prev_FAST[i];
                    // predict_dep_FAST[i] = prev_FAST_dep[i];

                    FAST_new_objs.push_back(i);
                    
                    continue;
                }
                
                spaceToPlane(pt2, pixel, m_camera[0]);
                // 也可以选择使用相机的运动预测模型来预估该物体在当前帧左图像的位置，即假设该物体是静态的
                if(inBorder(pixel))
                {
                    cur_FAST[i] = pixel;
                    predict_dep_FAST[i] = depth;
                    continue;
                }
                else
                {
                    // cur_FAST[i] = prev_FAST[i];
                    // predict_dep_FAST[i] = prev_FAST_dep[i];

                    FAST_new_objs.push_back(i);
                    
                    continue;
                }
            }
        }
    }
    FAST_pred_motion = true;

    // 是否需要为sift点估计预测位置
    // if(frame_count > 1)
    // {
    //     // 需要用运动模型为
    //     if(cal_pred_for_sift_track)
    //     {
    //         // todo
    //     }
    // }
}

// 使用光流模型来给出上一帧新物体在当前帧的预测点位置！
void FeatureTracker::Ptspredict_flow(const cv::Mat &flow_map, const cv::Mat &seg_map)
{
    if (seg_map.cols != flow_map.cols || seg_map.rows != flow_map.rows){
        fprintf(stderr, "seg_map and flow_map should have the same dimension!"); 
        abort();
    }

    float up_border_v = row - 0.1;
    float up_border_u = col - 0.1;

    float low_border = 0.0;
    for (auto &id:FAST_new_objs)
    {
        // 上一帧的FAST点是经过严格筛选的，位于边界区域的点都不会选取！
        float prev_x = prev_FAST[id].x;
        float prev_y = prev_FAST[id].y;
        // !!要注意，at的坐标索引是(row, col)！flow_map的值代表 后一帧点坐标 - 前一帧点坐标
        float flow_x = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[0];
        float flow_y = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[1];
        
        float pred_x = max(low_border, min(prev_x+flow_x, up_border_u));
        float pred_y = max(low_border, min(prev_y+flow_y, up_border_v));

        // 不管点的label值，因为可能存在误检或漏检等情况，而且这里也只是匹配点的预测值
        // uchar cls_label = seg_map.at<Vec2f>(int(pred_x), int(pred_y))[0];

        cur_FAST[id].x = pred_x;
        cur_FAST[id].y = pred_y;
    }
    // sift点直接使用暴力匹配，不需要预测值。但是似乎有的话会更好？这样可以帮忙筛选那些可能暴力匹配错误得比较离谱的点？？
    // for (int i = 0; i < prev_sift.size(); ++i){
    //     float prev_x = prev_sift[i].x;
    //     float prev_y = prev_sift[i].y;
    //     float flow_x = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[0];
    //     float flow_y = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[1];
    //     float pred_x = max(low_border, min(prev_x+flow_x, up_border_u));
    //     float pred_y = max(low_border, min(prev_y+flow_y, up_border_v));

    //     // uchar cls_label = seg_map.at<Vec2f>(int(pred_x), int(pred_y))[0];
    //     cv::Point2f pred_pts(pred_x, pred_y);
    //     pred_sift.emplace_back(pred_pts);
    // }
}

// CPU上跟踪FAST特征点（注意此函数中暂时没有左右特征点匹配，以及左图像中新特征点的检测）
// 跟踪上一帧的特征点在当前帧上的位置，8个维度分别表示特征点的3维归一化平面坐标（即z轴坐标为1）、像素坐标、二维点速度（即前后两帧的匹配点在归一化平面上在u和v方向上的向量），以及该点在前后两帧间属于哪个运动物体(id)
// 返回类型中，第一个int表示该特征点（在当前滑窗内）的全局ID，第2个int表示该特征点在当前帧的观测点是在左相机（0）还是右相机（1)
// 注意，VINS-Fusion中没有对双目相机进行立体校正，因此此处左右图像间的匹配点无法通过视差来直接计算深度！但在MVIO项目中，默认已经进行了立体校正
void FeatureTracker::trackImage(int frame_count, const cv::Mat &_img, bool &end_flow_post, const cv::Mat &seg_map, bool &end_FAST_track,  
                                const cv::Mat &prev_dep_map,const cv::Mat &flow_map, const vector<Vector3d> &Ps, const vector<Matrix3d> &Rs)
{
    TicToc t_r;
    // FAST检测只能在8UC1或者32FC1 的灰度图像上。而光流跟踪只能在8UC1灰度图像上！
    // if(_img.channels() == 3)
    //     cvtColor(_img, cur_img, COLOR_BGR2GRAY);
    // else
    //     //cur_img = _img.clone();
    //     cur_img = _img;
    
    frame_cnt = frame_count;
    assert(cur_img.type() == CV_8UC1 && "Must provide an image with CV_8UC1 type for tracking and detection of FAST!");

    if(frame_count == 0)
    {
        end_FAST_track = true;
        return;
    }
    // 是否要进行图像预处理
    //cv::Mat rightImg = _img1;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */

    if(add_new_sift_in_next_frame)
    {
        // 这里不需要等到sift_select函数完成，只需要其中将上一帧有立体匹配（但是无当前帧跟踪）的点加入到prev_FAST即可，后续这两个函数的操作互相不再影响
        // while(!done_select_sift)
        while(!add_new_FAST_from_sift)
        {
            usleep(300);
        }
        
        add_new_FAST_from_sift = false;
    }

    TicToc t_o;

    num_track_FAST_bg = 0;
    num_track_FAST_static = 0;
    num_sta_FAST_long_track = 0;

    // FAST新的背景跟踪点只在下半部分图像中检测，即近处的点
    int num_new_FAST_detect[3][6];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 6; ++j)
            num_new_FAST_detect[i][j] = 0;

    int num_orig_FAST = prev_FAST.size();
    int num_add_new = 0;
    
    int thres_pt_id;
    
    if(add_new_sift_in_next_frame)
    {
        // rectangle(x_left_top. y_left_top, width, height)
        Rect targetRect(0, 0, col, 3*60);
        // 给定像素的值一定要是显式的Scalar类型，不能直接给定0这种int值！！否则编译器会报错
        Mat allZeroZone(targetRect.height, targetRect.width, CV_8UC1, Scalar(0));
        // 获取要涂黑的图像区域的ptr
        Mat dest_zone = mask_bg_prev(targetRect);
        allZeroZone.copyTo(dest_zone);
        
        for(int i = 3; i < 6; ++i)
        {
            for(int j = 0; j < 6; ++j)
            {
                // 虽然bloc中的背景sift跟踪点中有些被排除了，但是由于总体要采样的点足够多，因此应该不会导致最终的背景跟踪点太少吧？
                int num_pt = num_flow_pt_in_bloc[i][j] + num_long_track_FAST_in_bloc[i][j] + num_temp_flow_pt_in_bloc[i][j];
                int rest_num;
                if(i == 3)
                    rest_num = 7 - num_pt;
                else
                    rest_num = NUM_FEA_IN_BLOC - num_pt;
                
                if(rest_num > 0)
                {
                    num_add_new += rest_num;
                    num_new_FAST_detect[i-3][j] = rest_num;
                }
                else
                {
                    int w_bloc = 200, h_bloc = 60;
                    if(i == 5) h_bloc = row - 60*5;

                    if(j == 5) w_bloc = col - 200*5;

                    cv::Rect targetRectBloc(200*j, 60*i, w_bloc, h_bloc);
                    Mat allZeorZoneBloc(h_bloc, w_bloc, CV_8UC1, Scalar(0));
                    Mat dest_zone_bloc = mask_bg_prev(targetRectBloc);
                    allZeorZoneBloc.copyTo(dest_zone_bloc);
                }
            }
        }
        
        // 最少需要15个新FAST跟踪点
        int total_num = std::max(15,num_add_new);

        // 当需要补充的点数足够多时，检测新点
        // if(total_num > 4)
        {
            vector<Point2f> new_FAST_bg_temp;
            // cv::goodFeaturesToTrack(prev_img, new_FAST_bg_temp, total_num * 1.5, 0.03, 30, mask_bg_prev);

            // 使用此版本的函数，其中会输出各个角点的质量
            vector<float> quality_FAST;
            cv::goodFeaturesToTrack(prev_img, new_FAST_bg_temp, total_num * 1.0, 0.01, 30, mask_bg_prev, quality_FAST);
            int num_new = new_FAST_bg_temp.size();
            cout << "original num of new detected fea: " << num_new << endl;
            
            float l_x, l_y;
            Point2f prev_un_pt;
            int num_fea = 0;

            bool sort_by_quality = false;
            // 是否要根据检测点的quality先进行排序
            if(sort_by_quality)
            {
                multimap<float,int,greater<float>> fea_qua_id;
                float quality;
                for(int j = 0; j < num_new; ++j)
                {
                    quality = quality_FAST[j];
                    // map不允许有多个相同的key，如果相同，这里后插入的会把之前的替换
                    // 因此使用multimap
                    fea_qua_id.insert(make_pair(quality,j));
                }

                // 新点中最高的quality数值
                float max_quality = fea_qua_id.begin()->first;
                auto last_it = fea_qua_id.end();
                // 注意，从后往前数的时候是--
                last_it--;
                // 最低的quality数值
                float min_quality = last_it->first;
                // 直接用最高分和最低分的调和比例来获取阈值的话，容易导致参与F矩阵估计的点太少。应该使用中位数
                // float thres_qua = max_quality * 1.0/2 + min_quality * 2.0/2;
                int thres_num = 3.0/5 * num_new;
                bool higher_thres = true;
                
                multimap<float,int,greater<float>>::iterator it, iter;
                // 点的quality已经从大到小排列，质量越高的点越先加入
                // upper_bound(k)的意思是跳到比 给定key值k 大的key的第一个元素的iterator或者直接整个multimap的end()
                // 因为不同key值的之间的元素的iterator不能直接用++来连接
                
                for(it = fea_qua_id.begin(); it != fea_qua_id.end(); it = fea_qua_id.upper_bound(it->first))
                {
                    auto iter = it;
                    quality = iter->first;
                    // 记录第一个不大于thres_qua的点的序号，其及之后的新点不会参与F矩阵的估计.
                    // 改为后一半的新点不参与F矩阵的估计
                    // if(quality <= thres_qua && higher_thres) 
                    if(num_fea >= thres_num && higher_thres)
                    {
                        thres_pt_id = num_fea + num_orig_FAST;
                        higher_thres = false;
                    }
                    
                    int nums = fea_qua_id.count(quality);
                    while(nums--) 
                    {
                        Point2f &p = new_FAST_bg_temp[iter->second];
                        // 位于左图像最左和最右边小部分区域的点暂不考虑
                        if (!inBorder(p)) continue;
                        // 记录之后需要检测右观测点的上一帧点!
                        // new_FAST_bg.push_back(p);
                        
                        int row_bloc = p.y/60;

                        if(row_bloc < 3) continue;

                        int col_bloc = p.x/200;
                        if(col_bloc == 6) col_bloc = 5;
                        if(row_bloc == 6) row_bloc = 5;

                        if(num_new_FAST_detect[row_bloc-3][col_bloc] == 0) continue;

                        num_new_FAST_detect[row_bloc-3][col_bloc] -= 1;
                        
                        // 将所有的点加入cur_FAST，进行前后2帧的跟踪
                        prev_FAST.push_back(p);

                        prev_FAST_global_obj_id.push_back(0);
                        track_cnt_FAST.push_back(1);
                        ids_FAST.push_back(n_id);
                        // status_FAST.push_back(1);
                        obj_cls_id_FAST.emplace_back(0,0);
                        // 这些新点在完成跟踪之后会再检测其右跟踪点
                        prev_FAST_dep.push_back(-1.0);
                        prevLeftFeaMap[n_id] = p;

                        undistortedPts(p, prev_un_pt, m_camera[0]);
                        prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);

                        ++n_id;
                        iter++;
                        ++num_fea;
                    }
                }
            }
            else
            {
                for(auto &p: new_FAST_bg_temp)
                {
                    // 位于左图像边界区域的点暂不考虑
                    if (!inBorder(p)) continue;
                    // 记录之后需要检测右观测点的上一帧点!
                    // new_FAST_bg.push_back(p);
                    
                    int row_bloc = p.y/60;

                    if(row_bloc < 3) continue;

                    int col_bloc = p.x/200;
                    if(col_bloc == 6) col_bloc = 5;
                    if(row_bloc == 6) row_bloc = 5;

                    if(num_new_FAST_detect[row_bloc-3][col_bloc] == 0) continue;

                    num_new_FAST_detect[row_bloc-3][col_bloc] -= 1;
                    
                    // 将所有的点加入cur_FAST，进行前后2帧的跟踪
                    prev_FAST.push_back(p);

                    prev_FAST_global_obj_id.push_back(0);
                    track_cnt_FAST.push_back(1);
                    ids_FAST.push_back(n_id);
                    // status_FAST.push_back(1);
                    obj_cls_id_FAST.emplace_back(0,0);
                    // 后续再为这些上一帧新点检测其右匹配点
                    prev_FAST_dep.push_back(-1.0);
                    prevLeftFeaMap[n_id] = p;

                    undistortedPts(p, prev_un_pt, m_camera[0]);
                    prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);

                    ++n_id;
                    ++num_fea;
                }
            }
            cout << "final num of new detected fea: " << num_fea << endl;
        }
    }

    // 对所有上一帧的FAST点进行跟踪
    if (prev_FAST.size() > 0)
    {
        // 如果是在当前帧才添加上一帧的新FAST点，则要在此处再为所有FAST点提供flow预测值
        if(add_new_sift_in_next_frame)
        {
            // 使用恒速运动模型假设来估计静态点和动态物体点的位置，对于新物体，可以使用运动预测或者后续使用flow_map
            if(use_motion_to_pred_fea_pos)
            {
                Ptspredict_motion(frame_count, Rs[0], Ps[0], Rs[1], Ps[1], false, true);
            }
            else
            {
                int num_pt = prev_FAST.size();
                // predict_dep_FAST.clear();
                cur_FAST.resize(num_pt,Point2f(0.0,0.0));
                
                FAST_new_objs.resize(num_pt);
                std::iota(FAST_new_objs.begin(), FAST_new_objs.end(), 0);
                // 第二帧的点的深度预测要怎么给出？这里直接用上一帧该点的深度值是否合理。后面直接用了depth_map？
                // 注意，此时部分上一帧的点还没插入depth值，因此这里不应该直接赋值
                // predict_dep_FAST = prev_FAST_dep;
                // FAST_pred_motion = true;
            }
        }
        else
        {
            while(!FAST_pred_motion)
            {
                usleep(300);
            }
            FAST_pred_motion = false;
        }
        
        // 如果当前帧中还有特征点(如上一帧的新物体点）没有像素坐标预测，则意味着需要用flow_map来获取预测值。
        // todo: 也可以使用极线约束来提供一定的约束
        if(!FAST_new_objs.empty())
        {
            while(!end_flow_post)
            {
                usleep(300);
            }

            if (!seg_map.empty() && !flow_map.empty())
            {
                // predict_FAST.clear();
                //predict_sift.clear();
                // 这里只用flow_map来给出上一帧的"新物体点"提供预测点，非新物体则是之前就用物体的运动模型来变换和投影得到预测点
                Ptspredict_flow(flow_map, seg_map);
            }
            else
            {
                fprintf(stderr, "For predict of FAST in the second frame should provide the seg_map and flow_map!");
                abort(); 
            }
        }
        
        hasPrediction = true;
        vector<float> err;
        // 是否有对当前帧特征点所在位置的预测（可以是根据上一帧中计算的特征点的二维速度来推测，但是由于KITTI等数据集中的相机帧率低，且可能有动态物体，所以最好是使用物体的3D运动模型来投影上一帧的点作为预测）
        // 针对背景点 和 动态物体点，分别进行flow估计。如果都能有预估位置，则可以统一使用相同的参数（窗口大小 和 最大层数），否则（应该是指前两帧）对于动态物体，应该一开始就增大窗口？
        if(hasPrediction)
        {
            // cur_FAST = predict_FAST;
            // 参数1表示所要使用的最大的图像金字塔层数，0意味着不使用金字塔，1表示最多有2层的金字塔（可以拥有一定的尺度不变性）。因为有预测值，所以一开始尝试层数较少的金字塔
            // status和prev_pts的长度是一样的，表征上一帧每个点在当前帧中是否能被光流追踪到
            
            // todo:可以根据err的大小来作为匹配质量的排序
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_FAST, cur_FAST, status_FAST, err, cv::Size(21, 21), 2, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            // int succ_num = 0;
            // for (size_t i = 0; i < status_FAST.size(); ++i)
            // {
            //     if (status_FAST[i])
            //         ++succ_num;
            // }
            // 如果跟踪的点数太少，则增加金字塔的最大层数到4层
            // if (succ_num < 5)
            //     //  重新计算时，是否要重新为cur_FAST赋值？
            //     // 如果不设置cv::OPTFLOW_USE_INITIAL_FLOW，则默认是copy prev_FAST到cur_FAST作为初始值
            //    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_FAST, cur_FAST, status_FAST, err, cv::Size(21, 21), 3);
               
            hasPrediction = false;
        }
        else
        {
            cur_FAST = prev_FAST;
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_FAST, cur_FAST, status_FAST, err, cv::Size(21, 21), 3);
        }

        FLOW_BACK = true;
        // reverse check  逆向光流计算，用上面光流计算出的当前帧中特征点位置 再 计算到上一帧的光流匹配，如果得到检测位置和上一帧的点原始位置距离相差很小，则认为光流成功
        if(FLOW_BACK)
        {
            vector<float> err_c_p;
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_FAST;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_FAST, reverse_pts, reverse_status, err_c_p, cv::Size(15, 15), 2, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_FAST, reverse_pts, reverse_status, err_c_p, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status_FAST.size(); ++i)
            {
                if(status_FAST[i] && reverse_status[i] && distance(prev_FAST[i], reverse_pts[i]) < 0.5)
                {
                    err[i] = err[i] + err_c_p[i];
                    continue;
                }
                else
                    status_FAST[i] = 0;
            }
        }
        
        // 如果所有待跟踪点在上一帧就已经确定（VINS的思路），则需要等待其他线程把这些点在上一帧的mask中进行标注
        if(!add_new_sift_in_next_frame)
        {
            // 等待完成绘制上一帧特征点的mask，因此其中需要使用到obj_cls_id_FAST等变量，这些变量在下面会被修改为存储当前帧点的信息
            while(!use_prev_fea)
            {
                usleep(300);
            }
        }

        // 按照err对跟踪点进行排序
        vector<pair<float,int>> err_id_FAST;
        float error;
        int num_track = cur_FAST.size();
        for(int j = 0; j < num_track; ++j)
        {
            if(status_FAST[j] == 0) continue;
            error = err[j];
            // map不允许有多个相同的key，如果相同，这里后插入的会把之前的替换
            // 因此使用multimap
            err_id_FAST.emplace_back(error,j);
        }

        // 按err从小到大排序
        sort(err_id_FAST.begin(), err_id_FAST.end(), [](const pair<float, int> &a, const pair<float, int> &b)
        {
            return a.first < b.first;
        });
        
        float x, y, disp_x, disp_y;
        Vec2b pt_info;
        int num_orig_track = err_id_FAST.size();
        int num_final_track = 0;
        
        // 基于极线约束，使用预测的相机运动构建F矩阵来过滤静态点的匹配（注意，这对动态物体点无效，因为动态点仍可能满足相机的极线约束，参考rigidmask）
        bool check_flow_with_epi = true;
        if(!use_motion_to_pred_fea_pos) check_flow_with_epi = false;
        
        Matrix3d cam_F;
        if(check_flow_with_epi)
        {
            if(frame_count > 1)
            {
                // 构建有效的F矩阵需要非0位移
                if(P_cam_motion.norm() > 0.08)
                {
                    // Quaterniond delta_Q(R_cam_motion);
                    // float delta_angle = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

                    // 是否需要 预测的旋转较小?
                    // if(delta_angle < 0.6)
                    {
                        Matrix3d t_up;
                        t_up << 0.0, -P_cam_motion(2), P_cam_motion(1), P_cam_motion(2), 0.0, -P_cam_motion(0), -P_cam_motion(1), P_cam_motion(0), 0.0;
                        // 本质矩阵到关键矩阵
                        cam_F = K_trans_inv * t_up * R_cam_motion * K_inv;
                    }
                    // else
                    // {
                    //     check_flow_with_epi = false;
                    // }
                }
                else
                    check_flow_with_epi = false;
            }
            else
                check_flow_with_epi = false;
        }
        
        // for (int i = 0; i < cur_FAST.size(); ++i)
        for (int k = 0; k < num_orig_track; ++k)
        {
            int i = err_id_FAST[k].second;
            
            if (status_FAST[i])
            {
                Point2f &pt_prev = prev_FAST[i];
                Point2f &pt_cur = cur_FAST[i];

                x = pt_cur.x;
                y = pt_cur.y;
                pt_info = seg_map.at<Vec2b>(y, x);

                // 在左图像的左侧的一些区域内的点要么不可能在右图像中有观测，要么其深度超过了相应的阈值
                // 使用seg_map来查询点的类别
                uchar cls_label = pt_info[0];

                if (cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7) 
                {
                    status_FAST[i] = 0;
                    continue;
                }
                // if (cls_label == 0 && x <= bg_left_border_left_img) 
                // {
                //     status_FAST[i] = 0;
                //     continue;
                // }

                uchar prev_cls = obj_cls_id_FAST[i].first;

                if ((prev_cls > 0 || cls_label > 0) && x <= obj_left_border_left_img)
                {
                    status_FAST[i] = 0;
                    continue;
                }
                
                // 如果上一帧为背景点，且该点之前已经被跟踪了至少2帧（则上上帧也最终一定是背景点），则其当前帧不可能变为物体点
                if (prev_cls == 0 && cls_label != 0 && track_cnt_FAST[i] > 1) 
                {
                    status_FAST[i] = 0;
                    continue;
                }

                // 判断是否会超出上下边界
                if (inBorder(pt_cur))
                {
                    bool is_close_static_fea = false;

                    int prev_obj_state = obj_cls_id_FAST[i].second;
                    int gl_id = ids_FAST[i];

                    bool is_invalid_sta_fea = false;

                    if(check_flow_with_epi)
                    {
                        int gl_obj_id = prev_FAST_global_obj_id[i];

                        if(gl_obj_id > 0 && status_objs_prev.find(gl_obj_id) == status_objs_prev.end())
                        {
                            cout << "Weired!" << endl;
                            exit(-1);
                        }

                        // 静态点需要离预测的极线不能太远
                        if(gl_obj_id == 0 || status_objs_prev[gl_obj_id] == 1)
                        {
                            // 极线的参数
                            float a = cam_F(0,0) * pt_prev.x + cam_F(0,1) * pt_prev.y + cam_F(0,2);
                            float b = cam_F(1,0) * pt_prev.x + cam_F(1,1) * pt_prev.y + cam_F(1,2);
                            float c = cam_F(2,0) * pt_prev.x + cam_F(2,1) * pt_prev.y + cam_F(2,2);

                            float num = a * pt_cur.x + b * pt_cur.y + c;

                            float den = a * a + b * b;

                            // F矩阵无效，一般是因为相机位移t为0? 也有可能是误差造成的
                            if(den != 0) 
                            {
                                // 点到线的距离 的平方
                                float dsqr = num*num/den;
                                
                                // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
                                // 由于这是预测的运动构成的极线，精度不高，所以不应该限制太大
                                if(dsqr > Th_epipolar_con * Th_epipolar_con) 
                                {
                                    // 只排除掉背景匹配，静态物体可能变成了动态物体？
                                    if(gl_obj_id == 0)
                                    {
                                        status_FAST[i] = 0;
                                        continue;
                                    }
                                    else
                                    {
                                        is_invalid_sta_fea = true;
                                    }
                                }
                                
                            }
                        }
                    }

                    // 确定该跟踪点是否能参与F或H矩阵的估计和验证
                    // 纯背景跟踪点
                    if(prev_cls == 0 && cls_label == 0)
                    {
                        is_close_static_fea = true;

                    }
                    else if(prev_cls > 0 && prev_cls == cls_label && prev_obj_state == 0)
                    {
                        // 静态物体的跟踪点，只使用其中在上一帧中有立体匹配的点（这样其深度值较为准确）来估计F或H矩阵
                        // 第3帧开始才会有相机的运动预测
                        if(frame_count > 1 && reject_with_F && prevRightFeaMap.find(gl_id) != prevRightFeaMap.end())
                        {
                            int prev_dep = prev_FAST_dep[i];
                            // 上一帧7m以内的静态物体点
                            // float th_sta_obj_fea = (float)max(mThDepthBg/2.0,7.0);
                            float th_sta_obj_fea = 14;
                            if(!is_invalid_sta_fea && prev_dep > 0 && prev_dep <= th_sta_obj_fea)
                            {
                                if(prev_un_Fea_map.find(gl_id) != prev_un_Fea_map.end())
                                {
                                    Vec4f &prev_fea = prev_un_Fea_map[gl_id];
                                    float prev_x = prev_fea(0) * prev_dep;
                                    float prev_y = prev_fea(1) * prev_dep;
                                    Vector3d prev_pt(prev_x,prev_y,prev_dep);
                                    
                                    Vector3d pred_cur = (R_cam_motion * prev_pt + P_cam_motion);
                                    if(pred_cur(2) > 0)
                                    {
                                        float pred_x = pred_cur(0)/pred_cur(2);
                                        float pred_y = pred_cur(1)/pred_cur(2);
                                        Vector2d pred_un_pt(pred_x,pred_y);

                                        Point2f cur_un_fea;
                                        undistortedPts(pt_cur,cur_un_fea,m_camera[0]);
                                        Vector2d cur_un_pt(cur_un_fea.x,cur_un_fea.y);
                                        
                                        float err = (pred_un_pt - cur_un_pt).norm();
                                        // 重投影误差小于8个像素点
                                        if(err < 8.0/FOCAL_LENGTH_X)
                                        {
                                            is_close_static_fea = true;
                                        }
                                    }

                                    // is_close_static_fea = true;
                                }
                            }
                        }
                    }

                    // 如果上一帧和当前帧都是静态点，则把这些点作为当前帧的背景静态点
                    //if (obj_cls_id_FAST[i].second == 0 && 0 == cls_label) ++num_track_FAST_bg;
                    // 如果上一帧和当前帧都是背景点，则把这些点作为当前帧的背景静态点

                    int obj_label = pt_info[1];
                    
                    // 第二个元素暂时存储当前帧的临时物体id，待物体关联完成后在修改为全局物体id。第一个元素存储该点的全局类别（基于上一帧对齐），但如果上一帧的点为背景点，则保存当前帧该点的检测类别
                    if (prev_cls == 0 || is_close_static_fea)
                    {
                        // 如果是某个物体在上一帧漏检了，则要求上一帧其漏检特征点必须具有该帧下的深度值（否则后续无法进行运动估计以判断该点是否为动态）
                        if(prev_cls == 0 && cls_label > 0)
                        {
                            if(prev_FAST_dep[i] <= 0)
                            {
                                if(prevRightFeaMap.find(gl_id) != prevRightFeaMap.end())
                                {
                                    float disp_x_prev = prevLeftFeaMap[gl_id].x - prevRightFeaMap[gl_id].x;
                                    if(disp_x_prev <= 0)
                                        assert(false);
                                    float dep = mbf/disp_x_prev;

                                    if(dep > 1.5 && dep < mThDepthObj)
                                        prev_FAST_dep[i] = dep;
                                    else
                                    {
                                        status_FAST[i] = 0;
                                        continue;
                                    }
                                }
                                else
                                {
                                    status_FAST[i] = 0;
                                    continue;
                                }
                            }
                        }

                        // if(cls_label == 0)
                        if(is_close_static_fea)
                        {
                            int row_in_bloc = pt_prev.y/60;

                            // 上半部分图像中拒绝那些flow绝对值太小的背景点匹配
                            if(row_in_bloc < 3 && prev_cls == 0 && cls_label == 0)
                            {
                                disp_x = pt_cur.x - pt_prev.x;
                                disp_y = pt_cur.y - pt_prev.y;

                                if(disp_x * disp_x + disp_y * disp_y < Min_dist_flow * Min_dist_flow)
                                {
                                    status_FAST[i] = 0;
                                    
                                    continue;
                                }
                            }

                            // 记录纯背景跟踪点用于校验要估计的F或H矩阵
                            if(prev_cls == 0)
                            {
                                // 先不在这里记录纯背景跟踪点，而是在下面对添加的上一帧新背景点寻找立体匹配后 再记录所有有效的背景跟踪点
                                // id_bg_track_FAST.push_back(i);
                                ++num_track_FAST_bg;
                            }
                            
                            if(reject_with_F && !only_use_track_sift_for_F)
                            {
                                int col_in_bloc = pt_prev.x/200;
                                
                                if(col_in_bloc == 6) col_in_bloc = 5;
                                if(row_in_bloc == 6) row_in_bloc = 5;

                                // 近处的点多添加一些
                                if(row_in_bloc >= 3)
                                {
                                    int num_total = 7;
                                    if(row_in_bloc > 3)
                                    {
                                        num_total = NUM_FEA_IN_BLOC;
                                        // 如果该bloc内有静态物体，适当减少特征点（物体上的特征点太多且集中）
                                        if(prev_cls > 0) 
                                        {
                                            num_total = 6;
                                            // cout << "Got a static obj fea tracking!" << endl;
                                        }
                                    }

                                    if(id_best_track_bloc[row_in_bloc][col_in_bloc] < num_total)
                                    {
                                        // 是否只选择多帧跟踪（这些点是否也有办法记录其quality?)，以及quality值足够高的新检测点用于估计F矩阵
                                        // 最后还是要根据点匹配的质量的大小来排序，而不是根据点检测的质量。匹配质量越好的点，越先放入temp_pts_for_F中！
                                        // if(i < thres_pt_id)
                                        {
                                            id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;
                                            temp_pts_for_F.push_back(i+1);
                                        }
                                        // pts_stereo_for_F;
                                    }
                                }
                                else
                                {
                                    int num_total = 5;
                                    float prev_dep = prev_FAST_dep[i];
                                    // 如果该bloc内包含了近处的背景点，则增加该bloc的点数，因为一般一个物体上不会只有一个特征点
                                    if(prev_dep > 0 && prev_dep <= 18)
                                    {
                                        num_total = 9;
                                    }

                                    // FAST的跟踪点中还是有在上半部分的。其来自于sift
                                    if(id_best_track_bloc[row_in_bloc][col_in_bloc] < num_total)
                                    {
                                        // 是否只选择多帧跟踪（这些点是否也有办法记录其quality?)，以及quality值足够高的新检测点用于估计F矩阵
                                        // 最后还是要根据点匹配的质量来选择，而不是根据点检测的质量
                                        // if(i < thres_pt_id)
                                        {
                                            id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;
                                            temp_pts_for_F.push_back(i+1);
                                        }
                                        // pts_stereo_for_F;
                                    }
                                }
                            }
                        }

                        if(prev_cls == 0)
                            // 上一帧为背景点，则当前帧跟踪点的cls label与其当前帧的检测类别相一致（要么背景要么物体）
                            obj_cls_id_FAST[i] = std::pair<uchar, int>(cls_label, obj_label);
                        else
                            obj_cls_id_FAST[i] = std::pair<uchar, int>(prev_cls, obj_label);
                    }
                    else
                    {
                        // 如果上一帧是物体点，需要该点在上一帧必须有深度值。物体的所有被跟踪点均在上一帧已经确定，且具有深度值！
                        if(prev_FAST_dep[i] <= 0) 
                        {
                            status_FAST[i] = 0;
                            continue;
                        }

                        // 上一帧为物体点，则当前帧跟踪点的cls label与上一帧的点一致，obj_id则仍然用临时检测物体的id
                        obj_cls_id_FAST[i] = std::pair<uchar, int>(prev_cls, obj_label);
                    }
                    
                    // 跟踪的FAST静态点(包括静态物体）数量。需要该点在上一帧为静态点（但是如果上一帧为背景点，而当前帧匹配的是物体点，则不算为静态点跟踪）
                    if (prev_obj_state == 0 && (prev_cls > 0 || cls_label == 0))
                    {
                        ++num_track_FAST_static;
                        // 长跟踪的静态FAST点数
                        if(track_cnt_FAST[i] > 1) ++num_sta_FAST_long_track;
                    }

                    // 对于跟踪成功的物体点，在上一帧中对其进行标注
                    if(add_new_sift_in_next_frame)
                    {
                        // 如果关联的两个点有一个不是背景点，则认为是物体点
                        if(prev_cls != 0 || cls_label != 0)
                        {
                            circle(mask_prev_fea_objs, prev_FAST[i], 3, 0, -1);
                        }
                    }
                    
                    ++num_final_track;

                    if(obj_cls_id_FAST[i].first == 0)
                    {
                        id_bg_track_FAST.push_back(i);
                    }
                }
                else 
                {
                    status_FAST[i] = 0;
                }
            }
        }

        // 根据跟踪结果，为临时添加的上一帧新FAST点检测右图像观测
        // 上一帧的物体FAST点 是否有立体匹配已经是在上一帧就决定了的
        if(add_new_sift_in_next_frame)
        {
            int num = prev_FAST.size();
            // 在上一帧处理后保留的点 在上一帧中的立体匹配是在上一帧中就尝试寻找了，这里只为那些在当前帧新添加的上一帧点（指的是上面检测的新背景点。不包括从sift转化来的新FAST点）
            if(num > num_orig_FAST)
            {
                vector<Point2f> tracked_new_pt;
                vector<int> id_for_r;
                for(int k = num_orig_FAST; k < num; ++k)
                {
                    if(status_FAST[k])
                    {
                        tracked_new_pt.push_back(prev_FAST[k]);
                        id_for_r.push_back(k);
                    }
                }
                
                if(tracked_new_pt.size() > 0)
                {
                    // 首先为这些新特征点设置右匹配点的预测值
                    float ave_disp_x_bg = mbf/(2.0/3*mMinDepthPt + 1.0/3*mThDepthBg);
                    float ave_disp_y_bg = Y_shift_right_image/(2.0/3*mMinDepthPt + 1.0/3*10);
                    vector<Point2f> pts_right;
                    float r_x, r_y;

                    for(auto &pt:tracked_new_pt)
                    {
                        float disp = prev_dep_map.at<float>(pt.y,pt.x);
                        if(disp <= 0)
                        {
                            // assert(disp>0 && "Why value in disp_map <= 0?");
                            r_x = pt.x - ave_disp_x_bg;
                            r_y = min(pt.y+ave_disp_y_bg, (float)(row-5));
                        }
                        else
                        {
                            float depth = mbf/disp;
                            float shift_y = Y_shift_right_image/depth;
                            r_x = pt.x - disp;
                            r_y = min(pt.y+shift_y, (float)(row-5));
                        }

                        if(r_x >= 5 && r_x <= bg_right_border_right_img)
                            pts_right.emplace_back(r_x, r_y);
                        else if (r_x < 5)
                            pts_right.emplace_back(5, r_y);
                        else
                            pts_right.emplace_back(bg_right_border_right_img, r_y);
                    }
                    
                    // 为上一帧每个新点寻找立体匹配
                    vector<uchar> statusLR;
                    vector<cv::Point2f> reverseLeftPts;
                    vector<uchar> statusRL;
                    vector<float> err;
                    Point2f prev_pt_r, prev_un_pt_r;
                    float disp_x, disp_y;

                    // float max_disp = mbf/mMinDepthPt;
                    float max_disp = mbf/1.5;

                    float min_disp = mbf/mThDepthBg;
                    // float min_disp = mbf/15.0;

                    float close_pt_disp = mbf/18.0;

                    // cout << "Start stereo match for FAST in right image!" << endl;
                    // // cur left ---- cur right
                    // // 注意，无论有没有预测值，cur_right_pts和status的长度都与cur_pts的是一样的，status的值会指示cur_right_pts中的对应元素是否为有效的光流估计匹配点
                    cv::calcOpticalFlowPyrLK(prev_img, prev_img_r, tracked_new_pt, pts_right, statusLR, err, cv::Size(21, 21), 2,
                                            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                    FLOW_BACK = 1;
                    if(FLOW_BACK)
                    {
                        vector<float> err_reserve;
                        reverseLeftPts = tracked_new_pt;
                        cv::calcOpticalFlowPyrLK(prev_img_r, prev_img, pts_right, reverseLeftPts, statusRL, err_reserve, cv::Size(15, 15), 2,
                                                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                    }
                    
                    uchar succ;
                    for(int i = 0; i < statusLR.size(); ++i)
                    {
                        if(FLOW_BACK)
                        {
                            succ = statusLR[i] * statusRL[i];
                        }
                        
                        Point2f &prev_pt = tracked_new_pt[i];
                        int row_in_bloc = prev_pt.x/60;
                        int id = id_for_r[i];

                        if(succ && distance(tracked_new_pt[i], reverseLeftPts[i]) < 0.6)
                        {
                            Point2f &prev_pt_r = pts_right[i];
                            disp_x = prev_pt.x - prev_pt_r.x;
                            disp_y = prev_pt.y - prev_pt_r.y;
                            
                            // 太远或太近的跟踪点放弃
                            // if(disp_x < min_disp || disp_x > max_disp) continue;
                            if(disp_x < min_disp || disp_x > max_disp) 
                            {
                                status_FAST[id] = 0;
                                continue;
                            }

                            // 大于18m的点仍为深度估计不够准确，放弃该立体匹配
                            if(disp_x < close_pt_disp)
                            {
                                // 只是放弃该立体匹配，但是不放弃该跟踪点
                                // status_FAST[id] = 0;
                                continue;
                            }

                            // 如果该立体匹配不太准确，则放弃该立体匹配，但是仍保留该跟踪点
                            // disp_y如何限制？KITTI的标定数据表明img2的相同点要比在img_3中的位置高（y坐标小），这个差距与点的depth有关
                            if(abs(disp_y) > 1.1) continue;

                            int g_id = ids_FAST[id];

                            float dep = mbf/disp_x;

                            // 基于该深度值，使用预测的相机位姿进行重投影，如果与flow估计的当前帧匹配点的误差大于阈值，则放弃该深度
                            if(frame_count > 1)
                            {
                                Vec4f &un_pt_prev = prev_un_Fea_map[g_id];
                                float p_X = un_pt_prev(0) * dep;
                                float p_Y = un_pt_prev(1) * dep;
                                
                                Vector3d prev_3d(p_X, p_Y, dep);
                                Vector3d proj_cur = R_cam_motion * prev_3d + P_cam_motion;
                                
                                if(proj_cur(2) == 0)
                                    continue;
                                else
                                {
                                    Point2f pred_cur;
                                    spaceToPlane(proj_cur, pred_cur, m_camera[0]);
                                    Point2f &cur_fea = cur_FAST[id];
                                    
                                    // 由于恒速运动模型得到的运动预测不是很精确，因此阈值反大一些
                                    if((pred_cur.x - cur_fea.x)*(pred_cur.x - cur_fea.x) + (pred_cur.y - cur_fea.y)*(pred_cur.y - cur_fea.y) > 18 * 18)
                                        continue;
                                }
                            }

                            prevRightFeaMap[g_id] = prev_pt_r;
                            undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);
                            prev_un_r_Fea_map[g_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0, 0.0);

                            if(!use_tria_stereo)
                            {                       
                                prev_FAST_dep[id] = dep;
                            }
                        }
                        else
                        {
                            // 上半部分的跟踪点只能是来自之前某帧的sift？
                            // 每一帧只在图像下半部检测新的背景跟踪点，上上帧新检测的FAST点应该不会在上一帧中出现在上半部图像，除非汽车倒退运动
                            // 因此上半部分的跟踪点不会是这里的新跟踪点
                            if(row_in_bloc < 3)
                            {
                                if(obj_cls_id_FAST[id].first > 0) 
                                    status_FAST[id] = 0;
                            }
                        }
                    }
                }
            }
        }

        // 记录纯背景跟踪点
        // 应该在上面遍历跟踪结果时记录，这样可以按照跟踪质量从好到坏先后记录
        // for(int i = 0; i < ids_FAST.size(); ++i)
        // {
        //     if(status_FAST[i] == 0) continue;
        //     if(obj_cls_id_FAST[i].first == 0)
        //     {
        //         id_bg_track_FAST.push_back(i);
        //     }
        // }
    }

    end_FAST_track = true;
    printf("optical flow of FAST costs: %fms \n", t_o.toc());
}

// 将当前帧检测和完成匹配的sift，分配到相应变量中，包括当前左图像中的sift点（与上一帧有匹配关系，不论其在当前帧中是否有深度匹配；以及当前左图像中的新sift点，需要有有效深度匹配）
void FeatureTracker::select_sift_V1(int frame_count, const cv::Mat &_img1, const cv::Mat &seg_map_prev, const cv::Mat &seg_map_cur, 
                                    const cv::Mat &map_depth_prev, const cv::Mat &flow_map, bool &end_flow_post, bool init_succ_IMU, bool use_mask_img)
{
    cout << "Start selcct sift!" << endl;
    TicToc t_o;

    frame_cnt = frame_count;
    // add_new_FAST_from_sift = false;
    
    // 去除GPU线程中的postprocess()!将sift的后处理（选择有效的match）放在此处，减少GPU中其他任务的等待时间。
    // 对于sift点，其允许检测的最小深度值改为1.5m，因为暴力匹配允许更大的视差估计（1.5m对应的视差值为256）
    // float max_shift_y = Y_shift_right_image/mMinDepthPt;
    float max_shift_y = Y_shift_right_image/1.5;
    vector<int> temp_flow_pt_id;

    // 选择sift跟踪点 和 潜在跟踪点
    if(frame_cnt > 0) 
        Sift_->select_flow_matching(Y_shift_right_image/mMinDepthPt, seg_map_prev, seg_map_cur, mask_bg_prev, num_flow_pt_in_bloc, num_temp_flow_pt_in_bloc, 
                                num_long_track_FAST_in_bloc, prev_FAST, track_cnt_FAST, temp_flow_pt_id, prev_color_img_l, use_mask_img);
    
    Sift_->select_stereo_matching(Y_shift_right_image/mMinDepthPt);
    
    num_track_sift_bg = 0;
    num_track_sift_static = 0;
    num_sta_sift_long_track = 0;
    
    if(!cur_sift.empty()) cur_sift.clear();
    
    cur_sift_index.clear();
    
    int shift_index = 10000;

    sift_no_stereo_bg.clear();
    
    // 在上一帧中就已经被跟踪的背景点或物体点（在当前帧如果再被跟踪，则为long track），以及上一帧添加的新物体点（有深度值）
    set<int> tracked_objs_long_bg;
    
    vector<uchar> is_tracked;
    // 前后帧中sift跟踪点的筛选
    // prev_sift中保存的点，是上一帧中的跟踪点；即每一帧在prev_sift中只保留跟踪点，而不放入新的sift点（也可以选择放入，代码有这样的设计）。
    // 因此，如果是prev_sift中的点在当前帧仍被观测到，则意味是观测至少连续3帧的点
    if (prev_sift.size() > 0)
    {
        // status_sift.resize(prev_sift.size(),0);
        int* valid_match_flow_ptr;
        SiftData *sift_data;
        SiftPoint *h1_siftpts;
        int init_index, end_index;
        int shift_id;
        int num_sift_bg_prev_ = num_sift_bg_prev;
        
        status_sift.resize(prev_sift_index.size(), 0);

        int num_prev_sift = prev_sift.size();

        cur_sift.resize(prev_sift_index.size(), cv::Point2f(0, 0));
        cur_sift_index.resize(prev_sift_index.size(), 0);

        for(int n_img = 0; n_img < 2; ++n_img)
        {
            // 设置参数的值
            if(n_img == 0)
            {
                valid_match_flow_ptr = Sift_->img1.h_matching_pts_flow;
                sift_data = &(Sift_->siftData1);
                h1_siftpts = Sift_->siftData1.h_data;
                init_index = 0;
                // 对于原始图像上的sift跟踪，由于上一帧masked_img上的跟踪点排在原始图像上new sift点前面，因此这里只能全部遍历
                // end_index = num_sift_bg_prev_;
                end_index = num_prev_sift;
                shift_id = 0;
            }
            else if(use_mask_img)
            {
                valid_match_flow_ptr  = Sift_->img4.h_matching_pts_flow;
                sift_data = &(Sift_->siftData4);
                h1_siftpts = Sift_->siftData4.h_data;
                init_index = num_sift_bg_prev_;
                end_index = num_prev_sift;
                shift_id = shift_index;
            }
            else
                break;
            
            int num_match_flow = valid_match_flow_ptr[0];
            if(num_match_flow == 0) continue;

            vector<int> valid_match_flow;
            valid_match_flow.reserve(num_match_flow);
            // 用reserve和assign来快速将数组中的元素复制到vector中，注意，assign中的第二个元素应该是对应vector的end()位置，因此应该是被最后一个被复制数组元素的下一位
            valid_match_flow.assign(valid_match_flow_ptr+1, valid_match_flow_ptr+num_match_flow+1);

            if(n_img == 0) is_tracked.resize(num_match_flow,0);

            const vector<float> &valid_disp_x = sift_data->valid_disp_x;
            const vector<float> &valid_disp_y = sift_data->valid_disp_y;
            Vec2b info_pt;
            int id_in_pts;
            Point2f cur_pt;

            // TODO：对于sift点，应该也可以计算其在当前帧的预测像素坐标，这样就可以计算出flow_u、flow_v以及disp_x和disp_y的预估值，用于这里匹配点的筛选！！
            for (int i = init_index; i < end_index; ++i)
            {
                // int validpts_id = valid_match_flow[i];
                // 只从上一帧的有效sift点（包括与上上帧的匹配点，也包括没与上上帧匹配，但是在上一帧中有立体匹配且深度值符合要求的sift点）中选取与当前帧的匹配点
                // 上一帧的img2和当前帧的img1的数据是完全相同的，每帧保留img2中保留的sift点（包括与img1的匹配以及img2中的新sift点）的index，这些点就是当前img1的待跟踪点
                // vector<int>::iterator iter = std::find(prev_sift_index.begin(),prev_sift_index.end(), validpts_id); 
                // if (iter==prev_sift_index.end()) {
                //     continue;
                // }

                int validpts_id = prev_sift_index[i];
                // 上一帧的某些跟踪点，是通过上上帧的sift检测点（有立体匹配）和光流金字塔的得到的匹配点，其本质应该是上一帧的FAST点了
                if(validpts_id < 0)
                {
                    continue;
                }

                if(validpts_id >= shift_index)
                {
                    if(n_img == 0) continue;
                    validpts_id -= shift_index;
                }

                vector<int>::iterator iter = std::find(valid_match_flow.begin(),valid_match_flow.end(),validpts_id);
                // 如果跟踪点不在上一帧保留的sift点集中
                if (iter == valid_match_flow.end())
                {
                    continue;
                }
                
                if(n_img == 0)
                {
                    int id = std::distance(valid_match_flow.begin(),iter);
                    is_tracked[id] = 1;
                }

                //int cord_x = h1_siftpts[validpts_id].xpos;
                //int cord_y = h1_siftpts[validpts_id].ypos;
                
                float match_x = h1_siftpts[validpts_id].match_xpos;
                float match_y = h1_siftpts[validpts_id].match_ypos;
                info_pt = seg_map_cur.at<Vec2b>(match_y,match_x);
                uchar cls_label = info_pt[0];
                
                // don't consider points of person, rider, bicycle and train. Consider points of car, bus, truck
                if (cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7) continue;

                cur_pt.x = match_x;
                cur_pt.y = match_y;

                if(!inBorder(cur_pt)) continue;
                
                uchar obj_id = info_pt[1];

                float disp_x = valid_disp_x[validpts_id];
                float disp_y = valid_disp_y[validpts_id];
                // int id_index = std::distance(prev_sift_index.begin(), iter);
                int id_index = i;
                uchar cls_prev = obj_cls_id_sift[id_index].first;

                // obj_cls_id_sift :fist - class label of point; second - dynamic object id (static object and background - 0)
                // 前后两帧都是背景点，则光流值应该很小。太过近处的静态SIFT即使会被错误滤除，也很可能可以用FAST点来替补。
                // 直接舍弃连续两帧yolo都漏检同个动态物体的那部分特征点！
                //if (cls_label == obj_cls_id_sift[id_index].first && obj_id == 0 && abs(disp_x) < 1280/10.0 && abs(disp_y) < 384/6.0)
                // 匹配的disp其实已经在select_matching中进行筛选过了，但是这里再根据所属物体的动态性来细化点的匹配可信度
                // if (cls_label == obj_cls_id_sift[id_index].first && cls_label == 0)
                if (cls_label == obj_cls_id_sift[id_index].first && cls_label == 0 && abs(disp_x) < 1280/10.0 && abs(disp_y) < 384/6.5)
                {
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    // 当前点的匹配点在其siftdata检测点集中的index,这里是siftdata2
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    // 后续根据物体的运动状态的判定结果和这里的局部obj_id来更改sift点的全局id！obj_cls_id_sift这个量由上一帧的量修改为表示当前帧的，然后用reduceVector来去除多余
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(0, obj_id);
                    // 记录背景上的sift点个数
                    ++num_track_sift_bg;
                    // 记录跟踪的静态sift点数
                    ++num_track_sift_static;
                    // 记录长跟踪静态点的个数
                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;

                    if(n_img == 0) id_bg_track_sift.push_back(id_index);

                }
                // 上一帧的静态物体sift点（即上一帧的全局obj id为0，但cls不为0），即使其（所在物体）在当前两帧之间变为动态的，其光流值也不会太大（物体刚启动运动）。
                // 这些sift点暂时不归为背景sift点，等到确定该物体是否运动后再归类！
                else if (cls_label == obj_cls_id_sift[id_index].first && cls_label > 0 && obj_cls_id_sift[id_index].second == 0 && abs(disp_x) < 1280/10.0 && abs(disp_y) < 384/6.5)
                {
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    // 物体id按当前帧的局部编号来，等物体关联之后再等其进行更新。
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_label, obj_id);
                    ++num_track_sift_static;

                    // id_bg_track_sift.push_back(id_in_pts);

                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;
                }
                // 上上帧与上一帧之间是动态或新的物体，其在上一帧到当前帧之间也可能变成静态的。这种情况在这里可以也按动态物体的标准来衡量（虽然可能会造成一定的误匹配，但概率会比较小）。
                // 后续根据物体运动状态的判定来更改其对应sift点的全局id。
                else if (cls_label == obj_cls_id_sift[id_index].first && obj_cls_id_sift[id_index].second > 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/5.0)
                //else if (cls_label == obj_cls_id_sift[id_index].first && obj_id > 0 && abs(disp_x) < 1280/4.0 && abs(disp_y) < 384/2.0)
                {
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    // 物体id按当前帧的局部编号来，等物体关联之后再等其进行更新。
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_label, obj_id);
                }
                // 漏检或者错检的物体的配对，在系统前两帧暂时不考虑，因为首帧无法得知各物体的运动状态！！对于首帧的误检物体的特征点匹配，依赖于flow_map提供FAST点的预测和匹配。
                // 上一帧点为运动或新的物体，且与当前帧匹配点类别不同（当前帧可能为背景，也可能为物体），则可能是当前帧漏检该运动物体（为背景点）或者错分类该物体，则按照动态光流来约束
                else if (cls_label != obj_cls_id_sift[id_index].first && obj_cls_id_sift[id_index].second > 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/5.0)
                {
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    uchar prev_cls = obj_cls_id_sift[id_index].first;
                    // 物体id按当前帧的局部编号来，等物体关联之后再对其进行更新。
                    // 物体类别则与上一帧的匹配物体点对齐，当前帧匹配点的检测分类可以从seg_map中查询
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(prev_cls, obj_id);
                }
                // 上一帧点为静态物体(明确不是背景），且与当前帧匹配点类别不同（当前帧可能为背景或其他物体），则可能是当前帧漏检该静态物体（为背景点）或者错分类该物体，则按照静态光流来约束
                else if (cls_label != obj_cls_id_sift[id_index].first && cls_label >0 && obj_cls_id_sift[id_index].second == 0 && abs(disp_x) < 1280/10.0 && abs(disp_y) < 384/6.5)
                {
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    uchar prev_cls = obj_cls_id_sift[id_index].first;
                    // 物体id按当前帧的局部编号来，等物体关联之后再对其进行更新。
                    // 物体类别则与上一帧的匹配物体点对齐，当前帧匹配点的检测分类可以从seg_map中查询
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(prev_cls, obj_id);
                    ++num_track_sift_static;
                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;
                }
                // 上一帧点为背景点，当前帧点为物体点，则可能是上一帧漏检了该物体且该点没有与上上帧的物体SIFT点相关联（因为没有被校正cls），则此时无法确定该物体是否为动态的。还是采用动态光流的阈值
                else if (cls_label != obj_cls_id_sift[id_index].first && obj_cls_id_sift[id_index].first == 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/5.0)
                {
                    // 如果该背景点被多帧观测（多帧观测的情况下则确定该点原本为背景点），则不太可能当前帧突然变为物体点
                    if (track_cnt_sift[id_index] > 1) continue;
                    // 如果该点在上一帧中没有深度值，则不采用该跟踪点，因为这样无法参与估计该漏检物体的运动！
                    if(prev_sift_dep[id_index] <= 0) continue;
                    ++num_track_sift;
                    cur_sift[id_index] = cur_pt;
                    cur_sift_index[id_index] = (h1_siftpts[validpts_id].match+shift_id);
                    status_sift[id_index] = 1;
                    // 物体id按当前帧的局部编号来，等物体关联之后再对其进行更新。
                    // 上一帧的背景点对齐到当前帧的物体点时，使用当前帧的物体类别，后续则可以根据上一帧的点的全局id来判断该点是否为背景点
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_label, obj_id);
                }

                if(add_new_sift_in_next_frame)
                {
                    // 如果关联的两个点有一个不是背景点，则认为是物体点
                    if(cls_prev != 0 || cls_label != 0)
                    {
                        circle(mask_prev_fea_objs, prev_sift[id_index], 3, 0, -1);
                    }
                    
                    tracked_objs_long_bg.insert(validpts_id);
                }
            }
            // 这个值不是单指背景上的跟踪点，而是在原始的seg_map上的跟踪点，包含了背景和物体的跟踪点
            if(n_img == 0) num_sift_bg_cur = num_track_sift;
            // 重置图像的有效跟踪点数，以免下帧没有sift跟踪点（可能在别处该点没重置？）
            if(n_img == 1 && use_mask_img) valid_match_flow_ptr[0] = 0;
        }
    }
    
    // 如果下面要进行rejectWithF，则这里先不删除外点，而是等到rejectWithF中进行。
    // 算了，即使不进行极线约束的筛选，跟踪的sift点也不一定全部都会被保留，这取决于FAST跟踪点和sift跟踪点的优先级，有些情况下FAST跟踪点需要先标注，这可能会覆盖某些sift跟踪点。sift无效点的删除放在之后
    // if(!ids_sift.empty())
    // {
    //     if(!reject_with_F)
    //         reduce_invalid_fea(true);
    // }

    // 添加当前帧图像中有效的背景跟踪sift点中 不在上一帧保留点中 的点
    // 这部分点其实占当前帧sift跟踪点的大多数，如果要通过三角化测量来获取更多有深度值的点，则必须添加这部分点
    int prev_pts_with_stereo = 0;
    set<int> pt_stereo_with_flow;
    if(frame_count > 0)
    {
        int* valid_match_flow_ptr = Sift_->img1.h_matching_pts_flow;
        SiftData *sift_data = &(Sift_->siftData1);
        SiftPoint *h1_siftpts = Sift_->siftData1.h_data;

        int num_match_flow = valid_match_flow_ptr[0];
        
        const vector<float> &valid_disp_x = sift_data->valid_disp_x;
        const vector<float> &valid_disp_y = sift_data->valid_disp_y;
        Point2f prev_pt, prev_pt_r, cur_pt, prev_un_pt, prev_un_pt_r, cur_un_pt;

        int id_in_pts = cur_sift.size();

        bool has_track = !is_tracked.empty();
        // bool first_add = true;
        float disp_prev, dep_prev;
        float match_xpos_r, match_ypos_r, disp_match_x_r, disp_match_y_r;

        vector<Point2f> temp_track;
        vector<int> temp_track_pt_g_id, temp_track_pt_l_id;

        for(int k = 0; k < num_match_flow; ++k)
        {
            int pts_id = valid_match_flow_ptr[(k+1)];
            // 在上面已经添加了的跟踪点
            if(has_track && is_tracked[k] == 1) 
            {
                if(find(temp_flow_pt_id.begin(),temp_flow_pt_id.end(),pts_id) != temp_flow_pt_id.end())
                {
                    pt_stereo_with_flow.insert(pts_id);
                    ++prev_pts_with_stereo;
                }    
                
                continue;
            }

            float prev_x = h1_siftpts[pts_id].xpos;
            float prev_y = h1_siftpts[pts_id].ypos;
            float match_x = h1_siftpts[pts_id].match_xpos;
            float match_y = h1_siftpts[pts_id].match_ypos;
            
            // 均匀化分布的设置已经在match_selecting()函数中完成了。这里无需再次标注和查看

            // uchar cls_label_prev = mask_bg_prev.at<uchar>(prev_y,prev_x);
            // uchar cls_label_cur = mask_bg.at<uchar>(match_y,match_x);
            // // 只增加背景中的跟踪点，且不能和上一帧保留的bg上的sift点靠得太近
            // if (cls_label_prev == 0 || cls_label_cur == 0) continue;
            
            uchar cls = seg_map_prev.at<Vec2b>(prev_y, prev_x)(0);
            Vec2b pt_cur = seg_map_cur.at<Vec2b>(match_y, match_x);
            // 物体的跟踪点中，所有在上一帧有深度估计的点均在上面已经选取了(其中有深度的点包括sift立体匹配，也包括在上一帧用光流金字塔估计得到的深度值，或者用估计的运动来计算深度值)
            // 即每一帧保留的物体sift点都必须是有深度值的，包括跟踪点和新点
            if(cls != 0) continue;

            // 非考虑类别的物体
            uchar cls_cur = pt_cur(0);
            if(cls_cur == 1 || cls_cur == 2 || cls_cur == 4 || cls_cur == 7) continue;

            auto iter = find(temp_flow_pt_id.begin(),temp_flow_pt_id.end(),pts_id);
            bool valid_prev_stereo, valid_prev_dep_pred;
            valid_prev_stereo = false;
            // 如果该跟踪点在上一帧没有立体匹配
            if(iter == temp_flow_pt_id.end())
            {
                temp_track.emplace_back(prev_x, prev_y);
                temp_track_pt_g_id.push_back(n_id);
                temp_track_pt_l_id.push_back(id_in_pts);

                // 用上一帧的depth_map给上一帧的点提供深度值，后续用预测的2帧间相机运动（IMU积分或者相机恒速模型）来排除明显不符合的深度值
                disp_prev = map_depth_prev.at<float>(prev_y,prev_x);
                valid_prev_dep_pred = true;
                if(disp_prev <= 0 || disp_prev >= 192) valid_prev_dep_pred = false;
                dep_prev = mbf/disp_prev;
                if(dep_prev < mMinDepthPt || dep_prev > mThDepthBg) valid_prev_dep_pred = false;
            }
            else
            {
                pt_stereo_with_flow.insert(pts_id);

                float *prev_info_ptr = Sift_->prev_stereo_match_info + 5 * pts_id;
                match_xpos_r = prev_info_ptr[3]; 
                match_ypos_r = prev_info_ptr[4];

                disp_match_x_r = prev_x - match_xpos_r;
                disp_match_y_r = prev_y - match_ypos_r;
                assert(disp_match_x_r > 0);
                dep_prev = mbf/disp_match_x_r;
                valid_prev_stereo = true;
            }

            prev_pt.x = prev_x;
            prev_pt.y = prev_y;
            cur_pt.x = match_x;
            cur_pt.y = match_y;

            undistortedPts(prev_pt, prev_un_pt, m_camera[0]);

            prev_sift.emplace_back(prev_x, prev_y);

            cur_sift.emplace_back(match_x,match_y);

            // 对于用depth_map获取的上一帧的sift点深度，就不添加右观测点了。后续如果参与LBA，则只会根据前后帧的投影误差来优化首帧点深度
            // if(prevLeftFeaMap.empty()) assert(false && "Weired! There is no any fea in last frame!");
            prevLeftFeaMap[n_id] = prev_pt;
            prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);
            ids_sift.push_back(n_id);
            
            // 在上一帧中是否保留了有立体匹配的新sift点的信息。这里倾向于不保存
            if(0 && new_sift_stereo_prev.find(pts_id) != new_sift_stereo_prev.end())
            {
                Vec<float,8> &prev_new = new_sift_stereo_prev[pts_id];
                prevRightFeaMap[n_id] = Point2f(prev_new(4), prev_new(5));
                prev_un_r_Fea_map[n_id] = Vec4f(prev_new(6), prev_new(7), 0.0 ,0.0);
                
                if(!use_tria_stereo)
                {
                    assert(prev_x == prev_new(0) && "something worng with new_sift_stereo_prev!");
                    dep_prev = mbf/(prev_x - prev_new(4));
                    valid_prev_dep_pred = true;
                }

                if(has_track) is_tracked[k] = 1;
                new_sift_stereo_prev.erase(pts_id);
            }

            // if(has_track) is_tracked[k] = 1;
            if(valid_prev_stereo)
            {
                prev_pt_r.x = match_xpos_r;
                prev_pt_r.y = match_ypos_r;
                undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);

                prevRightFeaMap[n_id] = prev_pt_r;
                prev_un_r_Fea_map[n_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0 ,0.0);
            }

            // if(first_add)
            // {
            //     id_new_track_sift = n_id;
            //     first_add = true;
            // }

            // 是否要从depth_map中获取深度值。
            // 这部分点寻求使用2帧的三角化来获得在上一帧中的深度值，包括首帧左右2帧或者前后2帧
            if(use_tria_stereo)
            {
                prev_sift_dep.push_back(-1.0);
            }
            else
            {
                if(valid_prev_stereo)
                {
                    prev_sift_dep.push_back(dep_prev);
                }
                else
                    prev_sift_dep.push_back(-1.0);
            }

            // 记录当前帧该跟踪点在当前帧的检测sift点集中的序号
            cur_sift_index.push_back(h1_siftpts[pts_id].match);
            prev_sift_global_obj_id.push_back(0);
            // 因为上一帧该点是背景点，因此按照当前帧匹配点的检测cls 和 local obj_id
            obj_cls_id_sift.emplace_back(pt_cur(0),pt_cur(1));
            track_cnt_sift.push_back(1);
            status_sift.push_back(1);
            ++n_id;

            // ++num_sift_bg_cur;
            ++num_track_sift;

            // 记录用于估计F矩阵的sift跟踪点在cur_sift中的序号
            
            // 如果上一帧和当前帧都是背景点，则是背景点
            if(pt_cur(0) == 0) id_bg_track_sift.push_back(id_in_pts);
            ++id_in_pts;
        }

        // 为跟踪的sift点在上一帧寻找右观测点
        if(!temp_track.empty())
        {
            float ave_disp_x_bg = mbf/(2.0/3*mMinDepthPt + 1.0/3*mThDepthBg);
            float ave_disp_y_bg = Y_shift_right_image/(2.0/3*mMinDepthPt + 1.0/3*10);
            vector<Point2f> temp_track_r;
            float r_x, r_y;
            for(auto &pt:temp_track)
            {
                float disp = map_depth_prev.at<float>(pt.y,pt.x);
                if(disp <= 0)
                {
                    // assert(disp>0 && "Why value in disp_map <= 0?");
                    r_x = pt.x - ave_disp_x_bg;
                    r_y = min(pt.y+ave_disp_y_bg, (float)(row-5));
                }
                else
                {
                    float depth = mbf/disp;
                    float shift_y = Y_shift_right_image/depth;
                    r_x = pt.x - disp;
                    r_y = min(pt.y+shift_y, (float)(row-5));
                }
                if(r_x >= 5 && r_x <= bg_right_border_right_img)
                    temp_track_r.emplace_back(r_x, r_y);
                else if (r_x < 5)
                    temp_track_r.emplace_back(5, r_y);
                else
                    temp_track_r.emplace_back(bg_right_border_right_img, r_y);
            }

            vector<float> err;
            vector<uchar> status_temp;
            cv::calcOpticalFlowPyrLK(prev_img, prev_img_r, temp_track, temp_track_r, status_temp, err, cv::Size(21, 21), 3, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            // FLOW_BACK = false;
            // reverse check  逆向光流计算，用上面光流计算出的当前帧中特征点位置 再 计算到上一帧的光流匹配，如果得到检测位置和上一帧的点原始位置距离相差很小，则认为光流成功
            FLOW_BACK = true;
            if(FLOW_BACK)
            {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = temp_track;
                cv::calcOpticalFlowPyrLK(prev_img_r, prev_img, temp_track_r, reverse_pts, reverse_status, err, cv::Size(15, 15), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_FAST, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status_temp.size(); ++i)
                {
                    if(status_temp[i] && reverse_status[i] && distance(temp_track[i], reverse_pts[i]) <= 0.5)
                    {
                        continue;
                    }
                    else
                        status_temp[i] = 0;
                }
            }

            Point2f prev_un_pt_r;
            for (size_t i = 0; i < status_temp.size(); ++i)
            {
                if (!status_temp[i])
                {
                    int l_id = temp_track_pt_l_id[i];
                    // 物体点在上一帧的漏检点如果没有深度值，则放弃该点（其实也可以不用放弃，用作2个物体的点关联，以便开启像素关联验证）
                    if(obj_cls_id_sift[l_id].first > 0) 
                        status_sift[l_id] = 0;
                    // 如果背景跟踪点在上一帧没有立体匹配，且此系统不允许2-frame的三角化深度测量，则放弃此点
                    else if(!USE_TRIANGULATE_TWO_FRAME)
                        status_sift[l_id] = 0;
                    continue;
                }
                
                Point2f &prev_pt = temp_track[i];
                Point2f &prev_pt_r = temp_track_r[i];
                float disp_r = prev_pt.x - prev_pt_r.x;

                // 大于最大深度，或小于1.5m的点，都不要了
                if(disp_r <= mbf/mThDepthBg || disp_r > 256) 
                {
                    // assert(false && "Find a right point with r_x less than r_l!");
                    continue;
                }

                ++prev_pts_with_stereo;
                int gl_id = temp_track_pt_g_id[i];
                prevRightFeaMap[gl_id] = prev_pt_r;
                undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);
                prev_un_r_Fea_map[gl_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0 ,0.0);

                // 如果不需要三角化才得到深度值（即认为立体校正足够好）
                if(!use_tria_stereo) 
                {
                    int l_id = temp_track_pt_l_id[i];
                    float dep = mbf/disp_r;
                    prev_sift_dep[l_id] = dep;
                }
            }
        }

        valid_match_flow_ptr[0] = 0;
    
    
        // 注意，temp_flow_pt_id中前面部分点是跟踪点，即在pt_stereo_with_flow中，应该略去！
        // 这些上一帧有立体匹配的sift点，采用与FAST一样的跟踪方式，因此就相当于是上一帧的新FAST点了！
        // 因此，FAST的跟踪要在这里完成之后才进行。
        bool find_flow_for_stereo_match = true;
        if(find_flow_for_stereo_match)
        {
            // vector<Point2f> temp_track_prev, temp_track_cur, temp_track_prev_r, temp_track_cur_r;
            Point2f prev_pt, prev_un_pt, prev_pt_r, prev_un_pt_r;
            if(!temp_flow_pt_id.empty())
            {
                SiftPoint *all_pt_ptr = Sift_->siftData1.h_data;
                float prev_x, prev_y, prev_x_r, prev_y_r;
                for(auto iter: temp_flow_pt_id)
                {
                    if(pt_stereo_with_flow.find(iter) != pt_stereo_with_flow.end()) continue;

                    float *prev_info_ptr = Sift_->prev_stereo_match_info + 5 * iter;
                    prev_x = all_pt_ptr[iter].xpos;
                    prev_y = all_pt_ptr[iter].ypos;
                    
                    prev_x_r = prev_info_ptr[3];
                    prev_y_r = prev_info_ptr[4];
                    
                    assert(prev_x - prev_x_r > 0);

                    prev_FAST.emplace_back(prev_x,prev_y);
                    prev_FAST_global_obj_id.push_back(0);
                    track_cnt_FAST.push_back(1);
                    ids_FAST.push_back(n_id);
                    // status_FAST.push_back(1);
                    obj_cls_id_FAST.emplace_back(0,0);

                    prev_pt.x = prev_x;
                    prev_pt.y = prev_y;
                    prevLeftFeaMap[n_id] = prev_pt;

                    undistortedPts(prev_pt, prev_un_pt, m_camera[0]);
                    prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);

                    prev_pt_r.x = prev_x_r;
                    prev_pt_r.y = prev_y_r;
                    prevRightFeaMap[n_id] = prev_pt_r;

                    undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);
                    prev_un_r_Fea_map[n_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0, 0.0);

                    if(use_tria_stereo)
                        prev_FAST_dep.push_back(-1.0);
                    else
                    {
                        assert(prev_x-prev_x_r > 0);
                        float dep = mbf/(prev_x-prev_x_r);
                        prev_FAST_dep.push_back(dep);
                    }

                    ++n_id;
                }
            }
        }
        // 通知FAST track线程已经完成了来自sift的新FAST的添加。
        // 之后再在FAST的跟踪函数中查看各个bloc中特征点数是否足够，如果不够，添加上一帧图像中检测的FAST新点（有立体匹配的在前，无立体匹配的在后）（可以不在上一时刻检测，而是在当前帧中再从上一帧的图像剩余区域检测FAST点，这样就不需要排除多余的FAST点）
        add_new_FAST_from_sift = true;

        // 为上一帧有立体匹配的sift新点寻找跟踪点。这些点其实相当于是FAST点了！即跟踪方式与FAST相同了
        // 这里暂时不使用new_sift_stereo_prev来存储每一帧中的只有立体匹配的sift点
        if(0 && find_flow_for_stereo_match) 
        {
            int id_in_pts = cur_sift.size();

            vector<Point2f> temp_track_prev, temp_track_cur;
            vector<int> id_temp;
    
            if(!new_sift_stereo_prev.empty())
            {
                for(auto &iter: new_sift_stereo_prev)
                {
                    Vec<float,8> &pt_prev = iter.second;

                    if(pt_prev(1) < (1.0/7*row) && (pt_prev(0) > 2.0/5*col && pt_prev(0) < 3.0/5*col))
                    {
                        continue;
                    }

                    temp_track_prev.emplace_back(pt_prev(0),pt_prev(1));
                    id_temp.push_back(iter.first);
                }
            }

            // 用什么来设置预测值？运动模型还是光流图？
            while(!end_flow_post)
            {
                usleep(300);
            }
            
            float up_border_v = row - 3.0;
            float up_border_u = col - 3.0;
            float low_border = 3.0;
            for(auto &pt:temp_track_prev)
            {
                float prev_x = pt.x;
                float prev_y = pt.y;
                // !!要注意，at的坐标索引是(row, col)！flow_map的值代表 后一帧点坐标 - 前一帧点坐标
                float flow_x = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[0];
                float flow_y = flow_map.at<Vec2f>(int(prev_y), int(prev_x))[1];
                
                float pred_x = max(low_border, min(prev_x+flow_x, up_border_u));
                float pred_y = max(low_border, min(prev_y+flow_y, up_border_v));

                // 不管点的label值，因为可能存在误检或漏检等情况，而且这里也只是匹配点的预测值
                // uchar cls_label = seg_map.at<Vec2f>(int(pred_x), int(pred_y))[0];
                
                temp_track_cur.emplace_back(pred_x, pred_y);
            }

            // 使用光流来跟踪这些在上一帧中有立体匹配的点
            vector<float> err;
            vector<uchar> status_temp;
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, temp_track_prev, temp_track_cur, status_temp, err, cv::Size(21, 21), 2, 
                                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            // FLOW_BACK = false;
            // reverse check  逆向光流计算，用上面光流计算出的当前帧中特征点位置 再 计算到上一帧的光流匹配，如果得到检测位置和上一帧的点原始位置距离相差很小，则认为光流成功
            FLOW_BACK = true;
            if(FLOW_BACK)
            {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = temp_track_prev;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, temp_track_cur, reverse_pts, reverse_status, err, cv::Size(15, 15), 2, 
                                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_FAST, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status_temp.size(); ++i)
                {
                    if(status_temp[i] && reverse_status[i] && distance(temp_track_prev[i], reverse_pts[i]) <= 0.5)
                    {
                        continue;
                    }
                    else
                        status_temp[i] = 0;
                }
            }

            for (size_t i = 0; i < status_temp.size(); ++i)
            {
                if (!status_temp[i])
                    continue;
                
                Point2f &cur_pt = temp_track_cur[i];
                if(cur_pt.y < (1.0/7*row) && (cur_pt.x > 2.0/5*col && cur_pt.x < 3.0/5*col))
                {
                    continue;
                }

                uchar cls_label_cur = mask_bg_cur.at<uchar>(cur_pt.y,cur_pt.x);
                // 只增加背景中的跟踪点，且不能和上一帧保留的bg上的sift点靠得太近
                // 如果当前帧是物体点，是否保留该上一帧的物体漏检点？
                if (cls_label_cur == 0) continue;

                ++prev_pts_with_stereo;

                Vec<float,8> &temp_pt = new_sift_stereo_prev[id_temp[i]];
                prev_sift.push_back(temp_track_prev[i]);
                cur_sift.push_back(temp_track_cur[i]);
                
                // 对于用depth_map获取的上一帧的sift点深度，就不添加右观测点了。后续如果参与LBA，则只会根据前后帧的投影误差来优化首帧点深度
                // if(prevLeftFeaMap.empty()) assert(false && "Weired! There is no any fea in last frame!");
                prevLeftFeaMap[n_id] = temp_track_prev[i];
                prev_un_Fea_map[n_id] = Vec4f(temp_pt(2), temp_pt(3), 0.0, 0.0);
                ids_sift.push_back(n_id);

                prevRightFeaMap[n_id] = Point2f(temp_pt(4), temp_pt(5));
                prev_un_r_Fea_map[n_id] = Vec4f(temp_pt(6), temp_pt(7), 0.0 ,0.0);
                ++n_id;

                float dep_prev;
                if(!use_tria_stereo)
                {
                    dep_prev = mbf/(temp_pt(0) - temp_pt(4));
                }
                else
                {
                    dep_prev = -1.0;
                }
                
                prev_sift_dep.push_back(dep_prev);

                cur_sift_index.push_back(-1);
                prev_sift_global_obj_id.push_back(0);
                obj_cls_id_sift.emplace_back(0,0);
                track_cnt_sift.push_back(1);
                // 最后该点在当前帧的状态是取决于后续在当前帧是否有立体匹配
                status_sift.push_back(1);
                
                // ++num_sift_bg_cur;
                ++num_track_sift;
                // 这部分跟踪点的精度应该是不及sift匹配的，所以这部分是否要参与F矩阵的估计？
                id_bg_track_sift.push_back(id_in_pts);
                ++id_in_pts;
            }
            new_sift_stereo_prev.clear();
        }

        cout << "Num of tracked prev pts with stereo match: " << prev_pts_with_stereo << endl;

        // 是否需要根据极线约束去除异常匹配点
        if(frame_count > 0 && reject_with_F)
        {
            rejectWithFV1(true, init_succ_IMU);
        }
        
        if(!track_cnt_sift.empty())
        {
            for(auto &n : track_cnt_sift)
                ++n;
        }
        // assert(num_track_sift == ids_sift.size() && "Something wrong with num_track_sift!");

        num_track_sift = cur_sift.size();
        cout << "num of tracked sift in cur frame: " << num_track_sift << endl;
    
        if(num_track_sift > 0)
        {
            cv::Point2f temp_pt(0.0,0.0);
            cur_right_sift.resize(num_track_sift,temp_pt);
            cur_sift_dep.resize(num_track_sift,-1.0);

            // 可能还未删除cur_sift中的无效跟踪点
            // if(!status_sift.empty()) status_sift.clear();
            // status_sift.resize(num_track_sift,0);
        }
    }

    // 当前帧跟踪到的点中的最大全局id。其中有些跟踪点是无效的（已经被确认无效，或者之后setMask时会跟FAST过近而无效），但因为每个点的id都是独一无二的，因此后面的新点的id肯定不会比这里的跟踪点大
    if(!ids_sift.empty())
        last_id_track_fea_cur = *(std::max_element(ids_sift.begin(),ids_sift.end()));
    
    if(!obj_fea_disp_num.empty()) 
        obj_fea_disp_num.clear();

    num_bg_sift_with_dep = 0;
    
    // 寻找跟踪sift点的右图像匹配点。并添加新的sift点（有立体匹配或者深度值的物体点）
    if(STEREO && !_img1.empty())
    {
        int* valid_match_stereo;
        SiftData* sift_data;
        
        SiftPoint *h2_siftpts;
        int init_track, num_track;
        int num_sift_bg_cur_ = num_sift_bg_cur;
        int shift_id;
        
        num_new_sift_bg = 0;
        int num_new_sift = 0;

        // todo： 暂不添加当前帧中有立体匹配的背景sift新点，改为下一帧根据flow匹配结果添加
        // add_new_sift_in_next_frame = true;

        for(int n_img = 0; n_img < 2; ++n_img)
        {
            if(n_img == 0)
            {
                valid_match_stereo  = Sift_->img2.h_matching_pts_stereo;
                sift_data = &(Sift_->siftData2);
                h2_siftpts = Sift_->siftData2.h_data;
                init_track = 0;
                num_track = num_track_sift;
                shift_id = 0;
            }
            else if(use_mask_img)
            {
                valid_match_stereo  = Sift_->img5.h_matching_pts_stereo;
                sift_data = &(Sift_->siftData5);
                h2_siftpts = Sift_->siftData5.h_data;
                // init_track = num_sift_bg_cur_;
                init_track = 0;
                num_track = num_track_sift;
                shift_id = shift_index;
            }
            else 
                break;
            
            int num_match_stereo = valid_match_stereo[0];
            if(num_match_stereo == 0) continue;

            vector<uchar> status_sift_stereo_flow(num_match_stereo, 0);
            Vec2b info_pt;
            if(num_track > 0)
            {
                // 也可以像上面一样使用reserve和assign函数来转化为vector，只是下面这种方式更简洁
                // 同样的，vetcor vec(ptr_begin, ptr_end)，其中ptr_end位置处的元素是不算的
                vector<int> match_stereo(valid_match_stereo + 1, valid_match_stereo + 1 + num_match_stereo);

                assert(match_stereo.size() == num_match_stereo && "Something wrong with stereo-matching number!");

                uchar obj_cls;
                int obj_id;
                //printf("stereo image; track feature on right image\n");
                // cur left ---- cur right
                for (int i = init_track; i < num_track; ++i)
                {
                    if(status_sift[i] == 0) continue;

                    int id_pt = cur_sift_index[i];
                    if(id_pt > shift_index) id_pt -= shift_index;
                    vector<int>::iterator iter = std::find(match_stereo.begin(),match_stereo.end(),id_pt);
                    // 使用跟踪点的cls而不是obj_id来判断,是因为当前帧有些背景点可能关联的是上一帧的物佑(当前帧漏检)
                    uchar obj_cls = obj_cls_id_sift[i].first;
                    if (iter == match_stereo.end()) 
                    {
                        // 对于左图像中没有在右图像找到匹配的sift跟踪点，后续是否要尝试用depth_map来获取其深度估计？
                        // 对于物体点可以，但对于背景点则没必要，精度没法保证
                        if (obj_cls > 0)
                        {
                            id_sift_no_depth.push_back(i);
                            status_sift[i] = 1;
                            continue;
                        }
                        else
                        {
                            // 是否保留该跟踪点是在上面已经决定了，这里无需再判断。这里只记录该跟踪点在当前帧是否有立体匹配

                            // 此处不能只用prev_sift_dep或use_tria_stereo判断该点在上一帧是否有立体匹配（因为有些点跟踪点是当前帧才临时添加的，还没通过左右2帧的三角化得到深度值），应该查看该点是否在Map_prev_r中
                            // int obj_id = ids_sift[i];
                            // bool has_stereo_prev = (prevRightFeaMap.find(obj_id) != prevRightFeaMap.end());
                            // if(USE_TRIANGULATE_TWO_FRAME || has_stereo_prev)
                            {
                                // 记录没有右匹配的背景跟踪点在cur_sift中的序号 
                                sift_no_stereo_bg.push_back(i);
                                // 当前帧没有深度值的bg跟踪点的状态记为2
                                status_sift[i] = 2;
                            }
                            continue;
                        }
                    }
                    int index_valid_match = std::distance(match_stereo.begin(), iter);
                    status_sift_stereo_flow[index_valid_match] = 1;
                    int id_siftdata = match_stereo[index_valid_match];
                    float cord_x = h2_siftpts[id_siftdata].xpos;
                    float cord_y = h2_siftpts[id_siftdata].ypos;
                    float match_x = h2_siftpts[id_siftdata].match_xpos;
                    float match_y = h2_siftpts[id_siftdata].match_ypos;

                    // 对于n_img == 1,已经在cudaSift中完成了此项过滤
                    // 右匹配点超过了规定的图像边界，那么是否应该保存其中的左图像的bg跟踪点？在右图像中快超出图像，意味着该点下一帧很可能离开相机视野？sift跟踪点的作用不是用于长跟踪，而是为当前帧跟踪提供3D-2D的PnP
                    if (n_img == 0 && !(inBorder(cv::Point2f(match_x,match_y)))) 
                    {
                        int gl_id = ids_sift[i];
                        
                        // if(obj_cls == 0 && prev_sift_dep[i] > 0)
                        if(obj_cls == 0)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                        }
                        continue;
                    }
                    
                    // 在左图像的左侧的一些区域内的点要么不可能在右图像中有观测，要么其深度超过了相应的阈值;同时右图像的右侧区域也不可能在左侧有观测点
                    if (obj_cls == 0) 
                    {
                        if(cord_x <= bg_left_border_left_img || match_x >= bg_right_border_right_img)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                    }
                    else if(cord_x <= obj_left_border_left_img || match_x >= obj_right_border_right_img)
                        continue;

                    // 左右图像的立体视差为 左图像点 - 右图像点
                    float disp_x = Sift_->siftData2.valid_disp_x[index_valid_match];
                    float disp_y = Sift_->siftData2.valid_disp_y[index_valid_match];
                    
                    if (disp_x <= 0) 
                    {
                        cout << "Weired! Get a disp_x < 0 for sift!" << endl;
                        if (obj_cls == 0)
                        {
                        
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                        
                            continue;
                        }
                        else
                        {
                            status_sift[i] = 1;
                            id_sift_no_depth.push_back(i);
                            continue;
                        }
                    }
                        
                    float depth = mbf/disp_x;

                    if (obj_cls == 0)
                    {
                        // if (depth >= mThDepthBg || depth < mMinDepthPt) continue;
                        if (depth >= mThDepthBg || depth < 1.5) 
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                        // 左相机坐标系中按照估计深度得到的3D点是否在右相机的视锥体内部。如果不在，则此立体匹配是错误的
                        // float err = (r_cam_3D_plane[0]*cur_un_sift[i].x+r_cam_3D_plane[1]*cur_un_sift[i].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }
                    // 当前帧的物体无论是否运动，都只取25m内的物体
                    else 
                    {
                        // if (depth >= mThDepthObj || depth < mMinDepthPt) continue;
                        // 对于特征点匹配，可以适当减小深度值
                        if (depth >= mThDepthObj || depth < 1.5) continue;
                        // float err = (r_cam_3D_plane[0]*cur_un_sift[i].x+r_cam_3D_plane[1]*cur_un_sift[i].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }

                    float y_shift = Y_shift_right_image/depth;

                    // 右匹配点的v坐标超出图像下边界,则直接放弃该跟踪点
                    if ((cord_y+y_shift) > row - 5) continue;

                    // 从相机2和相机3的校正矩阵P_rect_xx来看,右相机的点的v坐标应该始终比左相机的对象点的v坐标要大,因此disp应该是小于0的!
                    // if(abs(disp_y + y_shift) > 1.0)
                    if(abs(disp_y) > 1.0)
                    {
                        if (obj_cls == 0)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                        else
                        {
                            status_sift[i] = 1;
                            id_sift_no_depth.push_back(i);
                            continue;
                        }
                    }
                    
                    // 是否要直接使用右图像中的匹配点的v坐标?
                    cur_right_sift[i] = cv::Point2f(match_x, match_y);
                    // cur_right_sift[i] = cv::Point2f(match_x, cord_y+y_shift);
                    
                    // 之后记得给当前帧那些来自上一帧跟踪但是在当前帧却没有右图像sift的点 分配深度值（通过depth_map）
                    if(obj_cls == 0)
                    {
                        if(use_tria_stereo)
                            cur_sift_dep[i] = -1.0;
                        else
                            cur_sift_dep[i] = depth;
                    }
                    else
                        cur_sift_dep[i] = depth;
                    
                    status_sift[i] = 1;

                    obj_id = obj_cls_id_sift[i].second;
                    if(obj_id > 0)
                    {
                        if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                        {
                            obj_fea_disp_num[obj_id] = make_pair(disp_x,1);
                        }
                        else
                        {
                            obj_fea_disp_num[obj_id].first += disp_x;
                            obj_fea_disp_num[obj_id].second += 1;
                        }
                    }
                    else
                    {
                        ave_dep_bg_cur_frame += depth;
                        ++num_bg_with_dep;
                        ++num_bg_sift_with_dep;
                    }
                }
            }
            // int num_track_sift_depth = cur_right_sift.size();
            
            Point2f tmp_undist_pt;
            
            vector<float> &valid_disp_x = sift_data->valid_disp_x;
            vector<float> &valid_disp_y = sift_data->valid_disp_y;
            float disp_x, disp_y;
            float cord_x, cord_y, match_x, match_y;
            int validpts_id;
            
            // cout << "start add new stereo sift!" << endl;
            // 往当前帧添加新的sift点，需要该点具有有效的右图像匹配点（深度估计）
            Point2f cur_un_new, cur_un_new_r;
            
            for(int i = 0; i < num_match_stereo; ++i)
            {
                int num_sift_add = cur_sift.size();
                bool no_stereo = false;
                bool invalid_depth = false;
                // 跟踪的点已经算过了
                if(status_sift_stereo_flow[i] == 1) continue;

                validpts_id = valid_match_stereo[i+1];

                cord_x = h2_siftpts[validpts_id].xpos;
                cord_y = h2_siftpts[validpts_id].ypos;
                match_x = h2_siftpts[validpts_id].match_xpos;
                match_y = h2_siftpts[validpts_id].match_ypos;

                info_pt = seg_map_cur.at<Vec2b>(cord_y,cord_x);
                uchar cls_label = info_pt[0];
                
                // 如果sift背景点新点要到下一帧再决定，则这里跳过
                if(add_new_sift_in_next_frame && cls_label == 0) continue;

                // do not consider points of person, rider or bicycle or train
                if(cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7) continue;

                // 太靠近图像顶端的区域，有可能是树木之类的，也可能是建筑，去除这部分区域的点，会损失一部分点，但是可以提高运动估计的精度？
                // 最好是使用全景分割来得到更广泛的类别区分
                // 这里最后是只针对物体了
                if(cord_y < (1.0/6*row) && (cord_x > 1.0/6*col && cord_x < 5.0/6*col))
                {
                    continue;
                }
                
                if(cls_label == 0 && (cord_x <= bg_left_border_left_img || match_x >= bg_right_border_right_img)) continue;
                if(cls_label > 0 && (cord_x <= obj_left_border_left_img || match_x >= obj_right_border_right_img)) continue;

                // 3通道uchar则用Vec3b，单通道uchar则用uchar
                if(!inBorder(cv::Point2f(match_x,match_y))) 
                    continue;
                else
                {
                    disp_x = valid_disp_x[i];
                    disp_y = valid_disp_y[i];
                    // 人为sift匹配是较为准确的立体匹配，即y方向上没有视差。disp_x<=0已经在sift后处理阶段筛除过了
                    // disp_x is the (x_left_img - x_right_img), so disp_x should be > 0
                    //if (disp_x <= 0) continue;

                    if(disp_x <= 0)
                    {
                        cout << "Weired! Get disp_x < 0 for a sift!" << endl;
                        continue;
                    }

                    float depth = mbf/disp_x;

                    float y_shift = Y_shift_right_image/depth;

                    if((cord_y+y_shift) > row - 5) continue;

                    // 当前帧背景点
                    if(n_img == 0 && cls_label == 0)
                    {
                        // 当计算出来的深度太小时，是否可信？sift的立体匹配可信度较高，但是光流匹配就不一定（因为缺少预测值，只能暴力匹配）
                        // 对于新的背景点，如果深度为1m，则下一帧它很可能就不在相机前方了（相机的频率为100hz，如果按10m/s速度前行，那么两帧之间间隔距离为1m）
                        // 但是也可能相机处于静止状态，所以这里也可以取1.5m为最小深度
                        // if(depth >= mThDepthBg || depth < mMinDepthPt) continue;
                        if(depth >= mThDepthBg || depth < 1.5) continue;
                        
                        if(abs(disp_y) > 1.0) continue;

                        // cv::Point2f new_sift(cord_x, cord_y);
                        // undistortedPts(new_sift, tmp_undist_pt, m_camera[0]);
                        // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                        
                        // ++num_new_sift_bg;
                    }
                    // 新的sift点中，即使有静态物体，这里也暂时只选择那些在动态物体深度阈值（2m-25m）内的静态物体点，这样的深度估计比较精确。
                    // 事实上用sift是可以得到较小深度的左右匹配，但是这部分点很可能不够多，但是又无法使用像素点的匹配（因为太近的点的depth_map估计结果肯定是不准确的）
                    // 同时深度太小的点在下一帧不容易被跟踪到（因为靠得近的点光流大）。对于动态物体而言，其可能是朝着远离相机的方向运动（即同向但速度比相机快），那么下一帧它还是可能出现在视野内的
                    // 可以保留物体上深度值较小的点，但需要当前帧该物体的特征跟踪点数较多（至少8个），否则后续无法进行运动估计
                    // 另外分割模型不太准确，不在物体上的点应该如何排除？（物体匹配期间有外点排除机制）
                    else
                    {
                        // 这里应该都是物体上的点
                        assert(cls_label > 0 && "Weirde! Why here get a tracked sift in bg?");
                        // if(depth >= mThDepthObj || depth < mMinDepthPt) continue;
                        if(depth >= mThDepthObj || depth < 1.5) continue;

                        // 是否要保留这部分物体点，后续用depth_map查看是否能获得深度估计。可以
                        // if(abs(disp_y + y_shift) > 0.6)
                        if(abs(disp_y) > 1.0)
                        {
                            // 记录没有右匹配的新sift物体点，后期尝试用depth_map来寻找其深度值
                            id_sift_no_depth.push_back(num_sift_add);
                            invalid_depth = true;
                            // 在id_sift_no_depth后这里千万不能continue,否则下面就忘记添加这个点的右观测点信息了
                            // continue
                        }
                        
                        // cv::Point2f new_sift(cord_x, cord_y);
                        // undistortedPts(new_sift, tmp_undist_pt, m_camera[0]);
                        // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }

                    // 在不插入背景中新sift点的情况下，是否还要额外记录这些有立体匹配的新点
                    if(!add_new_sift_in_next_frame && no_add_new_sift)
                    {
                        if(cls_label == 0)
                        {
                            // continue;

                            Point2f cur_new(cord_x, cord_y);
                            undistortedPts(cur_new, cur_un_new, m_camera[0]);
                            Point2f cur_new_r(match_x, match_y);
                            undistortedPts(cur_new_r, cur_un_new_r, m_camera[1]);
                            new_sift_stereo_prev[validpts_id] = Vec<float,8>(cord_x, cord_y, cur_un_new.x, cur_un_new.y, match_x, match_y, cur_un_new_r.x, cur_un_new_r.y);
                            continue;
                        }
                    }

                    int obj_id = (int)info_pt[1];
                    cur_sift_index.push_back(validpts_id+shift_id);
                    ids_sift.push_back(n_id++);
                    cur_sift.emplace_back(cord_x, cord_y);
                    track_cnt_sift.push_back(1);
                    obj_cls_id_sift.emplace_back(cls_label,obj_id);
                    // 记录当前帧sift点所匹配的上一帧中的sift点的全局obj id，当一个物体在两帧间的特征点匹配数足够多时，可以直接关联此两物体！
                    // 此处为各个物体的新sift点，在进行物体关联之后，要来修改当前帧各个特征点所属物体的全局id，尤其是当前帧新添加的点！！
                    prev_sift_global_obj_id.push_back(obj_id);
                    if(no_stereo)
                    {
                        cur_right_sift.emplace_back(0, 0);
                        // cur_right_sift.emplace_back(match_x, cord_y+y_shift);
                        cur_sift_dep.push_back(-1.0);
                        // 当前帧没有深度值的bg新点的状态记为3
                        status_sift.push_back(3);
                    }
                    else
                    {
                        cur_right_sift.emplace_back(match_x, match_y);
                        // cur_right_sift.emplace_back(match_x, cord_y+y_shift);
                        if(invalid_depth)
                        {
                            // status_sift.push_back(0);
                            status_sift.push_back(1);
                            cur_sift_dep.push_back(-1.0);
                        }
                        else
                        {
                            status_sift.push_back(1);
                            if(use_tria_stereo && cls_label == 0)
                                cur_sift_dep.push_back(-1.0);
                            else
                                cur_sift_dep.push_back(depth);
                        }
                    }

                    ++num_new_sift;

                    // 物体点如果有有效的深度估计,则记录
                    if(obj_id > 0 && !invalid_depth)
                    {
                        if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                        {
                            obj_fea_disp_num[obj_id] = make_pair(disp_x,1);
                        }
                        else
                        {
                            obj_fea_disp_num[obj_id].first += disp_x;
                            obj_fea_disp_num[obj_id].second += 1;
                        }
                    }
                    else if(obj_id == 0 && !no_stereo)
                    {
                        ave_dep_bg_cur_frame += depth;
                        ++num_bg_with_dep;
                        ++num_bg_sift_with_dep;
                    }
                }   
            }
        }
        
        if(!add_new_sift_in_next_frame && !no_add_new_sift)
            cout << "Num of detected new sift (for bg and objs) in cur frame is: " << num_new_sift << endl;
        else
            cout << "Num of detected new sift (only for objs) in cur frame  is: " << num_new_sift << endl;
    }
    
    done_select_sift = true;
    // 当前帧特征点"关联阶段"的最后一个sift（包含跟踪与新检测的）的全局id。注意，实际上当前帧新sift点的最大全局id不是这里的值，因为后续sift中的跟踪外点还会被转为新点！
    // last_id_sift_cur = n_id - 1; 

    // 后处理，将当前帧的部分信息保存为prev
    Sift_->Postprocess();
    
    // ------------------------------------------------------------------------
    printf("Sift select costs: %fms \n", t_o.toc());
}


// 新版的select_sift函数，去除了前期的一些尝试，例如单独在物体mask中检测物体点，简化了函数逻辑；
// 添加了一些新的操作，如在每个bloc中选择ambi值最低的2个点参与F矩阵的估计，如果每个bloc中有深度小于3m的点，则全部加入
void FeatureTracker::select_sift_V2(int frame_count, const cv::Mat &_img1, const cv::Mat &seg_map_prev, const cv::Mat &seg_map_cur, 
                                    const cv::Mat &map_depth_prev, const cv::Mat &flow_map, bool &end_flow_post, bool init_succ_IMU, bool use_mask_img)
{
    cout << "Start selcct sift!" << endl;
    TicToc t_o;

    frame_cnt = frame_count;
    // add_new_FAST_from_sift = false;
    
    // 去除GPU线程中的postprocess()!将sift的后处理（选择有效的match）放在此处，减少GPU中其他任务的等待时间。
    // 对于sift点，其允许检测的最小深度值改为1.5m，因为暴力匹配允许更大的视差估计（1.5m对应的视差值为256）
    // float max_shift_y = Y_shift_right_image/mMinDepthPt;
    float max_shift_y = Y_shift_right_image/1.5;
    vector<int> temp_flow_pt_id;

    // 选择sift跟踪点 和 潜在跟踪点
    if(frame_cnt > 0)
        Sift_->select_flow_matching(Y_shift_right_image/mMinDepthPt, seg_map_prev, seg_map_cur, mask_bg_prev, num_flow_pt_in_bloc, num_temp_flow_pt_in_bloc, 
                                num_long_track_FAST_in_bloc, prev_FAST, track_cnt_FAST, temp_flow_pt_id, prev_color_img_l, use_mask_img);
    
    // 挑选当前帧中的有效立体匹配点
    Sift_->select_stereo_matching(Y_shift_right_image/mMinDepthPt);
    
    num_track_sift_bg = 0;
    num_track_sift_static = 0;
    num_sta_sift_long_track = 0;
    
    int* valid_match_flow_ptr = Sift_->img1.h_matching_pts_flow;
    int num_match_flow = valid_match_flow_ptr[0];
    
    set<int> pt_stereo_with_flow;
    int prev_pts_with_stereo = 0;
    
    // vector<int> id_long_tracked_sift;
    
    vector<int> gl_id_stat_obj_fea;
    
    // 从第2帧开始有跟踪点
    if(frame_count > 0)
    {
        // 部分跟踪点是上一帧就已经被跟踪了，为long_tracked sift
        int num_prev_sift = prev_sift.size();

        assert(num_prev_sift == prev_sift_index.size());
        
        cur_sift.resize(num_prev_sift, cv::Point2f(0, 0));
        cur_sift_index.resize(num_prev_sift, 0);
        status_sift.resize(num_prev_sift,0);

        int id_in_pts = num_prev_sift;

        SiftData *sift_data = &(Sift_->siftData1);
        SiftPoint *h1_siftpts = Sift_->siftData1.h_data;
        
        const vector<float> &valid_disp_x = sift_data->valid_disp_x;
        const vector<float> &valid_disp_y = sift_data->valid_disp_y;
        Point2f prev_pt, prev_pt_r, cur_pt, prev_un_pt, prev_un_pt_r, cur_un_pt;

        float disp_prev, dep_prev;
        float match_xpos_r, match_ypos_r, disp_match_x_r, disp_match_y_r;

        Vec2b info_pt_cur;

        vector<Point2f> temp_track;
        vector<int> temp_track_pt_g_id, temp_track_pt_l_id;

        // vector<int> orig_prev_sift_index = prev_sift_index;

        // 让特征点尽可能均匀地分布在图像中
        // 此外，每个bloc中选取一定数量的跟踪点用于估计F或H矩阵并分解得到R和t
        for(int i = 0; i < 6; ++i)
            for(int j = 0; j < 6; ++j)
                id_best_track_bloc[i][j] = 0;

        // vector<int> pts_stereo_for_F;
        vector<int> id_pts_flow_for_F;
        int l_pt_id, match_id;
        bool old_track = false;
        bool is_stat_obj_fea_track;
        int col_in_bloc, row_in_bloc;

        // 基于极线约束，使用预测的相机运动构建F矩阵来过滤静态点的匹配（注意，这对动态物体点无效，因为动态点仍可能满足相机的极线约束，参考rigidmask）
        bool check_flow_with_epi = true;
        if(!use_motion_to_pred_fea_pos) check_flow_with_epi = false;

        Matrix3d cam_F;
        if(check_flow_with_epi)
        {
            if(frame_count > 1)
            {
                // 构建有效的F矩阵需要非0位移
                if(P_cam_motion.norm() > 0.06)
                {
                    // Quaterniond delta_Q(R_cam_motion);
                    // float delta_angle = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

                    // todo:是否需要预测的旋转较小
                    // if(delta_angle < 0.7)
                    {
                        Matrix3d t_up;
                        t_up << 0.0, -P_cam_motion(2), P_cam_motion(1), P_cam_motion(2), 0.0, -P_cam_motion(0), -P_cam_motion(1), P_cam_motion(0), 0.0;
                        // 本质矩阵到关键矩阵
                        cam_F = K_trans_inv * t_up * R_cam_motion * K_inv;
                    }
                    // else
                    // {
                    //     check_flow_with_epi = false;
                    // }
                }
                else
                    check_flow_with_epi = false;
            }
            else
                check_flow_with_epi = false;
        }

        // 跟踪点的信息记录
        for(int k = 0; k < num_match_flow; ++k)
        {
            old_track = false;
            is_stat_obj_fea_track = false;
            dep_prev = -1.0;
            int validpts_id = valid_match_flow_ptr[(k+1)];

            float prev_x = h1_siftpts[validpts_id].xpos;
            float prev_y = h1_siftpts[validpts_id].ypos;
            float cur_x = h1_siftpts[validpts_id].match_xpos;
            float cur_y = h1_siftpts[validpts_id].match_ypos;

            // 注意，valid_disp_的索引是用k，不是用validpts_id！！
            float disp_x = valid_disp_x[k];
            float disp_y = valid_disp_y[k];
            
            info_pt_cur = seg_map_cur.at<Vec2b>(cur_y,cur_x);
            uchar cls_cur = info_pt_cur[0];

            if (cls_cur == 1 || cls_cur == 2 || cls_cur == 4 || cls_cur == 7) continue;
            
            prev_pt.x = prev_x;
            prev_pt.y = prev_y;
            cur_pt.x = cur_x;
            cur_pt.y = cur_y;

            uchar cls_prev;

            uchar det_cls_prev = seg_map_prev.at<Vec2b>(prev_y,prev_x)(0);

            if(reject_with_F)
            {
                col_in_bloc = prev_x/200;
                row_in_bloc = prev_y/60;

                if(col_in_bloc == 6) col_in_bloc = 5;
                if(row_in_bloc == 6) row_in_bloc = 5;
            }
            
            auto iter = find(prev_sift_index.begin(),prev_sift_index.end(),validpts_id);

            // if it is a long-tracked sift
            if(iter != prev_sift_index.end())
            {
                if(!inBorder(cur_pt)) 
                {
                    if(reject_with_F && det_cls_prev == 0 && cls_cur == 0) num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;

                    continue;
                }

                uchar obj_id = info_pt_cur[1];

                int id_index = std::distance(prev_sift_index.begin(),iter);
                // 注意，如果是上一帧的保留点，则判断其在上一帧的cls应该要用cls_prev而不是det_cls_prev，因为保留点的cls可能与其检测的cls不一致
                cls_prev = obj_cls_id_sift[id_index].first;
                int obj_status_prev = obj_cls_id_sift[id_index].second;
                
                bool invalid_sta_obj_fea = false;
                old_track = true;
                // 纯背景跟踪点 或者 静态物体跟踪点，需要前后2帧的物体标签一致
                if((cls_cur == cls_prev && cls_cur == 0) || (cls_cur == cls_prev && cls_cur > 0 && obj_status_prev == 0))
                {
                    if(check_flow_with_epi)
                    {
                        int gl_obj_id = prev_sift_global_obj_id[id_index];

                        if(gl_obj_id > 0 && status_objs_prev.find(gl_obj_id) == status_objs_prev.end())
                        {
                            cout << "Weired!" << endl;
                            exit(-1);
                        }

                        if(gl_obj_id > 0 && status_objs_prev[gl_obj_id] != 1)
                        {
                            cout << "Weired!" << endl;
                            exit(-1);
                        }

                        int succ = check_flow_with_F(cam_F, prev_pt, cur_pt);

                        // 静态点需要离预测的极线不能太远
                        // 对于不成立的点，只删除纯背景点
                        // 静态物体点可能变为了动态点
                        if(succ <= 0)
                        {
                            if(cls_cur == cls_prev && cls_cur == 0)
                            {
                                status_sift[id_index] = 0;

                                if(reject_with_F)
                                    num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;
                                
                                continue;
                            }
                            else
                            {   
                                invalid_sta_obj_fea = true;
                            }
                        }
                    }
                }

                // 下面遍历的这些条件，很可能不能覆盖所有的条件，为了避免某些遗漏条件下的上一帧点被保留下来，在每一条件下单独修改status，即status_sift[id_index] = 1;
                if (cls_cur == cls_prev && cls_cur == 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/6.0)
                {
                    // 后续根据物体的运动状态的判定结果和这里的局部obj_id来更改sift点的全局id！obj_cls_id_sift这个量由上一帧的量修改为表示当前帧的，然后用reduceVector来去除多余
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(0, obj_id);
                    // 记录背景上的sift点个数
                    ++num_track_sift_bg;
                    // 记录跟踪的静态sift点数
                    ++num_track_sift_static;
                    // 记录长跟踪静态点的个数
                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;

                    if(reject_with_F && only_use_track_sift_for_F) id_bg_track_sift.push_back(id_index);
                }
                // 上一帧的静态物体sift点（即上一帧的全局obj id为0，但cls不为0），即使其（所在物体）在当前两帧之间变为动态的，其光流值也不会太大（物体刚启动运动）。
                // 这些sift点暂时不归为背景sift点，等到确定该物体是否运动后再归类！
                else if (cls_cur == cls_prev && cls_cur > 0 && obj_status_prev == 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/6.0)
                {
                    // 物体id按当前帧的局部编号来，等物体关联之后再等其进行更新。
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_prev, obj_id);
                    ++num_track_sift_static;

                    // id_bg_track_sift.push_back(id_in_pts);

                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;

                    // 系统第3帧开始才会有相机运动的预测
                    // 该静态物体点需要在上面通过极线约束的检验
                    if(frame_count > 1 && reject_with_F && !invalid_sta_obj_fea)
                    {
                        // 静态物体的跟踪点，如果基于预测的相机运动的重投影误差小于阈值，
                        // 则可以参与当前帧相机运动的F矩阵的估计（因为有些情景除了路边停的车，近处几乎没有任何可靠的点，甚至没有车道线），前提是上一帧的该物体点有立体匹配（深度的精度较高）
                        int gl_id = ids_sift[id_index];
                        // 只选择那些在上一帧有立体匹配的静态物体点
                        if(prevRightFeaMap.find(gl_id) != prevRightFeaMap.end())
                        {
                            int prev_dep = prev_sift_dep[id_index];
                            // 上一帧12m以内的静态物体点。需要该点在上一帧中立体三角化成功（对于有立体匹配的最新跟踪点，其深度值只来自于立体三角化或前后帧三角化，不来自于运动变换更新）
                            if(prev_dep > 0 && prev_dep < 12.0)
                            {
                                // 对每个map进行索引时，最好先判断一下相关的key是否存在，否则如果不在的话，只要进行了索引就会创建默认的键值对，value是随机的！
                                if(prev_un_Fea_map.find(gl_id) != prev_un_Fea_map.end())
                                {
                                    Vec4f &prev_fea = prev_un_Fea_map[gl_id];
                                    float prev_x = prev_fea(0) * prev_dep;
                                    float prev_y = prev_fea(1) * prev_dep;
                                    Vector3d prev_pt(prev_x,prev_y,prev_dep);
                                    
                                    Vector3d pred_cur = (R_cam_motion * prev_pt + P_cam_motion);
                                    if(pred_cur(2) > 0)
                                    {
                                        float pred_x = pred_cur(0)/pred_cur(2);
                                        float pred_y = pred_cur(1)/pred_cur(2);

                                        Point2f cur_un_fea;
                                        undistortedPts(cur_pt,cur_un_fea,m_camera[0]);

                                        float err = (pred_x - cur_un_fea.x)*(pred_x - cur_un_fea.x) + (pred_y - cur_un_fea.y)*(pred_y - cur_un_fea.y);
                                        // 重投影误差小于5个像素点
                                        if(err < (5.0/FOCAL_LENGTH_X)*(5.0/FOCAL_LENGTH_X))
                                        {
                                            is_stat_obj_fea_track = true;
                                            
                                            if(only_use_track_sift_for_F)
                                                id_bg_track_sift.push_back(id_index);
                                            // else
                                            // {
                                            //     // 记录该静态物体跟踪点的全局id，因为后续要先reduceVector，使用其全局id查看其在剩下vector中的序号
                                            //     gl_id_stat_obj_fea.push_back(ids_sift[id_index]);
                                            // }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // 上上帧与上一帧之间是动态或新的物体，其在上一帧到当前帧之间也可能变成静态的。这种情况在这里可以也按动态物体的标准来衡量（虽然可能会造成一定的误匹配，但概率会比较小）。
                // 后续根据物体运动状态的判定来更改其对应sift点的全局id。
                else if (cls_cur == cls_prev && obj_status_prev > 0 && abs(disp_x) < 1280/6.5 && abs(disp_y) < 384/4.5)
                //else if (cls_label == obj_cls_id_sift[id_index].first && obj_id > 0 && abs(disp_x) < 1280/4.0 && abs(disp_y) < 384/2.0)
                {
                    // 物体id按当前帧的局部编号来，等物体关联之后再等其进行更新。
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_prev, obj_id);
                    // 对于上一帧的新物体的跟踪点，这里暂时不加入
                }
                // 漏检或者错检的物体的配对，在系统前两帧暂时不考虑，因为首帧无法得知各物体的运动状态！！对于首帧的误检物体的特征点匹配，依赖于flow_map提供FAST点的预测和匹配。
                // 上一帧点为运动或新的物体，且与当前帧匹配点类别不同（当前帧可能为背景，也可能为物体），则可能是当前帧漏检该运动物体（为背景点）或者错分类该物体，则按照动态光流来约束
                else if (cls_cur != cls_prev && obj_status_prev > 0 && abs(disp_x) < 1280/6.5 && abs(disp_y) < 384/4.5)
                {
                    // 物体id按当前帧的局部编号来，等物体关联之后再对其进行更新。
                    // 物体类别则与上一帧的匹配物体点对齐，当前帧匹配点的检测分类可以从seg_map中查询
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_prev, obj_id);
                }
                // 上一帧点为静态物体(明确不是背景），且与当前帧匹配点类别不同（当前帧可能为背景或其他物体），则可能是当前帧漏检该静态物体（为背景点）或者错分类该物体，则按照静态光流来约束
                else if (cls_cur != cls_prev && cls_prev >0 && obj_status_prev == 0 && abs(disp_x) < 1280/8.0 && abs(disp_y) < 384/6.0)
                {
                    // 物体类别则与上一帧的匹配物体点对齐，当前帧匹配点的检测分类可以从seg_map中查询
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_prev, obj_id);
                    ++num_track_sift_static;
                    if(track_cnt_sift[id_index] > 1) ++num_sta_sift_long_track;
                }
                // 上一帧点为背景点，当前帧点为物体点，则可能是上一帧漏检了该物体且该点没有与上上帧的物体SIFT点相关联（因为没有被校正cls），则此时无法确定该物体是否为动态的。还是采用动态光流的阈值
                else if (cls_cur != cls_prev && cls_prev == 0 && abs(disp_x) < 1280/6.5 && abs(disp_y) < 384/4.5)
                {
                    // 如果该背景点被多帧观测（多帧观测的情况下则确定该点原本为背景点），则不太可能当前帧突然变为物体点
                    // 如果该点在上一帧中没有深度值，则不采用该跟踪点，因为这样无法参与估计该漏检物体的运动！
                    if (track_cnt_sift[id_index] > 1) 
                    {
                        continue;
                    }
                    
                    // 对于上一帧的漏检物体点，要求其有深度值。
                    if(prev_sift_dep[id_index] <= 0)
                    {
                        int gl_id = ids_sift[id_index];
                        if(prevRightFeaMap.find(gl_id) == prevRightFeaMap.end())
                        {
                            continue;
                        }
                        else
                        {
                            float dep;
                            float prev_x_r = prevRightFeaMap[gl_id].x;
                            dep = mbf/(prev_x - prev_x_r);
                            if(dep > 1.5 && dep < mThDepthObj)
                                prev_sift_dep[id_index] = dep;
                            else
                                continue;
                        }
                    }
                    
                    // 上一帧的背景点对齐到当前帧的物体点时，使用当前帧的物体类别，后续则可以根据上一帧的点的全局id来判断该点是否为背景点
                    obj_cls_id_sift[id_index] = std::pair<uchar, int>(cls_cur, obj_id);
                }
                else
                {
                    if(reject_with_F && det_cls_prev == 0 && cls_cur == 0) 
                    {
                        num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;
                    }
                    
                    continue;
                }
                
                status_sift[id_index] = 1;
                l_pt_id = id_index;
                ++num_track_sift;
                cur_sift[id_index] = cur_pt;
                // 当前点的匹配点在其siftdata检测点集中的index,这里是siftdata2
                match_id = h1_siftpts[validpts_id].match;
                cur_sift_index[id_index] = match_id;

                // 注意，上一帧的就已被跟踪的sift的stereo不一定是来自sift的暴力匹配。
                // 对于背景点，是来自于暴力匹配；而对于物体点，其立体匹配可能是来自于depth_map（其实也可以让depth_map只提供深度值，而不提供右图像观测）
                // int gl_id = ids_sift[id_index];
                // if(prevRightFeaMap.find(gl_id) != prevRightFeaMap.end()) 
                // {
                //     // has_stereo = true;
                //     dep_prev = prev_sift_dep[id_index];
                // }

                if(add_new_sift_in_next_frame)
                {
                    // 如果关联的两个点有一个不是背景点，则认为是物体跟踪点
                    if(cls_prev != 0 || cls_cur != 0)
                    {
                        // 此mask是标注上一帧中所有被当前帧跟踪到的点，为了后续物体关联时采样用于运动估计的像素点关联而用
                        circle(mask_prev_fea_objs, prev_sift[id_index], 3, 0, -1);
                    }
                    // tracked_objs_long_bg.insert(validpts_id);
                }
            }
            // 如果被跟踪点是上一帧的新点。这只会是纯背景跟踪点
            else
            {
                // 只保留上一帧新的背景跟踪点
                // cls_prev = seg_map_prev.at<Vec2b>(prev_y, prev_x)(0);
                cls_prev = det_cls_prev;
                
                // 上一帧被检测的物体的点，无论是被跟踪点还是新点，都已经在上一帧添加了，所以不会出现在当前帧的新跟踪点中。
                if(cls_prev != 0) 
                {
                    continue;
                }

                // 纯背景跟踪点需要通过极线约束的检验
                if(check_flow_with_epi && cls_cur == 0)
                {
                    int succ = check_flow_with_F(cam_F, prev_pt, cur_pt);

                    // 背景点需要离预测的极线不能太远
                    if(succ <= 0)
                    {
                        // status_sift[id_index] = 0;

                        if(reject_with_F)
                            num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;
                        
                        continue;
                    }
                }

                // 是否有立体匹配
                auto it = find(temp_flow_pt_id.begin(),temp_flow_pt_id.end(),validpts_id);
                
                bool valid_prev_stereo = false;
                // 如果该跟踪点在上一帧没有立体匹配
                if(it == temp_flow_pt_id.end())
                {
                    temp_track.emplace_back(prev_x, prev_y);
                    temp_track_pt_g_id.push_back(n_id);
                    temp_track_pt_l_id.push_back(id_in_pts);
                }
                else
                {
                    pt_stereo_with_flow.insert(validpts_id);

                    float *prev_info_ptr = Sift_->prev_stereo_match_info + 5 * validpts_id;
                    match_xpos_r = prev_info_ptr[3];
                    match_ypos_r = prev_info_ptr[4];

                    disp_match_x_r = prev_x - match_xpos_r;
                    // disp_match_y_r = prev_y - match_ypos_r;
                    assert(disp_match_x_r > 0);
                    dep_prev = mbf/disp_match_x_r;

                    // 图像上半部分的跟踪点，要求深度值较小
                    int row_in_bloc = prev_y/60;

                    // todo: 是否可以接受上半图像中深度较大的点？有些路况基本无法在地面上检测到特征点，因此还是需要较远的点
                    if(0 && row_in_bloc < 3 && dep_prev > 5.0)
                    {
                        // 图像上半部分的跟踪点数不需要再统计，因为不会在该区域检测新的FAST背景点
                        // if(reject_with_F && det_cls_prev == 0 && cls_cur == 0) num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;
                        continue;
                    }
                    
                    // 根据ORB-SLAM2的设置，大于40倍基线的深度的背景点为远点，这些远点的立体匹配一般比较不准确（会影响估计F和H以及PnP和LBA），因此放弃它们的立体匹配，而是用前后2帧三角化来计算它们的深度
                    // 这里取得比40倍基线(KITTI基线长度为0.54m，40倍为21m)
                    if(cls_cur == 0) 
                    {
                        if(dep_prev > mMinDepthPt && dep_prev < 21)
                            valid_prev_stereo = true;
                    }
                    else
                    {
                        // 如果漏检的物体点在上一帧的深度超过阈值，则放弃该物体点跟踪
                        if(dep_prev > mMinDepthPt && dep_prev <= mThDepthObj)
                            valid_prev_stereo = true;
                        else
                            continue;
                    }
                }
                
                undistortedPts(prev_pt, prev_un_pt, m_camera[0]);

                prev_sift.emplace_back(prev_x, prev_y);
                cur_sift.emplace_back(cur_x,cur_y);
                prevLeftFeaMap[n_id] = prev_pt;
                prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);
                ids_sift.push_back(n_id);

                // 添加上一帧的右匹配点
                if(valid_prev_stereo)
                {
                    prev_pt_r.x = match_xpos_r;
                    prev_pt_r.y = match_ypos_r;
                    undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);

                    prevRightFeaMap[n_id] = prev_pt_r;
                    prev_un_r_Fea_map[n_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0 ,0.0);
                }

                // 这部分点寻求使用2帧的三角化来获得在上一帧中的深度值，包括首帧左右2帧或者前后2帧
                if(use_tria_stereo && cls_cur == 0)
                {
                    prev_sift_dep.push_back(-1.0);
                }
                else
                {
                    // 如果匹配的当前点为物体点，或者背景点可以在这里直接三角化
                    if((cls_cur > 0 || !use_tria_stereo) && valid_prev_stereo)
                    {
                        prev_sift_dep.push_back(dep_prev);
                    }
                    else
                    {
                        // 等待后续完成深度预测
                        prev_sift_dep.push_back(-1.0);
                    }
                }

                // 记录当前帧该跟踪点在当前帧的检测sift点集中的序号
                match_id = h1_siftpts[validpts_id].match;
                cur_sift_index.push_back(match_id);
                prev_sift_global_obj_id.push_back(0);
                // 因为上一帧该点是背景点，因此按照当前帧匹配点的检测cls 和 local obj_id
                obj_cls_id_sift.emplace_back(cls_cur,info_pt_cur(1));
                track_cnt_sift.push_back(1);
                status_sift.push_back(1);
                ++n_id;

                l_pt_id = id_in_pts;
                // ++num_sift_bg_cur;
                ++num_track_sift;

                // 记录用于估计F矩阵的sift跟踪点在cur_sift中的序号
                
                // 如果上一帧和当前帧都是背景点，则是背景点
                if(cls_cur == 0) 
                {
                    if(reject_with_F && only_use_track_sift_for_F) 
                        id_bg_track_sift.push_back(id_in_pts);
                }

                ++id_in_pts;
            }

            // 记录跟踪点中有立体匹配的点
            if(find(temp_flow_pt_id.begin(),temp_flow_pt_id.end(),validpts_id) != temp_flow_pt_id.end())
            {
                // set不会有重复元素
                pt_stereo_with_flow.insert(validpts_id);
                ++prev_pts_with_stereo;
            }

            // 选取背景跟踪点用于计算相机运动的F矩阵
            if(reject_with_F)
            {
                // 如果是纯背景跟踪点，或者静态物体的跟踪点
                if((cls_prev == 0 && cls_cur == 0) || is_stat_obj_fea_track)
                {   
                    // 上半部分的跟踪点要求深度较小（一般都是路杆、路牌之类的），这类点一般较少
                    if(row_in_bloc < 3)
                    {
                        // 上半部分的跟踪点是否要参与F矩阵的估计？对估计R的影响大吗？
                        // continue;
                        
                        // 近处的物体点不允许出现在图像上半部分
                        if(is_stat_obj_fea_track) continue;

                        // 上半部分每个bloc只添加5个点。由于CudaSift中对上半图像中没有深度或者深度较大的点 有数量限制，因此这里可以适当增大bloc允许的点数
                        int num_total = 5;
                        // 上半部分中如果有近点，则一般不会是只有一个点，可以多添加这部分点非地面点，有利于F矩阵的估计
                        // todo:但是深度误估计（即sift立体匹配错误点）的点较多怎么办？
                        if(dep_prev > 0)
                        {
                            // if(dep_prev < 4.5)
                            //     num_total = 10;
                            // else if(dep_prev < 16)
                            //     num_total = 9;

                            if(dep_prev < 16)
                                num_total = 9;
                        }
                        
                        if(id_best_track_bloc[row_in_bloc][col_in_bloc] < num_total)
                        {
                            id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;
                            
                            if(only_use_track_sift_for_F) 
                                pts_for_cal_F.insert(-l_pt_id);
                            else
                                id_pts_flow_for_F.push_back(match_id);
                        }
                    }
                    else
                    {
                        int num_total = 6;
                        // 近处的点多添加一些
                        // 最下面2行的每个框增加多一些点
                        if(row_in_bloc > 3)
                        {
                            num_total = NUM_FEA_IN_BLOC;
                            // 如果该框中出现了静态物体，则一般而言物体会占据比较大的区域，则适当减少一下点
                            if(is_stat_obj_fea_track)
                                num_total = 6;
                        }
                        
                        if(id_best_track_bloc[row_in_bloc][col_in_bloc] < num_total)
                        {
                            id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;

                            if(only_use_track_sift_for_F) 
                                pts_for_cal_F.insert(-l_pt_id);
                            else
                                id_pts_flow_for_F.push_back(match_id);
                        }
                    }

                    // 记录有立体匹配的背景跟踪点，用于选择分解得到的R和t，并估计t的真实尺度
                    // if(has_stereo)
                    // {
                    //     pts_stereo_for_F.push_back(l_pt_id);
                    // }
                }
            }
        }

        // 为部分跟踪的sift点（即背景中的新跟踪点）在上一帧寻找右观测点
        if(!temp_track.empty())
        {
            float ave_disp_x_bg = mbf/(2.0/3*mMinDepthPt + 1.0/3*mThDepthBg);
            float ave_disp_y_bg = Y_shift_right_image/(2.0/3*mMinDepthPt + 1.0/3*mThDepthBg);
            vector<Point2f> temp_track_r;
            float r_x, r_y;
            for(auto &pt:temp_track)
            {
                float disp = map_depth_prev.at<float>(pt.y,pt.x);
                if(disp <= 0)
                {
                    // assert(disp>0 && "Why value in disp_map <= 0?");
                    r_x = pt.x - ave_disp_x_bg;
                    r_y = min(pt.y+ave_disp_y_bg, (float)(row-5));
                }
                else
                {
                    float depth = mbf/disp;
                    float shift_y = Y_shift_right_image/depth;
                    r_x = pt.x - disp;
                    r_y = min(pt.y+shift_y, (float)(row-5));
                }

                if(r_x >= 5 && r_x <= bg_right_border_right_img)
                    temp_track_r.emplace_back(r_x, r_y);
                else if (r_x < 5)
                    temp_track_r.emplace_back(5, r_y);
                else
                    temp_track_r.emplace_back(bg_right_border_right_img, r_y);
            }

            vector<float> err;
            vector<uchar> status_temp;
            cv::calcOpticalFlowPyrLK(prev_img, prev_img_r, temp_track, temp_track_r, status_temp, err, cv::Size(21, 21), 2, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            // FLOW_BACK = false;
            // reverse check  逆向光流计算，用上面光流计算出的当前帧中特征点位置 再 计算到上一帧的光流匹配，如果得到检测位置和上一帧的点原始位置距离相差很小，则认为光流成功
            FLOW_BACK = true;
            if(FLOW_BACK)
            {
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = temp_track;
                vector<float> reverse_err;
                cv::calcOpticalFlowPyrLK(prev_img_r, prev_img, temp_track_r, reverse_pts, reverse_status, reverse_err, cv::Size(15, 15), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_FAST, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status_temp.size(); ++i)
                {
                    if(status_temp[i] && reverse_status[i] && distance(temp_track[i], reverse_pts[i]) < 0.8)
                    {
                        continue;
                    }
                    else
                        status_temp[i] = 0;
                }
            }

            Point2f prev_un_pt_r;
            for (size_t i = 0; i < status_temp.size(); ++i)
            {
                Point2f &prev_pt = temp_track[i];
                Point2f &prev_pt_r = temp_track_r[i];
                float disp_x_stereo = prev_pt.x - prev_pt_r.x;
                float disp_y_stereo = prev_pt.y - prev_pt_r.y;
                int l_id = temp_track_pt_l_id[i];
                uchar cls_ = obj_cls_id_sift[l_id].first;
                int row_in_bloc = prev_pt.y/60;
                // 没有或者不正确的匹配
                if (!status_temp[i] || abs(disp_y_stereo) > 1.2 || disp_x_stereo <= 0)
                {
                    // 物体点在上一帧的漏检点如果没有深度值或者深度值，则放弃该点（其实也可以不用放弃，用作2个物体的点关联，以便开启像素关联验证?）
                    if(cls_ > 0) 
                        status_sift[l_id] = 0;
                    // 如果背景跟踪点在上一帧没有立体匹配，且此系统不允许2-frame的三角化深度测量，则放弃此点
                    else if(!USE_TRIANGULATE_TWO_FRAME)
                        status_sift[l_id] = 0;
                    else
                    {
                        // todo:对于图像上半部分的跟踪点，如果在上一帧没有立体匹配，是否放弃该跟踪点？应该要保留，有利于估计矩阵的旋转
                        if(0 && row_in_bloc < 3)
                        {
                            status_sift[l_id] = 0;
                        }
                    }
                    continue;
                }

                // 对于图像上半部分的点，对深度要求进一步加强
                // if(row_in_bloc < 3)
                // {
                //     if(cls_ == 0)
                //     {
                //         // 对于纯背景跟踪点，如果上一帧该点深度大于18.0m，则放弃该立体匹配（认为不够准确），但不放弃该跟踪点！小于1.5的点还是可以保留的，例如距离汽车很近的杆，且汽车静止
                //         if(disp_x_stereo < mbf/18.0)
                //         {
                //             // 上半部分的跟踪点数不需要再统计
                //             // if(reject_with_F) num_flow_pt_in_bloc[row_in_bloc][col_in_bloc] -= 1;
                //             // status_sift[l_id] = 0;
                //             continue;
                //         }
                //     }
                // }
                
                // 物体点在上一帧的漏检点如果深度值超过阈值，则放弃该点
                if(cls_ > 0)
                {
                    if(disp_x_stereo <= mbf/mThDepthObj || disp_x_stereo > 256)
                    {
                        status_sift[l_id] = 0;
                        continue;
                    }
                }
                // 纯背景点的深度值不能太大，超过20m就认为不可靠；也不能太小
                else if(disp_x_stereo < mbf/20.0 || disp_x_stereo > 256) 
                    continue;

                ++prev_pts_with_stereo;
                int gl_id = temp_track_pt_g_id[i];
                prevRightFeaMap[gl_id] = prev_pt_r;
                undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);
                prev_un_r_Fea_map[gl_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0 ,0.0);

                // 如果不需要三角化才得到深度值（即认为立体校正足够好）
                if(cls_ > 0 || !use_tria_stereo) 
                {
                    float dep = mbf/disp_x_stereo;
                    prev_sift_dep[l_id] = dep;
                }
                
                // if(obj_cls_id_sift[l_id].first == 0)
                // {
                //     pts_stereo_for_F.push_back(l_id);
                // }
            }
        }

        valid_match_flow_ptr[0] = 0;
        
        // 这些上一帧有立体匹配的sift点，采用与FAST一样的跟踪方式，因此就相当于是上一帧的新FAST点了！
        // 因此，FAST的跟踪要在这里完成之后才进行。注意，这部分有立体匹配的sift点可以出现在图像的上半部分，此时其深度值比较小
        bool find_flow_for_stereo_match = true;
        if(find_flow_for_stereo_match)
        {
            if(!temp_flow_pt_id.empty())
            {
                SiftPoint *all_pt_ptr = Sift_->siftData1.h_data;
                float prev_x, prev_y, prev_x_r, prev_y_r;
                for(auto iter: temp_flow_pt_id)
                {
                    // 注意，temp_flow_pt_id中前面部分点是跟踪点，即在pt_stereo_with_flow中，应该略去！
                    if(pt_stereo_with_flow.find(iter) != pt_stereo_with_flow.end()) continue;

                    float *prev_info_ptr = Sift_->prev_stereo_match_info + 5 * iter;
                    prev_x = all_pt_ptr[iter].xpos;
                    prev_y = all_pt_ptr[iter].ypos;
                    
                    prev_x_r = prev_info_ptr[3];
                    prev_y_r = prev_info_ptr[4];
                    
                    assert(prev_x - prev_x_r > 0);

                    float dep = mbf/(prev_x-prev_x_r);
                    if(dep > 24) continue;

                    prev_FAST.emplace_back(prev_x,prev_y);
                    prev_FAST_global_obj_id.push_back(0);
                    track_cnt_FAST.push_back(1);
                    ids_FAST.push_back(n_id);
                    // status_FAST.push_back(1);
                    obj_cls_id_FAST.emplace_back(0,0);

                    prev_pt.x = prev_x;
                    prev_pt.y = prev_y;
                    prevLeftFeaMap[n_id] = prev_pt;

                    undistortedPts(prev_pt, prev_un_pt, m_camera[0]);
                    prev_un_Fea_map[n_id] = Vec4f(prev_un_pt.x, prev_un_pt.y, 0.0, 0.0);

                    prev_pt_r.x = prev_x_r;
                    prev_pt_r.y = prev_y_r;
                    prevRightFeaMap[n_id] = prev_pt_r;

                    undistortedPts(prev_pt_r, prev_un_pt_r, m_camera[1]);
                    prev_un_r_Fea_map[n_id] = Vec4f(prev_un_pt_r.x, prev_un_pt_r.y, 0.0, 0.0);

                    if(use_tria_stereo)
                        prev_FAST_dep.push_back(-1.0);
                    else
                    {
                        float dep = mbf/(prev_x-prev_x_r);
                        prev_FAST_dep.push_back(dep);
                    }

                    ++n_id;
                }
            }
        }

        // 通知FAST track线程已经完成了来自sift的新FAST的添加。
        // 之后再在FAST的跟踪函数中查看各个bloc中特征点数是否足够，如果不够，添加上一帧图像中检测的FAST新点（有立体匹配的在前，无立体匹配的在后）（可以不在上一时刻检测，而是在当前帧中再从上一帧的图像剩余区域检测FAST点，这样就不需要排除多余的FAST点）
        add_new_FAST_from_sift = true;

        cout << "Num of tracked prev sift fea with stereo match: " << prev_pts_with_stereo << endl;

        // 是否需要根据极线约束去除异常匹配点
        // 如果需要，是否只使用sift跟踪来估计F矩阵？
        if(reject_with_F && only_use_track_sift_for_F)
        {
            if(pts_for_cal_F.size() > 8)
                rejectWithFV1(true, init_succ_IMU, pts_for_cal_F);
            else
                rejectWithFV1(true, init_succ_IMU);
        }
        else
        {
            // 删除无有效跟踪的prev_sift点
            reduce_invalid_fea(true);
        }
        
        if(!track_cnt_sift.empty())
        {
            for(auto &n : track_cnt_sift)
                ++n;
        }
        // assert(num_track_sift == ids_sift.size() && "Something wrong with num_track_sift!");

        num_track_sift = cur_sift.size();
        cout << "num of tracked sift in cur frame: " << num_track_sift << endl;
        
        if(num_track_sift > 0)
        {
            cv::Point2f temp_pt(0.0,0.0);
            cur_right_sift.resize(num_track_sift,temp_pt);
            cur_sift_dep.resize(num_track_sift,-1.0);

            // 如果上面删减了失败的prev_sift点，则剩下的点都是有效跟踪点了，因此这里可以把所有点的status赋为0，下面根据立体匹配结果来最后决定是否保留该跟踪点。
            // 也可以换种思路，不重置，下面根据跟踪点是否有立体匹配来决定是否删除该点
            if(!status_sift.empty()) status_sift.clear();
            status_sift.resize(num_track_sift,0);

            // 使用背景跟踪点中的部分点来估计F矩阵
            if(reject_with_F && !only_use_track_sift_for_F)
            {
                vector<int> valid_track;
                // id_pts_flow_for_F保存的是按照ambi score从小到大排序的有效跟踪点（包含纯背景点，和近处的静态物体点）的sift match的index
                // 为每个bloc选取最大固定数量的最佳跟踪点来估计F矩阵
                for(int k = 0; k < id_pts_flow_for_F.size(); ++k)
                {
                    int index_in_flow_match = id_pts_flow_for_F[k];
                    auto it_index = find(cur_sift_index.begin(),cur_sift_index.end(),index_in_flow_match);

                    // 有些跟踪点在寻找上一帧立体匹配时可能被删除了
                    // assert(it_index != cur_sift_index.end());
                    if(it_index == cur_sift_index.end()) continue;

                    int l_index = std::distance(cur_sift_index.begin(),it_index);
                    
                    valid_track.push_back(l_index);

                    // id_bg_track_sift中只添加纯背景跟踪点，而不添加静态物体跟踪点（它们只用来估计F或H矩阵），因为即使物体点是F或H估计的外点，也不删除该物体跟踪点（因为该物体可能变为动态的
                    if(obj_cls_id_sift[k].first == 0)
                    {
                        id_bg_track_sift.push_back(k);
                        ++num_sift_bg_cur;
                    }

                    // uchar cls_cur = obj_cls_id_sift[l_index].first;
                    // 跟踪点的cls_cur为0，意味着上一帧和当前帧该点均位于背景中。但是添加的静态跟踪点中还包含静态物体点！
                    // if(cls_cur == 0)
                    // {
                        // 这里不需要再进行分区挑选了，因为在上面已经进行了

                        // Point2f &prev_pt = prev_sift[l_index];
                        // int col_in_bloc = prev_pt.x/200;
                        // int row_in_bloc = prev_pt.y/60;

                        // if(col_in_bloc == 6) col_in_bloc = 5;
                        // if(row_in_bloc == 6) row_in_bloc = 5;
                        
                        // if(row_in_bloc < 3)
                        // {
                        //     // continue;

                        //     // 上半部分每个bloc只添加一个点
                        //     if(id_best_track_bloc[row_in_bloc][col_in_bloc] < 1)
                        //     {
                        //         id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;

                        //         pts_for_cal_F.insert(-l_index);
                        //     }

                        // }
                        // else
                        // {
                        //     int num_total = 2;
                        //     if(row_in_bloc > 3) num_total = 3;
                        //     // 近处的点多添加一些，每个bloc三个点
                        //     if(id_best_track_bloc[row_in_bloc][col_in_bloc] < num_total)
                        //     {
                        //         id_best_track_bloc[row_in_bloc][col_in_bloc] += 1;

                        //         pts_for_cal_F.insert(-l_index);
                        //     }
                        // }

                        // 由于id_pts_flow_for_F中的跟踪是按照匹配score从小到大排序放入的，因此这里pts_for_cal_F的点也是越靠前匹配质量越高！
                        // pts_for_cal_F.insert(-l_index);
                    // }
                }

                // 优先取前 n% 的点参与F或H矩阵的估计
                int num_valid = valid_track.size();
                int num_ada = num_valid * 4/5.0;
                // todo:是否要专门添加上半图像中的点？不需要，如果近点实在不足，自然会添加远点
                bool add_up_half_img_fea = false;
                num_up_half = 0;
                int total_add = 0;
                for(int i = 0; i < num_valid; ++i)
                {
                    int id = valid_track[i];
                    float y = cur_sift[id].y;
                    // 如果已经添加了足够多的sift跟踪点，则看是否有足够多的远点
                    // if(i > num_ada && total_add > 20)
                    if(i > num_ada)
                    {
                        // 选取一定的远点用以估计R
                        if(add_up_half_img_fea && num_up_half < 5)
                        {
                            if(y >= 3)
                                continue;
                        }
                        else
                        {
                            break;
                        }
                    }

                    pts_for_cal_F.insert(-id);
                    ++total_add;

                    if(y/60 < 3) ++num_up_half;
                }
            }
        }
    }

    // 当前帧跟踪到的点中的最大全局id。其中有些跟踪点是无效的（已经被确认无效，或者之后setMask时会跟FAST过近而无效），但因为每个点的id都是独一无二的，因此后面的新点的id肯定不会比这里的跟踪点大
    if(!ids_sift.empty())
        last_id_track_fea_cur = *(std::max_element(ids_sift.begin(),ids_sift.end()));
    
    // 记录各个检测物体上的特征点的平均深度和点数，当没有使用depth_map时可以提供各个物体上的点的近似深度值预测
    // if(use_motion_to_pred_fea_dep)
    {
        if(!obj_fea_disp_num.empty()) 
            obj_fea_disp_num.clear();
    }

    num_bg_sift_with_dep = 0;
    
    // 寻找跟踪sift点在当前帧的右图像匹配点。并添加新的sift点（有立体匹配或者深度值的物体点）
    if(STEREO && !_img1.empty())
    {
        int* valid_match_stereo = Sift_->img2.h_matching_pts_stereo;
        SiftData* sift_data = &(Sift_->siftData2);
        
        SiftPoint *h2_siftpts = Sift_->siftData2.h_data;
        int init_track = 0, num_track = num_track_sift;
        int num_sift_bg_cur_ = num_sift_bg_cur;
        
        num_new_sift_bg = 0;
        int num_new_sift = 0;   

        int num_match_stereo = valid_match_stereo[0];
        // todo： 暂不添加当前帧中有立体匹配的背景sift新点，改为下一帧根据flow匹配结果添加
        // add_new_sift_in_next_frame = true;

        if(num_match_stereo > 0)
        {
            vector<uchar> status_sift_stereo_flow(num_match_stereo, 0);
            Vec2b info_pt;
            if(num_track > 0)
            {
                // 也可以像上面一样使用reserve和assign函数来转化为vector，只是下面这种方式更简洁
                // 同样的，vetcor vec(ptr_begin, ptr_end)，其中ptr_end位置处的元素是不算的
                vector<int> match_stereo(valid_match_stereo + 1, valid_match_stereo + 1 + num_match_stereo);

                assert(match_stereo.size() == num_match_stereo && "Something wrong with stereo-matching number!");

                uchar obj_cls;
                int obj_id;
                //printf("stereo image; track feature on right image\n");
                // cur left ---- cur right
                for (int i = init_track; i < num_track; ++i)
                {
                    // 这里是否要滤除，取决于上面是否将status_sift所有值进行重置
                    // if(status_sift[i] == 0) continue;

                    int id_pt = cur_sift_index[i];

                    vector<int>::iterator iter = std::find(match_stereo.begin(),match_stereo.end(),id_pt);
                    // 使用跟踪点的cls而不是obj_id来判断,是因为当前帧有些背景点可能关联的是上一帧的物体(当前帧漏检)，这样可以表示该跟踪点是否为纯背景跟踪点
                    uchar obj_cls = obj_cls_id_sift[i].first;
                    // 该点在当前帧没有立体匹配
                    if (iter == match_stereo.end()) 
                    {
                        // 对于左图像中没有在右图像找到匹配的sift跟踪点，后续是否要尝试用depth_map来获取其深度估计？
                        // 对于物体点可以，但对于背景点则没必要，精度没法保证
                        
                        if (obj_cls > 0)
                        {
                            id_sift_no_depth.push_back(i);
                            status_sift[i] = 2;
                            continue;
                        }
                        else
                        {
                            // 是否保留该跟踪点是在上面已经决定了，这里无需再判断。这里只记录该跟踪点在当前帧是否有立体匹配

                            // 此处不能只用prev_sift_dep或use_tria_stereo判断该点在上一帧是否有立体匹配（因为有些点跟踪点是当前帧才临时添加的，还没通过左右2帧的三角化得到深度值），应该查看该点是否在Map_prev_r中
                            // int obj_id = ids_sift[i];
                            // bool has_stereo_prev = (prevRightFeaMap.find(obj_id) != prevRightFeaMap.end());
                            // if(USE_TRIANGULATE_TWO_FRAME || has_stereo_prev)
                            {
                                // 记录没有右匹配的背景跟踪点在cur_sift中的序号 
                                sift_no_stereo_bg.push_back(i);
                                // 当前帧没有深度值的bg跟踪点的状态记为2
                                status_sift[i] = 2;
                            }
                            continue;
                        }
                    }

                    int index_valid_match = std::distance(match_stereo.begin(), iter);
                    status_sift_stereo_flow[index_valid_match] = 1;
                    int id_siftdata = match_stereo[index_valid_match];
                    float cord_x = h2_siftpts[id_siftdata].xpos;
                    float cord_y = h2_siftpts[id_siftdata].ypos;
                    float match_x = h2_siftpts[id_siftdata].match_xpos;
                    float match_y = h2_siftpts[id_siftdata].match_ypos;

                    // 对于n_img == 1,已经在cudaSift中完成了此项过滤
                    // 右匹配点超过了规定的图像边界，那么是否应该保存其中的左图像的bg跟踪点？在右图像中快超出图像，意味着该点下一帧很可能离开相机视野？sift跟踪点的作用不是用于长跟踪，而是为当前帧跟踪提供3D-2D的PnP
                    if (!(inBorder(cv::Point2f(match_x,match_y)))) 
                    {
                        // int gl_id = ids_sift[i];
                        
                        // if(obj_cls == 0 && prev_sift_dep[i] > 0)
                        if(obj_cls == 0)
                        {
                            // sift_no_stereo_bg.push_back(i);
                            // status_sift[i] = 2;
                            
                            status_sift[i] = 0;
                        }
                        else
                            status_sift[i] = 0;
                        
                        continue;
                    }
                    
                    // 在左图像的左侧的一些区域内的点要么不可能在右图像中有观测，要么其深度超过了相应的阈值;同时右图像的右侧区域也不可能在左侧有观测点
                    if (obj_cls == 0) 
                    {
                        if(cord_x <= bg_left_border_left_img || match_x >= bg_right_border_right_img)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                    }
                    else if(cord_x <= obj_left_border_left_img || match_x >= obj_right_border_right_img)
                    {
                        // 物体点如果当前帧没有深度值，则放弃该点
                        // 被放弃的该跟踪点可能是静态物体点，但是它仍然可以参与F矩阵的估计。只不过后续它不会被加入地图，也就无法参与优化t的尺度
                        status_sift[i] = 0;
                        continue;
                    }

                    // 左右图像的立体视差为 左图像点 - 右图像点
                    float disp_x = Sift_->siftData2.valid_disp_x[index_valid_match];
                    float disp_y = Sift_->siftData2.valid_disp_y[index_valid_match];
                    
                    if (disp_x <= 0) 
                    {
                        cout << "Weired! Get a disp_x < 0 for sift!" << endl;
                        if (obj_cls == 0)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                        
                            continue;
                        }
                        else
                        {
                            // 物体点跟踪点没有立体匹配，但是一定要有深度值
                            status_sift[i] = 2;
                            id_sift_no_depth.push_back(i);
                            continue;
                        }
                    }
                    
                    float depth = mbf/disp_x;

                    if (obj_cls == 0)
                    {
                        // 太远的点认为立体匹配不够准确，后续该点的深度依赖于2帧三角化和运动变换更新
                        // if (depth >= mThDepthBg || depth < 1.5) 
                        if (depth > 21.0 || depth < 1.5) 
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                        // 左相机坐标系中按照估计深度得到的3D点是否在右相机的视锥体内部。如果不在，则此立体匹配是错误的
                        // float err = (r_cam_3D_plane[0]*cur_un_sift[i].x+r_cam_3D_plane[1]*cur_un_sift[i].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }
                    // 当前帧的物体无论是否运动，都只取25m内的物体
                    else 
                    {
                        // if (depth >= mThDepthObj || depth < mMinDepthPt) continue;
                        // 对于特征点匹配，可以适当减小深度值
                        if (depth >= mThDepthObj || depth < 1.5) 
                        {
                            status_sift[i] = 0;
                            continue;
                        }
                        // float err = (r_cam_3D_plane[0]*cur_un_sift[i].x+r_cam_3D_plane[1]*cur_un_sift[i].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }

                    float y_shift = Y_shift_right_image/depth;

                    // 右匹配点的v坐标超出图像下边界,则直接放弃该跟踪点
                    if ((cord_y+y_shift) > row - 5) 
                    {
                        status_sift[i] = 0;
                        continue;
                    }

                    // 从相机2和相机3的校正矩阵P_rect_xx来看,右相机的点的v坐标应该始终比左相机的对象点的v坐标要大,因此disp应该是小于0的!
                    // if(abs(disp_y + y_shift) > 1.0)
                    if(abs(disp_y) > 1.2)
                    {
                        if (obj_cls == 0)
                        {
                            sift_no_stereo_bg.push_back(i);
                            status_sift[i] = 2;
                            
                            continue;
                        }
                        else
                        {
                            status_sift[i] = 2;
                            id_sift_no_depth.push_back(i);
                            continue;
                        }
                    }
                    
                    // 是否要直接使用右图像中的匹配点的v坐标?
                    cur_right_sift[i] = cv::Point2f(match_x, match_y);
                    // cur_right_sift[i] = cv::Point2f(match_x, cord_y+y_shift);
                    
                    // 之后记得给当前帧那些来自上一帧跟踪但是在当前帧却没有右图像sift的点 分配深度值（通过depth_map）
                    if(obj_cls == 0)
                    {
                        if(use_tria_stereo)
                            cur_sift_dep[i] = -1.0;
                        else
                            cur_sift_dep[i] = depth;
                    }
                    else
                        cur_sift_dep[i] = depth;
                    
                    status_sift[i] = 1;

                    // 记录已有的各个物体（包括背景）的平均特征点深度或视差，为后续的新点提供预测值
                    // if(use_motion_to_pred_fea_dep)
                    {
                        obj_id = obj_cls_id_sift[i].second;
                        if(obj_id > 0)
                        {
                            if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                            {
                                obj_fea_disp_num[obj_id] = make_pair(disp_x,1);
                            }
                            else
                            {
                                obj_fea_disp_num[obj_id].first += disp_x;
                                obj_fea_disp_num[obj_id].second += 1;
                            }
                        }
                        else
                        {
                            ave_dep_bg_cur_frame += depth;
                            ++num_bg_with_dep;
                            ++num_bg_sift_with_dep;
                        }
                    }
                }
            }
            // int num_track_sift_depth = cur_right_sift.size();
            
            Point2f tmp_undist_pt;
            
            vector<float> &valid_disp_x = sift_data->valid_disp_x;
            vector<float> &valid_disp_y = sift_data->valid_disp_y;
            float disp_x, disp_y;
            float cord_x, cord_y, match_x, match_y;
            int validpts_id;
            
            // cout << "start add new stereo sift!" << endl;
            // 往当前帧添加新的sift点，需要该点具有有效的右图像匹配点（深度估计）
            Point2f cur_un_new, cur_un_new_r;
            int num_sift_add = num_track_sift;
            for(int i = 0; i < num_match_stereo; ++i)
            {
                bool no_stereo = false;
                bool invalid_depth = false;
                // 跟踪的点已经算过了
                if(status_sift_stereo_flow[i] == 1) continue;
                
                validpts_id = valid_match_stereo[i+1];

                cord_x = h2_siftpts[validpts_id].xpos;
                cord_y = h2_siftpts[validpts_id].ypos;
                match_x = h2_siftpts[validpts_id].match_xpos;
                match_y = h2_siftpts[validpts_id].match_ypos;

                info_pt = seg_map_cur.at<Vec2b>(cord_y,cord_x);
                uchar cls_label = info_pt[0];
                
                // 如果sift背景点新点要到下一帧再决定，则这里跳过
                if(add_new_sift_in_next_frame && cls_label == 0) continue;

                // do not consider points of person, rider or bicycle or train
                if(cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7) continue;

                // 太靠近图像顶端的区域，有可能是树木之类的，也可能是建筑，去除这部分区域的点，会损失一部分点，但是可以提高运动估计的精度？
                // 最好是使用全景分割来得到更广泛的类别区分
                // 这里最后是只针对物体了
                if(cord_y < (1.0/6*row) && (cord_x > 1.0/6*col && cord_x < 5.0/6*col))
                {
                    continue;
                }
                
                if(cls_label == 0 && (cord_x <= bg_left_border_left_img || match_x >= bg_right_border_right_img)) continue;
                if(cls_label > 0 && (cord_x <= obj_left_border_left_img || match_x >= obj_right_border_right_img)) continue;

                if(!inBorder(cv::Point2f(match_x,match_y))) 
                    continue;
                else
                {
                    disp_x = valid_disp_x[i];
                    disp_y = valid_disp_y[i];
                    // 人为sift匹配是较为准确的立体匹配，即y方向上没有视差。disp_x<=0已经在sift后处理阶段筛除过了
                    // disp_x is the (x_left_img - x_right_img), so disp_x should be > 0
                    //if (disp_x <= 0) continue;

                    if(disp_x <= 0)
                    {
                        cout << "Weired! Get disp_x < 0 for a sift!" << endl;
                        continue;
                    }

                    float depth = mbf/disp_x;

                    float y_shift = Y_shift_right_image/depth;

                    if((cord_y+y_shift) > row - 5) continue;

                    // 当前帧背景点
                    if(cls_label == 0)
                    {
                        // 当计算出来的深度太小时，是否可信？sift的立体匹配可信度较高，但是光流匹配就不一定（因为缺少预测值，只能暴力匹配）
                        // 对于新的背景点，如果深度为1m，则下一帧它很可能就不在相机前方了（相机的频率为100hz，如果按10m/s速度前行，那么两帧之间间隔距离为1m）
                        // 但是也可能相机处于静止状态，所以这里也可以取1.5m为最小深度
                        // 太远的点认为立体匹配不够准确，后续该点的深度依赖于被跟踪且2帧三角化
                        // if(depth >= mThDepthBg || depth < 1.5 || abs(disp_y) > 1.0)
                        if(depth > 21 || depth < 1.5 || abs(disp_y) > 1.2)
                        {
                            if(!add_new_sift_in_next_frame && USE_TRIANGULATE_TWO_FRAME)
                            {
                                no_stereo = true;
                                sift_no_stereo_bg.push_back(num_sift_add);
                            }
                            else
                                continue;
                        }

                        // cv::Point2f new_sift(cord_x, cord_y);
                        // undistortedPts(new_sift, tmp_undist_pt, m_camera[0]);
                        // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                        
                        // ++num_new_sift_bg;
                    }
                    // 新的sift点中，即使有静态物体，这里也暂时只选择那些在动态物体深度阈值（2m-25m）内的静态物体点，这样的深度估计比较精确。
                    // 事实上用sift是可以得到较小深度的左右匹配，但是这部分点很可能不够多，但是又无法使用像素点的匹配（因为太近的点的depth_map估计结果肯定是不准确的）
                    // 同时深度太小的点在下一帧不容易被跟踪到（因为靠得近的点光流大）。对于动态物体而言，其可能是朝着远离相机的方向运动（即同向但速度比相机快），那么下一帧它还是可能出现在视野内的
                    // 可以保留物体上深度值较小的点，但需要当前帧该物体的特征跟踪点数较多（至少8个），否则后续无法进行运动估计
                    // 另外分割模型不太准确，不在物体上的点应该如何排除？（物体匹配期间有外点排除机制）
                    else
                    {
                        // if(depth >= mThDepthObj || depth < mMinDepthPt) continue;
                        if(depth >= mThDepthObj || depth < 1.5) continue;

                        // 是否要保留这部分物体点，后续用depth_map查看是否能获得深度估计。可以
                        // if(abs(disp_y + y_shift) > 0.6)
                        if(abs(disp_y) > 1.2)
                        {
                            // 记录没有右匹配的新sift物体点，后期尝试用depth_map来寻找其深度值
                            id_sift_no_depth.push_back(num_sift_add);
                            invalid_depth = true;
                            // 在id_sift_no_depth后这里千万不能continue,否则下面就忘记添加这个点的右观测点信息了
                            // continue
                        }
                        
                        // cv::Point2f new_sift(cord_x, cord_y);
                        // undistortedPts(new_sift, tmp_undist_pt, m_camera[0]);
                        // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                        // if (err <= 0) continue;
                    }

                    int obj_id = (int)info_pt[1];
                    cur_sift_index.push_back(validpts_id);
                    ids_sift.push_back(n_id++);
                    cur_sift.emplace_back(cord_x,cord_y);
                    track_cnt_sift.push_back(1);
                    obj_cls_id_sift.emplace_back(cls_label,obj_id);
                    // 记录当前帧sift点所匹配的上一帧中的sift点的全局obj id，当一个物体在两帧间的特征点匹配数足够多时，可以直接关联此两物体！
                    // 此处为各个物体的新sift点，在进行物体关联之后，要来修改当前帧各个特征点所属物体的全局id，尤其是当前帧新添加的点！！
                    prev_sift_global_obj_id.push_back(obj_id);
                    // 没有立体匹配的背景新点
                    if(cls_label == 0)
                    {
                        if(no_stereo)
                        {
                            cur_right_sift.emplace_back(0, 0);
                            // cur_right_sift.emplace_back(match_x, cord_y+y_shift);
                            cur_sift_dep.push_back(-1.0);
                            // 当前帧没有深度值的bg新点的状态记为3
                            status_sift.push_back(3);
                        }
                        else
                        {
                            cur_right_sift.emplace_back(match_x,match_y);
                            status_sift.push_back(1);
                            if(use_tria_stereo)
                                cur_sift_dep.push_back(depth);
                            else
                                cur_sift_dep.push_back(-1.0);
                        }
                    }
                    else
                    {
                        // cur_right_sift.emplace_back(match_x, cord_y+y_shift);
                        // 没有立体匹配的物体新点
                        if(invalid_depth)
                        {
                            cur_right_sift.emplace_back(0, 0);
                            // 使用depth_map为物体点获取深度值，但是不会将该点作为立体匹配，该深度只是为了估计物体的3D运动。
                            // 没有立体匹配的新物体点与背景点一样，status都标记为3
                            // status_sift.push_back(0);
                            status_sift.push_back(3);
                            cur_sift_dep.push_back(-1.0);
                        }
                        else
                        {
                            cur_right_sift.emplace_back(match_x,match_y);
                            status_sift.push_back(1);
                            // 物体点必须在这里就给出深度值，即认为立体校正足够准确
                            cur_sift_dep.push_back(depth);
                        }
                    }

                    ++num_new_sift;
                    ++num_sift_add;

                    // if(use_motion_to_pred_fea_dep)
                    {
                        // 物体点如果有有效的深度估计,则记录
                        if(obj_id > 0 && !invalid_depth)
                        {
                            if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                            {
                                obj_fea_disp_num[obj_id] = make_pair(disp_x,1);
                            }
                            else
                            {
                                obj_fea_disp_num[obj_id].first += disp_x;
                                obj_fea_disp_num[obj_id].second += 1;
                            }
                        }
                        else if(obj_id == 0 && !no_stereo)
                        {
                            ave_dep_bg_cur_frame += depth;
                            ++num_bg_with_dep;
                            ++num_bg_sift_with_dep;
                        }
                    }
                }   
            }
        }
        
        if(!add_new_sift_in_next_frame && !no_add_new_sift)
            cout << "Num of detected new sift (for bg and objs) in cur frame is: " << num_new_sift << endl;
        else
            cout << "Num of detected new sift (only for objs) in cur frame  is: " << num_new_sift << endl;
    }
    
    done_select_sift = true;
    // 当前帧特征点"关联阶段"的最后一个sift（包含跟踪与新检测的）的全局id。注意，实际上当前帧新sift点的最大全局id不是这里的值，因为后续sift中的跟踪外点还会被转为新点！
    // last_id_sift_cur = n_id - 1; 

    // 后处理，将当前帧的部分信息保存为prev
    Sift_->Postprocess();
    
    // ------------------------------------------------------------------------
    printf("Sift select costs: %fms \n", t_o.toc());
}

// 根据已有的sift（跟踪的）和FAST（跟踪的）形成mask，并在当前左图像中检测新的FAST点。对所有左图像FAST点进行右图像点的跟踪,保存有效的立体跟踪点
void FeatureTracker::det_new_FAST_objs(const int &frame_count, const int num_solid_obj, const cv::Mat &seg_map, const cv::Mat &cls_map, const cv::Mat & depth_map, 
                                        bool initial_succ, const bool &marg_old_prev, bool &stereo_match_done, const cv::Mat &_img1)
{   
    TicToc t_det_new_and_assign;
    
    // 当前帧新检测的FAST点的全局id不可能小于last_fea_id这个数（排在当前帧新检测的sift之后）
    // int last_fea_id = n_id;

    // Mat rightImg = _img1;
    if(!_img1.empty())
    {
        if(_img1.channels() == 3)
            cvtColor(_img1, cur_img_r, COLOR_BGR2GRAY);
        else
            // cur_img_r = _img1.clone();
            cur_img_r = _img1;
    }
    
    // 下面这些对于初始帧图像也会执行（仅检测新特征点）
    if (1)
    {
        // 等待sift点的跟踪和新点采集结束
        while(!done_select_sift)
        {
            usleep(300);
        }

        done_select_sift = false;

        // 如果物体是仅前向位移（或同时仅有很小的旋转），则可以过滤flow
        bool filter_flow_with_small_rot = true;
        Quaterniond delta_Q(R_cam_motion);
        float delta_angle = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

        // if(filter_flow_with_small_rot)
        // {
        //     if(frame_count > 1)
        //     {
        //         if(delta_angle >= 0.5)
        //         {
        //             filter_flow_with_small_rot = false;
        //         }
        //         else
        //         {
        //             // 如果位移很小，则要么旋转较大，要么物体静止，则不过滤
        //             if(P_cam_motion.norm() >= 0.25)
        //             {
        //                 // 根据kitti_odometry 0007序列的最开始一系列帧来看，42帧（4.1s）内汽车转向了90度！平均1s转22度！也就是每2帧间转了2.2度！
        //                 // 需要预测的旋转较小。1s内旋转小于8度？
        //                 if(delta_angle >= 0.5)
        //                     filter_flow_with_small_rot = false;
        //             }
        //             else if(P_cam_motion.norm() >= 0.1)
        //             {
        //                 if(delta_angle >= 0.3)
        //                     filter_flow_with_small_rot = false;
        //             }
        //             else
        //             {
        //                 filter_flow_with_small_rot = false;
        //             }
        //         }
        //     }
        //     else if(frame_count == 1)
        //     {
        //         //todo: 前2帧没有运动预测，是否假设为小旋转并过滤？
        //         filter_flow_with_small_rot = true;
        //     }
        //     else
        //     {
        //         filter_flow_with_small_rot = false;
        //     }
        // }
        
        int num_valid_bg_sift = 0, num_valid_bg_FAST = 0;
        int num_bg_sift = id_bg_track_sift.size();
        if(filter_flow_with_small_rot)
        {
            // 统计所有背景跟踪点的flow直线 与 图像中心点的距离， 以及各个跟踪点的flow线的方向
            // 如果相机近似直线前进，则所有背景点的光流应该近似相交于图像的中心点。如果同时有小的旋转，则应该也近似成立。是否可以使用预测的相机旋转运动来直接判断物体是否近似直线运动？
            // 如果相机有较大的旋转（同时可以有向前的位移），那么所有光流不一定汇聚于一点（也不会是图像中心），光流的方向也不一定一致。
            // 例如相机向前运动同时向左转，那么左半图像的点的光流朝向很复杂（前向运动导致点在图像中的位置向左边两个角运动，但是旋转又会使所有点位置右移动，最终使得点的位置很难有一致的规律）！
            bool filter_succ = false;

            int num_dist_15 = 0, num_dist_30 = 0;
            int num_right_dir = 0;
            vector<int> outliers_15, outliers_30;
            for(int k = 0; k < num_bg_sift; ++k)
            {
                int id = id_bg_track_sift[k];
                if(status_sift[id] == 0) 
                {
                    continue;
                }
                ++num_valid_bg_sift;
                // 计算图像中心点 到 该点光流直线 的距离，并且验证上一帧该点是否处于 当前帧该点与图像中心点的连线之间的区域（即要近似满足 当前点指向上一帧点的射线 再往前穿过中心点） 
                Point2f &cur_pt = cur_sift[id];
                Point2f &prev_pt = prev_sift[id];
                float dist = cal_dist_img_center_to_flow_line(cur_pt,prev_pt);
                
                // cout << "dist: " << dist << endl;

                if(dist < 15) 
                    ++num_dist_15;
                // else
                else if(dist >= 25)
                {
                    outliers_15.push_back(-id);
                }
                
                if(dist < 30) 
                    ++num_dist_30;
                // else
                else if(dist >= 40)
                    outliers_30.push_back(-id);

                // flow直线不满足要求的点
                if(dist != 10000)
                    ++num_right_dir;
            }
            
            // 认为sift的匹配精度较高，因此如果sift点数较多，且flow交于图像中心的点较多，则认为相机近似直线向前运动
            // if(num_bg_sift > 9 && num_dist_15 >= 0.45 * num_bg_sift)
            if((num_valid_bg_sift > 20 && (num_right_dir >= 0.75*num_valid_bg_sift && num_dist_15 >= 0.55 * num_right_dir)) || (num_valid_bg_sift < 20 && (num_right_dir > 0.5 * num_valid_bg_sift && num_dist_15 >= 0.45 * num_right_dir)))
            {
                for(auto pt_id: outliers_15)
                {
                    status_sift[-pt_id] = 0;
                }

                for(auto id: id_bg_track_FAST)
                {
                    if(status_FAST[id] == 0) continue;
                    ++num_valid_bg_FAST;
                    Point2f &cur_pt = cur_FAST[id];
                    Point2f &prev_pt = prev_FAST[id];
                    float dist = cal_dist_img_center_to_flow_line(cur_pt,prev_pt);
                    if(dist >= 25)
                        status_FAST[id] = 0;
                    else
                        ++num_dist_15;
                }

                cout << "filter bg flow match: " << (num_valid_bg_FAST+num_valid_bg_sift) << " -> " << num_dist_15 << endl;
                filter_succ = true;
            }
            // else if(num_bg_sift > 9 && num_dist_30 >= 0.65 * num_bg_sift)
            else if((num_valid_bg_sift > 20 && (num_right_dir >= 0.75*num_valid_bg_sift && num_dist_30 > 0.6 * num_right_dir)) || (num_valid_bg_sift < 20 && (num_right_dir > 0.55 * num_valid_bg_sift && num_dist_30 > 0.5 * num_right_dir)))
            {
                for(auto pt_id: outliers_30)
                {
                    status_sift[-pt_id] = 0;
                }

                for(auto id: id_bg_track_FAST)
                {
                    if(status_FAST[id] == 0) continue;
                    ++num_valid_bg_FAST;
                    Point2f &cur_pt = cur_FAST[id];
                    Point2f &prev_pt = prev_FAST[id];
                    float dist = cal_dist_img_center_to_flow_line(cur_pt,prev_pt);
                    if(dist >= 40)
                        status_FAST[id] = 0;
                    else
                        ++num_dist_30;
                }

                cout << "filter bg flow match: " << (num_valid_bg_FAST+num_valid_bg_sift) << " -> " << num_dist_30 << endl;
                filter_succ = true;
            }
            else
            {
                // 否则，再统计FAST背景跟踪点的情况
                int num_bg_FAST = id_bg_track_FAST.size();
                for(int k = 0; k < num_bg_FAST; ++k)
                {
                    int id = id_bg_track_FAST[k];
                    if(status_FAST[id] == 0) continue;
                    ++num_valid_bg_FAST;
                    Point2f &cur_pt = cur_FAST[id];
                    Point2f &prev_pt = prev_FAST[id];
                    float dist = cal_dist_img_center_to_flow_line(cur_pt,prev_pt);
                    // cout << "dist: " << dist << endl;
                    
                    if(dist < 15) 
                        ++num_dist_15;
                    // else
                    else if(dist >= 25)
                        outliers_15.push_back(id+1);
                    
                    if(dist < 30)
                        ++num_dist_30;
                    // else
                    else if(dist >= 40)
                        outliers_30.push_back(id+1);

                    if(dist != 10000)
                        ++num_right_dir;
                }

                int total_num_bg = num_valid_bg_FAST + num_valid_bg_sift;
                if((total_num_bg > 40 && (num_right_dir >= 0.65*total_num_bg && num_dist_15 >= 0.45 * num_right_dir)) || (total_num_bg < 40 && (num_right_dir >= 0.5 * total_num_bg && num_dist_15 >= 0.35 * num_right_dir)))
                {
                    for(auto pt_id: outliers_15)
                    {
                        if(pt_id <= 0)
                            status_sift[-pt_id] = 0;
                        else
                            status_FAST[pt_id-1] = 0;
                    }

                    cout << "filter bg flow match: " << (total_num_bg) << " -> " << num_dist_15 << endl;
                    filter_succ = true;
                }
                else if((total_num_bg > 40 && (num_right_dir >= 0.70*total_num_bg && num_dist_30 >= 0.60 * num_right_dir)) || (total_num_bg < 40 && (num_right_dir >= 0.55 * total_num_bg && num_dist_30 >= 0.4 * num_right_dir)))
                {
                    for(auto pt_id: outliers_30)
                    {
                        if(pt_id <= 0)
                            status_sift[-pt_id] = 0;
                        else
                            status_FAST[pt_id-1] = 0;
                    }

                    cout << "filter bg flow match: " << (total_num_bg) << " -> " << num_dist_30 << endl;
                    filter_succ = true;
                }
            }

            // todo:如果没有过滤成功，则只去除那些明显匹配错误的点。如果P够大，R够小，则flow应该有同一的方向？
            if(!filter_succ)
            {

            }
        }
        else
        {
            for(auto id: id_bg_track_FAST)
            {
                if(status_FAST[id] == 0) continue;
                ++num_valid_bg_FAST;
            }
        }
        
        // 最少要18个点参与F矩阵的估计;
        if(!temp_pts_for_F.empty())
        {
            // int num_bg_match = id_bg_track_FAST.size();
            // 优先选择前n%匹配质量的背景跟踪点来估计F矩阵
            int middle = num_valid_bg_FAST * 1.0/3;
            int num_ori = pts_for_cal_F.size();
            for(auto id: pts_for_cal_F)
            {
                if(status_sift[-id] == 0) --num_ori;
            }
            
            int all_ = num_ori + middle;
            // 最少要40个点参与F或H矩阵的估计
            int all = max(40,all_);

            int rest = all - num_ori;
            // int rest = 40 - num_ori;

            int k = 0;

            bool add_up_half_img_fea = false;
            // 注意，temp_pts_for_F中其实还可能包含了部分静态物体的跟踪点，它们并不包含在id_bg_track_FAST（纯背景跟踪点）中。因此即使num_bg_match为0，这里还是有可能会添加静态跟踪点
            if(rest > 0)
            {
                for(auto it: temp_pts_for_F)
                {
                    float y = cur_FAST[(it-1)].y/60.0;
                    if(rest <= 0 && k >= middle) 
                    {
                        if(add_up_half_img_fea && num_up_half < 5)
                        {
                            if(y >= 3) continue;
                        }
                        else
                            break;
                    }

                    // 部分新添加的上一帧新背景FAST点在上面寻找立体匹配时可能被删除了
                    if(status_FAST[(it-1)] == 0) continue;

                    pts_for_cal_F.insert(it);
                    --rest;
                    ++k;

                    if(y < 3) ++num_up_half;
                }
            }
            else
            {
                // 再添加8个FAST跟踪点
                rest = 10;
                
                // todo:是否要优先添加上半图像的点直到满足最小值？
                // 没必要，当近点严重不足时，自然会添加远处的点！
                if(add_up_half_img_fea)
                {
                    if(num_up_half < 5)
                    {
                        for(auto &it: temp_pts_for_F)
                        {
                            if(status_FAST[(it-1)] == 0) continue;

                            if(num_up_half >= 5) break;

                            float y = cur_FAST[(it-1)].y/60.0;
                            if(y < 3)
                            {
                                --rest;
                                ++num_up_half;
                                pts_for_cal_F.insert(it);

                                it = 0;
                            }
                        }
                    }
                }

                for(auto &it: temp_pts_for_F)
                {
                    if(rest <= 0) break;
                    if(it == 0) continue;
                    if(status_FAST[(it-1)] == 0) continue;
                    
                    --rest;
                    pts_for_cal_F.insert(it);
                }
            }
        }

        // 根据F矩阵排除FAST中的跟踪外点
        if(frame_count > 0 && reject_with_F)
        {
            if(only_use_track_sift_for_F)
                rejectWithFV1(false);
            else
                rejectWithFV2(initial_succ, pts_for_cal_F);
        }

        int Max_th_num_old = 65;
        int Min_th_num_new = 15;

        // 限制背景跟踪点的数量，以便限制LBA的数量（太多跟踪点也很难保证估计精度）。总数不能超过70.
        // 如果自车运动较大，则增加新点的数量
        // if(cal_Mat_F_H || frame_count > 1)
        // {
        //     float delta_ang = delta_angle;
        //     if(cal_Mat_F_H)
        //     {
        //         Quaterniond delta_q(R_from_E);
        //         delta_ang = fabs(acos(delta_q.w()) * 2.0 / 3.1416 * 180.0);
        //     }
            
        //     if(P_cam_motion.norm() >= 0.7 || delta_ang > 0.8)
        //     {
        //         Max_th_num_old = 35;
        //         Min_th_num_new = 35;
        //     }
        //     else if(P_cam_motion.norm() <= 0.25 && delta_ang < 0.4)
        //     {
        //         Max_th_num_old = 60;
        //         Min_th_num_new = 10;
        //     }
        // }
        
        int num_old_track = Max_th_num_old;
        int num_new_track = Min_th_num_new;

        vector<int> new_track_sift, new_track_FAST;
        vector<int> del_sift;
        int num_sift = 0;
        for(int k = 0; k < num_bg_sift; ++k)
        {
            int id = id_bg_track_sift[k];
            if(status_sift[id] == 0) 
            {
                continue;
            }

            // 如果是静态物体点，则不算入
            if(obj_cls_id_sift[id].first > 0) continue;

            // todo:是否要保留一定的FAST跟踪点？
            // 因为FAST点较容易形成多帧跟踪，所以还是要保留一定FAST点以便可以形成LBA
            if(0 && num_sift > 50)
            {
                // status_sift[id] = 0;
                del_sift.push_back(id);
                continue;
            }
            
            int g_id = ids_sift[id];
            if(!reserve_bg_track_pt_id.empty())
                if(find(reserve_bg_track_pt_id.begin(),reserve_bg_track_pt_id.end(),g_id) != reserve_bg_track_pt_id.end())
                    continue;

            // 包含旧跟踪点和新跟踪点
            ++num_sift;
            
            if(track_cnt_sift[id] > 2) 
            {
                if(num_old_track > 0)
                {
                    --num_old_track;
                }
                else
                    status_sift[id] = 0;
            }
            else
            {
                new_track_sift.push_back(id);
            }
        }

        int num_FAST = 0;
        for(int k = 0; k < id_bg_track_FAST.size(); ++k)
        {
            int id = id_bg_track_FAST[k];
            if(status_FAST[id] == 0)
            {
                continue;
            }

            if(obj_cls_id_FAST[id].first > 0) continue;

            int g_id = ids_FAST[id];
            if(!reserve_bg_track_pt_id.empty())
                if(find(reserve_bg_track_pt_id.begin(),reserve_bg_track_pt_id.end(),g_id) != reserve_bg_track_pt_id.end())
                    continue;
            
            if(track_cnt_FAST[id] > 2) 
            {
                if(num_old_track > 0)
                {
                    --num_old_track;
                    
                    ++num_FAST;
                }
                else
                    status_FAST[id] = 0;
            }
            else
            {
                new_track_FAST.push_back(id);
            }
        }

        // 已添加的长跟踪点数，这些点将参与LBA
        num_old_track_fea = Max_th_num_old - num_old_track;

        // 如果原本旧跟踪点充足，则应该保证至少保留12个（可以参与LBA）
        if(frame_count > 1 && num_old_track_fea < 12)
        {
            for(auto &it: del_sift)
            {
                if(num_old_track_fea >= 12)
                {
                    if(track_cnt_sift[it] > 2)
                    {
                        ++num_old_track_fea;
                        --num_old_track;
                        // 已处理的点
                        it = -1;
                    }
                }
                else
                {
                    // sift的点已经足够多了
                    break;
                }
            }
        }

        // 保证上一帧有深度值的点的最低数量（为了后续尺度的恢复和PnP)，包括旧跟踪点和部分新跟踪点
        num_track_fea_with_dep_prev = num_old_track_fea;

        // 保持总的跟踪点数不超过80
        if(num_old_track > 0)
            num_new_track += num_old_track;
        
        // 如果FAST点数达不到最低要求，则先添加FAST点以达到最低要求
        // if(num_FAST < 20)
        if(num_FAST < 12)
        {
            for(auto &it: new_track_FAST)
            {
                if(num_new_track > 0 && num_FAST < 12)
                {
                    // 表示该点已处理
                    it = -1;
                    --num_new_track;
                    ++num_FAST;
                    int g_id = ids_FAST[it];
                    if(prevRightFeaMap.find(g_id) != prevRightFeaMap.end())
                    {
                        ++num_track_fea_with_dep_prev;
                    }
                }
                // else if(num_new_track <= 0)
                // {
                //     it = -1;
                //     // 如果已经达到最大跟踪点数，则删除剩余的点
                //     status_FAST[it] = 0;
                // }
                else
                {
                    break;
                }
            }
        }

        // 剩下的sift中，在new_track_sift中的点质量更高，优先
        for(auto &it: new_track_sift)
        {
            // 如果此时还未达到最大跟踪点数
            if(num_new_track > 0)
            {
                --num_new_track;
                int g_id = ids_sift[it];
                // 上一帧有深度值的新跟踪点
                if(prevRightFeaMap.find(g_id) != prevRightFeaMap.end())
                {
                    ++num_track_fea_with_dep_prev;
                }
            }
            else
            {
                int g_id = ids_sift[it];
                float dep_prev = prev_sift_dep[it];
                // 上一帧有深度值的新跟踪点
                if(num_track_fea_with_dep_prev < Min_num_bg_track_with_dep_prev && (dep_prev > 0 || prevRightFeaMap.find(g_id) != prevRightFeaMap.end()))
                {
                    ++num_track_fea_with_dep_prev;
                }
                else
                    status_sift[it] = 0;
            }
        }
        
        // 之后的不管是old还是new，按照质量排序先后保留
        for(auto &it: del_sift)
        {
            if(it == -1) continue;
            if(num_new_track > 0)
            {
                int cnt = track_cnt_sift[it];
                // 如果长跟踪点数已经达到了最大值，则放弃该点
                if(cnt > 2 && num_old_track <= 0)
                {
                    status_sift[it] = 0;
                    continue;
                }

                --num_new_track;

                if(cnt > 2) 
                {
                    --num_old_track;
                    ++num_old_track_fea;
                }

                int g_id = ids_sift[it];
                // 上一帧有深度值的跟踪点，可能是旧点或新点
                if(cnt > 2 || prevRightFeaMap.find(g_id) != prevRightFeaMap.end())
                {
                    ++num_track_fea_with_dep_prev;
                }
            }
            else
            {
                int g_id = ids_sift[it];
                float dep_prev = prev_sift_dep[it];
                // 上一帧有深度值的新跟踪点数还没达到最小阈值
                if(num_track_fea_with_dep_prev < Min_num_bg_track_with_dep_prev && (dep_prev || prevRightFeaMap.find(g_id) != prevRightFeaMap.end()))
                {
                    ++num_track_fea_with_dep_prev;
                }
                else
                    status_sift[it] = 0;
            }
        }

        for(auto &it: new_track_FAST)
        {
            if(it == -1) continue;
            if(num_new_track > 0)
            {
                --num_new_track;
                int g_id = ids_FAST[it];
                if(prevRightFeaMap.find(g_id) != prevRightFeaMap.end())
                {
                    ++num_track_fea_with_dep_prev;
                }
            }
            else
            {
                int g_id = ids_FAST[it];
                float dep_prev = prev_FAST_dep[it];
                if(num_track_fea_with_dep_prev < Min_num_bg_track_with_dep_prev && (dep_prev > 0 || prevRightFeaMap.find(g_id) != prevRightFeaMap.end()))
                {
                    ++num_track_fea_with_dep_prev;
                }
                else
                    status_FAST[it] = 0;
            }
        }

        // printf("set mask begins \n");
        TicToc t_m;
        
        // 设置图像mask，在特征点处画小黑点。目的是后续不在已经跟踪匹配到的特征点处再次检测
        // 根据cur_sift和cur_FAST来设置黑点区域。去除cur_FAST中的没有完成匹配的点 和 处在cur_sift所造成的黑点区域的匹配点。
        // 另外还可以对cur_FAST中的点根据全局跟踪次数进行排列（用处?不采用)
        // 采用新版的函数
        setMask(initial_succ, USE_IMU);
        printf("set mask for new FAST detection costs %fms \n", t_m.toc());
        
        if (!ids_FAST.empty())
        {
            last_id_track_FAST_cur = *(std::max_element(ids_FAST.begin(),ids_FAST.end()));
            // 当前帧跟踪点集中的最大点id，注意，sift点id不一定比FAST的小，因为上一帧新的sift点的id不一定比新FAST点的id小！
            last_id_track_fea_cur = std::max(last_id_track_fea_cur,last_id_track_FAST_cur);
            
            //printf("track cnt %d\n", num_track_FAST);
            // cur_FAST中剩下的是跟踪到的点
            for (auto &n : track_cnt_FAST)
                ++n;
        }

        // status_FAST.clear();
        num_track_FAST = cur_FAST.size();
        // cout << "original num of tracked FAST: " << num_track_FAST << endl;

        // start detect new FAST

        // 是否要在每一帧完成跟踪后检测和添加新的背景点
        if(!add_new_sift_in_next_frame)
        {
            num_track_fea_static = num_track_FAST_static + num_track_sift_static;
            num_long_track_fea_stat = num_sta_FAST_long_track + num_sta_sift_long_track;

            //printf("Begin detecting new FAST feature! \n");
            // TicToc t_t;
            // MAX_CNT_PTS_BG值为300，仅指每一帧背景上的跟踪点和新检测点的总数（背景中的Shi-Thomas角点是作为最后备用的，优先使用背景和静态物体的sift点，以及物体上的Shi-Thomas点，因为距离较近）
            // int n_max_cnt_bg = MAX_CNT_PTS_BG - num_track_FAST_bg - num_track_sift_bg - num_new_sift_bg;
            // 新的sift点数量太多，这里不考虑其影响，保证FAST的数量
            // int n_max_cnt_bg = MAX_CNT_PTS_BG - num_track_FAST_bg - num_track_sift_bg;
            // 新sift点中最多不超过1/4的点会被下一帧跟踪到的！
            // int n_max_cnt_bg = MAX_CNT_PTS_BG - num_track_FAST_bg - num_track_sift_bg - 1.0/2 * num_new_sift_bg;
            int n_max_cnt_bg = MAX_CNT_PTS_BG - num_track_FAST_bg - num_track_sift_bg - num_new_sift_bg;

            // FAST比较容易连续多帧跟踪；而sift采用随机检测和暴力匹配，很难保证相同的点在前后两帧都会出现！
            // 系统前2帧直接规定背景中的新检测FAST点数最小值
            if(frame_count < 2) 
            {
                if(num_track_FAST_static < 30)
                    n_max_cnt_bg = std::max(n_max_cnt_bg, 50);
                else
                    n_max_cnt_bg = std::max(n_max_cnt_bg, 25);
            }
            // 从第3帧开始就可以查看长跟踪（至少3帧观测）的静态特征点数
            // 如果滑窗还未满，则不会去除任何一帧的跟踪内点和新特征点
            else if(frame_count < WINDOW_SIZE)
            {
                if(num_long_track_fea_stat < 30)
                {
                    // 要保证后续静态FAST的跟踪点数不能太少
                    // 跟踪的静态点数大于阈值，则说明上一帧的新检测点被跟踪的还比较多
                    if(num_track_FAST_static >= 25)
                    {
                        n_max_cnt_bg = std::max(n_max_cnt_bg, 25);
                    }
                    else
                    {
                        n_max_cnt_bg = std::max(n_max_cnt_bg, 50);
                    }
                }
            }
            // 如果滑窗已满，则不论是否已初始化，看上一滑窗是marg次新帧还是最老帧（如果是当前帧滑窗刚满，则上一帧实际也还没marg，但是其值会有参考性）
            else if(marg_old_prev)
            {
                // 如果上一帧的选择是需要marg最老帧，则说明最新几帧之间的视差足够大，跟踪点可能比较少
                // 增加判断条件，提高阈值
                if(num_track_fea_static < 70 || num_long_track_fea_stat < 40)
                {
                    if(num_track_FAST_static < 30)
                    {
                        n_max_cnt_bg = std::max(n_max_cnt_bg, 50);
                    }
                    else
                    {
                        n_max_cnt_bg = std::max(n_max_cnt_bg, 30);
                    }
                }
            }
            // 如果上一帧需要marg次新帧，则说明最新几帧间的视差很小，应该就不需要对new FAST的数量做什么要求了吧？其实此时还可以考虑相机是否有运动，如果是静止的话，则可以适当减少新特征点的数量？
            else
            {
                // todo
            }
            
            // 上面最后得出的需要检测的新点数不一定大于0
            if (n_max_cnt_bg > 0)
            {
                if(mask_bg_cur.empty())
                {
                    cout << "mask is empty " << endl;
                    abort();
                }
                if (mask_bg_cur.type() != CV_8UC1)
                {
                    cout << "mask type wrong " << endl;
                    abort();
                }

                // cv::Mat seg_mask(row, col, CV_8UC1);
                cout << "Start detect new FAST in bg!" << endl;
                n_max_cnt_bg = 100;
                // 将当前帧背景的特征点数补充到MAX_CNT_PTS_BG
                // 注意，opencv的此函数检测的其实是Harris或其改进版本Shi-Tomasi角点，其不具备尺度不变性（没有设计多尺度），具有旋转不变性和对光照不敏感（因为采用的是亮度梯度）！
                // 另外，Shi-Tomasi角点只能提取角点（并且强度较低的角点也不能检测），不能提取边缘或者斑点等特征点！浪费了许多特征点！
                // https://blog.csdn.net/weixin_34910922/article/details/119045533
                // 而FAST关键点（即ORB特征的角点）则具有一定（但是其鲁棒性不如SIFT和SURF）的尺度不变性（设计了金字塔）和旋转不变性（因为brief描述子中采用了方向向量），对光照变化也不太敏感。
                // https://blog.csdn.net/wuchaohuo724/article/details/117919862
                // FAST特征可以检测角点、边缘点或光斑（因为是数圆上连续大于某阈值的点数，边缘点一般是一半点）。
                // ORB中的检测和匹配速度都很快(整体比SIFT快两个数量级），但是FAST特征点本身不够鲁棒（对于噪声的鲁棒性较差），甚至不如Shi-Tomasi角点，但是其检测速度确实非常快。
                // 如果是考虑精度更多，那么SIFT最佳，其次Shi-Tomasi角点+光流金字塔（再加上有预测点）则是次佳；ORB最大的优点是速度快！
                // 虽然goodFeaturesToTrack函数的运行速度比较慢，但是每一帧背景需要新检测的点并不会很多，总的值为300-跟踪的特征点-新检测的sift点
                // cv::goodFeaturesToTrack(cur_img, n_FAST_bg, n_max_cnt_bg, 0.02, MIN_DIST_BG*1.0/2, mask_bg);

                // 新FAST点的特征要有足够高的辨识度，且尽可能提高点和点之间的距离，这样才能为下一帧的sift跟踪点保留一些空间
                cv::goodFeaturesToTrack(cur_img, n_FAST_bg, 1.5*n_max_cnt_bg, 0.04, 25, mask_bg_cur);
            }
        }
        
        vector<cv::Point2f> n_FAST_obj;
        // detect new fea of objs
        if (num_solid_obj > 0)
        {
            int num_objs_fea_track = (cur_FAST.size() - num_track_FAST_bg) + (cur_sift.size() - num_track_sift_bg - num_new_sift_bg);
            int n_max_cnt_obj = 0;
            if(num_objs_fea_track*1.0/num_solid_obj < 10)
                // 期望平均每个物体上能保留到MAX_CNT_PTS_OBJ个特征点（后续不够的话可以用像素点匹配来凑？）
                n_max_cnt_obj = MAX_CNT_PTS_OBJ * num_solid_obj - num_objs_fea_track;
            
            if (n_max_cnt_obj > 0)
            {
                cout << "num of FAST fea need to be detected: " << n_max_cnt_obj << endl;
                
                if(mask_solid_objs.empty())
                {
                    cout << "mask is empty " << endl;
                    abort();
                }
                if (mask_solid_objs.type() != CV_8UC1)
                {
                    cout << "mask type wrong " << endl;
                    abort();
                }
                
                // 这部分操作已经在build_seg_map中完成了
                // cv::Mat seg_mask(row, col, CV_8UC1);
                // cv::compare(cls_map, 0, seg_mask, cv::CMP_EQ);
                // // mask_solid_objs中进一步排除了背景区域
                // mask_solid_objs.setTo(0, seg_mask);

                // cout << "Start detect new FAST of objs!" << endl;
                TicToc time_for_objs_fea_detect;
                // 将当前帧中的FAST特征点数补充到MAX_CNT
                // 是否需要单独为各个物体的区域单独创建一个mask，在各自的区域内进行FAST的检测？应该不需要，mask了背景区域，再规定了特征点间隔就有类似的效果

                // n_max_cnt_obj = 0;
                // while(true)
                // {
                //     cv::imshow("mask of solid objs with tracked fea", mask_solid_objs);
                //     // 一直等待用户按下ESC键（ASCI码为27）
                //     if(waitKey(0) == 27)
                //     {
                //         break;
                //     }
                // }

                // 物体点不再用检测点的quality进行排序
                // vector<float> quality;
                // cv::goodFeaturesToTrack(cur_img, n_FAST_obj, n_max_cnt_obj, 0.02, MIN_DIST_OBJ, mask_solid_objs, quality);
                cv::goodFeaturesToTrack(cur_img, n_FAST_obj, n_max_cnt_obj, 0.02, MIN_DIST_OBJ, mask_solid_objs);
                printf("Detect new FAST fea of objs costs: %fms \n", time_for_objs_fea_detect.toc());
            }
        }
        // printf("detect new FAST feature costs: %f ms \n", t_t.toc());

        // start add right match predict for tracked fea

        // cur_right_FAST.clear();
        float pred_u_r, pred_disp, depth, shift_y;
        float l_x, l_y, r_x, r_y;
        
        // 记录各个临时物体上的FAST跟踪特征点的视差值之和以及特征点数，以便后续给物体上的新检测点提供视差的参考值
        // 如果是系统首帧则不会有track特征点，也就不会有各个跟踪点的预测深度
        // float ave_disp_x_bg = mbf/(3.0/4*mMinDepthPt + 1.0/4*mThDepthBg);
        // float ave_disp_y_bg = Y_shift_right_image/(3.0/4*mMinDepthPt + 1.0/4*mThDepthBg);
        // FAST点就选择近一些的点，不然容易选到很远处的树叶
        float ave_disp_x_bg = mbf/(1.0/2*mMinDepthPt + 1.0/2*10);
        float ave_disp_y_bg = Y_shift_right_image/(1.0/2*mMinDepthPt + 1.0/2*10);

        float ave_disp_x_objs = mbf/(2.0/3*mMinDepthPt + 1.0/3*mThDepthObj);
        float ave_disp_y_objs = Y_shift_right_image/(2.0/3*mMinDepthPt + 1.0/3*mThDepthObj);

        num_new_FAST_bg = 0;
        Vec2b pt_info;
        uchar cls_label;
        int obj_id;

        // 跟踪特征点可以使用depth_map或恒速运动模型来设置当前帧的右匹配点预测
        if(cur_FAST.size() > 0)
        {
            // 首帧需要使用depth_map的结果来获取预测值
            // 之后的帧如果需要，也可以是选择使用depth_map的结果作为深度预测值
            if(frame_count < 2 || use_motion_to_pred_fea_dep == 0)
            {
                while(!stereo_match_done)
                {
                    usleep(300);
                }
           
                int i = 0;
                float th_x_right_img;
                for(auto &p: cur_FAST)
                {
                    // 上面刚reduceVector，应该是不会出现此情况的
                    if (!status_FAST[i++]) 
                    {
                        assert(status_FAST[i-1]);
                        continue;
                    }

                    l_x = p.x;
                    l_y = p.y;
                
                    pred_disp = depth_map.at<float>(l_y,l_x);
                    if (pred_disp <= 0) 
                    {
                        r_x = max(5.0f,l_x-ave_disp_x_bg);
                        r_y = min(l_y+ave_disp_y_bg, (float)(row-5));
                        cur_right_FAST.emplace_back(r_x,r_y);

                        continue;
                    }
                    depth = mbf/pred_disp;
                    shift_y = Y_shift_right_image/depth;
                    r_x = l_x - pred_disp;
                    
                    pt_info = seg_map.at<Vec2b>(l_y, l_x);
                    cls_label = pt_info[0];
                    
                    if(cls_label == 0)
                        th_x_right_img = bg_right_border_right_img;
                    else
                        th_x_right_img = obj_right_border_right_img;
                    
                    if(r_x >= 5 && r_x <= th_x_right_img)
                        cur_right_FAST.emplace_back(r_x, l_y+shift_y);
                    else if (r_x < 5)
                        cur_right_FAST.emplace_back(5, l_y+shift_y);
                    else
                        cur_right_FAST.emplace_back(th_x_right_img, l_y+shift_y);
                }
            }
            else
            {
                // 如果系统是使用恒速运动模型为每个点预测了在当前帧的（深度值和）右匹配点
                // 此时predict_dep_FAST应该不为空
                assert(!predict_dep_FAST.empty());
                int obj_id;
                uchar cls;
                
                // 设置当前帧跟踪到的特征点的右图像预测点坐标
                for(int i = 0; i < cur_FAST.size(); ++i)
                {
                    l_x = cur_FAST[i].x;
                    l_y = cur_FAST[i].y;
                    // first是与所跟踪的上一帧特征点的全局cls所对齐的（除非上一帧是背景点，而当前帧是物体，则保留为物体cls）。如果是初始帧，则都是当前帧各个点的检测类别
                    cls = obj_cls_id_FAST[i].first;
                    obj_id = obj_cls_id_FAST[i].second;
                    
                    depth = predict_dep_FAST[i];
                    bool bad_dep = false;
                    // 这种情况是上一帧的没有深度值的背景点（可以是上一帧的新背景点，或者不是新背景点，但是之前没有三角化成功）
                    if(depth <= 0) 
                    {
                        if(obj_fea_disp_num.find(obj_id) != obj_fea_disp_num.end())
                        {
                            assert(obj_id > 0 && "bg fea should not be in obj_fea_disp_num！");
                            pred_disp = obj_fea_disp_num[obj_id].first/obj_fea_disp_num[obj_id].second;
                            depth = mbf/pred_disp;
                            bad_dep = true;
                        }
                        else
                        {
                            if(cls == 0)
                            {
                                // 最后我们规定FAST的深度必须比较近，最远不超过10m，因为FAST点的检测和匹配精度均比较低
                                // if(ave_dep_bg_cur_frame != 0 && num_bg_with_dep > 15) 
                                // {
                                //     float ave_dep = ave_dep_bg_cur_frame/num_bg_with_dep;
                                //     float ave_disp_x = mbf/ave_dep;
                                //     float ave_disp_y = Y_shift_right_image/ave_dep;
                                //     // 检查超出边界
                                //     r_x = max(5.0f,l_x-ave_disp_x);
                                //     r_y = min(l_y+ave_disp_y, (float)(row-5));
                                //     cur_right_FAST.emplace_back(r_x,r_y);
                                // }
                                // else
                                {
                                    r_x = max(5.0f,l_x-ave_disp_x_bg);
                                    r_y = min(l_y+ave_disp_y_bg, (float)(row-5));
                                    cur_right_FAST.emplace_back(r_x,r_y);
                                }
                            }
                            else
                            {
                                r_x = max(5.0f,l_x-ave_disp_x_objs);
                                r_y = min(l_y+ave_disp_y_objs, (float)(row-5));
                                cur_right_FAST.emplace_back(r_x,r_y);
                            }
                            
                            continue;
                        }
                    }
                    else
                        pred_disp = mbf/depth;
                    
                    shift_y = Y_shift_right_image/depth;
                    
                    // 检验该深度是否合适
                    r_x = l_x - pred_disp;
                    r_y = min(l_y+shift_y, (float)(row-5));
                    if(cls == 0)
                    {
                        
                        if(r_x >= 5 && r_x <= bg_right_border_right_img)
                            cur_right_FAST.emplace_back(r_x, r_y);
                        else if(r_x < 5)
                            cur_right_FAST.emplace_back(5, r_y);
                        else
                            cur_right_FAST.emplace_back(bg_right_border_right_img, r_y);
                    }
                    else
                    {   
                        if(r_x >= 5 && r_x <= obj_right_border_right_img)
                            cur_right_FAST.emplace_back(r_x, r_y);
                        else if(r_x < 5)
                            cur_right_FAST.emplace_back(5, r_y);
                        else
                            cur_right_FAST.emplace_back(obj_right_border_right_img, r_y);
                        
                        // 只保存检测出来的物体的跟踪特征点的视差值，对于当前帧漏检的物体的跟踪点，只能用预测的深度值或者直接用左图像点坐标 来计算右观测点（否则这里会把背景点也记录进去，而背景点的深度的差异是很大的！）
                        if(obj_id != 0 && !bad_dep)
                        {
                            if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                            {
                                obj_fea_disp_num[obj_id] = std::pair<float,int>(pred_disp,1);
                            }
                            else
                            {
                                // 可以这样做加法并赋值吗？
                                obj_fea_disp_num[obj_id].first  += pred_disp;
                                obj_fea_disp_num[obj_id].second += 1;
                                // obj_fea_disp_num[obj_id].first  = obj_fea_disp_num[obj_id].first + pred_disp;
                                // obj_fea_disp_num[obj_id].second = obj_fea_disp_num[obj_id].second + 1;
                            }
                        }
                    }
                }
            }
        }

        // start add new FAST and set their right match predict

        // int tmp_id = n_id + 1000000;
        // int num_FAST = num_track_FAST;
        

        // while(!stereo_match_done)
        // {
        //     usleep(300);
        // }

        // 添加背景新点。新点在当前帧的右匹配点预测只能通过depth_map获取
        if(!add_new_sift_in_next_frame)
        {
            for(auto &p: n_FAST_bg)
            {
                bool has_no_stereo = false;
                // 位于左图像最左和最右边小部分区域的点暂不考虑
                if (!inBorder(p)) continue;
                l_x = p.x;
                l_y = p.y;
                // 图像的边界用左右相机重叠视野以及考虑的深度范围来确定。左图像的最左边部分不可能有右图像观测，右图像最右边不可能有左图像观测！
                pt_info = seg_map.at<Vec2b>(l_y, l_x);
                cls_label = pt_info[0];
                // assert(cls_label == 0 && "Why detect a new FAST which belongs to obj!");
                if (cls_label != 0) continue;

                // if (l_x <= bg_left_border_left_img) continue;
                // ++count;
                // if(l_x <= bg_left_border_left_img) 
                // {
                //     // has_no_stereo = true;
                //    continue
                // }
                
                obj_id = pt_info[1];
                
                // 新特征点的cls label为当前帧的检测类别，后续等物体匹配完成后再改为全局cls label
                obj_cls_id_FAST.push_back(std::pair<uchar, int>(cls_label, obj_id));
                // 注意，在完成物体关联和确定物体的全局id之后，要对这个量进行修改，保存当前帧每个特征点的所属全局物体的id，用于与下一帧进行物体关联
                prev_FAST_global_obj_id.push_back(obj_id);
                cur_FAST.push_back(p);
                // 该特征地图点的（滑窗内）临时全局id，后续确定这部分新检测的FAST点中哪些需要被添加，再将其id减去100000
                //ids_FAST.push_back(tmp_id++);
                ids_FAST.push_back(n_id++);
                // 新检测的特征地图点，所以跟踪次数设置为1。其实这里的变量应该称为obser_cnt，即该点被观测的帧数，而不是被跟踪的次数（每一个跟踪就需要两帧观测）！
                track_cnt_FAST.push_back(1);
                //left_id_of_right_FAST.push_back(num_FAST++);
                // 背景中其实有可能某些点是属于漏检物体点的，后续发现该漏检物体之后需要修正这个值?
                ++num_new_FAST_bg;
                // 给定右匹配点的预测值
                // 如果是首帧，则使用depth_map；
                // 对于后续帧，也可以使用depth_map来设置
                if(frame_count == 0 || !use_motion_to_pred_fea_dep)
                {
                    pred_disp = depth_map.at<float>(l_y,l_x);
                    if (pred_disp <= 0) 
                    {
                        // if(ave_dep_bg_cur_frame != 0 && num_bg_with_dep > 15) 
                        // {
                        //     float ave_dep = ave_dep_bg_cur_frame/num_bg_with_dep;
                        //     float ave_disp_x = mbf/ave_dep;
                        //     float ave_disp_y = Y_shift_right_image/ave_dep;
                        //     // 检查超出边界
                        //     r_x = max(5.0f,l_x-ave_disp_x);
                        //     r_y = min(l_y+ave_disp_y, (float)(row-5));
                        //     cur_right_FAST.emplace_back(r_x,r_y);
                        // }
                        // else
                        {
                            r_x = max(5.0f,l_x-ave_disp_x_bg);
                            r_y = min(l_y+ave_disp_y_bg, (float)(row-5));
                            cur_right_FAST.emplace_back(r_x,r_y);
                        }
                        // cur_right_FAST.emplace_back(l_x,l_y);
                        continue;
                    }
                    depth = mbf/pred_disp;
                }
                // 对于非首帧，不等待depth_map的结果，那么背景中的新FAST特征点就没法给定较为精确的视差预测值！
                else
                {
                    // sift背景点的深度不能用来为FAST点提供平均值，我们设定FAST点需要比较近，因为其检测精度很有限
                    // if(ave_dep_bg_cur_frame != 0 && num_bg_with_dep > 15) 
                    // {
                    //     float ave_dep = ave_dep_bg_cur_frame/num_bg_with_dep;
                    //     float ave_disp_x = mbf/ave_dep;
                    //     float ave_disp_y = Y_shift_right_image/ave_dep;
                    //     // 检查超出边界
                    //     r_x = max(5.0f,l_x-ave_disp_x);
                    //     r_y = min(l_y+ave_disp_y, (float)(row-5));
                    //     cur_right_FAST.emplace_back(r_x,r_y);
                    // }
                    // else
                    {
                        r_x = max(5.0f,l_x-ave_disp_x_bg);
                        r_y = min(l_y+ave_disp_y_bg, (float)(row-5));
                        cur_right_FAST.emplace_back(r_x,r_y);
                    }
                    continue;
                }
                shift_y = Y_shift_right_image/depth;
                r_x = l_x - pred_disp;
                
                if(r_x >= 5 && r_x <= bg_right_border_right_img)
                    cur_right_FAST.emplace_back(r_x, l_y+shift_y);
                else if (r_x < 5)
                    cur_right_FAST.emplace_back(5, l_y+shift_y);
                else(r_x > bg_right_border_right_img);
                    cur_right_FAST.emplace_back(bg_right_border_right_img, l_y+shift_y);
            }
            n_FAST_bg.clear();
            //printf("feature cnt after add %d\n", (int)ids_FAST.size());
        }

        // must add new objs fea with depth in every frame

        // cout << "Num of original detected new FAST of objs: " << n_FAST_obj.size() << endl;
        // int num_new_FAST_obj = 0;
        for(auto &p : n_FAST_obj)
        {
            if (!inBorder(p)) continue;
            l_x = p.x;
            l_y = p.y;
            pt_info = seg_map.at<Vec2b>(l_y, l_x);
            cls_label = pt_info[0];

            // cout << "Class label of obj: " << (int)cls_label << endl;

            if (cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7 || cls_label == 0) continue;
            // 不可能在右图像中观测到的特征点。物体点在每一帧都需要有立体匹配，即深度估计
            if (l_x <= obj_left_border_left_img) continue;
            
            // ++num_new_FAST_obj;

            obj_id = pt_info[1];

            obj_cls_id_FAST.push_back(std::pair<uchar, int>(cls_label, obj_id));
            prev_FAST_global_obj_id.push_back(obj_id);
            cur_FAST.push_back(p);
            ids_FAST.push_back(n_id++);
            // 新检测的特征地图点，所以跟踪次数设置为1。其实这里的变量应该称为obser_cnt，即该点被观测的帧数，而不是被跟踪的次数（每一个跟踪就需要两帧观测）！
            track_cnt_FAST.push_back(1);
            // left_id_of_right_FAST.push_back(num_FAST++);

            // 设置右图像点的预测
            if(frame_count == 0 || !use_motion_to_pred_fea_dep)
            {
                pred_disp = depth_map.at<float>(l_y,l_x);
                bool false_disp = false;
                if (pred_disp <= 0 || pred_disp >= 192) 
                {
                    // assert(pred_disp > 0 && "Weired! disp in depth map is <= 0!");
                    false_disp =true;
                    // 利用之前已经有有效深度的检测点的视差值
                    if(obj_fea_disp_num.find(obj_id) != obj_fea_disp_num.end())
                        pred_disp = obj_fea_disp_num[obj_id].first/obj_fea_disp_num[obj_id].second;
                    else
                    {
                        r_x = max(5.0f,l_x-ave_disp_x_objs);
                        r_y = min(l_y+ave_disp_y_objs, (float)(row-5));
                        // 那这些点是否可以等到全部遍历完再查看是否有同一obj id的点的视差值可提供参考？
                        cur_right_FAST.emplace_back(r_x,r_y);
                        continue;
                    }
                }
                depth = mbf/pred_disp;

                if(!false_disp)
                {
                    // 记录每个检测物体上的新特征点的视差估计，为那些有明显错误视差值的点提供参考值
                    if(obj_fea_disp_num.find(obj_id) == obj_fea_disp_num.end())
                    {
                        obj_fea_disp_num[obj_id] = std::pair<float,int>(pred_disp,1);
                    }
                    else
                    {
                        // 可以这样做加法并赋值吗？
                        obj_fea_disp_num[obj_id].first  += pred_disp;
                        obj_fea_disp_num[obj_id].second += 1;
                        // obj_fea_disp_num[obj_label].first  = obj_fea_disp_num[obj_label].first + pred_disp;
                        // obj_fea_disp_num[obj_label].second = obj_fea_disp_num[obj_label].second + 1;
                    }
                }
            }
            else
            {
                if(obj_fea_disp_num.find(obj_id) != obj_fea_disp_num.end())
                {
                    pred_disp = obj_fea_disp_num[obj_id].first/obj_fea_disp_num[obj_id].second;
                    depth = mbf/pred_disp;
                }
                // 如果是当前帧出现的新物体，则只能用左图像点当作预测值。或者用平均深度对应的平均视差？
                else
                {
                    r_x = max(5.0f,l_x-ave_disp_x_objs);
                    r_y = min(l_y+ave_disp_y_objs, (float)(row-5));
                    // cur_right_FAST.emplace_back(l_x,l_y);
                    cur_right_FAST.emplace_back(r_x,r_y);
                    continue;
                }
            }
            
            shift_y = Y_shift_right_image/depth;
            r_x = l_x - pred_disp;
            r_y = min(l_y+shift_y, (float)(row-5));
            if(r_x >= 5 && r_x <= obj_right_border_right_img)
                cur_right_FAST.emplace_back(r_x, r_y);
            else if (r_x < 5)
                cur_right_FAST.emplace_back(5, r_y);
            else
                cur_right_FAST.emplace_back(obj_right_border_right_img, r_y);
        }
        // cout << "Num of original detected new FAST of objs: " << num_new_FAST_obj << endl;
        n_FAST_obj.clear();
        obj_fea_disp_num.clear();
        
        printf("num of cur_FAST: %d num of cur_right_FAST: %d\n", cur_FAST.size(), cur_right_FAST.size());
    }
    // cout << "Number of all FAST:" << cur_FAST.size() << endl;
    
    // cur_un_FAST.clear();
    // 将点从像素坐标去畸变并提升到归一化平面坐标
    // undistortedPts(cur_FAST, cur_un_FAST, m_camera[0]);

    // 可以将跟右图像相关的变量在此处用完后及时释放所占据的内存？（调用clear()后再调用Vector的shrink_to_fit()）。当前帧的右图像点的信息到最后还有用！

    // 如果是双目，则还要计算左右图像之间的光流跟踪以获得在右图像中的特征点位置
    // 上面先确定了当前帧左图像中的特征点，然后再确定右图像能跟踪到左图像中的哪些点（对于其中的新地图点，如果能在右图像中找到匹配，则在后续估计这些点的深度时是使用当前帧左右相机的位姿估计来进行三角化）
    if(!cur_img_r.empty() && stereo_cam)
    {
        if(!cur_FAST.empty())
        {
            // 后续对此处还没有深度的点进行补充深度值（如果是背景或静态物体点，则可以用三角化测量的结果；如果是动态物体的点，则只能用dep_map的视差值），然后复制给prev_FAST_dep
            cur_FAST_dep.resize(cur_FAST.size(), -1.0);

            //printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> statusRightLeft;
            vector<float> err, err_rl;
            
            cout << "Start stereo match for FAST in current right image!" << endl;
            
            // cur left ---- cur right
            // 不给出预测的右图像中的匹配点吗？因为不给出预测点位置，所以需要使用3层的图像金字塔来进行光流估计。那为什么不直接使用双目立体匹配的结果作为匹配的初值或者最终值呢？
            // 其实对于双目立体图像而言，这里使用3层光流应该就足以跟踪了，因为只要点的深度不要太大，它其实很好寻找（就沿着x轴）。另外，这里使用普通的光流跟踪，可以适配非立体校准图像对的情况！
            // 注意，无论有没有预测值，cur_right_pts和status的长度都与cur_pts的是一样的，status的值会指示cur_right_pts中的对应元素是否为有效的光流估计匹配点
            // TODO: 右图像中的FAST点不应该跟SIFT的点相重叠，这点后续要如何排除？可以再给定一个在SIFT点周围画黑点区域的右图像的mask，用来查询这里的FAST匹配是否在黑点区域外?T  太麻烦了

            // 如果用cv::OPTFLOW_USE_INITIAL_FLOW指明了cur_right_FAST有初始估计值，则cur_FAST和cur_right_FAST的size需要一致
            assert(cur_right_FAST.size() == cur_FAST.size());
            cv::calcOpticalFlowPyrLK(cur_img, cur_img_r, cur_FAST, cur_right_FAST, statusLeftRIght, err, cv::Size(21, 21), 2,
                                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            // cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_FAST, cur_right_FAST, statusLeftRIght, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            FLOW_BACK = 1;
            if(FLOW_BACK)
            {
                reverseLeftPts = cur_FAST;
                cv::calcOpticalFlowPyrLK(cur_img_r, cur_img, cur_right_FAST, reverseLeftPts, statusRightLeft, err_rl, cv::Size(21, 21), 1,
                                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                // cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_FAST, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
            }
            
            float r_x, r_y, disp_x, disp_y, shift_y;
            
            bool keep = false;
            // 如果不使用depth_map作为预测值，则新物体上的FAST点比较不容易寻找右匹配点
            keep = (frame_count > 0 && use_motion_to_pred_fea_dep);
            
            int detected_new_FAST = 0;
            int num_tracked_bg_FAST_with_dep = 0;
            uchar cls;
            for(size_t i = 0; i < statusLeftRIght.size(); ++i)
            {
                uchar find_track_l_r = 0;

                if(FLOW_BACK)
                {
                    find_track_l_r = statusLeftRIght[i] * statusRightLeft[i];
                }
                else
                {
                    find_track_l_r = statusLeftRIght[i];
                }
                
                // 采用当前帧的cls来判断是否为物体,因为跟踪点的cls是与上一帧的全局物体的cls对齐的(除非上一帧为背景而当前帧为物体)
                cls = obj_cls_id_FAST[i].first;

                // 如果是最先的版本，即在每一最新中都检测新的背景和物体点，则在此处排除图像最顶部区域的点
                if(!add_new_sift_in_next_frame)
                {
                    // 室外场景图像的中顶部区域一般是天空或者比较远的建筑，两侧顶部还可能是建筑
                    // TODO：其实最好的办法还是与激光雷达相结合，把深度明显太远的方向的图像区域全部去除！
                    // 光靠图像的话只能用全景分割来区别更一般的区域！例如将天空、树木等物体区域去除
                    // 注意，前面的分数要有一个数是浮点，否则该分数实际结果为0！！！
                    // if(cls == 0 && (cur_FAST[i].y < 1.0/6*row || (cur_FAST[i].x > 2.0/5*col && cur_FAST[i].x < 3.0/5*col)))
                    // 路面其实也可以检测特征点，但是很难跟踪，参考SOFT2！！
                    if(cls == 0 && (cur_FAST[i].y < 1.0/6*row))
                    {
                        statusLeftRIght[i] = 0;
                        continue;
                    }
                }
                
                // 通过CPU光流找到左右匹配点
                if(find_track_l_r)
                {
                    if (!(inBorder(cur_right_FAST[i])))
                    {
                        // 如果是物体点且没有立体匹配（位于图像边缘），则放弃该点
                        if(cls != 0)
                            statusLeftRIght[i] = 0;
                        //invalid_cur_FAST++;
                        continue;
                    }

                    if(FLOW_BACK && distance(cur_FAST[i], reverseLeftPts[i]) > 0.6)
                    {
                        // 跟踪点如果在当前帧没有立体匹配
                        if(i < num_track_FAST)
                        {
                            // 背景跟踪点是否保留取决于该点上一帧是否有深度值，而不是取决于当前帧是否有立体匹配
                            if(cls == 0)
                            {
                                statusLeftRIght[i] = 2;
                                FAST_no_stereo_bg.push_back(i);
                                // invalid_cur_FAST++;
                            }
                            else
                            {
                                // 如果该物体点的右观测点预测值是通过运动模型来设置的，则可能匹配得不是很准确，后续可以使用depth_map直接获取深度值
                                if(keep)
                                {
                                    id_FAST_no_depth.push_back(i);
                                    statusLeftRIght[i] = 2;
                                }
                                else
                                    statusLeftRIght[i] = 0;
                            }
                        }
                        else
                        {
                            if(cls == 0)
                            {
                                // 如果在当前帧检测了新的FAST背景点，是否允许其没有立体匹配取决于是否使用前后2帧的三角测量来恢复深度
                                if(USE_TRIANGULATE_TWO_FRAME)
                                {
                                    statusLeftRIght[i] = 3;
                                    FAST_no_stereo_bg.push_back(i);
                                }
                                else
                                    statusLeftRIght[i] = 0;
                            }
                            else
                            {
                                if(keep)
                                {
                                    id_FAST_no_depth.push_back(i);
                                    statusLeftRIght[i] = 3;
                                }
                                else
                                    statusLeftRIght[i] = 0;
                            }
                        }
                        
                        continue;
                    }
                    
                    r_x = cur_right_FAST[i].x;
                    r_y = cur_right_FAST[i].y;
                    // 右图像中的点不能大于指定深度范围内的左右相机重叠视野在右图像中的投影边界
                    if (cls == 0 && r_x > bg_right_border_right_img)
                    {
                        // 跟踪点是否保留取决于该点上一帧是否有深度值，而不是取决于当前帧是否有立体匹配
                        if(i < num_track_FAST)
                        {
                            statusLeftRIght[i] = 2;
                            FAST_no_stereo_bg.push_back(i);
                            // invalid_cur_FAST++;
                        }
                        else
                        {
                            // 匹配的点位于右图像边缘，认为该点下一帧很难被跟踪到了？
                            // statusLeftRIght[i] = 0;

                            if(USE_TRIANGULATE_TWO_FRAME)
                            {
                                statusLeftRIght[i] = 3;
                                FAST_no_stereo_bg.push_back(i);
                            }
                            else
                                statusLeftRIght[i] = 0;
                        }
                        //invalid_cur_FAST++;
                        continue;
                    }
                    else if (cls > 0 &&  r_x > obj_right_border_right_img)
                    {
                        statusLeftRIght[i] = 0;
                        //invalid_cur_FAST++;
                        continue;
                    }

                    disp_x = cur_FAST[i].x - r_x;
                    disp_y = cur_FAST[i].y - r_y;
                    // 认为FAST匹配是较为准确的立体匹配，即y方向上没有视差
                    // disp_x is the (x_left_img - x_right_img), so disp_x should be > 0
                    // 如果左右图像没有严格的立体校正，则y方向上的视差可能不会接近0。
                    if (disp_x <= 0) 
                    {
                        if (cls == 0)
                        {
                            // 当前帧没有深度值的FAST跟踪点的状态用2，新FAST点则用3
                            // 最后还是决定如果当前帧没有深度值，则必须是跟踪点且上一帧有深度值，才能保留该点
                            if(i < num_track_FAST)
                            {
                                statusLeftRIght[i] = 2;
                                FAST_no_stereo_bg.push_back(i);
                            }
                            else 
                            {
                                // 要不要保留当前帧背景中没有立体匹配的新sift点
                                if(USE_TRIANGULATE_TWO_FRAME)
                                {
                                    FAST_no_stereo_bg.push_back(i);
                                    statusLeftRIght[i] = 3;
                                } 
                                else
                                    statusLeftRIght[i] = 0;
                            }
                        }
                        else
                        {
                            if (keep) 
                            {
                                id_FAST_no_depth.push_back(i);
                                if(i < num_track_FAST)
                                    statusLeftRIght[i] = 2;
                                else
                                    statusLeftRIght[i] = 3;
                            }
                            else
                                statusLeftRIght[i] = 0;
                        }
                        continue;
                        //invalid_cur_FAST++;
                    }

                    float depth = mbf/disp_x;

                    // FAST点经常会把很远处的树叶作为特征点，并且深度估计也是错误的。因此限制FAST点的深度值
                    // 深度值太大时认为立体匹配不够准确，后续采用运动变换来计算深度
                    // if (cls == 0 && (depth >= mThDepthBg || depth < mMinDepthPt))
                    if (cls == 0 && (depth > 20 || depth < 1.6)) 
                    {
                        if(i < num_track_FAST)
                        {
                            statusLeftRIght[i] = 2;
                            FAST_no_stereo_bg.push_back(i);
                        }
                        else
                            // 深度不合适的新点就不要了，因为下一帧很难跟踪
                            statusLeftRIght[i] = 0;
                        
                        //invalid_cur_FAST++;
                        continue;
                    }
                    else if(cls > 0 && (depth >= mThDepthObj || depth < mMinDepthPt)) 
                    {
                        if (keep) 
                        {
                            id_FAST_no_depth.push_back(i);
                            if(i < num_track_FAST)
                                statusLeftRIght[i] = 2;
                            else
                                statusLeftRIght[i] = 3;
                        }
                        else
                            statusLeftRIght[i] = 0;
                    }

                    shift_y = Y_shift_right_image/depth;

                    //  FAST点的匹配在纵向上不够准确，是否要从depth_map获取该点的深度？不需要了，因为depth_map的结果很难比这里的flow optical要好
                    // if(abs(disp_y + shift_y) >= 0.5)
                    // if(0 && abs(disp_y + shift_y) >= 0.5)
                    // if(abs(disp_y + shift_y) >= 1.0)
                    
                    if(abs(disp_y) > 1.1)
                    {
                        if (cls == 0)
                        {
                            if(i < num_track_FAST)
                            {
                                statusLeftRIght[i] = 2;
                                FAST_no_stereo_bg.push_back(i);
                            }
                            else
                            {
                                if(USE_TRIANGULATE_TWO_FRAME)
                                {
                                    statusLeftRIght[i] = 3;
                                    FAST_no_stereo_bg.push_back(i);
                                }
                                else
                                    statusLeftRIght[i] = 0;
                            }
                            // invalid_cur_FAST++;
                        }
                        else
                        {
                            if (keep) 
                            {
                                id_FAST_no_depth.push_back(i);
                                if(i < num_track_FAST)
                                    statusLeftRIght[i] = 2;
                                else
                                    statusLeftRIght[i] = 3;
                            }
                            else
                                statusLeftRIght[i] = 0;
                        }

                        //invalid_cur_FAST++;
                        continue;    
                        
                    }

                    // 会出现这种情况吗？？所估计的深度值使得3D点位于右相机的视锥左侧面之外（更左边）且位于规定深度区域内，则排除，因为该点不可能在右图像观测到
                    // float err = (r_cam_3D_plane[0]*cur_un_FAST[i].x+r_cam_3D_plane[1]*cur_un_FAST[i].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                    // if (err <= 0) 
                    // {
                    //     statusLeftRIght[i] = 0;
                    //     //invalid_cur_FAST++;
                    //     continue;
                    // }

                    // 如果所估计的深度满足要求，则记录
                    // 是否直接默认为立体校对后的匹配，还是需要使用左右匹配进行三角化（一般还要同时优化左右相机的外参）
                    if(cls == 0)
                    {
                        if(use_tria_stereo)
                            cur_FAST_dep[i] = -1.0;
                        else
                            cur_FAST_dep[i] = depth;
                    }
                    else
                    {
                        // 能这么做是因为立体校正足够准确
                        cur_FAST_dep[i] = depth;
                    }
                    
                    if(i >= num_track_FAST)
                    {
                        ++detected_new_FAST;
                    }
                    
                    if(cls == 0)
                    {
                        if(!use_tria_stereo)
                        {
                            ave_dep_bg_cur_frame += depth;
                            ++num_bg_with_dep;
                        }

                        if(i < num_track_FAST) ++num_tracked_bg_FAST_with_dep;
                    }
                }
                else
                {
                    if(obj_cls_id_FAST[i].first == 0)
                    {
                        if(i < num_track_FAST)
                        {
                            statusLeftRIght[i] = 2;
                            FAST_no_stereo_bg.push_back(i);
                        }
                        else
                        {
                            if(USE_TRIANGULATE_TWO_FRAME)
                            {
                                statusLeftRIght[i] = 3;
                                FAST_no_stereo_bg.push_back(i);
                            }
                            else
                                statusLeftRIght[i] = 0;
                        }
                        // invalid_cur_FAST++;
                    }
                    else
                    {
                        if(keep)
                        {
                            id_FAST_no_depth.push_back(i);
                            if(i < num_track_FAST)
                                statusLeftRIght[i] = 2;
                            else
                                statusLeftRIght[i] = 3;
                            // ++detected_new_FAST;
                        }
                        else
                            statusLeftRIght[i] = 0;
                    }
                }
            }
            
            reverseLeftPts.clear();
            statusRightLeft.clear();

            if(add_new_sift_in_next_frame)
                cout << "num of new FAST with stereo match (only for objs) using CPU optical flow in current frame: " << detected_new_FAST << endl;
            else
                cout << "num of new FAST with stereo match (for bg and objs) using CPU optical flow in current frame: " << detected_new_FAST << endl;
            
            cout << "num of tracked FAST with stereo match in current frame: " << num_tracked_bg_FAST_with_dep << endl;
        }
    }
    
    // 用于替代INIT_DEPTH来为三角化失败的点赋予深度值
    // if(num_bg_with_dep > 1) ave_dep_bg_cur_frame = ave_dep_bg_cur_frame/num_bg_with_dep;
    if(num_bg_with_dep > 5)
    {
        cout << "Num of bg fea with stereo match: " << num_bg_with_dep << endl;
        cout << "Average dep of bg fea with stereo match: " << (ave_dep_bg_cur_frame/num_bg_with_dep) << endl;
    }
    
    // 标注新检测的背景FAST点，排除掉与已有点重合的新背景sift点！
    if(!add_new_sift_in_next_frame) set_new_fea_in_mask(initial_succ);

    printf("new FAST detection and depth eatimation for all FAST fea costs: %fms \n", t_det_new_and_assign.toc());
}

// 等待GPU立体匹配估计完成之后，使用depth_map补充没有深度估计的FAST特征点。最后将FAST点按照cls分配给当前帧各个临时物体。
void FeatureTracker::assign_fea_objs(int frame_count, const Mat &dep_map, bool &end_flow_post, bool &end_stereo_post, const Mat &_img1, const vector<int> &valid_obj_id, const Mat &obj_id_map, map<int, YoloV8::Box> &bbox_mask)
{
    TicToc t_o;

    cout << "num of detected objs: " << valid_obj_id.size() << endl;
    // cv::Mat rightImg = _img1;

    // 对于当前帧 物体 的sift跟踪点或新点，如果先前没有得到可靠的左右帧匹配，则在这里使用depth_map来获取深度
    if(!id_sift_no_depth.empty())
    {
        while (!end_stereo_post)
        {
            usleep(300);
        }

        float l_x, l_y, r_x, disp, depth, y_shift, err;
        int id_cur_sift;
        for(int i = 0; i < id_sift_no_depth.size(); ++i)
        {
            id_cur_sift = id_sift_no_depth[i];

            // status可以是2或3，即没有立体匹配的物体跟踪点或新点
            if(status_sift[id_cur_sift] == 0) continue;

            l_x = cur_sift[id_cur_sift].x;
            l_y = cur_sift[id_cur_sift].y;
            disp = dep_map.at<float>(l_y,l_x);
            if (disp <= 0) 
            {
                cout << "Weird! disp_x < 0!" << endl;
                status_sift[id_cur_sift] = 0;
                continue;
            }
            // 该点的全局cls label（对于跟踪点），如果是新检测点，则暂时使用seg_map中的检测类别
            uchar cls = obj_cls_id_sift[id_cur_sift].first;
            // 检验该深度是否合适
            r_x = l_x - disp;
            
            // 当前帧背景点也可能会被放进id_sift_no_depth中!即判断该背景点跟踪自上一帧的物体点!
            // 因此,会被放进id_sift_no_depth中的都是当前帧的物体点!
            // if (cls == 0 && (r_x <5 || r_x > bg_right_border_right_img)) continue;
            // if (cls > 0 && (r_x <5 || r_x > obj_right_border_right_img)) continue;

            if (r_x < 5 || r_x > obj_right_border_right_img)
            {
                status_sift[id_cur_sift] = 0;
                continue;
            }

            depth = mbf/disp;

            y_shift = Y_shift_right_image/depth;
            
            if ((l_y + y_shift) > row - 5)
            {
                status_sift[id_cur_sift] = 0;
                continue;
            }

            // if (cls == 0 && (depth >= mThDepthBg || depth < mMinDepthPt))
            if (cls == 0 && (depth >= mThDepthObj || depth < mMinDepthPt))
            {
                status_sift[id_cur_sift] = 0;
                continue;
            }
            else if(cls > 0 && (depth >= mThDepthObj || depth < mMinDepthPt))
            {
                status_sift[id_cur_sift] = 0;
                continue;
            }
            // err = (r_cam_3D_plane[0]*cur_un_sift[id_cur_sift].x+r_cam_3D_plane[1]*cur_un_sift[id_cur_sift].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
            // if (err <= 0)
            // {
            //     status_sift[id_cur_sift] = 0;
            //     continue;
            // }

            // 如果满足上面所有的条件，则认为通过dep_map找到的深度值可信；
            // 但是不会为此建立该点的右观测，因为如果它是静态物体，则其点可能在下一帧可以参与相机F矩阵和t尺度的估计，需要保证该点的深度是足够准确的！
            // 保持该点的status为2或3，即没有立体匹配的跟踪点或者新点
            // status_sift[id_cur_sift] = 1;
            // cur_right_sift[id_cur_sift] = cv::Point2f(r_x,l_y+y_shift);
            // cur_right_sift[id_cur_sift] = cv::Point2f(r_x,l_y);
            cur_sift_dep[id_cur_sift] = depth;
            // invalid_cur_FAST--;
        }
    }
    
    // 是否还要对sift跟踪点进行筛选？如果跟踪点与flow_map给出的结果相距太远，则拒绝该点？
    if(0 && use_motion_to_pred_fea_dep)
    {
        while(!end_flow_post)
        {
            usleep(400);
        }
        // todo
        for(int i = 0; i < cur_sift.size(); ++i)
        {

        }
    }

    // invalid_cur_FAST = 0;
    // 如果是首帧，则这里不会再进行任何FAST特征点的深度获取。当然，非首帧时这里也可能不需要（前面已经完成所有跟踪点的深度估计）
    // int invalid_FAST = id_FAST_no_depth.size();
    // cout << "Original num of FAST without depth from optical flow: " << id_FAST_no_depth.size() << endl;
    // 对于当前帧“物体”的新FAST点，尝试用depth_map来为其获取立体匹配
    if (!id_FAST_no_depth.empty())
    {
        while (!end_stereo_post)
        {
            usleep(300);
        }
        
        float l_x, l_y, r_x, r_y, disp, depth, y_shift, err;
        int id_cur_FAST;
        // 部分物体FAST点如果没法通过FAST光流找到左右匹配，则直接使用立体匹配图! 
        for(int i = 0; i < id_FAST_no_depth.size(); ++i)
        {
            // 视差值为正，即左点-右点
            id_cur_FAST = id_FAST_no_depth[i];
            l_x = cur_FAST[id_cur_FAST].x;
            l_y = cur_FAST[id_cur_FAST].y;

            disp = dep_map.at<float>(l_y,l_x);
            if (disp <= 0) 
            {
                cout << "Weird! disp_x < 0!" << endl;
                statusLeftRIght[id_cur_FAST] = 0;
                //++invalid_cur_FAST;
                continue;
            }

            uchar cls = obj_cls_id_FAST[id_cur_FAST].first;
            // 检验该深度是否合适
            r_x = l_x - disp;
            // 同样地,会加入id_FAST_no_depth的当前点都是物体点(要么跟踪自上一帧的物体点,不论当前帧是否为背景;要么是当前帧的检测物体上的新点)
            // if (cls == 0 && (r_x <5 || r_x > bg_right_border_right_img))
            if (cls == 0 && (r_x <5 || r_x > obj_right_border_right_img))
            {
                //++invalid_cur_FAST;
                statusLeftRIght[id_cur_FAST] = 0;
                continue;
            }

            if (cls > 0 && (r_x <5 || r_x > obj_right_border_right_img))
            {
                //++invalid_cur_FAST;
                statusLeftRIght[id_cur_FAST] = 0;
                continue;
            }

            depth = mbf/disp;

            y_shift = Y_shift_right_image/depth;
            
            if(l_y + y_shift > row - 5) 
            {
                statusLeftRIght[id_cur_FAST] = 0;
                continue;
            }


            // if (cls == 0 && (depth >= mThDepthBg || depth < mMinDepthPt))
            if (cls == 0 && (depth >= mThDepthObj || depth < mMinDepthPt))
            {
                //++invalid_cur_FAST;
                statusLeftRIght[id_cur_FAST] = 0;
                continue;
            }
            else if(cls > 0 && (depth >= mThDepthObj || depth < mMinDepthPt))
            {
                //++invalid_cur_FAST;
                statusLeftRIght[id_cur_FAST] = 0;
                continue;
            }

            // err = (r_cam_3D_plane[0]*cur_un_FAST[id_cur_FAST].x+r_cam_3D_plane[1]*cur_un_FAST[id_cur_FAST].y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
            // if (err <= 0) 
            // {
            //     //++invalid_cur_FAST;
            //     continue;
            // }
            
            // 如果满足上面所有的条件，则认为通过dep_map找到的右图像匹配点是可信的
            // statusLeftRIght[id_cur_FAST] = 1;
            // cur_right_FAST[id_cur_FAST] = cv::Point2f(r_x,l_y+y_shift);
            // cur_right_FAST[id_cur_FAST] = cv::Point2f(r_x,l_y);

            cur_FAST_dep[id_cur_FAST] = depth;
            // --invalid_FAST;
        }
        // // 左图像中的特征点也要进行reduce！因为我们只保留有深度估计的点！
        // reduceVector(cur_FAST, statusLeftRIght);
        // reduceVector(ids_FAST, statusLeftRIght);
        // reduceVector(track_cnt_FAST, statusLeftRIght);
        // reduceVector(obj_cls_id_FAST, statusLeftRIght);
        // reduceVector(cur_un_FAST, statusLeftRIght);
        // // 对于上一帧的FAST点也要进行删减，因此只截取来自跟踪的那部分FAST点的状态
        // vector<uchar> sub_vector(statusLeftRIght.begin(),statusLeftRIght.begin()+num_track_FAST);
        // reduceVector(prev_FAST_global_obj_id, sub_vector);
        // reduceVector(prev_FAST_dep, sub_vector);

        // // 更新当前帧中有效的FAST跟踪点数量
        // num_track_FAST = prev_FAST_global_obj_id.size();
        // ids_FAST_right = ids_FAST;
        
        // // 这里对于左图像中无法在右图像跟踪到的点，只在右图像中去除无法匹配的点，剩下的跟踪到的点可以在后续BA提供约束；而在左图像不操作，这样这些地图点在当前帧只有在左图像中有观测约束
        // reduceVector(cur_right_FAST, statusLeftRIght);
    }
    // cout << "Final num of FAST without depth: " << invalid_FAST << endl;

    // 是否再用depth_map筛选FAST和sift新点，对于太远的FAST则抛弃。增加这一步是因为上面CPU光流阶段的立体匹配并不准确，很多十分遥远的树叶或房子的点的深度估计错误，导致其被加入FAST点中！
    bool check_dep_with_dep_map = false;
    if(check_dep_with_dep_map)
    {
        while (!end_stereo_post)
        {
            usleep(300);
        }
        float dep_, disp_;
        for(int i = num_track_FAST; i < cur_FAST.size(); ++i)
        {
            if(statusLeftRIght[i] == 0 || statusLeftRIght[i] == 1) continue;
            disp_ = dep_map.at<float>(cur_FAST[i].y,cur_FAST[i].x);
            if(disp_ > 0)
            {
                dep_ = mbf/disp_;
                if(dep_ > 15)
                    statusLeftRIght[i] = 0;
            }
        }

        for(int i = num_track_sift; i < cur_sift.size(); ++i)
        {
            if(status_sift[i] == 0 || status_sift[i] == 1) continue;
            disp_ = dep_map.at<float>(cur_sift[i].y,cur_sift[i].x);
            if(disp_ > 0)
            {
                dep_ = mbf/disp_;
                if(dep_ > mThDepthBg)
                    status_sift[i] = 0;
            }

            // if (cur_sift[i].y < 1.0/6*row && cur_sift[i].x > 1.0/3*col && cur_sift[i].x < 2.0/3*col)
            // {
            //     status_sift[i] == 0;
            // }
        }
    }

    // 删除所有无效的sift跟踪点和新点
    // 但是如果这里删除无效sift点，会导致id_sift_no_depth中的序号无效！！！
    // reduce_invalid_fea(true);
    // 这里是否要再次删除无效的FAST点（即某些新的FAST点）?可以先不要，最后和运动估计外点一起删除（此外，如果要在此处删除，则statusLeftRIght也要同时删除其中的无效元素）
    // reduce_invalid_fea(false);

    if(!cur_sift.empty()) 
    {
        undistortedPts(cur_sift, cur_un_sift, m_camera[0], status_sift, false);
        // velocity与cur_sift等一样，里面可能存在这需要删除的点，包括跟踪的没有深度的点，以及后续在物体匹配完成后多余的点，以及RANSAC中的外点
        ptsVelocity(sift_velocity, ids_sift, cur_un_sift, prev_un_Fea_map, true, false, status_sift);
        
        // 注意，right特征点是跟左特征点集对齐的，其中有些点是无效的，后期需要delete
        undistortedPts(cur_right_sift, cur_un_right_sift, m_camera[1], status_sift, true);
        ptsVelocity(right_sift_velocity, ids_sift, cur_un_right_sift, prev_un_r_Fea_map, true, false, status_sift, true);
    }

    if(!cur_FAST.empty())
    {
        undistortedPts(cur_FAST, cur_un_FAST, m_camera[0], statusLeftRIght, false);
        
        // 计算当前帧中追踪到的特征点相对于上一帧的匹配特征点的速度（就是u和v方向的像素距离向量除以两帧时间间隔）
        // 此函数中顺便将当前帧中的特征点及其对应的地图点全局id放进cur_un_pts_map
        ptsVelocity(FAST_velocity, ids_FAST, cur_un_FAST, prev_un_Fea_map, true, false, statusLeftRIght);

        undistortedPts(cur_right_FAST, cur_un_right_FAST, m_camera[1], statusLeftRIght, true);
        
        // 上一帧右图像中的点直接用来跟当前帧右图像的点 计算速度？？为什么要计算这个量？又不计算前后帧两个右相机的相对位姿吧？而且上面也没用来提供预测值？
        // 另外，当前帧右图像和上一帧右图像所出现的特征地图点不一定完全重合吧？是的，如果当前帧某个特征地图点没有出现在上一帧右图像中，则该像素点速度设为0
        // 这里要计算的速度是否应该为当前帧左右图象跟踪点之间的速度？这里计算前后两个右图像对应点的速度是为了计算相机与IMU时间戳对齐时的插值时刻的图像特征点位置
        ptsVelocity(right_FAST_velocity, ids_FAST, cur_un_right_FAST, prev_un_r_Fea_map, true, false, statusLeftRIght, true);
    }

    // 此时prevLeftPtsMap还没更新，因此其保存的是上一帧左图像最终的特征像素点，用于表示画图表示当前帧左图像中跟踪上一帧的点的光流(用一个小箭头和两个像素位置处的小点表示)
    if(SHOW_TRACK)
    {
        drawTrack(cur_img, _img1, ids_FAST, cur_FAST, cur_right_FAST, prevLeftFeaMap);
        drawTrack(cur_img, _img1, ids_sift, cur_sift, cur_right_sift, prevLeftFeaMap);
    }
    
    // Mat img_for_debug = prev_color_img_l.clone();
    
    show_tracked_fea = false;
    // draw and show valid tracked fea in prev images!!
    if(show_tracked_fea && frame_count > 0)
    {
        Mat img_for_stereo = prev_color_img_l.clone();

        for(int i = 0; i < prev_sift.size(); ++i)
        {
            if(status_sift[i] == 0) continue;
            // 暂不显示物体/背景点
            if(obj_cls_id_sift[i].first != 0) continue;

            int gl_id = ids_sift[i];
            if(find(reserve_bg_track_pt_id.begin(),reserve_bg_track_pt_id.end(),gl_id) != reserve_bg_track_pt_id.end()) continue;
            Point2f &pt = prev_sift[i];
            // 实线的圈代表sift。红色的圈代表被跟踪sift点
            circle(prev_color_img_l, pt, 4, Scalar(0,0,255), 1, 16);
            Point2f &pt_cur = cur_sift[i];
            // 蓝色的线
            line(prev_color_img_l, pt, pt_cur, Scalar(255,0,0), 1, 16);

            if(prevRightFeaMap.find(gl_id) != prevRightFeaMap.end())
            {
                // 绿色的圈代表有立体匹配的sift点
                circle(img_for_stereo, pt, 4, Scalar(0,255,0), 1, 16);
                Point2f &pt_r = prevRightFeaMap[gl_id];
                line(img_for_stereo, pt, pt_r, Scalar(255,0,0), 1, 16);
            }
        }
        
        for(int i = 0; i < prev_FAST.size(); ++i)
        {
            if(statusLeftRIght[i] == 0) continue;
            // 暂不显示物体点
            if(obj_cls_id_FAST[i].first != 0) continue;

            int gl_id = ids_FAST[i];
            if(find(reserve_bg_track_pt_id.begin(),reserve_bg_track_pt_id.end(),gl_id) != reserve_bg_track_pt_id.end()) continue;
            Point2f &pt = prev_FAST[i];
            // 虚线的圈代表FAST.
            circle(prev_color_img_l, pt, 4, Scalar(255,0,255), 1, 8);
            Point2f &pt_cur = cur_FAST[i];
            // 线还是保持实线.FAST的线用绿色的
            line(prev_color_img_l, pt, pt_cur, Scalar(0,0,255), 1, 16);
            
            if(prevRightFeaMap.find(gl_id) != prevRightFeaMap.end())
            {
                circle(img_for_stereo, pt, 4, Scalar(255,255,0), 1, 8);
                Point2f &pt_r = prevRightFeaMap[gl_id];
                line(img_for_stereo, pt, pt_r, Scalar(0,0,255), 1, 16);
            }
        }

        // while(true)
        // {
        //     cv::imshow("mask of obj in cur frame", prev_mask_solid_objs);
        //     // 一直等待用户按下ESC键（ASCI码为27）
        //     if(waitKey(0) == 27)
        //     {
        //         break;
        //     }
        // }
        
        while(true)
        {
            cv::imshow("final all tracked fea in prev left image", prev_color_img_l);
            // 一直等待用户按下ESC键（ASCI码为27）
            if(waitKey(0) == 27)
            {
                break;
            }
        }
        
        while(true)
        {
            cv::imshow("final tracked fea with stereo match in prev left image", img_for_stereo);
            if(waitKey(0) == 27)
            {
                break;
            }
        }
    }
    
    // ————————————————————————————ₔ—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    // 根据当前帧局部obj_id分配sift点
    int valid_sift = 0;
    int num_sift_track = 0;
    int num_sift_new = 0;
    vector<pair<int,Vector8d>> test_vec;
    cout << "Start assign sift to local objs!" << endl;
    bool weired = false;
    uchar cls_;
    int l_id;
    for (int i = 0; i < cur_sift.size(); ++i)
    {
        // 跟踪sift点和新sift点中的无效点
        if (status_sift[i] == 0) continue;

        ++valid_sift;

        bool has_stereo = true;
        int feature_id = ids_sift[i];
        // 注意，第一个元素的index为0
        gl_id_index_map[feature_id] = -1*i;

        cls_ = obj_cls_id_sift[i].first;
        l_id = obj_cls_id_sift[i].second;

        // 当前帧下背景新sift点暂时不放入NewObjFeaFrame中,因为后面无需用到
        if(status_sift[i] == 1)
        {
            if(i < num_track_sift)
                ++num_sift_track;
            else
            {
                ++num_sift_new;
                if(cls_ == 0)
                    continue;
            }
        }
        else
        {
            // 背景或物体点中根据depth_map直接获取深度值,但是没有右观测，因此在这里暂时不修改此变量,等到用三角测量恢复点的深度值之后,再修改该值!
            if(status_sift[i] == 2)
            {
                // 没有立体匹配的背景或物体的跟踪点的status为2
                has_stereo = false;
                ++num_sift_track;

                // 如果纯背景跟踪点在当前帧没有立体匹配，则不将其加入到TrackObjFeaFrame中，因为它在objs-matching中没有用处
                if(cls_ == 0)
                    continue;
            } 

            if(status_sift[i] == 3) 
            {
                ++num_sift_new;
                has_stereo = false;
                // 背景或物体的没有立体匹配的新点的status都是3
                // 其中的背景新点无需加入NewObjFeaFrame中，因为后面无需用到
                if(cls_ == 0)
                    continue;
            }
            //status_sift[i] = 1;
        }

        // 该物体当前的临时id
        int obj_id = obj_cls_id_sift[i].second;

        int prev_obj_id = prev_sift_global_obj_id[i];
        
        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_statu;
        // 由于只添加有双目立体匹配的点，因此这里的z给定深度估计值，而不是原来的1.0
        // xyz_uv_velocity_statu << cur_un_sift[i].x, cur_un_sift[i].y, 1, cur_sift[i].x, cur_sift[i].y, sift_velocity[i].x, sift_velocity[i].y, obj_id;
        xyz_uv_velocity_statu << cur_un_sift[i].x, cur_un_sift[i].y, cur_sift_dep[i], cur_sift[i].x, cur_sift[i].y, sift_velocity[i].x, sift_velocity[i].y, prev_obj_id;
        
        // 将双目图像的右图像中的匹配点也作为左图像中所见地图点的一个观测，后续用于LBA对该地图点的优化约束，或者用于三角化得到该点的全局坐标（对于当前帧的新地图点而言）
        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_statu_r;
        // xyz_uv_velocity_statu_r << cur_un_right_sift[i].x, cur_un_right_sift[i].y, 1, cur_right_sift[i].x, cur_right_sift[i].y, right_sift_velocity[i].x, right_sift_velocity[i].y, obj_id;
        test_vec.emplace_back(0,xyz_uv_velocity_statu);
        if(has_stereo)
        {
            // 这里可以利用右匹配点的z来保存左图像点的上一帧匹配点的深度值！
            // 该点在上一帧不一定有有效的深度值，后续使用时需要判断其是否大于0
            xyz_uv_velocity_statu_r << cur_un_right_sift[i].x, cur_un_right_sift[i].y, 1.0, cur_right_sift[i].x, cur_right_sift[i].y, right_sift_velocity[i].x, right_sift_velocity[i].y, prev_obj_id;
        }
        
        // 记录当前帧局部物体所包含的特征点的id
        // index_sift_objs_cur[obj_id].push_back(i);
        
        if (i < num_track_sift) 
        {
            // 查看跟踪点具体情况
            // cout << "cls of tracked pt: " << (int)obj_cls_id_sift[i].first << endl;
            // cout << "prev_sift_global_obj_id: " << prev_obj_id << endl;
            // cout << "cur local obj id: " << obj_id << endl;
            // cout << "Get one track SIFT for obj " << obj_id << endl;
            
            if(has_stereo)
            {
                // 对于跟踪点，如果其有右图像观测，则在其中放入该特征点在上一帧相机坐标系下的深度值，方便后续物体关联时使用
                // 所有物体点的跟踪点都必须有立体匹配，只有背景点允许在当前帧的跟踪点没有立体匹配
                double z_prev_sift = prev_sift_dep[i];
                // 对于当前帧才添加的新的背景跟踪点，该点在上一帧中即使检测到了立体匹配，但是因为当前帧还没有进行立体三角化，因此还没有有效深度值
                if(z_prev_sift > 0) 
                {
                    xyz_uv_velocity_statu_r(2) = z_prev_sift;
                }
                else
                {
                    // 只有纯背景的跟踪点才可能在上一帧中没有有效深度值
                    assert(cls_ == 0);
                }

                test_vec.emplace_back(1, xyz_uv_velocity_statu_r);
            }
            
            if(TrackObjFeaFrame.find(obj_id) == TrackObjFeaFrame.end())
            {
                map<int, vector<pair<int,Vector8d>>> temp_map;
                temp_map.insert(std::make_pair(feature_id,test_vec));

                TrackObjFeaFrame.insert(std::make_pair(obj_id,temp_map));
            }
            else
            {
                // TrackObjFeaFrame[obj_id][feature_id].emplace_back(0,xyz_uv_velocity_statu);

                TrackObjFeaFrame[obj_id].insert(make_pair(feature_id,test_vec));
            }

            // 记录各个物体的sift点中分别来自于track和new detected的个数
            // num_obj_sift[obj_id].first = num_obj_sift[obj_id].first + 1;
            if(num_obj_sift.find(obj_id) == num_obj_sift.end())
                num_obj_sift.insert(make_pair(obj_id,std::pair<int,int>(1,0)));
            else
            {
                num_obj_sift[obj_id].first += 1;
                // num_obj_sift[obj_id].first = num_obj_sift[obj_id].first + 1;
            }

            // 如果在特征点关联阶段 存在 上一帧的背景点关联到当前帧的物体点
            if(!has_lost_obj_prev && prev_obj_id == 0 && cls_ > 0)
            {
                has_lost_obj_prev = true;
            }
        }
        else
        {
            // cout << "Get one new SIFT for obj " << obj_id << endl;
            if(has_stereo) test_vec.emplace_back(1,xyz_uv_velocity_statu_r);

            if(NewObjFeaFrame.find(obj_id) == NewObjFeaFrame.end())
            {
                map<int, vector<pair<int,Vector8d>>> temp_fea;

                temp_fea.insert(make_pair(feature_id,test_vec));

                NewObjFeaFrame.insert(std::make_pair(obj_id,temp_fea));
                
                //NewObjFeaFrame[obj_id].insert(std::make_pair(feature_id,test_vec));
            }
            else
            {
                // NewObjFeaFrame[obj_id][feature_id].emplace_back(0,xyz_uv_velocity_statu);

                NewObjFeaFrame[obj_id].insert(make_pair(feature_id,test_vec));
            }

            if(num_obj_sift.find(obj_id) == num_obj_sift.end())
                num_obj_sift.insert(make_pair(obj_id, std::pair<int,int>(0,1)));
            else
            {
                num_obj_sift[obj_id].second += 1;
                //num_obj_sift[obj_id].second = num_obj_sift[obj_id].second + 1;
            }
        }
        test_vec.clear();
    }
    
    // ---------------------------------------------------------------------------------------------------
    // 根据当前帧局部obj_id分配FAST点
    cout << "Start assign FAST to local objs!" << endl;
    int valid_FAST = 0;
    int num_FAST_track = 0;
    int num_FAST_new = 0;
    for (size_t i = 0; i < ids_FAST.size(); ++i)
    {
        if (statusLeftRIght[i] == 0) continue;

        ++valid_FAST;

        bool has_stereo = true;
        int feature_id = ids_FAST[i];
        // 为了与sift的第一个元素(index都为0)区分开来，这里对所有FAST点的index都+1
        gl_id_index_map[feature_id] = i+1;

        cls_ = obj_cls_id_FAST[i].first;
        l_id = obj_cls_id_FAST[i].second;

        // 当前帧下背景新FAST点暂时不放入NewObjFeaFrame中,因为后面无需用到
        if(statusLeftRIght[i] == 1)
        {
            if(i < num_track_FAST)
                ++num_FAST_track;
            else
            {
                ++num_FAST_new;
                if(cls_ == 0)
                    continue;
            }
        }
        else 
        {
            // 当前帧没有立体匹配的背景跟踪点，也暂时不加入TrackObjFeaFrame中
            if(statusLeftRIght[i] == 2) 
            {
                has_stereo = false;
                ++num_FAST_track;
                
                // 如果纯背景跟踪点在当前帧没有立体匹配，则不将其加入到TrackObjFeaFrame中，因为它在objs-matching中没有用处
                if(cls_ == 0)
                    continue;
            }

            // 没有立体匹配的背景或物体新点
            if(statusLeftRIght[i] == 3)
            {
                ++num_FAST_new;
                has_stereo = false;

                // 背景的新点不用加入NewObjFeaFrame
                if(cls_ == 0)
                    continue;
            }  
        }
        
        double x, y ,z;
        x = cur_un_FAST[i].x;
        y = cur_un_FAST[i].y;
        //z = 1;
        z = cur_FAST_dep[i];
        
        double p_u, p_v;
        p_u = cur_FAST[i].x;
        p_v = cur_FAST[i].y;
        //int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = FAST_velocity[i].x;
        velocity_y = FAST_velocity[i].y;

        int obj_id = obj_cls_id_FAST[i].second;
        // 对于新特征点而言，这个值其实是该点在当前帧所属的yolo检测物体的局部id
        int prev_obj_id = prev_FAST_global_obj_id[i];
        
        // 当前帧的各个物体包含了cur_FAST中的哪些点
        //index_FAST_objs_cur[obj_id].push_back(i);

        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_statu;
        // 归一化平面点三维坐标（z为1），二维像素坐标，该特征点在当前帧相对于前一帧的归一化平面点坐标的二维速度
        xyz_uv_velocity_statu << x, y, z, p_u, p_v, velocity_x, velocity_y, prev_obj_id;

        test_vec.emplace_back(0,xyz_uv_velocity_statu);
        
        Eigen::Matrix<double, 8, 1> xyz_uv_velocity_statu_r;
        
        //xyz_uv_velocity_statu_r << cur_un_right_FAST[i].x, cur_un_right_FAST[i].y, 1, cur_right_FAST[i].x, cur_right_FAST[i].y, right_FAST_velocity[i].x, right_FAST_velocity[i].y, obj_id;
        if(has_stereo) 
        {
            xyz_uv_velocity_statu_r << cur_un_right_FAST[i].x, cur_un_right_FAST[i].y, 1.0, cur_right_FAST[i].x, cur_right_FAST[i].y, right_FAST_velocity[i].x, right_FAST_velocity[i].y, prev_obj_id;
        }

        // 如果是跟踪点
        if (i < num_track_FAST)
        {
            // cout << "cls of tracked pt: " << (int)obj_cls_id_FAST[i].first << endl;
            // cout << "prev_FAST_global_obj_id: " << prev_obj_id << endl;
            // cout << "cur local obj id: " << obj_id << endl;
            // cout << "Get one track FAST for obj " << obj_id << endl;
            
            if(has_stereo) 
            {
                // 注意，这里可以利用右匹配点的z来保存左图像点的上一帧匹配点的深度值！
                double z_prev_FAST = prev_FAST_dep[i];
                if(z_prev_FAST > 0) 
                    xyz_uv_velocity_statu_r(2) = z_prev_FAST;
                else
                    assert(cls_ == 0);

                test_vec.emplace_back(1,xyz_uv_velocity_statu_r);
            }

            if(TrackObjFeaFrame.find(obj_id) == TrackObjFeaFrame.end())
            {
                map<int, vector<pair<int,Vector8d>>> temp_map;
                temp_map.insert(make_pair(feature_id,test_vec));

                TrackObjFeaFrame.insert(std::make_pair(obj_id,temp_map));
            }
            else
            {
                // TrackObjFeaFrame[obj_id][feature_id].emplace_back(0,xyz_uv_velocity_statu);
                // TrackObjFeaFrame[obj_id][feature_id].emplace_back(1,xyz_uv_velocity_statu_r);

                TrackObjFeaFrame[obj_id].insert(make_pair(feature_id,test_vec));
            }

            if(num_obj_FAST.find(obj_id) == num_obj_FAST.end())
                num_obj_FAST.insert(make_pair(obj_id, std::pair<int,int>(1,0)));
            else
            {
                num_obj_FAST[obj_id].first += 1;
                // num_obj_FAST[obj_id].first = num_obj_FAST[obj_id].first + 1;
            }

            if(!has_lost_obj_prev && prev_obj_id == 0 && cls_ > 0)
            {
                has_lost_obj_prev = true;
            }
        }
        // 如果是新的FAST点
        else
        {
            if(has_stereo)
            {
                test_vec.emplace_back(1,xyz_uv_velocity_statu_r);
            }

            if(NewObjFeaFrame.find(obj_id) == NewObjFeaFrame.end())
            {
                map<int, vector<pair<int,Vector8d>>> temp_fea;
                temp_fea.insert(make_pair(feature_id,test_vec));
                NewObjFeaFrame.insert(std::make_pair(obj_id,temp_fea));

                //NewObjFeaFrame[obj_id].insert(std::make_pair(feature_id,test_vec));
            }
            else
            {
                // NewObjFeaFrame[obj_id][feature_id].emplace_back(0,xyz_uv_velocity_statu);
                // NewObjFeaFrame[obj_id][feature_id].emplace_back(1,xyz_uv_velocity_statu_r);

                NewObjFeaFrame[obj_id].insert(std::make_pair(feature_id,test_vec));
            }
            
            if(num_obj_FAST.find(obj_id) == num_obj_FAST.end())
                num_obj_FAST.insert(make_pair(obj_id, std::pair<int,int>(0,1)));
            else
            {
                num_obj_FAST[obj_id].second += 1;
                // num_obj_FAST[obj_id].second = num_obj_FAST[obj_id].second + 1;
            }
        }
        test_vec.clear();
    }

    // 如果当前帧背景中既没有跟踪点也没有新检测点。则当前帧只能靠IMU的积分值来推导当前帧的相机位姿
    // if(TrackObjFeaFrame.find(0) == TrackObjFeaFrame.end() && NewObjFeaFrame.find(0) == NewObjFeaFrame.end())
    // {
    //     // assert(false && "There is no bg fea (tracked or new) in the cur frame!");
    //     cout << "There is no bg fea (tracked or new) in the cur frame!!" << endl;
    // }

    // 如果当前帧背景中没有跟踪点。则当前帧只能靠IMU的积分值来推导当前帧的相机位姿
    // 为了后面的objs_matching函数顺利执行（其实是懒得大幅修改函数逻辑了....)，这里添加一个无效的背景跟踪点
    if(frame_count > 0 && TrackObjFeaFrame.find(0) == TrackObjFeaFrame.end())
    {
        cout << "There is no tracked bg fea in the cur frame!!" << endl;
        Vector8d xyz_uv_velocity_statu;
        xyz_uv_velocity_statu << 0, 0, 0, 1, 1, 0, 0, 0;
        test_vec.emplace_back(0,xyz_uv_velocity_statu);
        map<int, vector<pair<int,Vector8d>>> temp_fea;
        temp_fea.insert(make_pair(-1,test_vec));
        TrackObjFeaFrame.insert(std::make_pair(0,temp_fea));
    }
    
    // 这里直接把yolo后处理得到的所有valid objs当作跟踪阶段的有效物体，通过obj-matching查看是否为被跟踪物体，或者通过像素点采样的ave-depth查看是否为新物体
    valid_detect_obj = valid_obj_id;

    if(frame_count > 0 && !valid_obj_id.empty())
    {
        set<int> find_tracked_objs;

        for(auto &obj_id: TrackObjFeaFrame)
        {
            if(obj_id.first > 0)
            {
                find_tracked_objs.insert(obj_id.first);
                // cout << "local id of found tracked obj: " << obj_id.first << endl;
            }
        }

        // 跟背景跟踪点类似，如果当前帧某个检测物体上并没有跟踪点，则在这里为该物体添加无效的跟踪点，以便其后续其可以参与二分图匹配
        // 因为当前帧某个物体没有任何的跟踪特征点，不一定都是因为该物体太远或太近（而放弃其所有跟踪点），而是纯粹没有跟踪到或检测到特征点。
        // 前提是上一帧必须有全局物体可以与当前帧的检测物体进行关联验证，或者存在上一帧的背景点与当前帧某个物体的点有关联（这样在后面就需要为该物体进行关联校正，则必须保证所有的临时检测物体都参与物体关联，这是物体关联函数的逻辑）
        if(!glob_obj_id_prev.empty() || has_lost_obj_prev)
        {
            for(auto &obj_id: valid_obj_id)
            {
                if(find_tracked_objs.find(obj_id) != find_tracked_objs.end()) continue;

                // cout << "Need to find a pixel of obj!" << endl;

                bool find = false;

                // 既然添加一个无效的跟踪点只是为了使得该物体可以参与二分图，该点不会被使用，
                // 那么就没必要遍历该物体的bbox区域了，既然该物体被检测到了，那么就只需要尝试将它与上一帧的某个物体关联
                // 另外注意，检测到的物体一定会有bbox,但是其在seg_map和mask map上不一定会有mask区域，这取决于置信度的取值！！如果物体在这两个map上没有标记，那么就没法得到其上的特征点！
                // 因此最方便的方法就是随便赋予一个无效的跟踪点

                // float left   = bbox_mask[obj_id].left;
                // float top    = bbox_mask[obj_id].top;
                // float bbox_W = bbox_mask[obj_id].right - left;
                // float bbox_H = bbox_mask[obj_id].bottom - top;
                // int start_x = (int)(left + bbox_W/2);
                // int start_y = (int)(top + bbox_H/2);
                // int half_W  = (int)(bbox_W/2);
                // int half_H  = (int)(bbox_H/2);
                
                // for(int i = 0; i < half_W; ++i)
                // {
                //     for(int j = 0; j < half_H; ++j)
                //     {
                //         if(obj_id_map.at<uchar>(start_y-j,start_x-i) == (uchar)obj_id || obj_id_map.at<uchar>(start_y+j,start_x+i) == (uchar)obj_id)
                //         {
                //             // cout << "find a pixel of obj!" << endl;
                //             Vector8d xyz_uv_velocity_statu;
                //             xyz_uv_velocity_statu << 0, 0, 0, 1, 1, 0, 0, 0;
                        
                //             // if(TrackObjFeaFrame.find(obj_id) == TrackObjFeaFrame.end())

                //             test_vec.emplace_back(0,xyz_uv_velocity_statu);
                //             map<int, vector<pair<int,Vector8d>>> temp_fea;
                //             // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                //             temp_fea.insert(make_pair(-1,test_vec));
                //             TrackObjFeaFrame.insert(std::make_pair(obj_id,temp_fea));
                //             find = true;
                //             break;
                //         }
                //         else
                //             continue;
                //     }
                //     if(find)
                //         break;
                // }

                if(!find)
                {
                    Vector8d xyz_uv_velocity_statu;
                    xyz_uv_velocity_statu << 0, 0, 0, 1, 1, 0, 0, 0;
                    test_vec.emplace_back(0,xyz_uv_velocity_statu);
                    map<int, vector<pair<int,Vector8d>>> temp_fea;
                    // 该物体的唯一跟踪点的id为-1，标志着该物体实际上没有跟踪特征点
                    temp_fea.insert(make_pair(-1,test_vec));
                    TrackObjFeaFrame.insert(std::make_pair(obj_id,temp_fea));
                }
            } 
            // if(obj_id > 0) valid_detect_obj.push_back(obj_id);
        }
        // assert(TrackObjFeaFrame.size() == (valid_obj_id.size()+1) && "Something wrong with var TrackObjFeaFrame!");
        cout << "Num of detected objs with tracked fea: " << find_tracked_objs.size() << endl;
        cout << "Num of detected objs without tracked fea: " << (valid_obj_id.size()-find_tracked_objs.size()) << endl;
    }

    // cout << "Num of (tracked or detected) FAST: " << valid_FAST << endl;
    cout << "Num of total tracked FAST (with or without depth) : " << num_FAST_track << endl;
    cout << "Num of total new FAST (with or without depth) : " << num_FAST_new << endl;
    // cout << "Num of (tracked or detected) FAST: " << valid_sift << endl;
    cout << "Num of total tracked sift (with or without depth) : " << num_sift_track << endl;
    cout << "Num of total new sift (with or without depth) : " << num_sift_new << endl;
    printf("feature assign costs: %fms\n", t_o.toc());
}

// 往当前帧跟踪特征点集合中添加跟踪和新检测的FAST点，以便跟踪点数满足位姿估计的要求
// 此方法使用情况为当前帧背景中跟踪和检测的特征点点（包括FAST和SIFT）总数可能大于规定的最大数量。
// ！！此方法已弃用。
void FeatureTracker::assign_FAST_objs(FASTFrame &FAST_frame, FeaObjFrame &TrackObjFeaFrame, FeaObjFrame &NewObjFeaFrame, vector<pair<int,int>> &num_obj_FAST)
{
    // int num_objs = TrackObjFeaFrame.size();

    int cnt_track_bg = MAX_CNT_PTS_BG - num_track_sift_bg - num_new_sift_bg;
    
    int num_FAST = 0;
    int num_FAST_left = ids_FAST.size();
    
    // 用于记录保留和丢弃的当前帧中的FAST点
    vector<uchar> status(cur_FAST.size(), 0);

    for (int i = 0; i < cur_FAST.size(); ++i)
    {
        int obj_id = obj_cls_id_FAST[i].second;
        // 先添加那些有多帧跟踪的ORB点（至少被连续3帧观测到）。部分点可能是没有右图像观测的
        if (track_cnt_FAST[i] > 2)
        {
            status[i] = 1;
            TrackObjFeaFrame[obj_id][ids_FAST[i]].assign(FAST_frame[ids_FAST[i]].begin(), FAST_frame[ids_FAST[i]].end());
            // 记录各个物体的FAST点中分别来自于track和new detected的个数
            num_obj_FAST[obj_id].first = num_obj_FAST[obj_id].first + 1;
            // 通过临时obj_id来判断是否为背景点
            if (obj_id == 0) ++num_FAST;
            continue;
        }
        // 如果当前帧纯背景上的跟踪的sift点足够多（后续还可以再添加静态物体上的SIFT点！），就只添加那些有多帧跟踪的FAST点用于LBA，或者添加FAST点直到达到背景点的最大规定数量
        if (obj_id == 0){
            if (cnt_track_bg <= 0) {
                status[i] = 0;
                continue;
            }
            else if (num_FAST >= cnt_track_bg){
                status[i] = 0;
                continue;
            }
            ++num_FAST;
        }
        status[i] = 1;
        // 观察帧数为2帧的点说明是上一帧中的新FAST点，在当前帧中被跟踪到
        if (track_cnt_FAST[i] == 2)
        {
            // 动态物体上要保留所有的特征点匹配
            TrackObjFeaFrame[obj_id][ids_FAST[i]] = FAST_frame[ids_FAST[i]];
            num_obj_FAST[obj_id].first = num_obj_FAST[obj_id].first + 1;
        }
        // 只有一帧观测的则为新FAST点，需要修改其全局id
        else
        {
            assert(ids_FAST[i] > 1000000 && "Wrong temporal obj_id of new detected FAST points!");
            vector<std::pair<int, Eigen::Matrix<double, 8, 1>>> tmp_obs = FAST_frame[ids_FAST[i]];
            // FAST_frame.erase(ids_FAST[i]);
            ids_FAST[i] = n_id++;
            //FAST_frame[ids_FAST[i]].emplace_back(tmp_obs);
            NewObjFeaFrame[obj_id][ids_FAST[i]] = tmp_obs;
            num_obj_FAST[obj_id].second = num_obj_FAST[obj_id].second + 1;
            // TODO：此处还需要修改ids_FAST_right中属于新点的id，需要与左图像中的对应新FAST点的新全局id相同（需要使用find(i)来从left_id_of_right_FAST中找到相应左图像中的点）
            // ids_FAST_right[index] = ids_FAST[i];
        }
    }
    // 去除掉当前帧中不需要的FAST点
    reduceVector(cur_FAST, status);
    // ids中保存的是当前帧特征点（包括从上一帧跟踪到的和当前帧中检测和新添的）所对应的(滑窗内的）特征地图点的全局id。当前帧的ids初始值是上一帧特征跟踪处理结束时的ids（即上一帧左图像中观测到的所有特征地图点）！
    // track_cnt保存的是对应ids的特征地图点在滑窗内被观测了几帧（包括当前帧的观测）
    // 对于特征点，只要该点被（连续）观测到的帧数大于2，那么就可以形成BA了！（3帧共视该点时，需要优化的变量是两个相对位姿和该点深度，3个帧互相之间的观测匹配可以形成3个约束）
    reduceVector(ids_FAST, status);
    reduceVector(track_cnt_FAST, status);
    reduceVector(obj_cls_id_FAST, status);
    reduceVector(prev_FAST_global_obj_id, status);
}

// Ps和Rs中要保存前一帧和当前帧相机的全局位姿。上一帧中各个物体在到当前帧之间的速度（恒速运动假设）保存在当前对象中
// 对于特征点数不够最小阈值的物体，从上一帧各个物体保留的m个普通像素点中采样以达到最小阈值数，然后直接使用flow_map进行匹配（当然还有保证其在当前帧中的cls要一致，可以是背景漏检点）
// 这也就意味着最后需要对当前帧每个物体区域内已经跟踪和检测的特征点进行mask，然后再从剩下的区域中采样像素点！
// FinalTrackObjFea保存最终上一帧各个物体（非背景）的匹配点对以及新检测出的特征点
void FeatureTracker::objs_matching_assign(int frame_count, double dt, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map, 
                                            bool initial_succ, bool has_pred_motion_objs_cam, const vector<Vector3d> &Ps, const vector<Matrix3d> &Rs)
{
    TicToc t_match_objs;
    ++total_frame;
    // 如果是首帧，或者当前帧与上一帧之间只会有背景点之间的跟踪（上一帧没有任何物体，且上一帧和当前帧之间只有背景上的跟踪点（不存在上一帧漏检的物体被当前帧发现）；当前帧只会有新物体）
    // if(frame_count == 0 || (!has_lost_obj_prev && TrackObjFeaFrame.size() == 1 && TrackObjFeaFrame.begin()->first == 0 && glob_obj_id_prev.empty()))
    if(frame_count == 0 || (TrackObjFeaFrame.size() == 1 && TrackObjFeaFrame.begin()->first == 0 && glob_obj_id_prev.empty()))
    {
        new_objs_cur = valid_detect_obj;
        if(frame_count == 0) cout << "Num of new obj in first frame: " << new_objs_cur.size() << endl;
        
        int index;
        if(!TrackObjFeaFrame.empty())
        {
            for(auto &pt:TrackObjFeaFrame[0])
            {
                // 如果当前帧也没有背景上的跟踪点
                if(pt.first == -1) break;

                index = gl_id_index_map[pt.first];
                if(index > 0)
                {
                    if(statusLeftRIght[(index-1)] == 0) continue;

                    if(TrackBgFea.empty())
                        TrackBgFea.insert(pt);
                    else
                        TrackBgFea[pt.first] = pt.second;
                }
                else if(status_sift[(-1*index)]!=0)
                {
                    if(TrackBgFea.empty())
                        TrackBgFea.insert(pt);
                    else
                        TrackBgFea[pt.first] = pt.second;
                }   
            }
        }
        printf("Matching objects for first frame or frame without any tracked objs costs: %fms\n", t_match_objs.toc());
        return;
    }
    
    // 当前帧没有背景上的跟踪点，这种极端情况在下面还没处理！
    if(frame_count > 0 && TrackObjFeaFrame.find(0) == TrackObjFeaFrame.end())
        assert(false && "No tracked bg fea in cur! Should process this situation!");

    const int num_detect_obj = TrackObjFeaFrame.size();

    const int num_prev_obj = glob_obj_id_prev.size();

    if(!detect_lost_objs_cur.empty()) detect_lost_objs_cur.clear();
    
    //FeaObjFrame FinalNewObjFea;
    
    // 注意，所有的容器类型的当前类对象的成员变量都要在每帧跟踪后如果没有后续用处，则应该在当前帧进行BA优化时进行clear！不要留到下一帧再进行，以节省时间和空间。

    // resize之后，系统应该就会为FinalTrackObj中每个元素创建一个空的pair<int,vector<int>>对象，下面可以引用每个对象？
    
    if(num_prev_obj > 0)
    {
        std::pair<int,vector<int>> temp_pair;
        temp_pair.first = 0;
        // 上一帧每个物体最多和当前帧的两个检测物体（还可以包括后面发现的物体的部分漏检）相关联
        temp_pair.second.resize(5,0);
        FinalTrackObj.resize(num_prev_obj,temp_pair);
        for (int k = 0; k < num_prev_obj; ++k)
        {
            // 这样就相当于为每个内部的vector进行了reserve
            FinalTrackObj[k].second.clear();
        }
    }
    
    // 完成关联的全局物体个数
    int num_track_g_obj = 0;

    // 记录最终当前帧的所有临时物体（不包含背景，也不包含之后发现的漏检物体）的关联情况，每个物体最多只与上一帧的一个非漏检物体相关联。如果有与上一帧背景的漏检点关联，则会记录在ParLostObjPrevBg或TotalLostObjPrevBg中
    // FinalTrackCurObj.clear();

    // std::vector<int> ParLostObjPrevBg;
    
    // FeaObjFrame TotalLostObjPrevBg;
    
    // std::vector<int> FinalLostObjPrev;
    
    // TicToc t_match_objs;
    
    //上一帧中的所有全局物体中需要进入二分图匹配的物体
    vector<int> prev_objs_assign_again;
    // 上一帧的物体中在当前帧已完成匹配的（有些物体在当前帧可能会有两个匹配点集，例如该物体在当前帧中被部分地漏检或错误地被分为两个物体，最后会需要把两部分点集合并为当前帧一个物体上，虽然这样的概率很小）
    map<int,vector<int>> matched_prev_objs_id;
    // 二分图匹配任务中的关联矩阵
    vector<std::vector<float>> association_mat;
    // 记录关联矩阵中是否有某一行或某一列全为0，有的话则该物体在当前帧消失了，或者是当前帧新出现的
    // vector<int> has_assoc_prev_objs;
    // vector<int> has_assoc_cur_objs;

    // 为每个局部obj预分配内存，某个obj最多与上一帧的全部物体以及背景有特征点关联。
    int num_total_objs_prev = num_prev_obj + 1;

    int reserve_num = num_total_objs_prev * 2;

    // 这种线性连续存储空间的不同部分可否被用于多个线程的并行访问？如果某一线程的vector由于元素数量过多而需要重新分配内存，是否会影响其他线程的内存访问？
    // 在openMP中，在parallel for循环外部的变量默认是shared属性的，其在各线程内部进行写操作时会被线程保护，即不会有两个线程同时对该变量进行写操作。
    // 但是如果临时变换了vector的整体内存区域，则某个正在读的子线程可能就会出现内存访问错误了！！所以我们预先给vector中的每个子vector留下足够的空间，避免在并行期间重新分配内存！
    // 当前帧的背景点中可能因为yolo漏检而有全局物体的点
    // 这里先为每个内部vector分配固定数量的元素，使得其分配一定大小的内存；然后再clear每个内部vector，这只会清除其中的元素，而不会改变各个vector的capacity，相当于为每个vector实现了reserve
    vector<int> temp_vec(reserve_num,0);
    vector<vector<int>> final_assign_id_cur_objs(TrackObjFeaFrame.size(),temp_vec);
    for (int i = 0; i < TrackObjFeaFrame.size(); ++i)
    {
        // 多预分配一些内存，保证后续多线程中的操作不会导致重新分配内存，也就不会导致多线程中的引用失效
        final_assign_id_cur_objs[i].clear();
    }

    // 记录前后两帧的背景之间的关联特征点
    vector<int> pts_two_bg;
    // 记录当前帧需要进入二分图匹配的obj
    vector<int> cur_obj_assign_again;
    // 记录上一帧背景点中需要被校正的点（属于被漏检的物体的部分特征点），key为特征点在当前帧所属的临时物体id(后续当前帧物体通过密集像素与上一帧物体完成匹配后，会改成全局物体id)，value为各个特征点的全局id
    map<int, vector<int>> lose_fea_prev_bg;
    // 当前帧背景中漏检物体的全局obj id，通过与上一帧的物体之间的特征点匹配来进行发现与关联
    // lose_objs_cur_bg.clear();
    // 记录当前背景中帧漏检的物体的被跟踪的特征点的id，与上面的lose_objs_cur_bg配对
    // fea_cur_lose_objs.clear();
    
    // cur_stat_objs.clear();

    // 此处可以开启OpenMP进行并行处理！
    // default(shared)会默认将parallel for外部的变量都作为各个线程的共享变量，包括函数的所有参数！
    int num_threads = NUM_THREADS < num_detect_obj ? NUM_THREADS : num_detect_obj;
    
    omp_set_num_threads(num_threads);

    cout << "Start objs matching for No." << (total_frame-1) << " frame!" << endl;
// https://blog.csdn.net/dcrmg/article/details/53888952  
// omp parallel指令下面的{}中如果没有别的 omp ...指令（例如omp for），则不是对给定的显式循环用多线程来执行，而是每个线程都要执行{}内所有的内容（即使内部有循环，则每个线程都要完整执行该循环）
#pragma omp parallel default(shared)
    {
#pragma omp for
        for (int i = 0; i < TrackObjFeaFrame.size(); ++i)
        {
            // cout << "matching for obj " << i << endl;
            // 没有上一帧各物体的速度模型，意味着当前为系统第二帧，或者该物体上一帧为新物体，则只能用特征点以及密集像素点的2D和3D匹配点集的相似性以及比例来进行物体关联
            // 如果非系统第2帧，则除了2D匹配，还能用特征点以及密集像素点的3D匹配的重投影误差来进行物体关联。
            // ！！！注意，这里默认当前帧的所有非背景被检测物体，要么全都没有跟踪点（即不在TrackObjFeaFrame中），要么就全都在TrackObjFeaFrame中（没有跟踪点的就添加一个无效点）。否则下面的索引key可能是不存在的！
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &track_fea_obj_i = TrackObjFeaFrame[i];

            int num_fea_obj_i = track_fea_obj_i.size();

            // 当前帧没有与上一帧有跟踪点的物体，很可能是新物体，也可能是仅仅是没有特征点匹配（后续可以用像素点匹配）
            // 不存在此情况，因为即使某个被检测物体没有跟踪点，也会为其添加一个无效的跟踪点
            if (num_fea_obj_i == 0) 
            {
                // final_assign_id_cur_objs[i].push_back(-1);
                // 在使用openMP进行for循环的并行化时，可以使用continue来结束单个循环。
                // 而在每个循环中不用简单使用break来达到退出全部循环的目的，需要使用一个全局变量和continue来变相实现break的功能
                // https://www.cnblogs.com/immortal-worm/p/9770661.html
                continue;
            }
            //vector<int> prev_global_id_pts_objs;
            vector<pair<int, int>> count_unique_id;
            map<int,vector<int>> assign_prev_id;
            // 这样得到的iter是一个个pair对象，直接用.first和.second来取map的key和value。如果是先是定义iterator来遍历，则用->first来取值
            for (auto &pt : track_fea_obj_i)
            {
                // 对于加进来的像素点，在这里是否要将其删除？会影响到多线程吗(会）？
                // 当前帧除了某个物体没有跟踪点，极端情况下背景上可能也没有跟踪点！
                if(pt.first < 0) continue;
                // 统计该物体的跟踪点中，属于上一帧各个全局物体的点
                int prev_id = pt.second[0].second(7);
                if(assign_prev_id.find(prev_id) == assign_prev_id.end())
                {
                    vector<int> temp_vec;
                    temp_vec.push_back(pt.first);
                    assign_prev_id.insert(make_pair(prev_id,temp_vec));
                }
                else
                    assign_prev_id[prev_id].push_back(pt.first);
            }

            // 如果该临时物体没有任何有效的跟踪点，则跳过，等待后续的二分图匹配
            if(assign_prev_id.empty()) continue;

            for (auto &pt : assign_prev_id)
            {
                // cout << "matched prev gl obj id: " << pt.first << endl;
                // 记录当前帧第i个obj上的特征点所关联到的前一帧的各个全局obj以及对应的点数
                count_unique_id.emplace_back(pt.first, pt.second.size());
            }
            
            // 模板函数隐式实例化
            // calc_num_unique_value(prev_global_id_pts_objs[i], count_unique_id);
            // 不自定义比较函数时，sort默认是从小到大排序。这里自定义比较函数，元素根据pair的第二个元素从大到小排序
            sort(count_unique_id.begin(), count_unique_id.end(), [](const pair<int, int> &a, const pair<int, int> &b)
            {
                return a.second > b.second;
            });

            map<int, pair<vector<Vec2f>,vector<Vec2f>>> match_pts_of_prev_obj;
            map<int, pair<vector<float>,vector<float>>> dep_match_pts_of_prev_obj;

            // bool has_pred_motion = true;
            // 参数has_pred_motion_objs_cam只有在系统第2帧时才会为false(表示此时相机和所有物体都没有运动先验），其它时候都为true
            bool has_pred_motion = has_pred_motion_objs_cam;
            bool cur_obj_is_stat = false;
            // 所查询的关联物体在两帧间的全局运动变换
            Vector3d P_12 = Vector3d::Zero();
            Matrix3d R_12 = Matrix3d::Identity();

            // 该临时物体的主要跟踪点所属的全局物体
            int prev_obj_id = count_unique_id[0].first;

            float ave_dep_cur_obj = 0.0;
            
            map<int,vector<int>> empty_assign_prev_id;

            // 对于当前帧的背景点集
            if (i == 0)
            {
                // 如果当前帧背景的所有特征点都和上一帧的某个Obj相关联
                if (count_unique_id.size() == 1)
                {
                    // 如果都是背景点。有一种非常极端的情况，就是图像中几乎所有的有效区域都被物体占据，且上一帧与当前帧都恰巧只漏检了同一个物体...这其实很可能会导致上一帧的相机位姿估计出现问题！！暂时不考虑此种情形
                    if (prev_obj_id == 0)
                        final_assign_id_cur_objs[i].push_back(prev_obj_id);
                    // 如果是当前帧的背景点全都关联到上一帧的某一个物体，这可能吗？可能性也很低，需要图像中绝大部分区域都被物体所占据，且当前帧漏检某个物体!
                    else
                    {
                        bool succ_matched = false;
                        // 这种情况下应该要求特征点匹配数尽可能地大，毕竟是从上一帧的某个物体上的点来寻找匹配，点数应该会是比较多的？
                        // 注意，当前帧背景中不一定有sift跟踪点
                        // if (num_fea_obj_i >= 1.2*min_num_fea_track_obj || (num_obj_sift.find(i) != num_obj_sift.end() && num_obj_sift[i].first >= 1.5*min_num_sift_track_obj))
                        if (num_fea_obj_i >= min_num_fea_track_obj)
                            // 采用特征点关联来校验两个物体的关联关系时，会提高校验的标准
                            succ_matched = check_match_two_objs(true, true, track_fea_obj_i, prev_obj_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, assign_prev_id);
                        // 只需要有很少的特征关联，就验证当前帧是否漏检了某个物体（用该物体上一帧的采样像素点）。如果连这点特征点关联都没有，那就放弃当前帧漏检的物体了
                        // 另外，两个物体的关联像素点数（以及比例）还要足够多才可以！
                        else if (num_fea_obj_i >= 1.2*thres_num_fea_lose_obj)
                            succ_matched = check_match_two_objs(false, true, track_fea_obj_i, prev_obj_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                        
                        // 如果匹配成功
                        if(succ_matched)
                        {
                            // 如果该物体在当前帧的平均深度超过阈值
                            if((ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt))
                            {
                                final_assign_id_cur_objs[i].clear();
                                continue;
                            }
                            // 确定关联的物体在这两帧间是否仍为静态的
                            else if (cur_obj_is_stat) 
                            {
                                // 尝试加锁，因为可能有多个线程对cur_stat_objs这个量进行写操作。使用lock是openMP中进行共享变量写操作时的同步方式之一，另外还可以使用临界区(omp critical)或者原子操作（atomic)
                                omp_set_lock(&mylock);
                                // 如果map中已有该key，则后续插入的相同key时会被忽略
                                cur_stat_objs.insert(std::pair<int,float>(prev_obj_id,ave_dep_cur_obj));
                                omp_unset_lock(&mylock);
                            }
                        }
                    }
                }
                // 当前帧背景的点与上一帧多个obj相关联
                else
                {
                    // 当前背景主要与上一帧的背景相关联，但当前帧还可能存在漏检的物体（全部或部分）
                    if (prev_obj_id == 0)
                    {
                        //if (count_unique_id[0].second >= 0.5 * num_fea_obj_i && ount_unique_id[0].second/count_unique_id[1].second > thres_rel_ratio_match_1_2)
                        final_assign_id_cur_objs[i].push_back(prev_obj_id);
                        bool succ_matched = false; 
                        // 当前帧背景是否存在漏检物体。可能会有多个漏检？
                        for (int j = 1; j < count_unique_id.size(); ++j)
                        {
                            ave_dep_cur_obj = 0.0;
                            int prev_id = count_unique_id[j].first;
                            if (count_unique_id[j].second >= 1.2*min_num_fea_track_obj)
                            {
                                succ_matched = check_match_two_objs(true, true, track_fea_obj_i, prev_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, assign_prev_id);
                                // 如果使用特征点作为校验，则使用后要清空这两个变量，以便后续使用像素点校验的多个循环可以复用同一个统计
                                if (!match_pts_of_prev_obj.empty()) 
                                {
                                    match_pts_of_prev_obj.clear();
                                    dep_match_pts_of_prev_obj.clear();
                                }
                            }
                            else if (count_unique_id[j].second > 1.0*thres_num_fea_lose_obj)
                            {
                                // 当前背景中可能漏检某物体，使用该物体在上一帧的采样像素点来确认是否匹配
                                succ_matched = check_match_two_objs(false, true, track_fea_obj_i, prev_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                            }
                            else
                                break;
                            
                            if(succ_matched)
                            {
                                if((ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt))
                                {
                                    final_assign_id_cur_objs[i].pop_back();
                                    continue;
                                }
                                // 确定关联的物体在这两帧间是否仍为静态的
                                else if (cur_obj_is_stat) 
                                {
                                    cur_stat_objs.insert(std::pair<int,float>(prev_id,ave_dep_cur_obj));
                                }
                            }
                        }
                    }
                    // 上一帧中与当前帧背景匹配点最多的不是背景，而是物体。这种情况出现的概率也不大，需要当前帧漏检，且某个漏检物体的特征跟踪点数大于上一帧纯背景的点数
                    else
                    {
                        for (int j = 0; j < count_unique_id.size(); ++j)
                        {
                            ave_dep_cur_obj = 0.0;
                            int prev_id = count_unique_id[j].first;
                            bool succ_matched = false;
                            // 当前帧的背景点集与上一帧匹配最的背景点数不占最多数且数量也少于阈值，则舍弃这少量的背景点-背景点匹配，之后使用静态物体点来补充背景点！
                            if (prev_id == 0) 
                            {
                                // // 背景和背景的关联特征点不应该太少？否则说明上一帧的背景区域中特征点实在是太少了，这些点不适合作为可靠的特征点？
                                // if (count_unique_id[j].second >= min_num_fea_track_bg)
                                // {
                                //     final_assign_id_cur_objs.push_back(prev_id);
                                //     // 这里必须continue以便排除最下面cur_obj_is_stat的影响
                                //     continue;
                                // }   
                                // else 
                                //     continue;
                                
                                // 算了，对背景跟踪点不设置点数要求，只要有，即加入背景的跟踪点集
                                final_assign_id_cur_objs[i].push_back(prev_id);
                                continue;
                            }
                            // 当前背景中可能漏检某物体，使用与该物体在上一帧的特征匹配点来确认物体关联。注意，上一帧可能会有多个漏检物体
                            else if(count_unique_id[j].second >= 1.2 * min_num_fea_track_obj)
                            {
                                succ_matched = check_match_two_objs(true, true, track_fea_obj_i, prev_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, assign_prev_id);
                                // 如果使用特征点作为校验，则使用后要清空这两个变量，以便后续使用像素点校验的多个循环可以复用同一个统计
                                if  (!match_pts_of_prev_obj.empty()) 
                                {
                                    match_pts_of_prev_obj.clear();
                                    dep_match_pts_of_prev_obj.clear();
                                }
                            }
                            // 如果特征点不够（可能该物体在当前帧只是部分漏检），使用该物体在上一帧的采样像素点来确认是否匹配
                            else if (count_unique_id[j].second >= 1.0 * thres_num_fea_lose_obj)
                            {
                                succ_matched = check_match_two_objs(false, true, track_fea_obj_i, prev_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                            }
                            else 
                                break;
                            
                            if(succ_matched)
                            {
                                if(ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt)
                                {
                                    final_assign_id_cur_objs[i].pop_back();
                                    continue;
                                }
                                else if (cur_obj_is_stat) 
                                {
                                    // 确定关联的物体在这两帧间是否仍为静态的
                                    cur_stat_objs.insert(std::pair<int,float>(prev_id,ave_dep_cur_obj));
                                } 
                            }
                        }
                    }
                }

                // pts_two_bg.clear();
                // set<int> ignore_prev_obj;
                // 如果当前帧出现了seg漏检且通过与上一帧的特征点关联被发现，则将背景上的这些漏检的物体提取出来
                if (final_assign_id_cur_objs[i].size() > 0)
                {
                    for (int k = 0; k < final_assign_id_cur_objs[i].size(); ++k)
                    {
                        int prev_obj_id = final_assign_id_cur_objs[i][k];
                        if (prev_obj_id == 0) 
                        {
                            pts_two_bg = assign_prev_id[prev_obj_id];
                            continue;
                        }
                        // 记录与当前帧背景相关联的上一帧物体，即当前帧的漏检物体点
                        lose_objs_cur_bg.push_back(prev_obj_id);

                        cout << "find a lose objs in cur bg!" << endl;
                        
                        // 当前的漏检物体点当中可能有异常点已经被排除，则这里不把这些点加入
                        // 但可以保留其中的sift跟踪点作为背景的新点？
                        vector<int> temp_lose_pt;
                        for(auto &pt: assign_prev_id[prev_obj_id])
                        {
                            int index = gl_id_index_map[pt];
                            if(index > 0 && statusLeftRIght[index-1] == 0) continue;
                            // 算了，背景上的匹配异常点还是直接删除好了，它作为异常点，很可能是因为其深度值估计错误；如果仅仅是因为匹配错误，那么删除这少数点也不会有多大损失
                            if(index <= 0 && status_sift[-1*index] == 0) 
                            {
                                // index = -1 * index;
                                // float depth = track_fea_obj_i[pt].second[0].second(2);
                                // // 保留的点的深度值要比较好
                                // if (depth > mThDepthObj || depth < mMinDepthPt) 
                                // {
                                //     continue;
                                // }
                                // status_sift[index] = 1;
                                // obj_cls_id_sift[index] = std::pair<uchar,int>(0,0);
                                // ids_sift[index] = n_id++;
                                // track_cnt_sift[index] = 1;
                                // prev_sift_global_obj_id[index] = 0;

                                continue;
                            }
                            // 匹配内点才保留到当前帧漏检物体集中
                            temp_lose_pt.push_back(pt);    
                        }

                        // fea_cur_lose_objs这个量只有i == 0的线程会使用，因此不需要加锁
                        if(!temp_lose_pt.empty())
                            fea_cur_lose_objs.push_back(temp_lose_pt);
                        else
                        {
                            // ignore_prev_obj.insert(prev_obj_id);
                            lose_objs_cur_bg.pop_back();
                        }
                    }
                    // assert(fea_cur_lose_objs.size() == lose_objs_cur_bg.size() && "Wrong operation of variable fea_cur_lose_objs!");
                }
                
                for(const auto pt_bg: track_fea_obj_i)
                {
                    if(pt_bg.first < 0) continue;
                    int matched_prev_obj = pt_bg.second[0].second(7);
                    // 已处理过的有效的上一帧匹配物体
                    if(find(lose_objs_cur_bg.begin(),lose_objs_cur_bg.end(),matched_prev_obj) != lose_objs_cur_bg.end()) continue;

                    // 当前背景中没有与上一帧物体完成有效匹配的点（跟踪自上一帧背景的背景点暂时被全部保留，而跟踪自上一帧物体点的当前帧背景点中有部分最终是无效的匹配)
                    // if (matched_prev_obj != 0 && std::find(lose_objs_cur_bg.begin(),lose_objs_cur_bg.end(),matched_prev_obj) == lose_objs_cur_bg.end())
                    if (matched_prev_obj != 0)
                    {
                        int index = gl_id_index_map[pt_bg.first];
                        // 如果是FAST点，则全部去除
                        if(index > 0)
                        {
                            statusLeftRIght[index-1] = 0;
                        }
                        else
                        {
                            index = -1*index;
                            // 保留这部分sift点。其实还有可能这部分点确实是漏检物体的，但是在当前帧没能完成匹配，那么这些点下一帧还有可能作为完全漏检物体的点被发现？
                            // 其实也没必要，直接当作跟踪中断。该物体下一帧成为新物体即可。
                            // 如果每一帧中需要添加新点，则判断这些点的深度是否太大，如果太大则直接抛弃，因为它当作背景点或者漏检物体点都不太合适
                            
                            if (add_new_sift_in_next_frame)
                            {
                                status_sift[index] = 0;
                                continue;
                            }
                            else
                            {
                                float depth = pt_bg.second[0].second(2);
                                // 如果当前帧需要添加背景中的新sift点
                                // 且该点深度符合要求，则保留此无效跟踪点为背景上的新点
                                if(depth < mThDepthBg && depth > mMinDepthPt)
                                {
                                    obj_cls_id_sift[index] = std::pair<uchar,int>(0,0);
                                    ids_sift[index] = n_id++;
                                    track_cnt_sift[index] = 1;
                                    prev_sift_global_obj_id[index] = 0;
                                }
                                else
                                {
                                    status_sift[index] = 0;
                                    continue;
                                }
                            }
                        }
                    }
                }
                
                // ignore_prev_obj.clear();

                // 至此，当前帧背景点中的跟踪点中的无效匹配点（包括匹配对了物体但是被视为极端异常点）都已经被修改了信息（要么statsu被置为-1，要被删除；要么修改为新的点），之后就无需再考虑背景中的这部分无效跟踪点了！
            }
            // 如果当前物体不是背景
            else
            {
                // 如果该物体没有跟踪点，则进入后续二分图匹配
                if(count_unique_id.empty()) continue;
                // 当前某一物体的特征点主要对应于上一帧的背景，则可能是上一帧某个物体被漏检了，其部分特征点通过与上上帧匹配得到obj id的校正，但是剩下新检测的特征点就归属于上一帧背景
                // 如果这部分跟踪的点数足够多，则计算两个匹配点集是否相似，如果相似，则保留这些特值点；否则，舍弃当前物体的这些与上一帧的背景匹配特征点。
                // 之后使用当前帧物体与上一帧其他全局id物体的关联（通过特征点或者像素），来将上面的上一帧部分背景点加入到该全局物体中（如果当前帧某个物体与上一帧的匹配中出现了背景点子集，则说明这部分子集是某个物体的一部分）
                // 如果上一帧该物体是完全漏检，则至少要4个点才能估计物体的运动。这里要求最少5个关联点
                if (prev_obj_id == 0 && count_unique_id[0].second > thres_num_fea_lose_obj)
                //if (prev_obj_id == 0 && count_unique_id[0].second >= thres_num_fea_lose_obj)
                {
                    float ave_depth_obj_prev = 0.0;

                    // 注意，只有这种情况下，此函数的第二个参数和has_pred_motion才会都是false，即需要人为判断是否要使用运动模型来进行3D点变换预测。这里上一帧漏检物体的未校正点不知道其是否为静态，因此不进行3D变换预测。
                    has_pred_motion = false;
                    
                    // 最后一个参数代表要估计上一帧的点的平均深度值，这里之所以要专门计算上一帧的点深度，是因为上一帧为背景点，其允许的深度值可能会大于物体允许的深度值
                    bool matchded = check_match_two_objs(true, false, track_fea_obj_i, prev_obj_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                        final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_depth_obj_prev, assign_prev_id, false, true);
                                                        
                    // 重置
                    has_pred_motion = has_pred_motion_objs_cam;
                    // 如果使用特征点作为校验，则使用后要清空这两个变量，以便后续使用像素点校验的多个循环可以复用同一个统计
                    if (!match_pts_of_prev_obj.empty()) 
                    {
                        match_pts_of_prev_obj.clear();
                        dep_match_pts_of_prev_obj.clear();
                    }
                    // 如果上一帧漏检的物体太远或太近，则放弃该关联
                    if(matchded && (ave_depth_obj_prev > mThDepthObj || ave_depth_obj_prev < mMinDepthPt))
                    {
                        if(!final_assign_id_cur_objs[i].empty())
                            final_assign_id_cur_objs[i].pop_back();
                    }

                    int num_asso = count_unique_id.size();
                    if (num_asso > 1)
                    {
                        // 是否能找到上一帧的一个非背景的关联物体。
                        // 因此对于当前帧的每个非背景物体，其"最多"与上一帧一个"非背景物体"相关联（另外还可以与上一帧的某些背景点关联）。反过来上一帧的某个物体，其在当前帧中可能与最多2个物体相关联，需要融合
                        for (int j = 1; j < num_asso; ++j)
                        {
                            // 上一帧被漏检的物体上，其被上上帧匹配所矫正的部分特征点也可能被当前帧匹配到。这里统一直接使用该物体的像素点进行匹配。当然还有可能是不同物体的特征点错配了，但是应该也可以用像素点匹配来滤除。
                            if (count_unique_id[j].second >= thres_num_fea_lose_obj)
                            // if (count_unique_id[j].second >= min_num_fea_track_obj/2)
                            {
                                ave_dep_cur_obj = 0.0;
                                int prev_ID = count_unique_id[j].first;
                                bool matched = check_match_two_objs(false, true, track_fea_obj_i, prev_ID, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                                
                                if(matched)
                                {
                                    if(ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt)
                                    {
                                        matched == false;
                                        if(!final_assign_id_cur_objs[i].empty())
                                            final_assign_id_cur_objs[i].pop_back();
                                    }    
                                    else if (cur_obj_is_stat) 
                                    {
                                        // 确定关联的物体在这两帧间是否仍为静态的
                                        cur_stat_objs.insert(std::pair<int,float>(prev_ID,ave_dep_cur_obj));
                                    }
                                }

                                // 一旦找到关联物体，则结束关联
                                if(matched) break;
                            }
                            else
                                break;    
                        }
                    }
                }
                else if(prev_obj_id == 0)
                {
                    // 匹配点数最多的仍是上一帧的背景，但时不超过4个特征点，则放弃该上一帧漏检的部分特征点。由于最大数量的匹配点也不超过4个，后面其他的物体关联也可以不用进行了。
                    continue;
                }
                // 主要是非背景物体点之间的关联（这俩物体可以是不同类别的，因为存在某一帧错分类的情况，我们只用obj id来做关联）。当前帧的物体与上一帧最多只有一个非背景关联物体
                else
                {
                    bool matched = false;
                    bool need_more_fea = false;
                    if (count_unique_id[0].second >= min_num_fea_track_obj)
                    {
                        // cout << "prev_obj_id: " << prev_obj_id << endl;
                        matched = check_match_two_objs(true, true, track_fea_obj_i, prev_obj_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                        final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, assign_prev_id);
                        // 如果使用特征点作为校验，则使用后要清空这两个变量，以便后续使用像素点校验的多个循环可以复用同一个统计
                        if(!match_pts_of_prev_obj.empty()) 
                        {
                            match_pts_of_prev_obj.clear();
                            dep_match_pts_of_prev_obj.clear();
                        }
                    }
                    // 特征匹配点数不够再用像素点来验证
                    else
                    {
                        // cout << "prev_obj_id: " << prev_obj_id << endl;
                        // 后续查看是否有上一帧该物体漏检的部分的特征点
                        need_more_fea = true;
                        matched = check_match_two_objs(false, true, track_fea_obj_i, prev_obj_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                            final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                    }
                    if(matched)
                    {
                        if(ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt)
                        {
                            matched == false;
                            if(!final_assign_id_cur_objs[i].empty())
                                final_assign_id_cur_objs[i].pop_back();
                        }
                    }

                    // 确定关联的物体在这两帧间是否仍为静态的
                    if (cur_obj_is_stat && matched) 
                    {
                        cur_stat_objs.insert(std::pair<int,float>(prev_obj_id,ave_dep_cur_obj));
                    } 

                    if ((need_more_fea || !matched) && count_unique_id.size() > 1)
                    {
                        for (int j = 1; j < count_unique_id.size(); ++j)
                        {
                            ave_dep_cur_obj = 0.0;
                            int prev_id = count_unique_id[j].first;
                            // 仍然有可能出现上一帧漏检的物体的未校正特征点在当前帧被检测到，如果这部分点足够多，则尝试加入到当前帧该物体中
                            if (prev_id == 0)
                            {
                                // 上一帧漏检时，则要求至少要5个点匹配点
                                if (count_unique_id[j].second > min_num_fea_track_obj/2)
                                {
                                    has_pred_motion = false;
                                    float ave_dep_prev_pts = 0.0;
                                    bool succ_matched = check_match_two_objs(true, false, track_fea_obj_i, prev_id, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                        final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_prev_pts, assign_prev_id, false, true);
                                    // 重置
                                    has_pred_motion = has_pred_motion_objs_cam;
                                    // 如果前面使用特征点作为校验，则使用后要清空这两个变量，以便后续使用像素点校验的多个循环可以复用同一个统计
                                    if(!match_pts_of_prev_obj.empty()) 
                                    {
                                        match_pts_of_prev_obj.clear();
                                        dep_match_pts_of_prev_obj.clear();
                                    }
                                    
                                    if(succ_matched && (ave_dep_prev_pts > mThDepthObj || ave_dep_prev_pts < mMinDepthPt))
                                    {
                                        succ_matched = false;
                                        if(!final_assign_id_cur_objs[i].empty())
                                            final_assign_id_cur_objs[i].pop_back();
                                    }
                                }
                                // 已经完成了上一帧漏检点的检验
                                need_more_fea = false;
                                // 找到上一帧的一个非背景关联物体后就结束关联
                                if (matched) break;
                            } 
                            // 如果当前物体在上一帧中还没有找到非背景的obj匹配。只需要找到一个匹配obj
                            else if (!matched)
                            {
                                if (count_unique_id[j].second >= thres_num_fea_lose_obj)
                                {
                                    matched = check_match_two_objs(false, true, track_fea_obj_i, count_unique_id[j].first, i, match_pts_of_prev_obj, dep_match_pts_of_prev_obj,
                                                    final_assign_id_cur_objs, seg_map, flow_map, depth_map, Ps, Rs, has_pred_motion, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_prev_id);
                                    if (matched)
                                    {
                                        if(ave_dep_cur_obj > mThDepthObj || ave_dep_cur_obj < mMinDepthPt)
                                        {
                                            matched = false;
                                            if(!final_assign_id_cur_objs[i].empty())
                                                final_assign_id_cur_objs[i].pop_back();
                                        }
                                        else if (cur_obj_is_stat)
                                        {
                                            // 确定关联的物体在这两帧间是否仍为静态的
                                            cur_stat_objs.insert(std::pair<int,float>(prev_id,ave_dep_cur_obj));
                                        }

                                        if (!need_more_fea) break;
                                    }
                                }
                                // else if (!need_more_fea) break;
                                // 之后的上一帧物体的跟踪点数太少，直接忽略
                                else 
                                    break;
                            }
                        }
                    }
                }
                
                // 如果当前帧某非背景obj已关联，则删除不属于上一帧所关联物体的匹配特征点(暂不执行，因为这里不是最终的物体关联结果，后续可能要重新参与二分图匹配）；
                // 如果有与上一帧背景的关联点：如果确认上一帧漏检，则记录上一帧漏检物体部分需要修改全局obj id的特征点；否则，删除所有与上一帧背景的关联点。
                // 注意，当前帧的非背景物体与上一帧最多只有一个非背景关联物体（同时还可能与上一帧的某些背景特征点关联，是上一帧该物体的漏检部分）
                // 由于可能当前帧物体没有实现任何关联，但其仍有来自上一帧背景点的跟踪点，则下面还是要进行，这里不应限制其是否有关联物体
                // if(final_assign_id_cur_objs[i].size() > 0 && i > 0)
                if(i > 0)
                {
                    // 遍历该物体在当前帧中的所有点。要注意，如果容器为空，那么其begin()和end()迭代器是等同的，即都为end()
                    for (auto &iter: assign_prev_id)
                    {
                        int id = iter.first;
                        bool is_matched_obj = true;
                        // 这里暂时不删除不属于两个关联物体的特征点匹配，因为此处还不是最终的关联关系（例如可能当前帧多个物体关联到了上一帧同个物体，这在后续需要重新关联）
                        if (final_assign_id_cur_objs[i].empty())
                        {
                            is_matched_obj = false;
                        }
                        else
                        {
                            if(std::find(final_assign_id_cur_objs[i].begin(), final_assign_id_cur_objs[i].end(), id) == final_assign_id_cur_objs[i].end())
                                is_matched_obj = false;
                        }
                        if(!is_matched_obj)
                        {
                            // const vector<int> &del_pts = iter.second;
                            // for (int j = 0; j < del_pts.size(); ++j)
                            // {
                            //     track_fea_obj_i.erase(del_pts[j]);
                            // }

                            // 对于没有与当前帧物体完成关联的上一帧物体，这里只先排除所有上一帧的背景关联点。物体关联点可能在之后还要参与二分图匹配
                            if(id == 0)
                            {
                                for (int k = 0; k < assign_prev_id[id].size(); ++k)
                                {
                                    int pt_id = assign_prev_id[id][k];
                                    int index = gl_id_index_map[pt_id];
                                    if(index > 0)
                                        statusLeftRIght[index-1] = 0;
                                    else
                                    {
                                        if(add_new_sift_in_next_frame)
                                        {
                                            status_sift[-1*index] = 0;
                                        }
                                        else
                                        {
                                            // todo: 可以尝试保留物体上的该sift点为新点，只要其深度估计较为可靠
                                            // continue;

                                            status_sift[-1*index] = 0;
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        // 设置临界区，如果上一帧漏检了该物体，且找到了需要校正的特征点，则记录它们，每个线程只能单独修改lose_fea_prev_bg变量！
                        // 临界区内代码在任何时刻最多只能有一个线程在执行，即此代码区在多线程中是“互斥”执行的，相当于加了锁
                        // 注意，指令critical可以定义一个任意大小的代码块作为临界区保护，而atomic原子操作应用在单条赋值语句中
                        else if(id == 0)
                        {
#pragma omp critical
                            {
                                // 记录上一帧中需要修改obj_id的背景点，key为当前帧中的物体临时id，后续根据该物体的全局匹配id来将滑动窗口中保存的这些背景点换到对应的物体点集中（如果有深度值的话）
                                for (int k = 0; k < assign_prev_id[id].size(); ++k)
                                {
                                    // 上一帧这些漏检点在匹配过程中可能有某些点被当作异常点，而提前删除了？是的，如果当前帧物体匹配到了上一帧的背景漏检点，则需要把其中的异常匹配点（2D和3D点集的分布异常点）的status置为0
                                    int pt_id = assign_prev_id[id][k];
                                    int index = gl_id_index_map[pt_id];
                                    if (index > 0 && statusLeftRIght[index-1]==0) continue;
                                    if (index <= 0 && status_sift[-1*index]==0) continue;

                                    // 记录当前帧某个临时物体在上一帧中的漏检点
                                    if(lose_fea_prev_bg.find(i) == lose_fea_prev_bg.end())
                                    {
                                        vector<int> temp_vec;
                                        temp_vec.push_back(pt_id);
                                        lose_fea_prev_bg.insert(make_pair(i,temp_vec));
                                    }
                                    else
                                        lose_fea_prev_bg[i].push_back(pt_id);
                                }
                                // 对于在上一帧中漏检的物体，其校正后的特征点在当前帧中被跟踪的点数可能少于4个，因此在上面没有使用上一帧该物体的采样像素点进行匹配验证。因此后续需要在二分图匹配用采样像素点来验证！上面也直接删除它们之间少量可能的特征点匹配。
                                // 要注意，对于上一帧漏检的物体，如何在上一帧保留它的密集像素点采样？需要上一帧和上上帧关于该物体有特征点关联，运动估计完成之后，使用运动模型来筛选上上帧与上一帧之间的密集像素点匹配（运动变换之后点的距离误差要比较小）！！！
                            }
                        }
                    }
                }
            }

            // 及时释放每个子线程的内存
            // vector或者string的内存可以用swap和空容器来彻底释放
            // https://cloud.tencent.com/developer/article/1383922
            if (!count_unique_id.empty())
            {
                // clear()仅删除元素，但内存还在
                count_unique_id.clear();
                vector<pair<int, int>>().swap(count_unique_id);

                // map的内存占用无法用swap来彻底释放
                // https://blog.csdn.net/sebeefe/article/details/123614590
                assign_prev_id.clear();
                map<int,vector<int>>().swap(assign_prev_id);
            }
            
            if (!match_pts_of_prev_obj.empty()) 
            {
                match_pts_of_prev_obj.clear();
                dep_match_pts_of_prev_obj.clear();
                map<int, pair<vector<Vec2f>,vector<Vec2f>>>().swap(match_pts_of_prev_obj);
                map<int, pair<vector<float>,vector<float>>>().swap(dep_match_pts_of_prev_obj);
            }

            // cout << "finished matching for obj " << i << endl;
        }
        
        // 等待所有子线程完成（所有循环）。omp barrier用于同步所有的线程，相当于设置了一个线程的集合点，所有线程都到达之后才能继续往下执行。这里是为了等待所有线程完成后，让主线程统计数据
        // 其实在每个调用了多线程的代码块（如omp for{}，omp parallel内的纯代码块{}）之后，都紧接着隐式地设置了barrier指令！如果想去除这个指令，则要在opm for或omp parallel等指令后接上 nowait 指令，如opm for nowait{}
#pragma omp barrier
        // 统计已经关联的非背景物体，以及上一帧和当前帧中所有未关联的物体
        // 只使用主线程进行统计，否则在omp parallel内的所有代码（块）都会被多个线程完整地执行！
#pragma omp master
        {
            // TODO：清除由于子线程而产生的堆上的内存占用，回收内存碎片。但这会把主线程中的一些暂时为empty的变量也去除内存吗？应该不会，因为还在作用域内
            // malloc_trim(0);
            // 记录上一帧需要修改obj_id的背景点，它们是上一帧被漏检的物体的部分特征点
            for (int k = 1; k < final_assign_id_cur_objs.size(); ++k)
            {
                int num_match_objs_prev = final_assign_id_cur_objs[k].size();
                // 记录需要用二分图匹配的当前帧物体
                if (num_match_objs_prev == 0)
                {
                    cur_obj_assign_again.push_back(k);
                    // 如果该物体在当前帧就没有任何跟踪点，只有一个添加进来的像素点，则这里删除该像素点
                    if(TrackObjFeaFrame[k].size() == 1 && TrackObjFeaFrame[k].begin()->first == -1)
                    {
                        // 清空属于物体k的跟踪点集map
                        TrackObjFeaFrame[k].clear();
                        // 是否要从整个物体点集map中删除物体k？
                        // TrackObjFeaFrame.erase(k);
                    }
                }
                // 只有一个匹配且为上一帧背景点的情况，在上面openML已经完成了，即将当前帧该物体加入到cur_obj_assign_again
                //else if(num_match_objs_prev > 1)
                // 如果当前帧的某物体只与上一帧的背景漏检物体关联，则要加入二分图匹配。最后再将关联到的上一帧的物体与lose_fea_prev_bg中的特征点匹配相融合
                else if(num_match_objs_prev == 1 && final_assign_id_cur_objs[k][0] == 0)
                {
                    cur_obj_assign_again.push_back(k);
                }
                else
                {
                    // 注意，对于当前帧的非背景obj，在上面的openML过程中，它的上一帧的匹配obj最多只有两个（如果是两个，则必然一个是物体，另一个是背景；如果只有一个，则只会是物体）
                    assert(num_match_objs_prev <= 2 && "Weired! Something wrong!");
                    if(num_match_objs_prev > 1)
                        assert(final_assign_id_cur_objs[k][0] == 0 || final_assign_id_cur_objs[k][1] == 0);
                    
                    vector<int> matched_cur_objs;
                    int prev_obj_id;
                    // 记录上一帧的物体中已经完成匹配的物体
                    for (int j = 0; j < num_match_objs_prev; ++j)
                    {
                        int prev_id = final_assign_id_cur_objs[k][j];
                        // 当前帧物体在上一帧中可能漏检，其部分特征点在上一帧的背景中。上一帧背景和漏检部分暂时不加入matched_prev_objs_id，而是在lose_fea_prev_bg中
                        if (prev_id == 0) continue;
                        prev_obj_id = prev_id;
                        matched_cur_objs.push_back(k);
                    }

                    if(matched_prev_objs_id.find(prev_obj_id) != matched_prev_objs_id.end())
                        matched_prev_objs_id[prev_obj_id].push_back(k);
                    else if(!matched_cur_objs.empty())
                        matched_prev_objs_id.insert(make_pair(prev_obj_id, matched_cur_objs));
                }
            }

            // 这些为了size==2的情况准备
            // 空的map平均大概占据40个字节，具体大小与编译器有关
            map<int, pair<vector<Vec2f>,vector<Vec2f>>> match_pts_of_objs;
            map<int, pair<vector<float>,vector<float>>> dep_match_pts_of_objs;
            vector<vector<int>> assign_id_cur_obj;
            Vector3d P_12 = Vector3d::Zero();
            Matrix3d R_12 = Matrix3d::Identity();
            bool cur_obj_is_stat = false;
            map<int,vector<int>> empty_assign_pt;

            if(!matched_prev_objs_id.empty())
            {
                // 如果出现上一帧某个物体与当前帧多个物体相关联的情况，则考虑将它们进行融合（如果只有两个非漏检物体的话，直接融合；如果有更多非漏检物体，则把该前一帧物体和这关联的所有当前帧物体都加入到二分图匹配）
                for(auto &iter: matched_prev_objs_id)
                {
                    int id_in_lost = -1;
                    int prev_id = iter.first;
                    bool check = (cur_stat_objs.find(prev_id) != cur_stat_objs.end());
                    vector<int>::iterator iter_lose = std::find(lose_objs_cur_bg.begin(),lose_objs_cur_bg.end(),prev_id);
                    if (iter_lose != lose_objs_cur_bg.end())
                    {
                        id_in_lost = std::distance(lose_objs_cur_bg.begin(), iter_lose);
                    }

                    bool first = true;

                    // 如果上一帧的该全局物体在当前帧是一对一配对（此外，其在当前帧背景中还允许有漏检的部分）
                    if (iter.second.size() == 1)
                    {
                        int cur_id = iter.second[0];
                        
                        // 遍历所匹配的当前帧物体上的跟踪点，记录其中与此全局物体的关联点
                        for(auto &pt: TrackObjFeaFrame[cur_id])
                        {
                            if (pt.second[0].second(7,0) == prev_id)
                            {
                                // 如果该物体在上面关联过程中被确认为静态物体，则其跟踪点中可能会有异常匹配点
                                if(check)
                                {
                                    int index = gl_id_index_map[pt.first];
                                    if (index > 0 && statusLeftRIght[index-1]==0) continue;
                                    if (index <= 0 && status_sift[-1*index]==0) continue;
                                }
                                if(first)
                                {
                                    map<int, vector<pair<int,Vector8d>>> temp_fea;
                                    // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                    temp_fea.insert(pt);
                                    FinalTrackObjFea.insert(make_pair(prev_id,temp_fea));
                                    first = false;
                                }
                                else
                                    // FinalTrackObjFea[prev_id][pt.first] = pt.second;
                                    FinalTrackObjFea[prev_id].insert(pt);
                            }
                        }
                        
                        FinalTrackObj[num_track_g_obj].first = prev_id;
                        FinalTrackObj[num_track_g_obj].second.push_back(cur_id);

                        if(id_in_lost != -1)
                        {
                            for (auto &pt:fea_cur_lose_objs[id_in_lost])
                            {
                                // 对于上一帧的静态物体，在上面的物体匹配过程中，对于上一帧 或 当前帧 漏检部分的匹配，如果匹配成功，则其中的异常匹配点已经被处理了！不会加入到fea_cur_lose_objs中
                                // if(check)
                                // {
                                //     int index = gl_id_index_map[pt.first];
                                //     if (index > 0 && statusLeftRIght[index-1]==0) continue;
                                //     if (index <= 0 && status_sift[-1*index]==0) continue;
                                // }

                                if(first)
                                {
                                    map<int, vector<pair<int,Vector8d>>> temp_fea;
                                    // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                    temp_fea.insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                                    FinalTrackObjFea.insert(make_pair(prev_id,temp_fea));
                                    first = false;
                                }
                                else
                                    // FinalTrackObjFea[prev_id][pt] = TrackObjFeaFrame[0][pt];
                                    FinalTrackObjFea[prev_id].insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                            }
                            
                            // lose_objs_cur_bg[id_in_lost] = lose_objs_cur_bg[id_in_lost] * (-1);
                            lose_objs_cur_bg[id_in_lost] = -1 * prev_id;
                            // 如果当前帧物体部分漏检，要把0放进去吗？
                            // 可以不用，只是记得后续在全局地图中搜索此id的点时注意要到背景的点集中搜索。这里为了后续方便，还是放进去了
                            FinalTrackObj[num_track_g_obj].second.push_back(0);
                        }

                        ++num_track_g_obj;

                        // 当前帧的每个物体最终所匹配的全局物体最多只有一个，如果其在上一帧有漏检，则漏检部分要么成为单独的新全局物体，要么与其他全局物体相融合
                        FinalTrackCurObj[cur_id] = prev_id;

                        // 将上一帧的背景上的某些漏检物体的特征点 与 其已校正过的物体特征点 统一到一起
                        if (lose_fea_prev_bg.find(cur_id) != lose_fea_prev_bg.end())
                        {
                            for(auto &pt_id: lose_fea_prev_bg[cur_id])
                            {
                                FinalTrackObjFea[prev_id][pt_id] = TrackObjFeaFrame[cur_id][pt_id];
                            }
                            // 此变量记录上一帧背景中“部分”漏检物体的id，用于后续在全局地图中将上一帧背景的这部分新特征点改变为物体点
                            // 在修改时是根据点的全局id从背景点集中寻找
                            ParLostObjPrevBg.insert(prev_id);
                            lose_fea_prev_bg[cur_id].clear();
                            lose_fea_prev_bg.erase(cur_id);
                        }
                        // 如果不是静态物体，则是潜在的动态物体。包括如果该物体上一帧是静态物体但是上面的关联过程中没通过静态检测，则暂时当作动态物体来看待
                        // 注意，cur_dyn_objs中记录的是各个被跟踪的动态的全局物体在FinalTrackObj中序号
                        if(cur_stat_objs.find(prev_id) == cur_stat_objs.end())
                            cur_dyn_objs.insert(num_track_g_obj-1);
                    }
                    // 当前某个物体被分割成两部分，是否应该要使用某些先验来验证它们是否为同一个物体？如两部分物体的空间距离要与它们的类别相关，火车或电车允许距离较大（但铰接结构为多部分刚体运动），本工作暂不考虑）？
                    // 这里还是使用前后两帧的该物体点集的方差（2d与3d）以及运动变换后的距离来作为判断，如果仍然满足要求，则融合
                    else if (iter.second.size() == 2)
                    {
                        // 融合。包括把该全局物体所关联的当前帧背景漏检物体点也加入到临时集合中进行验证
                        // 上一帧背景的漏检点则是直接加入与当前帧相同局部id的物体的关联集合中（因为上一帧的漏检点都是通过特征点匹配来验证联系的，可信度较高）
                        // 将所有物体的点集合在一起，然后进行前后两帧的物体关联的验证！如果通过，则直接加入最终的关联集合中！！如果不通过，当前帧和上一帧的相关检测物体加入二分图匹配
                        
                        for(int i = 0; i < 2; i++)
                        {
                            int cur_id = iter.second[i];
                            for(auto &pt: TrackObjFeaFrame[cur_id])
                            {
                                if (pt.second[0].second(7,0) == prev_id)
                                {
                                    if (check)
                                    {
                                        int index = gl_id_index_map[pt.first];
                                        if (index > 0 && statusLeftRIght[index-1]==0) continue;
                                        if (index <= 0 && status_sift[-1*index]==0) continue;
                                    }
                                    if(first)
                                    {
                                        map<int, vector<pair<int,Vector8d>>> temp_fea;
                                        // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                        temp_fea.insert(pt);
                                        FinalTrackObjFea.insert(make_pair(prev_id,temp_fea));
                                        first = false;
                                    }
                                    else
                                        // FinalTrackObjFea[prev_id][pt.first] = pt.second;
                                        FinalTrackObjFea[prev_id].insert(pt);
                                }
                            }
                        }

                        if(id_in_lost != -1)
                        {
                            for (auto &pt:fea_cur_lose_objs[id_in_lost])
                            {
                                // 对于上一帧 或 当前帧 漏检部分的匹配，如果匹配成功，则其中的异常匹配点已经被处理了！不会加入到fea_cur_lose_objs中
                                // if (cur_stat_objs.find(prev_id) != cur_stat_objs.end())
                                // {
                                //     int index = gl_id_index_map[pt.first];
                                //     if (index > 0 && statusLeftRIght[index-1]==0) continue;
                                //     if (index <= 0 && status_sift[-1*index]==0) continue;
                                // }

                                if(first)
                                {
                                    map<int, vector<pair<int,Vector8d>>> temp_fea;
                                    // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                    temp_fea.insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                                    FinalTrackObjFea.insert(make_pair(prev_id,temp_fea));
                                    first = false;
                                }
                                else
                                    // FinalTrackObjFea[prev_id][pt] = TrackObjFeaFrame[0][pt];
                                    FinalTrackObjFea[prev_id].insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                            }
                        }

                        map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &temp_fea = FinalTrackObjFea[prev_id];
                        // 静态物体根据平均距离进行排序！
                        float ave_dep_cur_obj = 0.0;
                        // 这里如果匹配成功，则要求把其中的异常匹配点直接从跟踪点集中删除
                        bool matched = check_match_two_objs(true, true, temp_fea, prev_id, -1, match_pts_of_objs, dep_match_pts_of_objs, assign_id_cur_obj, seg_map,
                                                            flow_map, depth_map, Ps, Rs, has_pred_motion_objs_cam, P_12, R_12, cur_obj_is_stat, ave_dep_cur_obj, empty_assign_pt, true);
                        
                        // 如果匹配成功，则保留FinalTrackObjFea中已经加入的匹配点
                        if(matched)
                        {
                            // 这里应该不需要再验证当前帧该物体的深度值了
                            // 如果该全局物体仍然是静态物体，则加入(map无需担心key重复，会覆盖)
                            if(cur_obj_is_stat && ave_dep_cur_obj > 0) 
                                cur_stat_objs.insert(std::pair<int,float>(prev_id,ave_dep_cur_obj));
                            else
                                cur_dyn_objs.insert(num_track_g_obj);

                            FinalTrackObj[num_track_g_obj].first = prev_id;
                            FinalTrackObj[num_track_g_obj].second = iter.second;

                            if(id_in_lost != -1) 
                            {
                                // lose_objs_cur_bg[id_in_lost] = lose_objs_cur_bg[id_in_lost] * (-1);
                                lose_objs_cur_bg[id_in_lost] = -1 * prev_id;
                                // 如果当前帧物体部分漏检，要把0放进去吗？放！
                                FinalTrackObj[num_track_g_obj].second.push_back(0);
                            }
                            // 注意，这个只是记录上一帧中的已知的全局物体，对于上一帧完全漏检的物体不计算在内（其单独保存在TotalLostObjPrevBg中）
                            num_track_g_obj++;

                            for(int i = 0; i < 2; i++)
                            {
                                int cur_id = iter.second[i];
                                FinalTrackCurObj[cur_id] = prev_id;
                                
                                // 上一帧的该物体的漏检部分是否有一些特征点被关联到
                                if (lose_fea_prev_bg.find(cur_id) != lose_fea_prev_bg.end())
                                {
                                    for(auto &pt_id: lose_fea_prev_bg[cur_id])
                                    {
                                        FinalTrackObjFea[prev_id].insert(make_pair(pt_id,TrackObjFeaFrame[cur_id][pt_id]));
                                    }
                                    // 此变量记录上一帧背景中“部分”漏检物体的id，用于后续在全局地图中将上一帧背景的这部分新特征点改变为物体点
                                    ParLostObjPrevBg.insert(prev_id);
                                    lose_fea_prev_bg[cur_id].clear();
                                    lose_fea_prev_bg.erase(cur_id);
                                }
                            }
                        }
                        // 否则都加入二分图
                        else
                        {
                            int cur_id_1 = iter.second[0];
                            int cur_id_2 = iter.second[1];
                            FinalTrackObjFea.erase(prev_id);
                            prev_objs_assign_again.push_back(prev_id);
                            cur_obj_assign_again.push_back(cur_id_1);
                            cur_obj_assign_again.push_back(cur_id_2);
                            iter.second.clear();
                            // 此时prev_id暂时还没有完成分配，则先把它从静态物体集中删除
                            if (cur_stat_objs.find(prev_id) != cur_stat_objs.end())
                                cur_stat_objs.erase(prev_id);
                        }
                    }
                    // 如果超过两个当前帧检测物体都关联到上一帧的某个物体，则集体加入二分图
                    else if (iter.second.size() > 2)
                    {
                        // 上一帧该物体与当前帧关联物体全部加入二分图等待重新匹配。需要将这些匹配从matched_prev_objs_id中删除
                        for(auto iters: iter.second)
                        {
                            cur_obj_assign_again.push_back(iters);
                            // 如果当前临时物体被加入二分图，则记得要先把其从最终关联关系中去除
                            //FinalTrackCurObj[iters-1] = -1;
                        }
                        prev_objs_assign_again.push_back(iter.first);
                        // 并非erase，不会导致iterator失效
                        iter.second.clear();
                        // 该全局物体暂时从静态物体中排除，后续最终关联完成后再确定
                        if (cur_stat_objs.find(prev_id) != cur_stat_objs.end())
                        {
                            cur_stat_objs.erase(prev_id);
                        }
                    }
                }
            }

            // 查找当前背景中是否还有漏检的物体，如果有则将其添加到当前帧的物体列表中
            // int repeat_obj = 0;
            if (lose_objs_cur_bg.size() > 0)
            {
                for(int k = 0; k < lose_objs_cur_bg.size(); ++k)
                {
                    // 表示该漏检部分已加入匹配集。这里给修改回来
                    if (lose_objs_cur_bg[k] < 0) 
                    {
                        lose_objs_cur_bg[k] = -1 * lose_objs_cur_bg[k];
                        continue; 
                    }
                    int prev_id = lose_objs_cur_bg[k];
                    // 添加当前帧的发现的漏检物体，根据其与上一帧关联的obj id。背景的obj id为0，其他物体依次增大，漏检的物体放在最后
                    // 先暂时增加物体数量，融合的步骤放到最终匹配完成后
                    if(matched_prev_objs_id.find(prev_id) != matched_prev_objs_id.end()) 
                        matched_prev_objs_id[prev_id].push_back(k+num_detect_obj);
                    else
                    {
                        vector<int> temp_vec;
                        temp_vec.push_back(k+num_detect_obj);
                        matched_prev_objs_id.insert(make_pair(prev_id,temp_vec));
                    }

                    // 没必要添加，浪费内存
                    // TrackObjFeaFrame[k+num_detect_obj] = fea_cur_lose_objs[k];

                    // 如果与当前帧该漏检物体相关联的上一帧物体 仅仅与该漏检物体相关联
                    // 因为上一帧有关联的物体已经都记录了，要出现此prev_id的size为1，要么该prev_id物体始终未被分配关联物体，要么之前被分配超过一个物体然而被清空（均加入二分图匹配）
                    // 此时matched_prev_objs_id中每个元素的size最大为3,即上一帧的该物体在当前帧被分成了三部分，其中两部分被检测为两个物体，一部分为漏在背景中（这种概率非常非常小）
                    if(matched_prev_objs_id[prev_id].size() == 1)
                    {
                        // 对于漏检的物体点集，可能它还有部分区域是被检测出来的但是又还没有与上一帧的物体进行匹配。如果存在这种情况，则在二分图匹配后，此处新增id的漏检物体点在后续会被融合进另一关联物体中
                        if (find(prev_objs_assign_again.begin(),prev_objs_assign_again.end(),prev_id) == prev_objs_assign_again.end())
                            prev_objs_assign_again.push_back(prev_id);
                    }
                }
                // 。。。todo？
            }
            
            // 寻找上一帧中在上述过程中没有得到任何待匹配（包括当前帧背景的漏检）的物体（glob_obj_id_prev中不包含背景），加入二分图任务
            // cout << "num of obj in prev frame: " << glob_obj_id_prev.size() << endl;
            for (auto &iter: glob_obj_id_prev)
            {
                if (matched_prev_objs_id.find(iter) == matched_prev_objs_id.end())
                {
                    if (find(prev_objs_assign_again.begin(),prev_objs_assign_again.end(),iter) == prev_objs_assign_again.end())
                        prev_objs_assign_again.push_back(iter);
                }
            }

            // 是否需要二分图匹配
            int num_rest_objs_prev = prev_objs_assign_again.size();
            int num_rest_objs_cur  = cur_obj_assign_again.size();
            if (num_rest_objs_prev > 0 && num_rest_objs_cur > 0)
            {
                vector<float> temp(num_rest_objs_cur, 0.0);
                // 关联矩阵，行数表示上一帧待关联物体数，列数表示当前帧待关联物体数
                association_mat.resize(num_rest_objs_prev, temp);
            }
            else 
                association_mat.clear();
        }
        
        if(association_mat.size() > 0)
        {
            // 如果上一帧(和当前帧均)只有一个待匹配物体，则直接查看它们的值
            // if(association_mat.size() == 1 && association_mat[0].size() == 1)
            if(association_mat.size() == 1)
            {
                // association_mat[0][0] = cur_obj_assign_again[0];
                int prev_id = prev_objs_assign_again[0];
                match_score_two_objs(prev_id, cur_obj_assign_again, association_mat[0], seg_map, flow_map, depth_map, Ps, Rs);
            }
            // 大于2个待匹配时，开启多线程
            else
            {
            // 对上一帧和当前帧未完成关联的物体构建二分图任务的关联矩阵
#pragma omp for
                {
                    // 对于上一帧的某一个未匹配物体，查看当前帧与其有关联关系的
                    for(int n = 0; n < association_mat.size(); ++n)
                    {
                        int prev_id = prev_objs_assign_again[n];
                        // 计算关联程度时仍然会剔除异常匹配点的影响，但是不会设置该点的status
                        match_score_two_objs(prev_id, cur_obj_assign_again, association_mat[n], seg_map, flow_map, depth_map, Ps, Rs);
                    }
                }
            }
        }
    }

    cout << "finished openML in objs-matching" << endl;
    
    // 上面结束openML
    int num_assoc_obj_prev = prev_objs_assign_again.size();
    int num_assoc_obj_cur  = cur_obj_assign_again.size();
    int final_assoc_obj_prev = num_assoc_obj_prev;
    int final_assoc_obj_cur  = num_assoc_obj_cur;
    vector<uchar> lost_obj_prev, is_new_obj_cur;
    vector<std::vector<float>> final_assoc_mat;

    // cout << "Original num of prev obj in bipartite graph: " << num_assoc_obj_prev << endl;
    // cout << "Original num of cur obj in bipartite graph: " << num_assoc_obj_cur << endl;

    if(num_assoc_obj_prev == 1 && num_assoc_obj_cur == 1)
    {
        // 拥有关联关系
        if(association_mat[0][0] > 0.0)
        {
            int id_obj_prev = prev_objs_assign_again[0];
            int id_obj_cur  = cur_obj_assign_again[0];
            
            // 当前帧临时物体与上一帧是一对一（如果还另外与上一帧的临时发现的漏检物体（部分）相关联，则另外放在别的变量中表示）
            FinalTrackCurObj[id_obj_cur] = id_obj_prev;

            if(matched_prev_objs_id.find(id_obj_prev) != matched_prev_objs_id.end()) 
                matched_prev_objs_id[id_obj_prev].push_back(id_obj_cur);
            else
            {
                vector<int> temp_vec;
                temp_vec.push_back(id_obj_cur);
                matched_prev_objs_id.insert(make_pair(id_obj_prev,temp_vec));
            }
        }
        else
        {
            // new_objs_cur.push_back(cur_obj_assign_again[0]);
            int id = cur_obj_assign_again[0];
            int num_assoc = final_assign_id_cur_objs[id].size();
            // 如果也没有和上一帧的背景漏检点相关联，则直接认为是新物体
            if (num_assoc == 0) 
            {   
                new_objs_cur.push_back(id);
            }
            else
            {
                // 如果当前帧该物体在上一帧有漏检点的关联，则将其作为上一帧完全漏检的物体
                if((num_assoc == 1 && lose_fea_prev_bg.find(id) != lose_fea_prev_bg.end()) || num_assoc > 1)
                {
                    bool first = true;
                    // 这就需要在特征点跟踪匹配时，允许上一帧的背景点匹配到当前帧中的物体点！！！FAST的光流跟踪时并不会设置mask，mask只是用于检测新特征点的！！
                    for(auto &iter: TrackObjFeaFrame[id])
                    {
                        if (iter.second[0].second(7) == 0)
                        {
                            if(first)
                            {
                                first = false;
                                map<int, vector<pair<int,Vector8d>>> temp_fea;
                                // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                temp_fea.insert(make_pair(iter.first,iter.second));
                                TotalLostObjPrevBg.insert(std::make_pair(id,temp_fea));
                            }
                            else
                                // 下面这两种map的插入方法哪个的效率更高？
                                TotalLostObjPrevBg[id].insert(iter);
                                // TotalLostObjPrevBg[id][iter.first] = iter.second;
                        }
                    }
                    lose_fea_prev_bg[id].clear();
                    lose_fea_prev_bg.erase(id);
                }
                // 如果该物体仅与上一帧的一个物体有关联点（指的是在上面完成关联检查）
                else
                {
                    new_objs_cur.push_back(id);
                }
            }
            
            for(auto iter: prev_objs_assign_again)
            {
                if (matched_prev_objs_id.find(iter) == matched_prev_objs_id.end()) 
                    FinalLostObjPrev.push_back(iter);
                // 这种情况是该物体重新加入二分图匹配但最终没有找到关联的检测物体，而且也没有与当前帧的漏检点相关联
                else if(matched_prev_objs_id[iter].size() == 0)
                {
                    FinalLostObjPrev.push_back(iter);
                    matched_prev_objs_id.erase(iter);
                }
            }
        }
    }
    // 需要进行二分图匹配。首先需要寻找上一帧和当前帧中的哪些物体完全没有待关联对象，去除它们
    else if (num_assoc_obj_prev > 0 && num_assoc_obj_cur > 0)
    {
        float max_score = 0.0;
        lost_obj_prev.resize(num_assoc_obj_prev,1);
        is_new_obj_cur.resize(num_assoc_obj_cur,1);
        // 上一帧物体中哪些与（关联矩阵中的）当前帧物体没有任何待关联
        for (int k = 0; k < num_assoc_obj_prev; ++k)
        {
            // 上一帧物体在当前帧中没有待匹配对象，则该物体在当前帧中已消失，不参与二分图匹配
            auto max_iter = std::max_element(association_mat[k].begin(),association_mat[k].end());
            if (*max_iter <= 0) 
            {
                lost_obj_prev[k] = 0;
                --final_assoc_obj_prev;
            }
            else if(*max_iter > max_score)
            {
                max_score = *max_iter;
            }
        }
        
        // 当前帧物体中哪些与（关联矩阵中的）上一帧物体没有任何待关联
        for (int i = 0; i < num_assoc_obj_cur; ++i)
        {
            for (int j = 0; j < num_assoc_obj_prev; ++j)
            {
                // 考虑当前帧的每个物体是否有待关联物体
                // 如果当前帧某个物体没有和上一帧任何检测物体建立临时的关联，则不参与二分图匹配
                if (association_mat[j][i] == 0) 
                {
                    if(j == num_assoc_obj_prev-1)
                    {
                        is_new_obj_cur[i] = 0;
                        --final_assoc_obj_cur;
                        
                        // 没有任何潜在关联的当前帧物体，其处理不在这里进行
                        // int id = cur_obj_assign_again[i];
                        // int num_assoc = final_assign_id_cur_objs[id].size();
                        // // 如果也没有和上一帧的背景漏检点相关联，则直接认为是新物体
                        // if (num_assoc == 0) 
                        // {   
                        //     new_objs_cur.push_back(id);
                        // }
                        // else
                        // {
                        //     // 如果当前帧该物体在上一帧有漏检点的关联，则将其作为上一帧完全漏检的物体
                        //     if((num_assoc == 1 && lose_fea_prev_bg.find(id) != lose_fea_prev_bg.end()) || num_assoc > 1)
                        //     {
                        //         bool first = true;
                        //         // 这就需要在特征点跟踪匹配时，允许上一帧的背景点匹配到当前帧中的物体点！！！FAST的光流跟踪时并不会设置mask，mask只是用于检测新特征点的！！
                        //         for(auto &iter: TrackObjFeaFrame[id])
                        //         {
                        //             if (iter.second[0].second(7) == 0)
                        //             {
                        //                 if(first)
                        //                 {
                        //                     first = false;
                        //                     map<int, vector<pair<int,Vector8d>>> temp_fea;
                        //                     // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                        //                     temp_fea.insert(make_pair(iter.first,iter.second));
                        //                     TotalLostObjPrevBg.insert(std::make_pair(id,temp_fea));
                        //                 }
                        //                 else
                        //                     // 下面这两种map的插入方法哪个的效率更高？
                        //                     TotalLostObjPrevBg[id].insert(iter);
                        //                     // TotalLostObjPrevBg[id][iter.first] = iter.second;
                        //             }
                        //         }
                        //         lose_fea_prev_bg.erase(id);
                        //     }
                        //     // 如果该物体仅与上一帧的一个物体有关联点（指的是在上面完成关联检查）
                        //     else
                        //     {
                        //         new_objs_cur.push_back(id);
                        //     }
                        // }
                    }
                    else
                        continue;
                }
                // 每一列中只要有一个元素不为0,则当前帧该物体与上一帧的物体存在关联
                else
                {
                    break;
                }
            }
        }
    
        // 需要进行关联矩阵的裁减，去除完全没有待关联对象的物体
        if ((final_assoc_obj_prev > 0 && final_assoc_obj_cur > 0) && (final_assoc_obj_prev < num_assoc_obj_prev || final_assoc_obj_cur < num_assoc_obj_cur))
        {
            vector<float> temp(final_assoc_obj_cur, 0.0);
            final_assoc_mat.resize(final_assoc_obj_prev, temp);
            int id_prev = 0;
            for (int i = 0; i < num_assoc_obj_prev; ++i)
            {
                if (lost_obj_prev[i] == 0) 
                {
                    int prev_obj = prev_objs_assign_again[i];
                    // 如果此全局物体没有与当前帧任何临时物体有关联（包括背景漏检部分）
                    if (matched_prev_objs_id.find(prev_obj) == matched_prev_objs_id.end()) 
                        FinalLostObjPrev.push_back(prev_obj);
                    // 这种情况是该物体重新加入二分图匹配但最终没有找到关联的检测物体，而且也没有与当前帧的漏检点相关联
                    else if(matched_prev_objs_id[prev_obj].size() == 0)
                    {
                        FinalLostObjPrev.push_back(prev_obj);
                        matched_prev_objs_id.erase(prev_obj);
                    }
                    
                    continue;
                }

                // float max_score = *(std::max_element(association_mat[i].begin(), association_mat[i].end()));
                float max_score_fix = 100;
                int id_cur = 0;
                for (int j = 0; j < num_assoc_obj_cur; ++j)
                { 
                    // 如果该当前帧物体没有与上一帧任何物体形成关联
                    if (is_new_obj_cur[j] == 0)
                    {
                        int cur_obj = cur_obj_assign_again[j];
                        // 注意，每一个没有任何潜在关联的当前帧物体，其在每一行中的该列都会出现is_new_obj_cur[j] == 0，其只需要被处理一次
                        if(find(new_objs_cur.begin(),new_objs_cur.end(),cur_obj) != new_objs_cur.end()) continue;
                        if(TotalLostObjPrevBg.find(cur_obj) != TotalLostObjPrevBg.end()) continue;
                        
                        int num_match_cur_obj = final_assign_id_cur_objs[cur_obj].size();
                        if (num_match_cur_obj == 0) 
                        {
                            new_objs_cur.push_back(cur_obj);
                        }
                        // 如果当前帧该物体之前有与上一帧的物体相匹配，则该物体是因为 与别的当前帧物体关联到 同一个上一帧物体，然后在二分图匹配中被排除了
                        else
                        {
                            // 当前帧物体只与上一帧背景中的漏检物体点相关联，则为上一帧完全漏检
                            if ((num_match_cur_obj == 1 && lose_fea_prev_bg.find(cur_obj) != lose_fea_prev_bg.end()) || num_match_cur_obj > 1)
                            {
                                bool first = true;
                                for(auto &pt: lose_fea_prev_bg[cur_obj])
                                {
                                    if(first)
                                    {
                                        first = false;
                                        map<int, vector<pair<int,Vector8d>>> temp_fea;
                                        // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                        temp_fea.insert(make_pair(pt,TrackObjFeaFrame[cur_obj][pt]));
                                        TotalLostObjPrevBg.insert(std::make_pair(cur_obj,temp_fea));
                                    }
                                    else
                                        // 下面这两种map的插入方法哪个的效率更高？
                                        TotalLostObjPrevBg[cur_obj].insert(make_pair(pt,TrackObjFeaFrame[cur_obj][pt]));
                                        // TotalLostObjPrevBg[iter][pt] = TrackObjFeaFrame[iter][pt];
                                }

                                lose_fea_prev_bg[cur_obj].clear();
                                lose_fea_prev_bg.erase(cur_obj);
                            }
                            // 如果不是只与上一帧背景相匹配，则将该物体作为新物体，就算有与上一帧背景相匹配，也放弃
                            else
                            {
                                new_objs_cur.push_back(cur_obj);
                                // cout << "cur obj id: " << cur_obj << endl;
                                // for(int n = 0; n < num_match_cur_obj; ++n)
                                // { 
                                //     int prev_id = final_assign_id_cur_objs[cur_obj][n];
                                //     cout << "matched prev obj id: " << prev_id << endl;
                                // }
                            }
                        }
                        continue;
                    }
                    else
                    {
                        // score为0的点则使取非常大的值，否则其值为0会直接影响分配。
                        // 获取所有上一帧物体与所有当前帧物体的score_match中的最大值，使得所有没有匹配的两个物体的score为此值的10倍
                        if(association_mat[i][j] == 0)
                            // final_assoc_mat[id_prev][id_cur] = 1000;
                            final_assoc_mat[id_prev][id_cur] = std::max(max_score*10,max_score_fix);
                        else
                            final_assoc_mat[id_prev][id_cur] = association_mat[i][j];
                        ++id_cur;
                    } 
                }
                ++id_prev;
            }
            // 去除二分图关联物体名单中的这些无关联物体
            reduceVector(prev_objs_assign_again,lost_obj_prev);
            reduceVector(cur_obj_assign_again,is_new_obj_cur);
        }
        else if(final_assoc_obj_prev == num_assoc_obj_prev && final_assoc_obj_cur == num_assoc_obj_cur)
        {
            for(auto &element: association_mat)
            {
                final_assoc_mat.push_back(std::move(element));
            }
        }
        else
        {
            final_assoc_mat.clear();
            // 此种情况下，是prev_objs_assign_again中的所有上一帧物体 或者 cur_obj_assign_again中的所有当前帧物体 都没有任何潜在关联物体（即score为0）
            // 则就是前后2帧之间没有任何潜在关联，不需要进行二分图了，这里也不需要reduce，而是所有物体均没有关联
            // reduceVector(prev_objs_assign_again,lost_obj_prev);
            // reduceVector(cur_obj_assign_again,is_new_obj_cur);
        }
        
        // 如果最终还需要二分图匹配
        if (final_assoc_mat.size() > 0)
        {
            // cout << "final num of prev objs in bipartite graph: " << final_assoc_mat.size() << endl;
            // cout << "final num of cur objs in bipartite graph: " << final_assoc_mat.size() << endl;

            // 开始进行二分图匹配
            UpdateCosts(final_assoc_mat, optimizer.costs());
            // entry of hungarian optimizer minimum-weighted matching
            optimizer.Minimize(&assignments);
            vector<uchar> status_obj_prev(final_assoc_obj_prev,1), status_obj_cur(final_assoc_obj_cur,1);
            // 注意，assignments中的每个配对是只保留非膨胀元素的
            for(auto &iter: assignments)
            {
                int id_obj_prev = prev_objs_assign_again[iter.first];
                int id_obj_cur  = cur_obj_assign_again[iter.second];
                
                // cout << "matched prev gl obj: " << id_obj_prev << "matched cur obj: " << id_obj_cur << endl;
                
                // 当前帧临时物体与上一帧是一对一（如果还另外与上一帧的临时发现的漏检物体（部分）相关联，则另外放在别的变量中表示）
                FinalTrackCurObj[id_obj_cur] = id_obj_prev;
                if(matched_prev_objs_id.find(id_obj_prev) != matched_prev_objs_id.end())
                    matched_prev_objs_id[id_obj_prev].push_back(id_obj_cur);
                else
                {
                    vector<int> temp_vec;
                    temp_vec.push_back(id_obj_cur);
                    matched_prev_objs_id.insert(make_pair(id_obj_prev,temp_vec));
                }
                // 基于上一帧和当前帧中在二分图匹配中最终找到匹配的物体(其实只需要查看哪个个数比较大即可?)
                status_obj_prev[iter.first] = 0;
                status_obj_cur[iter.second] = 0;
            }
            // 剩下的就是上一帧或者当前帧中没有找到匹配的物体，且这两个量中最多只有一个量的size不为0！！
            reduceVector(prev_objs_assign_again,status_obj_prev);
            reduceVector(cur_obj_assign_again,status_obj_cur);
        }

        // 如果是当前帧的物体有多余，则添加当前帧的新物体
        if(cur_obj_assign_again.size() > 0)
        {
            for(auto &iter: cur_obj_assign_again)
            {
                int num_match_cur_obj = final_assign_id_cur_objs[iter].size();
                if (num_match_cur_obj == 0) 
                {
                    new_objs_cur.push_back(iter);
                }
                // 如果当前帧该物体之前有与上一帧的物体相匹配，则该物体是因为 与别的当前帧物体关联到 同一个上一帧物体，然后在二分图匹配中被排除了
                else
                {
                    // 当前帧物体只与上一帧背景中的漏检物体点相关联，则为上一帧完全漏检;或者该物体在上一帧完全漏检
                    if ((num_match_cur_obj == 1 && lose_fea_prev_bg.find(iter) != lose_fea_prev_bg.end()) || num_match_cur_obj > 1)
                    {
                        bool first = true;
                        for(auto &pt: lose_fea_prev_bg[iter])
                        {
                            if(first)
                            {
                                first = false;
                                map<int, vector<pair<int,Vector8d>>> temp_fea;
                                // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                temp_fea.insert(make_pair(pt,TrackObjFeaFrame[iter][pt]));
                                TotalLostObjPrevBg.insert(std::make_pair(iter,temp_fea));
                            }
                            else
                                // 下面这两种map的插入方法哪个的效率更高？
                                TotalLostObjPrevBg[iter].insert(make_pair(pt,TrackObjFeaFrame[iter][pt]));
                                // TotalLostObjPrevBg[iter][pt] = TrackObjFeaFrame[iter][pt];
                        }

                        lose_fea_prev_bg[iter].clear();
                        lose_fea_prev_bg.erase(iter);
                    }
                    // 如果不是只与上一帧背景相匹配，则将该物体作为新物体，就算有与上一帧背景相匹配，也放弃
                    else
                    {
                        new_objs_cur.push_back(iter);
                        // cout << "cur obj id: " << iter << endl;
                        // for(int n = 0; n < num_match_cur_obj; ++n)
                        // { 
                        //     int prev_id = final_assign_id_cur_objs[iter][n];
                        //     cout << "matched prev obj id: " << prev_id << endl;
                        // }
                    }
                }
            }
        }
        
        if(prev_objs_assign_again.size() > 0)
        {
            for(auto iter: prev_objs_assign_again)
            {
                if (matched_prev_objs_id.find(iter) == matched_prev_objs_id.end()) 
                    FinalLostObjPrev.push_back(iter);
                // 这种情况是该物体重新加入二分图匹配但最终没有找到关联的检测物体，而且也没有与当前帧的漏检点相关联
                else if(matched_prev_objs_id[iter].size() == 0)
                {
                    FinalLostObjPrev.push_back(iter);
                    matched_prev_objs_id.erase(iter);
                }
            }
        }
    }
    // 如果没有二分图匹配，而是当前帧直接多出了一部分物体，则全部作为新物体（即当前帧两个物体的融合必须是在特征点关联阶段都关联到上一帧同一个物体）
    else if(num_assoc_obj_prev == 0 && num_assoc_obj_cur > 0)
    {
        // 对于加入到cur_obj_assign_again当前帧局部物体，其final_assign_id_cur_objs中的元素数要么是0，要么是1（只与上一帧的背景点临时相关联）
        for(auto iter: cur_obj_assign_again)
        {
            int num_match_cur_obj = final_assign_id_cur_objs[iter].size();
            // 当前帧某个物体没能与上一帧的任何物体（包括背景中漏检物体）相关联（可能性很小）
            if (num_match_cur_obj == 0)
            {
                new_objs_cur.push_back(iter);
            }
            // 这种情况下（只有当前帧物体加入到二分图）只可能是 当前帧物体只与上一帧的某些背景特征点（该物体在上一帧的部分漏检点）相关联，而上一帧的物体都已经有了唯一关联！
            // 则加入到待修正的“完整物体”列表中
            else if ((num_match_cur_obj == 1 && final_assign_id_cur_objs[iter][0] == 0) || num_match_cur_obj > 1)
            {
                assert(lose_fea_prev_bg.find(iter) != lose_fea_prev_bg.end() && "Something weired happened!");
                bool first = true;
                // 此变量中保存的是当前帧物体的局部id，该物体其实也会被当作新的物体，只不过起始帧是上一帧！这种情况只能依靠SIFT或者FAST（通过光流网络给出预测值）来进行关联！
                for(auto &pt: lose_fea_prev_bg[iter])
                {
                    if(first)
                    {
                        first = false;
                        map<int, vector<pair<int,Vector8d>>> temp_fea;
                        // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                        temp_fea.insert(make_pair(pt,TrackObjFeaFrame[iter][pt]));
                        TotalLostObjPrevBg.insert(std::make_pair(iter,temp_fea));
                    }
                    else
                        // 下面这两种map的插入方法哪个的效率更高？
                        TotalLostObjPrevBg[iter].insert(make_pair(pt,TrackObjFeaFrame[iter][pt]));
                        // TotalLostObjPrevBg[iter][pt] = TrackObjFeaFrame[iter][pt];
                }
                
                lose_fea_prev_bg[iter].clear();
                lose_fea_prev_bg.erase(iter);
            }
            else
            {
                new_objs_cur.push_back(iter);
                // cout << "cur obj id: " << iter << endl;
                // for(int n = 0; n < num_match_cur_obj; ++n)
                // { 
                //     int prev_id = final_assign_id_cur_objs[iter][n];
                //     cout << "matched prev obj id: " << prev_id << endl;
                // }
            }
        }
    }
    // 如果是上一帧直接多出了一部分物体，则全部作为离开视野的物体
    else if(num_assoc_obj_prev > 0 && num_assoc_obj_cur == 0)
    {
        for(auto &iter: prev_objs_assign_again)
        {
            // 把在当前帧中没有任何关联（包括背景漏检点）的上一帧物体归为丢失物体
            // 注意，上一帧的某个物体可能仅与当前帧背景中的某些特征点相关联（该物体在当前帧全部或部分漏检了），则该物体也算是关联成功，这种情况在matched_prev_objs_id中是查询得到iter的。
            if (matched_prev_objs_id.find(iter) == matched_prev_objs_id.end()) 
                FinalLostObjPrev.push_back(iter);
            // 这是针对上一帧该物体预关联时与超过1个的当前帧非背景物体相关联但是又全部被加入二分图匹配（融合失败），且最后该上一帧物体没有获得任何匹配（包括背景漏检点）
            else if(matched_prev_objs_id[iter].size() == 0)
            {
                FinalLostObjPrev.push_back(iter);
                matched_prev_objs_id.erase(iter);
            }
        }
    }
    
    cout << "finished matching of all objs!" << endl;
    
    // 两帧之间的相机运动
    Matrix3d R_12 = Matrix3d::Zero();
    Vector3d P_12 = Vector3d::Zero();
    // 只有系统第3帧开始才会有相机运动的预测（基于恒速运动模型）
    if(frame_count > 1)
    {
        // 当前帧相机坐标系下的两帧间相机运动预测值
        R_12 = Rs[1].transpose()*Rs[0];
        P_12 = Rs[1].transpose()*(Ps[0] - Ps[1]);
    }
    
    // 遍历物体匹配结果，如果有上一帧一个物体在当前帧有多个匹配物体的情况，则将这多个物体合为一个(保存到最终的FinalTrackObjFea中)！
    // 根据上面对prev_objs_assign_again的操作，这种情况只会发生在当前帧漏检物体（最多一个）和检测物体（最多两个）关联到上一帧同个物体的时候！
    for(int i = 0; i < glob_obj_id_prev.size(); ++i)
    {
        int id_prev = glob_obj_id_prev[i];
        bool first = true;
        // 找到了关联的上一帧物体(包括当前帧完全漏检），且该关联还没添加
        if (find(FinalLostObjPrev.begin(),FinalLostObjPrev.end(),id_prev) == FinalLostObjPrev.end() && FinalTrackObjFea.find(id_prev) == FinalTrackObjFea.end())
        {
            bool par_lost_prev = false;
            bool total_lost_cur = false;
            // 与当前帧的关联物体中，可能有背景中的漏检物体（添加为一个新的临时物体了）
            
            FinalTrackObj[num_track_g_obj].first = id_prev;
            
            assert(matched_prev_objs_id.find(id_prev) != matched_prev_objs_id.end());
            assert(matched_prev_objs_id[id_prev].size() > 0);

            // elem是与prev_id物体相关联的当前帧的局部物体们
            for(auto elem: matched_prev_objs_id[id_prev])
            {
                
                // 注意要小于号，num_detect_obj中包含了背景
                if (elem < num_detect_obj)
                {
                    FinalTrackObj[num_track_g_obj].second.push_back(elem);
                    // 如果该物体没有任何跟踪点，而是通过二分图找到了上一帧的匹配物体，则此处没有跟踪点可以添加。
                    // 因此，不一定在FinalTrackObj中存在的上一帧全局物体，其在FinalTrackObjFea就一定存在！但是经过下面的给动态跟踪物体添加像素点后，其在FinalTrackObjFea中就一定存在了！
                    if(!TrackObjFeaFrame[elem].empty())
                    {
                        for(auto &iter: TrackObjFeaFrame[elem])
                        {
                            if (iter.second[0].second(7) == id_prev)
                            {
                                if(first)
                                {
                                    map<int, vector<pair<int,Vector8d>>> temp_fea;
                                    // 这个点的id是-1，是用于让该物体参与物体关联期间的二分图匹配
                                    temp_fea.insert(iter);
                                    FinalTrackObjFea.insert(make_pair(id_prev,temp_fea));
                                    first = false;
                                }
                                else
                                    // FinalTrackObjFea[id_prev][iter.first] = iter.second;
                                    FinalTrackObjFea[id_prev].insert(iter);
                            }
                            // else
                            // {
                                    // 该临时物体的其他跟踪点是否在此处删除？下面还可能有该物体在上一帧背景的漏检点

                            // }   
                        }
                    }

                    // 将上一帧的背景上的某些漏检物体的特征点 与 其已校正过的物体特征点 统一到一起
                    if (lose_fea_prev_bg.find(elem) != lose_fea_prev_bg.end())
                    {
                        par_lost_prev = true;
                        for(auto &iter: lose_fea_prev_bg[elem])
                        {
                            if(first)
                            {
                                map<int, vector<pair<int,Vector8d>>> temp_fea;
                                temp_fea.insert(make_pair(iter,TrackObjFeaFrame[elem][iter]));
                                FinalTrackObjFea.insert(make_pair(id_prev,temp_fea));
                                first = false;
                            }
                            else
                                FinalTrackObjFea[id_prev].insert(make_pair(iter,TrackObjFeaFrame[elem][iter])) ;
                        }
                        // 此变量记录上一帧背景中“部分”漏检物体的id，用于后续在全局地图中将上一帧背景的这部分新特征点改变为物体点
                        // 在修改时是根据点的全局id从背景点集中寻找
                        ParLostObjPrevBg.insert(id_prev);
                        lose_fea_prev_bg[elem].clear();
                        lose_fea_prev_bg.erase(elem);
                    }
                }
                else
                {   
                    // 如果该物体在当前帧是完全漏检，则记录它
                    if (matched_prev_objs_id[id_prev].size() == 1) 
                    {
                        detect_lost_objs_cur.push_back(id_prev);
                        total_lost_cur = true;
                        cout << "find a total lost obj in cur frame!" << endl;
                    }

                    // 如果当前帧物体部分漏检，要把0放进去吗？？统一选择放进去
                    FinalTrackObj[num_track_g_obj].second.push_back(0);
                    int id = elem - num_detect_obj;
                    for (auto &pt:fea_cur_lose_objs[id])
                    {
                        if(first)
                        {
                            map<int, vector<pair<int,Vector8d>>> temp_fea;
                            temp_fea.insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                            FinalTrackObjFea.insert(make_pair(id_prev,temp_fea));
                            first = false;
                        }
                        else
                            // FinalTrackObjFea[id_prev][pt] = TrackObjFeaFrame[0][pt];
                            FinalTrackObjFea[id_prev].insert(make_pair(pt,TrackObjFeaFrame[0][pt]));
                    }
                }
            }
            
            // 如果是上一帧的全局静态物体且在当前帧（与上一帧之间）还未确认其是否为静态，则进行快速验证。至少需要是系统第3帧才会有相机的运动预测。
            // TODO:这里的快速验证是否还有必要？在之前的物体关联阶段没能直接为此静态物体实现关联（而是要用二分图来寻找匹配）就说明要么其匹配点集不是很准确，要么其在当前帧运动了，这里是否还有需要验证？
            // 另外，事实上对于上一帧的新物体，如果其在当前帧中找到关联，是否也可以使用相同的方法来验证其是否为静态物体？
            // 理论上应该是可以的，但是没有经过RANSAC的排除外点，其场景流估计是非常不准确的！即使计算出来的相似度大，也可能是由于匹配错误或者不准确的运动预测而导致的！而对于静态物体，我们是应该预期其相似性应该要足够大！
            if (frame_cnt > 1 && status_objs_prev[id_prev] == 1)
            {
                if (cur_stat_objs.find(id_prev) == cur_stat_objs.end())
                {
                    bool is_stat = false;
                    
                    float ave_dep_cur_obj = 0.0;
                    
                    if(FinalTrackObjFea.find(id_prev) != FinalTrackObjFea.end())
                    {
                        int num_fea = FinalTrackObjFea[id_prev].size();
                        // 如果不是上一帧部分漏检的物体，则要求特征点数不小于8；否则，只需要大于5
                        if ((!par_lost_prev && num_fea >= 8) || (par_lost_prev && num_fea >= 5))
                            // 如果是使用特征点匹配来验证静态，则验证成功的话会把其中的异常匹配点直接从FinalTrackObjFea中删除
                            is_stat = check_stat_obj(true, FinalTrackObjFea[id_prev], id_prev, FinalTrackObj[num_track_g_obj].second, P_12, R_12, ave_dep_cur_obj);
                        // 特征点不够时采用密集像素点（忽略上一帧或当前帧中属于该物体的部分漏检区域）。需要该物体在当前帧不是完全漏检，认为此时的像素光流匹配不是很准确？
                        else if(!total_lost_cur)
                            // 上一帧部分静态物体在当前帧可能跟踪特征点不足，而采用像素点来验证静态性，这样特征点中的异常匹配点就没办法排除，只能交由相机位姿估计来排除！
                            is_stat = check_stat_obj(false, FinalTrackObjFea[id_prev], id_prev, FinalTrackObj[num_track_g_obj].second, P_12, R_12, ave_dep_cur_obj, seg_map, flow_map, depth_map);
                    }
                    // 如果该物体在当前帧没有跟踪点，则可以使用像素点
                    else
                    {
                        FeaFrame empty_feaframe;
                        // 上一帧部分静态物体在当前帧可能跟踪特征点不足，而采用像素点来验证静态性，这样特征点中的异常匹配点就没办法排除，只能交由相机位姿估计来排除！
                        is_stat = check_stat_obj(false, empty_feaframe, id_prev, FinalTrackObj[num_track_g_obj].second, P_12, R_12, ave_dep_cur_obj, seg_map, flow_map, depth_map);
                    }
                    
                    if (is_stat && ave_dep_cur_obj > 0) 
                        cur_stat_objs.insert(std::pair<int,float>(id_prev,ave_dep_cur_obj));
                    else
                        cur_dyn_objs.insert(num_track_g_obj);
                }
            }
            // 否则该跟踪物体归为当前帧的潜在动态物体
            else
            {
                cur_dyn_objs.insert(num_track_g_obj);
            }
            ++num_track_g_obj;
        }
    }

    // 查看lose_fea_prev_bg中是否还有剩余元素！
    assert(lose_fea_prev_bg.size() == 0 && "There is something wrong about lose_fea_prev_bg member erase operation!");
    // 去除多余的元素。这个变量中只保留那些在当前帧中有跟踪匹配的上一帧的全局物体
    FinalTrackObj.resize(num_track_g_obj);
    
    {
        cout << "num of elem in FinalTrackCurObj: " << FinalTrackCurObj.size() << endl;
        cout << "num of elem in TotalLostObjPrevBg: " << TotalLostObjPrevBg.size() << endl;
        cout << "num of elem in new_objs_cur: " << new_objs_cur.size() << endl;

        // cout << "num of elem in FinalTrackObj: " << FinalTrackObj.size() << endl;
        // cout << "num of elem in FinalLostObjPrev: " << FinalLostObjPrev.size() << endl;
    }

    assert(FinalTrackCurObj.size() + TotalLostObjPrevBg.size() + new_objs_cur.size() == (num_detect_obj-1) && "Something wrong with vars FinalTrackCurObj and TotalLostObjPrevBg!");
    assert(FinalTrackObj.size() + FinalLostObjPrev.size() == glob_obj_id_prev.size() && "Something wrong with vars FinalTrackObj and FinalLostObjPrev!");

    // 为背景 和 各个非静态物体添加匹配点对直到满足数量要求
    
    // 将map转化为set，并按照平均深度值从小到大排列
    // 注意，set不论接受什么样的比较函数，对于先后的两个比较量a和b，只要比较函数返回true，则把b排在a后面；如果返回的是false，则把b排在a前面。对于set默认的比较函数，则是进行a<b的比较，即升序
    set<pair<int, float>, Cmp_depth_stat_objs> sorted_depth_sta_objs(cur_stat_objs.begin(), cur_stat_objs.end());

    // 添加静态特征点匹配
    int num_sta_objs = sorted_depth_sta_objs.size();
    int num_track_fea_bg = 0;
    // int num_track_fea_bg_prior = 0;
    // 首先添加前后两帧的背景的sift特征匹配点

    // 当VI未初始化时，直接将所有的静态特征点的当前帧跟踪（包括sift和FAST，无论其有多少帧观测）都加入地图！
    // 尤其是那些初始观测帧下有深度估计的点，它们将被用于当前帧相机的PnP估计！
    // 理论上是可以限制每一帧中参与相机运动估计的匹配数量，然后剩下的静态点直接加入，后续其中超过连续3帧观测的点会在滑窗满时进行LBA。但是为了避免外点对LBA的影响，这里还是直接让每一帧的静态匹配都参与运动估计以去除外点！
    // 另外，从系统第5帧起，如果上一滑窗中连续观测帧数大于等于3的点数小于阈值，则应该适当增加新的FAST点，因为FAST点使用光流来跟踪，比较容易形成多帧跟踪！

    // 之后修改的版本中由于允许当前帧的背景跟踪点没有深度估计
    // 因此对于每一帧的背景跟踪点（这里遍历的都是当前帧有立体匹配的点，但是上一帧该点不一定有！！）都加入地图，因为后续要对它们三角化以估计它们在观测首帧的深度；
    // 或者如果当前帧要marg次新帧，且该点首帧（上一帧）有深度估计，则后续要根据上一帧的深度和相机运动来更新该点首帧（即当前帧）中的深度
    // if (!initial_succ)
    {
        // 当前帧背景上不一定有与上一帧背景的跟踪点！这种情况下就不进行相机的视觉位姿估计了
        if(!pts_two_bg.empty())
        {
            for (auto &pt_id: pts_two_bg)
            {
                // 这里能否不使用复制,而是指针?复制大量的Vector8d也是挺浪费时间和内存的...或者仅复制点的全局id?
                if(TrackBgFea.empty())
                    TrackBgFea.insert(make_pair(pt_id,TrackObjFeaFrame[0][pt_id]));
                else
                    TrackBgFea[pt_id] = TrackObjFeaFrame[0][pt_id];
            }
        }
    }

    // else
    // {
    //     // int num_rest_FAST = MIN_CNT_PTS_TRACK_BG - num_track_fea_bg;
    //     for (auto &pt_id: pts_two_bg)
    //     {
    //         // map默认按照key从小到大排序，在插入pts_two_bg时是按照map的顺序！
    //         // 如果已经VI初始化成功，则加入背景中所有的超过2帧观测（包含当前帧）的sift点（其精度比FAST高）和FAST点，因为这些点才有可能在当前帧参与LBA！
    //         // 暂时不把只有两帧观测的sift点加入地图，因为这些点不可能参与当前帧的LBA，且当前滑窗可能是要marg次新帧（则上一帧观测则白添加了，后续还是要删除）！！！
    //         // 注意，id比last_id_track_fea_prev小的点中不一定都是有效"静态"观测帧数大于2的点!
    //         // 比如某个物体上上帧为动态，而上一帧变为静态，且上一滑窗marg了次新帧，则该静态点的有效静态观测其实只有上一帧，加上当前帧则只有2帧，但点的id仍沿用其为动态点时的id！将这部分点加入地图也无所谓，毕竟属于少数，后续连续帧数少于4帧时再从地图中删除即可。
    //         if (pt_id <= last_id_track_fea_prev)
    //         {
    //             TrackBgFea[pt_id] = TrackObjFeaFrame[0][pt_id];
    //             ++num_track_fea_bg;
    //             // 这个值有可能小于0了
    //             // --num_rest_FAST;
    //             // ++num_track_fea_bg_prior;
    //         }
    //         else
    //             break;
            
    //         // // 如果部分优先要求的特征点已经遍历完了，而点数还达不到相机运动估计的最小要求，则查看是否有静态物体。如果有，则先结束背景点的遍历；如果没有，则直接添加背景的剩余FAST静态跟踪点
    //         // else if(num_rest_FAST > 0)
    //         // {
    //         //     if(num_sta_objs == 0)
    //         //     {
    //         //         TrackBgFea[pt_id] = TrackObjFeaFrame[0][pt_id];
    //         //         --num_rest_FAST;
    //         //         // 静态点集最多添加完bg的FAST点，无论数量达不达到MIN_CNT_PTS_TRACK_BG
    //         //     }
    //         //     else
    //         //         break;
    //         // }
    //         // // 达到点数点数要求就停止
    //         // else
    //         //     break;
    //     }
    // }

    // invalid_stat_objs.clear();

    // 静态跟踪点至少需要MIN_CNT_PTS_TRACK_BG个
    num_rest_track = MIN_CNT_PTS_TRACK_BG - TrackBgFea.size();
    // 上一帧有立体匹配的静态跟踪点至少需要Min_num_bg_track_with_dep_prev个
    num_rest_track_stereo = Min_num_bg_track_with_dep_prev - num_track_fea_with_dep_prev;

    // 如果有静态物体。再从静态物体的关联点中选择超过2帧观测（包括当前帧）的sift点和FAST点。
    // 另外同样地，当VI初始化还没成功时，每一帧都把所有静态物体的仅有两帧观测的sift跟踪点也加入静态点集合！这样子可能会因为点数较多而导致前几帧相机运动估计的时间比较长，但同样地这意味着下一帧背景中的新FAST点的检测数量会减少甚至为0！
    // 物体点不太容易形成多帧跟踪,因此物体点在每一帧中都必须有立体匹配,这就大大减少了能形成多帧跟踪的物体点的数量!
    // 所以,短跟踪长度的静态物体点只能用于VI未初始化时,用于提高视觉位姿估计的精度;VI初始化后,长跟踪长度(大于等于4帧观测)的物体点可以加入地图参与LBA!
    // if (num_sta_objs > 0 && num_rest_sift > 0)

    if (num_sta_objs > 0)
    {
        float thres_dep_sta_obj_fea = min(mThDepthBg/2.0, 10.0);

        // cout << "num of static objs during objs-matching: " << num_sta_objs << endl;
        bool need_PnP_cur_frame = false;

        // 纯双目的话，需要指定PnP_per_frame;双目+IMU的话（本项目暂不考虑单目IMU），需要指定是否未初始化 或者 初始化后的use_pnp_after_imu_init/fea_filtered
        // need_PnP_cur_frame = (!USE_IMU && PnP_per_frame) || (USE_IMU && (!initial_succ || use_pnp_after_imu_init));
        need_PnP_cur_frame = (!USE_IMU && PnP_per_frame) || (USE_IMU && (!initial_succ || (!fea_filtered || use_pnp_after_imu_init)));

        assert(mThDepthObj > thres_dep_sta_obj_fea);

        for (auto stat_obj: sorted_depth_sta_objs)
        {
            float depth = stat_obj.second;
            // 这种情况其实不会出现，因为上一帧 或 当前帧 的漏检点在匹配完成后都进行了平均深度的检查，不满足条件的漏检物体都会放弃！！
            // if (depth > mThDepthObj || depth < mMinDepthPt)
            // if (depth > mThDepthObj || depth < 1.5)
            if (depth > mThDepthObj)
            {
                // 后续在物体运动估计线程中需要遍历相关的当前帧物体的点并将其删除！
                invalid_stat_objs.insert(stat_obj.first);
                continue;
            }
            // 只添加近处的静态物体的跟踪点到地图中
            // todo:对于添加进地图的静态物体是否要记录？
            else if(depth < thres_dep_sta_obj_fea)
            {
                if(Limit_num_static_track)
                {
                    if(num_rest_track <= 0 && num_rest_track_stereo <= 0)
                    {
                        // 如果参与LBA的点数满足最小阈值，则退出所有遍历
                        if(num_old_track_fea >= 12)
                        {
                            // 这里应该是continue而不是break，这样就可以继续遍历所有物体并将平均深度大于阈值的物体放入invalid_stat_objs
                            continue;
                            // break;
                        }
                    }
                }
                
                int prev_id = stat_obj.first;
                // 加入各个静态物体的sift跟踪点 和 观察帧数大于2的FAST点
                // 该静态物体不一定有跟踪特征点（即通过像素匹配来完成关联）,所以要判断
                if(FinalTrackObjFea.find(prev_id) != FinalTrackObjFea.end())
                {
                    for(auto &pt: FinalTrackObjFea[prev_id])
                    {
                        int id_pt = pt.first;
                        
                        if(Limit_num_static_track)
                        {
                            if(num_rest_track <= 0 && num_rest_track_stereo <= 0) 
                            {
                                if(num_old_track_fea >= 12)
                                    break;
                                else
                                {
                                    if(sta_obj_fea_in_map.find(id_pt) == sta_obj_fea_in_map_cur.end()) 
                                        continue;
                                }
                            }

                            // 需要该点在上一帧中具有立体匹配
                            if(prevRightFeaMap.find(id_pt) == prevRightFeaMap.end()) continue;

                            int index = gl_id_index_map[id_pt];
                            float dep_pt = 0.0;
                            if(index > 0)
                                dep_pt = cur_FAST_dep[(index-1)];
                            else
                                dep_pt = cur_sift_dep[(-index)];
                            
                            // 具体特征点的深度值不能太大
                            if(dep_pt > thres_dep_sta_obj_fea) continue;

                            sta_obj_fea_in_map_cur.insert(id_pt);

                            if(TrackBgFea.empty())
                                TrackBgFea.insert(make_pair(id_pt,pt.second));
                            else
                                TrackBgFea[id_pt] = pt.second;

                            // 如果该静态物体跟踪点在上一帧就已经加入地图，则其是长期点，会参与LBA
                            if(sta_obj_fea_in_map.find(id_pt) != sta_obj_fea_in_map_cur.end()) ++num_old_track_fea;

                            --num_rest_track;
                            --num_rest_track_stereo;
                        }
                        else
                        {
                            int index = gl_id_index_map[id_pt];
                            float dep_pt = 0.0;
                            if(index > 0)
                                dep_pt = cur_FAST_dep[(index-1)];
                            else
                                dep_pt = cur_sift_dep[(-index)];
                            
                            // 具体特征点的深度值不能太大
                            if(dep_pt > thres_dep_sta_obj_fea) continue;
                            
                            // 观测帧数大于2的sift和FAST点优先添加，不论当前帧之前是否已经完成VI初始化。
                            // 注意，对于静态物体点，其实际观测帧数 与 其在地图中的保留观测帧数 不一定相等，因为静态物体点不是在其一开始被跟踪时就能被加入地图
                            // 但是，总体而言，被跟踪越久的点，其可靠性就越高。因此这里优先添加跟踪帧数更多的物体点（即使其在地图中的帧数不一定足够参与当前帧的LBA）
                            // 另一方面，上面限制了平均深度小于7m的物体才可以考虑加入地图，由于默认自车不会倒退，因此物体点在当前帧的深度只会比上一帧更小（或不变），这就使得其很大概率还是会被继续加入地图
                            if (id_pt <= last_id_track_fea_prev)
                            {
                                // 对于有超过2帧观测的点，不论其在上一帧有没有立体匹配，都加入地图，因为它们可以参与LBA
                                sta_obj_fea_in_map.insert(id_pt);
                                
                                if(TrackBgFea.empty())
                                    TrackBgFea.insert(make_pair(id_pt,pt.second));
                                else
                                    TrackBgFea[id_pt] = pt.second;
                                // 暂时不管跟踪点的数量限制
                                // --num_rest_sift;
                            }
                            // 如果当前帧需要PnP得到相机位姿的初始估计
                            else if(need_PnP_cur_frame)
                            {
                                // 对于只有2帧观测的物体点，需要其在上一帧中有立体匹配，这样其深度值比较可靠
                                if(prevRightFeaMap.find(id_pt) == prevRightFeaMap.end()) continue;

                                sta_obj_fea_in_map.insert(id_pt);
                                // if(num_rest_sift <= 0) break;
                                if(TrackBgFea.empty())
                                    TrackBgFea.insert(make_pair(id_pt,pt.second));
                                else
                                    TrackBgFea[id_pt] = pt.second;
                                // --num_rest_sift;
                            }
                            // 否则遍历下一个物体
                            else 
                                break;
                        }
                    }
                }
            }
            else
            {
                continue;
            }

            // if (num_rest_sift <= 0) break;
        }

        //added_short_track_Fea.clear();

        // 如果已添加的sift和FAST点数大于等于相机位姿估计所需要最少的点数，则不再需要剩下的FAST点（上一帧的新FAST点）;否则，优先从静态物体中选取上一帧的新FAST点的跟踪补充得到阈值.
        // 上面的想法已放弃。因为在VI未初始化期间，我们希望所有的两帧间的静态点跟踪都参与运动估计，排除其中的外点，这样当所有点都加入地图，部分跟踪时间较长的点后续参与LBA时有比较好的初始估计！另外，直接添加所有跟踪点的做法比较简单方便！
        // if (MIN_CNT_PTS_TRACK_BG > TrackBgFea.size())
        // {
        //     // 如果已经初始化成功，则暂时不把只有两帧观测的FAST点加入地图，因为这些点不可能参与当前帧的LBA，且当前滑窗可能是要marg次新帧（则上一帧观测则白添加了，后续还是要删除）
        //     // 如果还没初始化成功，则还需要足够多的最新两帧的特征匹配来估计两帧间相机的运动。
        //     if (initial_succ) continue;
            
        //     int num_rest_FAST = MIN_CNT_PTS_TRACK_BG - TrackBgFea.size();
        //     if (num_sta_objs > 0)
        //     {
        //         for (auto stat_obj: sorted_depth_sta_objs)
        //         {
        //             // 选取尽可能近的静态物体？太远的物体可能还不如纯背景中的FAST点？
        //             // if (stat_obj.second > mThDepthObj)
        //             if (stat_obj.second > 20)
        //                 break;
        //             else
        //             {
        //                 int prev_id = stat_obj.first;
        //                 for (auto &pt: FinalTrackObjFea[prev_id])
        //                 {
        //                     if (num_rest_FAST == 0) break;
        //                     int id_pt = pt.first;
        //                     if (gl_id_index_map[pt.first]>0 && id_pt > last_id_track_fea_prev)
        //                     {
        //                         TrackBgFea[id_pt] = pt.second;
        //                         //added_short_track_Fea.insert(id_pt);
        //                         --num_rest_FAST;
        //                     }
        //                 }
        //             }
        //             if (num_rest_FAST == 0) break;
        //         }
        //     }
        //     // 如果数量仍达不到最小阈值，则最后添加纯bg中的只有两帧观测的FAST点
        //     if (num_rest_FAST > 0)
        //     {
        //         for (int i = num_track_fea_bg_prior; i < pts_two_bg.size(); ++i)
        //         {
        //             if (num_rest_FAST == 0) break;
        //             int pt_id = pts_two_bg[i];
        //             if (gl_id_index_map[pt_id]<=0) break;
        //             TrackBgFea[pt_id] = TrackObjFeaFrame[0][pt_id];
        //             //added_short_track_Fea.insert(pt_id);
        //             --num_rest_FAST;
        //         }
        //     }
        //     // 至此就不再添加静态跟踪点了，无论数量是否达到MIN_CNT_PTS_TRACK_BG
        // }
    }

    // cout << "Num of tracked static fea (with stereo match):  " << TrackBgFea.size() << endl;

    sta_obj_fea_in_map.clear();
    sta_obj_fea_in_map = sta_obj_fea_in_map_cur;

    // 记录在物体跟踪阶段就已经确认为静态点的总数！之所以说是大概的数（approximate），是因为有些上一帧的静态物体此时还无法确认状态，需要后续进行运动估计，因此这里没有计算它们的跟踪点。
    // appro_num_track_stat_fea = 0;
    appro_num_track_stat_fea += pts_two_bg.size();
    if(num_sta_objs > 0)
    {
        for (const auto &stat_obj: cur_stat_objs)
        {
            if(invalid_stat_objs.find(stat_obj.first) != invalid_stat_objs.end()) continue;
            if(FinalTrackObjFea.find(stat_obj.first) != FinalTrackObjFea.end()) appro_num_track_stat_fea += FinalTrackObjFea[stat_obj.first].size();
        }
    }

    set<int, greater<int>> dyn_obj_id_del;
    // 遍历各个动态的关联物体，如果其关联点数小于最小阈值，则添加普通像素
    if (cur_dyn_objs.size() > 0)
    {
        // cout << "original num of detected dyn objs during obj-matching: " << cur_dyn_objs.size() << endl;
        
        int temp_pt_id = -1;
        for (auto id_tracked_obj: cur_dyn_objs)
        {
            // 当前跟踪的全局物体的id
            int id_obj = FinalTrackObj[id_tracked_obj].first;
            // cout << "matched prev obj id: " << id_obj << endl;
            vector<int> &match_cur_objs = FinalTrackObj[id_tracked_obj].second;
            
            // for(int n = 0; n < match_cur_objs.size(); ++n)
            //     cout << "matched cur obj id from matching: " << match_cur_objs[n] << endl;

            int num_pixel;
            bool total_lost = false;
            bool no_track = false;
            // 如果上一帧的物体在当前帧是完全漏检的
            if(std::find(detect_lost_objs_cur.begin(), detect_lost_objs_cur.end(), id_obj) != detect_lost_objs_cur.end()) total_lost = true;
            if (!total_lost)
            {
                if(FinalTrackObjFea.find(id_obj) == FinalTrackObjFea.end())
                {
                    no_track = true;
                    num_pixel = MIN_CNT_PTS_TRACK_OBJ;
                }
                else
                    num_pixel = MIN_CNT_PTS_TRACK_OBJ - FinalTrackObjFea[id_obj].size();
            }
            else
            {
                // 对于完全漏检的物体，要求的跟踪点数更少，因为像素点光流匹配很不精确，而特征点只要够精确，即使点数少也没关系！此外还有一点，像素点本身还要保证其深度值可靠，这点需要像matching函数中那样去除点集中的外点！太麻烦了！
                if(FinalTrackObjFea.find(id_obj) == FinalTrackObjFea.end())
                {
                    assert(false && "This branch should not occur!");
                    // no_track = true;
                    // num_pixel = 8;
                }
                else
                {
                    // 对于在当前帧完全漏检的跟踪物体，后续如何为其添加像素点跟踪？只能相信flow_map的估计足够好，认为基于光流且关联到背景中的点也是漏检点
                    num_pixel = 10 - FinalTrackObjFea[id_obj].size();
                }
            }
            
            // 匹配点数满足要求的物体就不需要添加像素点匹配
            if (num_pixel <= 0) continue;

            // 添加像素点匹配，从pixel_objs_prev中选择，且像素点在mask_prev_fea_objs中不能为黑！
            cv::Point2f un_pt_cur,un_pt_prev;
            float* ptr_pixel = pixel_objs_prev[id_obj];
            int num_valid_pixel = (int)ptr_pixel[-1];
            // cout << "num of sampled pixel of gl obj in prev frame: " << num_valid_pixel << endl;
            float prev_u, prev_v, cur_u, cur_v, flow_u, flow_v, disp, depth;
            int temp_obj_id;
            
            int cout_pixel = 0;
            for(int i = 0; i < num_valid_pixel; ++i)
            {
                if (num_pixel <= 0) break;

                // 对于当前帧完全漏检的物体，需要把上一帧所有不重复的采样像素点匹配加入点集中用于位姿估计，保留其中的内点作为该物体在当前帧的密集像素点集
                // 对于非漏检或者部分漏检的物体，添加像素匹配数直到跟踪点数达到最低要求。算了不设置最低数量要求
                // if (num_pixel == 0) break;
                // 上一帧的物体像素点
                prev_u = ptr_pixel[(i*3)];
                prev_v = ptr_pixel[(i*3+1)];
                // 该像素点是否在上一帧的特征点附近
                if (mask_prev_fea_objs.at<uchar>(prev_v,prev_u) == 0) continue;
                
                // 检查一下flow_map的结果vec2f是(du,dv)还是(dv,du)。一般来说是前者
                flow_u = flow_map.at<Vec2f>(prev_v,prev_u)[0];
                flow_v = flow_map.at<Vec2f>(prev_v,prev_u)[1];
                // flow_u = flow_map.at<Vec2f>(prev_v,prev_u)[1];
                // flow_v = flow_map.at<Vec2f>(prev_v,prev_u)[0];

                cur_u = flow_u + prev_u;
                cur_v = flow_v + prev_v;
                
                // if(cout_pixel < 5)
                // {
                //     cout << "prev_u: " << prev_u << endl;
                //     cout << "prev_v: " << prev_v << endl;
                //     cout << "flow_u: " << flow_u << endl;
                //     cout << "flow_v: " << flow_v << endl;
                // }
                
                if(inBorder(Point2f(cur_u,cur_v)))
                    disp = depth_map.at<float>(cur_v,cur_u);
                else
                    continue;
                
                if(disp <= 0) continue;
                depth = mbf/disp;
                // if(cout_pixel < 5)
                // {
                //     cout << "depth: " << depth << endl;
                // }
                
                if (depth >= mThDepthObj || depth <= mMinDepthPt) continue;
                // 判断物体obj_id是否一致
                temp_obj_id = seg_map.at<Vec2b>(cur_v,cur_u)[1];

                // cout << "matched cur obj id from flow: " << temp_obj_id << endl;
                // ++cout_pixel;
                
                // 对于当前帧非完全漏检的物体，如果上一帧的物体像素点匹配到当前帧的背景点，则放弃，因为认为普通像素的光流估计不准确，不足以发现漏检物体点！
                // 而对于当前帧完全漏检的物体，则直接保留那些匹配到当前帧背景中的像素点...
                if ((temp_obj_id == 0 && !total_lost) || std::find(match_cur_objs.begin(),match_cur_objs.end(),temp_obj_id) == match_cur_objs.end()) continue;

                undistortedPts(cv::Point2f(cur_u,cur_v),un_pt_cur,m_camera[0]);
                undistortedPts(cv::Point2f(prev_u,prev_v),un_pt_prev,m_camera[0]);

                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_statu;
                // 归一化平面点三维坐标（z为1），二维像素坐标，该特征点在当前帧相对于前一帧的归一化平面点坐标的二维速度。
                // 注意，这些像素点仅仅会用于位姿的初始化估计，因此不需要用到速度，直接给定上一帧点的归一化坐标！！
                // 物体上的像素点只会用于物体的运动估计，只使用左相机的两帧观测，因此这里后三个量不会被用到，这里我们使用最后一个量（原本应该是保存所关联的上一帧的点全局物体id）来保存上一帧像素点的深度！！
                // xyz_uv_velocity_statu << un_pt_cur.x, un_pt_cur.y, z, cur_u, cur_v, un_pt_prev.x, un_pt_prev.y, id_obj;

                // 当前帧归一化坐标，当前帧点深度，当前帧像素坐标，上一帧归一化坐标，上一帧点深度
                xyz_uv_velocity_statu << un_pt_cur.x, un_pt_cur.y, depth, cur_u, cur_v, un_pt_prev.x, un_pt_prev.y, ptr_pixel[(i*3+2)];

                vector<pair<int,Vector8d>> temp_pair;
                temp_pair.emplace_back(0,xyz_uv_velocity_statu);
                // 像素点的全局id也都是为负的！
                if(no_track)
                {
                    map<int,vector<pair<int,Vector8d>>> temp_map;
                    // 像素跟踪点的id是负的
                    temp_map.insert(make_pair(temp_pt_id,temp_pair));
                    FinalTrackObjFea.insert(make_pair(id_obj,temp_map));
                    no_track = false;
                    --temp_pt_id;
                }
                else
                {
                    // FinalTrackObjFea[id_obj][temp_pt_id--].emplace_back(0,xyz_uv_velocity_statu);
                    FinalTrackObjFea[id_obj].insert(make_pair(temp_pt_id,temp_pair));
                    --temp_pt_id;
                }
                    
                --num_pixel;
            }

            // 如果关联点数实在太少，则放弃关联，当前物体作为新物体
            if(FinalTrackObjFea[id_obj].size() < 8)
            {
                dyn_obj_id_del.insert(id_tracked_obj);
            }
        }

        if(!dyn_obj_id_del.empty())
        {
            int index;
            // dyn_obj_id_del中元素是从大到小排序，所以下面可以删除FinalTrackObj的元素
            for(auto it = dyn_obj_id_del.begin(); it != dyn_obj_id_del.end(); ++it)
            {
                int gl_obj_id = FinalTrackObj[*it].first;
                // cout << "erased gl_obj_id: " << gl_obj_id << endl;
                vector<int> &match_cur_objs = FinalTrackObj[*it].second;
                for(int i = 0; i < match_cur_objs.size(); ++i)
                {
                    int cur_obj_id = match_cur_objs[i];
                    // 该物体在当前帧有漏检点怎么办？需要在这里把那些跟踪的当前帧背景点删除
                    if(cur_obj_id != 0)
                    {
                        // 当前帧的关联物体归为新物体
                        new_objs_cur.push_back(cur_obj_id);
                        FinalTrackCurObj.erase(cur_obj_id);
                    }
                    else
                    {
                        for(auto &pt: TrackObjFeaFrame[0])
                        {
                            if(pt.second[0].second(7) == gl_obj_id)
                            {
                                index = gl_id_index_map[pt.first];
                                if(index > 0)
                                {
                                    statusLeftRIght[(index-1)] = 0;
                                }
                                else
                                {
                                    index = -1*index;
                                    if(add_new_sift_in_next_frame)
                                        status_sift[index] = 0;
                                    else
                                    {
                                        // 是否保留背景跟踪sift点为新点？
                                        status_sift[index] = 0;

                                    }
                                }
                            }
                        }
                    }
                }

                // 这里不能删除FinalTrackObj中对应的整个子vector，不然后续cur_dyn_objs中的id无法正确对应到FinalTrackObj中的元素
                // 但可以清空其中的元素
                FinalTrackObj[*it].second.clear();
                if(FinalTrackObjFea.find(gl_obj_id) != FinalTrackObjFea.end())
                {
                    FinalTrackObjFea[gl_obj_id].clear();
                    FinalTrackObjFea.erase(gl_obj_id);
                }

                assert(matched_prev_objs_id.find(gl_obj_id) != matched_prev_objs_id.end());
                matched_prev_objs_id.erase(gl_obj_id);
                
                // 该物体在当前帧可能是全漏检
                auto iter_ = std::find(detect_lost_objs_cur.begin(), detect_lost_objs_cur.end(), gl_obj_id);
                if(iter_ != detect_lost_objs_cur.end())
                {
                    detect_lost_objs_cur.erase(iter_);
                }
                
                FinalLostObjPrev.push_back(gl_obj_id);

                //vector这类内存连续的容器，其iterator可以进行加减法；而map或set这类的iterator只能用++或--的操作
                auto iter = cur_dyn_objs.find(*it);
                if(iter != cur_dyn_objs.end())
                    cur_dyn_objs.erase(iter);
            }
        }
    }

    cout << "num of new objs cur during objs-matching: " << new_objs_cur.size() << endl;
    // for(auto iter:new_objs_cur)
    // {
    //     cout << "id of new objs: " << iter << endl;
    // }
    cout << "num of dyn tracked objs during objs-matching: " << cur_dyn_objs.size() << endl;
    // for(auto iter:cur_dyn_objs)
    // {
    //     if(FinalTrackObj[iter].second.size() > 1) cout << "There is more than 1 cur objs matched with the same prev obj!" << endl;
    //     for(auto it: FinalTrackObj[iter].second)
    //         cout << "id of dyn objs: " << it << endl;
    // }
    cout << "num of static tracked objs during objs-matching: " << cur_stat_objs.size() << endl;
    // for(auto iter:cur_stat_objs)
    // {
    //     cout << "id of sta objs: " << iter.first << endl;
    // }
    
    assert(new_objs_cur.size() + FinalTrackCurObj.size() == (num_detect_obj-1));
    // 至此，物体的关联函数结束！后续在完成RANSAC的位姿初步估计后，把外点也加入剔除的范围，然后对cur_sift等一系列变量进行删减
    
    // ROS_DEBUG("Matching objects costs: %fms", t_match_objs.toc());
    printf("Matching objects costs: %fms\n", t_match_objs.toc());

    final_assign_id_cur_objs.clear();
    prev_objs_assign_again.clear();
    matched_prev_objs_id.clear();
    association_mat.clear();
    pts_two_bg.clear();
    cur_obj_assign_again.clear();
    lose_fea_prev_bg.clear();
}

// cal_3D_reproj_err的目的是指定是否进行3D刚体欧氏变换（根据预测的速度模型）并计算误差，它与has_pred_motion息息相关，它是专门为系统前两帧的所有物体、上一帧新出现的物体以及一种特殊情况设置的：
// 上一帧的新物体（特殊的是系统第2帧，连相机的运动先验都没有），或者漏检某物体（其全部或部分特征点没有并校正为obj id（因为没有与上上帧形成匹配），因此它们在上一帧最终被当成背景点）。当前帧没有漏检该物体，但是无法知道该物体是否为动态（或者直接连相机运动都不知道），因此不能直接用相机的运动模型来进行3D变换。
bool FeatureTracker::check_match_two_objs(bool use_fea, bool cal_3D_pred_err, map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &fea_cur_obj, const int id_checked_prev_obj, const int id_checked_cur_obj,
                                            map<int, pair<vector<Vec2f>,vector<Vec2f>>> &match_pts_of_prev_obj, map<int, pair<vector<float>,vector<float>>> &dep_match_pts_of_prev_obj,
                                            vector<vector<int>> &final_assign_id_cur_objs, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map_2, const vector<Vector3d> &Ps, 
                                            const vector<Matrix3d> &Rs, bool has_pred_motion, Vector3d &P_12, Matrix3d &R_12, bool &cur_sta_obj, float &ave_depth, 
                                            map<int,vector<int>> &assign_prve_id, bool direct_erase_pts, bool cal_ave_depth_prev_pts)
{
    int match_index;
    cur_sta_obj = false;

    float factor_fea = 1.0;

    vector<int> fea_id;

    // 如果特征关联点数足够多
    if (use_fea)
    {
        // 使用特征点进行关联校验前清空这两个变量
        if (!match_pts_of_prev_obj.empty()) 
        {
            match_pts_of_prev_obj.clear();
            dep_match_pts_of_prev_obj.clear();
        }
        
        factor_fea = factor_fea * 0.85;

        int num_fea = fea_cur_obj.size();
        vector<float> temp_dep(num_fea,0);
        vector<Vec2f> temp_fea(num_fea,Vec2f(0,0));;
        
        pair<vector<float>,vector<float>> temp_pair_dep(temp_dep,temp_dep);
        pair<vector<Vec2f>,vector<Vec2f>> temp_pair_fea(temp_fea,temp_fea);

        // 清除元素，但是保留了开辟的内存，避免频繁重新分配内存
        temp_pair_dep.first.clear();
        temp_pair_dep.second.clear();
        temp_pair_fea.first.clear();
        temp_pair_fea.second.clear();

        if (id_checked_cur_obj == -1)
        {
            match_index = 0;
            float depth_cur, depth_prev;
            
            for (auto &iter: fea_cur_obj)
            {
                // 有些上一帧的静态物体，可能在初步物体关联期间已经排除了其中的特征点异常匹配；后面如果该静态物体需要和别的物体进行融合，则这里就直接先把该异常匹配排除，免得之后进行计算两个点集的相似性时还要再次排除它！
                int index = gl_id_index_map[iter.first];
                if (index > 0) 
                {
                    if(statusLeftRIght[index-1] == 0)
                        continue;
                }
                else
                {
                    if(status_sift[-index] == 0)
                        continue;
                }

                depth_cur = iter.second[0].second(2,0);
                if (depth_cur <= 0) continue;

                // 物体点在每一帧中不一定有立体匹配。但是物体点一定有深度值
                if (iter.second.size() == 2)
                {
                    depth_prev = iter.second[1].second(2,0);
                }
                else
                {
                    if (index > 0)
                        depth_prev = prev_FAST_dep[index-1];
                    else
                        depth_prev = prev_sift_dep[-index];
                }
                
                // 有些点可能没有深度值，但这些匹配点可以参与2D点集的相似度计算
                if (depth_prev <= 0) continue;

                // 归一化平面坐标
                float pts1_x = iter.second[0].second(0,0);
                float pts1_y = iter.second[0].second(1,0);
                // match_pts_of_prev_obj[match_index].second.emplace_back(pts1_x, pts1_y);
                temp_pair_fea.second.emplace_back(pts1_x, pts1_y);

                float prev_x = pts1_x - iter.second[0].second(5,0) * (cur_time - prev_time);
                float prev_y = pts1_y - iter.second[0].second(6,0) * (cur_time - prev_time);
                // match_pts_of_prev_obj[match_index].first.emplace_back(prev_x, prev_y);
                temp_pair_fea.first.emplace_back(prev_x, prev_y);

                // 这种寻找dep所在元素id的效率太慢，尤其是当匹配点较多时
                // vector<int>::iterator iter_;
                // iter_ = std::find(ids_FAST.begin(), ids_FAST.end(), g_id_pt);
                // if(iter_ != ids_FAST.end())
                // {
                //     int index = std::distance(ids_FAST.begin(),iter_);
                //     if (prev_FAST_dep[index] > 0 && cur_FAST_dep[index] > 0) 
                //     {
                //         dep_match_pts_of_prev_obj[match_index].first.push_back(prev_FAST_dep[index]);
                //         dep_match_pts_of_prev_obj[match_index].second.push_back(cur_FAST_dep[index]);
                //     }
                // }
                // else
                // {
                //     iter_ = std::find(ids_sift.begin(), ids_sift.end(), g_id_pt);
                //     int index = std::distance(ids_sift.begin(),iter_);
                //     if (prev_sift_dep[index] > 0 && cur_sift_dep[index] > 0)
                //     {
                //         dep_match_pts_of_prev_obj[match_index].first.push_back(prev_sift_dep[index]);
                //         dep_match_pts_of_prev_obj[match_index].second.push_back(cur_sift_dep[index]);
                //     }
                // }

                temp_pair_dep.first.push_back(depth_prev);
                temp_pair_dep.second.push_back(depth_cur);
                
                fea_id.push_back(iter.first);
            }

            if(!fea_id.empty())
            {
                dep_match_pts_of_prev_obj.insert(make_pair(match_index,temp_pair_dep));
                match_pts_of_prev_obj.insert(make_pair(match_index,temp_pair_fea));
            }
        }
        // 给定的当前物体的特征点集是否与上一帧多个物体相关联
        else
        {
            // 是否已经分配好了当前物体的匹配点（按照上一帧的物体id）
            if(!assign_prve_id.empty())
            {
                assert(assign_prve_id.find(id_checked_prev_obj) != assign_prve_id.end());
                float depth_prev, depth_cur;
                for(int i = 0; i < assign_prve_id[id_checked_prev_obj].size(); i++)
                {
                    int g_id_pt = assign_prve_id[id_checked_prev_obj][i];
                    
                    int index = gl_id_index_map[g_id_pt];
                    if (index > 0) 
                    {
                        if(statusLeftRIght[index-1] == 0)
                            continue;
                    }
                    else
                    {
                        if(status_sift[-index] == 0)
                            continue;
                    }

                    Eigen::Matrix<double, 8, 1> &pt = fea_cur_obj[g_id_pt][0].second;
                    depth_cur = pt(2);
                    if (depth_cur <= 0) continue;

                    // 如果该点有立体匹配
                    if (fea_cur_obj[g_id_pt].size() == 2) 
                    {
                        depth_prev = fea_cur_obj[g_id_pt][1].second(2);
                    }
                    else
                    {
                        if(index > 0)
                            depth_prev = prev_FAST_dep[index-1];
                        else
                            depth_prev = prev_sift_dep[-index];
                    }
                    
                    if (depth_prev <= 0) continue;

                    // dep_match_pts_of_prev_obj[id_checked_prev_obj].first.push_back(depth_prev);
                    // dep_match_pts_of_prev_obj[id_checked_prev_obj].second.push_back(depth_cur);
                    temp_pair_dep.first.push_back(depth_prev);
                    temp_pair_dep.second.push_back(depth_cur);

                    // 给定左图像中匹配点的归一化平面坐标
                    float pts1_x = pt(0);
                    float pts1_y = pt(1);
                    // match_pts_of_prev_obj[id_checked_prev_obj].second.emplace_back(pts1_x, pts1_y);
                    temp_pair_fea.second.emplace_back(pts1_x, pts1_y);

                    float prev_x = pts1_x - pt(5) * (cur_time - prev_time);
                    float prev_y = pts1_y - pt(6) * (cur_time - prev_time);
                    // match_pts_of_prev_obj[id_checked_prev_obj].first.emplace_back(prev_x, prev_y);
                    temp_pair_fea.first.emplace_back(prev_x, prev_y);
                    fea_id.push_back(g_id_pt);
                }

                // if(dep_match_pts_of_prev_obj.find(match_index) == dep_match_pts_of_prev_obj.end())
                {
                    dep_match_pts_of_prev_obj.insert(make_pair(id_checked_prev_obj,temp_pair_dep));
                    match_pts_of_prev_obj.insert(make_pair(id_checked_prev_obj,temp_pair_fea));
                }
            }
            else
            {
                float depth_prev,depth_cur;
                for (auto &iter: fea_cur_obj)
                {
                    // 需要确认该点的obj id为需要的id_checked_prev_obj！
                    if (iter.second[0].second(7,0) != id_checked_prev_obj) continue;
                    
                    int g_id_pt = iter.first;
                    
                    int index = gl_id_index_map[g_id_pt];
                    if (index > 0) 
                    {
                        if(statusLeftRIght[index-1] == 0)
                            continue;
                    }
                    else
                    {
                        if(status_sift[-index] == 0)
                            continue;
                    }

                    float depth_cur  = iter.second[0].second(2,0);
                    if (depth_cur <= 0) continue;

                    // 该点在上一帧是否有立体匹配
                    if (iter.second.size() == 2)
                    {
                        depth_prev = iter.second[1].second(2,0);
                    }
                    else
                    {
                        if(index > 0)
                            depth_prev = prev_FAST_dep[index-1];
                        else
                            depth_prev = prev_sift_dep[-index];
                    }
                    
                    if (depth_prev <= 0) continue;

                    // dep_match_pts_of_prev_obj[id_checked_prev_obj].first.push_back(depth_prev);
                    // dep_match_pts_of_prev_obj[id_checked_prev_obj].second.push_back(depth_cur);
                    temp_pair_dep.first.push_back(depth_prev);
                    temp_pair_dep.second.push_back(depth_cur);

                    // 给定左图像中匹配点的去畸变归一化平面坐标
                    float pts1_x = iter.second[0].second(0,0);
                    float pts1_y = iter.second[0].second(1,0);
                    // match_pts_of_prev_obj[id_checked_prev_obj].second.emplace_back(pts1_x, pts1_y);
                    temp_pair_fea.second.emplace_back(pts1_x, pts1_y);

                    float prev_x = pts1_x - iter.second[0].second(5,0) * (cur_time - prev_time);
                    float prev_y = pts1_y - iter.second[0].second(6,0) * (cur_time - prev_time);
                    // match_pts_of_prev_obj[id_checked_prev_obj].first.emplace_back(prev_x, prev_y);
                    temp_pair_fea.first.emplace_back(prev_x, prev_y);

                    fea_id.push_back(iter.first);
                }

                // if(dep_match_pts_of_prev_obj.find(match_index) == dep_match_pts_of_prev_obj.end())
                {
                    dep_match_pts_of_prev_obj.insert(make_pair(id_checked_prev_obj,temp_pair_dep));
                    match_pts_of_prev_obj.insert(make_pair(id_checked_prev_obj,temp_pair_fea));
                }
            }
            match_index = id_checked_prev_obj;
        }   
    }
    // 如果关联的特征点不够多，则使用上一帧采样的物体像素点进行关联程度计算
    else
    {
        factor_fea = factor_fea * 0.98;
        int num_prev_pixel = 0;
        // 是否需要先要寻找指定的两个物体之间的配对像素点，要注意这时是采用当前帧的物体临时id作为map的key
        if (match_pts_of_prev_obj.empty())
            num_prev_pixel = track_pixels_one_prev_obj(id_checked_prev_obj, seg_map, flow_map, depth_map_2, match_pts_of_prev_obj, dep_match_pts_of_prev_obj);
        else
        {
            for (const auto &iter:dep_match_pts_of_prev_obj)
            {
                num_prev_pixel += iter.second.first.size();
            }
        }

        if (match_pts_of_prev_obj.find(id_checked_cur_obj) == match_pts_of_prev_obj.end()) 
        {
            //final_assign_id_cur_objs[id_checked_cur_obj].push_back(-1);
            return false;
        }
        // 可能是当前帧的某个物体部分漏检，则要求该部分的像素点不少于整个物体有效像素跟踪数的1/3，否则直接抛弃这部分关联
        // 或者该物体在上一帧的采样像素点中有效跟踪数太少。注意，由于不是所有物体的采样点数都能够达到规定的阈值，因此不宜用该阈值的一半来作为限制
        // else if(match_pts_of_prev_obj[id_checked_cur_obj].second.size() < 0.3 * num_prev_pixel || num_prev_pixel < NUM_SAMPLED_PIXEL_OBJ*0.5)
        else if(match_pts_of_prev_obj[id_checked_cur_obj].second.size() < 0.3 * num_prev_pixel || num_prev_pixel < 25)
        {
            //final_assign_id_cur_objs[id_checked_cur_obj].push_back(-1);
            return false;
        }
        match_index = id_checked_cur_obj;
    }

    // 两帧的相机全局位姿（w_T_cam)
    Vector3d Ps_c1;
    Vector3d Ps_c2;
    Matrix3d Rs_c1;
    Matrix3d Rs_c2;
    
    bool has_cam_motion = false;
    bool has_obj_motion = true;

    if(has_pred_motion)
    {
        if(!Ps.empty())
        {
            Ps_c1 = Ps[0];
            Ps_c2 = Ps[1];
            Rs_c1 = Rs[0];
            Rs_c2 = Rs[1];
            has_cam_motion = true;
        }
        else
        {
            has_pred_motion = false;
        }
    }

    bool prev_sta_obj = false;
    // 给定相机坐标下的 该物体的帧间运动，即将该物体在上一帧相机坐标系下的点 变换到 当前帧相机坐标下的该对应点
    auto iter = RP_objs_pred.find(id_checked_prev_obj);
    if (iter != RP_objs_pred.end())
    {
        // 注意，RP_objs_pred保存的就是该物体的点在两帧相机坐标下间的运动变换，在基于运动的FAST点位姿预测之前就已经计算好了，这里无需再次变换
        // R_12 = Rs_c2.transpose()*RP_objs_pred[id_checked_prev_obj].first*Rs_c1;
        // P_12 = Rs_c2.transpose()*(RP_objs_pred[id_checked_prev_obj].first*Ps_c1 + RP_objs_pred[id_checked_prev_obj].second - Ps_c2);
        R_12 = RP_objs_pred[id_checked_prev_obj].first;
        P_12 = RP_objs_pred[id_checked_prev_obj].second;
    }
    else 
    {
        // 如果待验证关联的上一帧物体不是动态物体
        auto iter_ = status_objs_prev.find(id_checked_prev_obj);
        // 上一帧新出现的物体（例如当前帧为系统第二帧，则首帧出现的所有物体都是新物体）。status_objs_prev中为0的物体一定在RP_objs_prev中有区别于相机的运动模型
        // 上一帧如果是背景点（且当前帧为物体，即为校验上一帧是否漏检），两帧之间还没有物体运动估计（系统前2帧），或者上一帧为新物体
        if (id_checked_prev_obj == 0 || !has_cam_motion || (iter_ != status_objs_prev.end() && iter_->second == 2))
        {
            has_pred_motion = false;
            // 注意，此函数默认不会进行两帧之间的纯背景点的关联验证，因此如果上一帧的被验证物体为背景，则当前帧的被验证物体一定不再是背景
            if(id_checked_prev_obj == 0 && id_checked_cur_obj != -1)
                assert(id_checked_cur_obj != 0);
        }
        // 如果校验的上一帧物体为已有的静态物体(1)
        // 另外，系统第2帧时没有相机的运动先验，则此时所有物体都不会有运动先验，这就需要函数调用者给定此时的has_pred_motion为false！
        else if(has_cam_motion)
        {
            R_12 = Rs_c2.transpose()*Rs_c1;
            P_12 = Rs_c2.transpose()*(Ps_c1 - Ps_c2);
            prev_sta_obj = true;
        }
    }
    // 计算两个匹配点集合之间的匹配程度
    // 这里的has_pred_motion包含了无法得知上一帧背景中的漏检物体是否运动，根据此函数调用者给定的值来决定是否使用相机运动来进行3D点的变换
    vector<float> dist;
    set<int> outliers;
    if (has_pred_motion && cal_3D_pred_err)
    {
        // 默认如果使用特征点匹配进行关联，则给定的都是特征点的归一化平面坐标
        bool is_pixel = !use_fea;
        outliers = calcu_2d_3d_pts_dist(dist, is_pixel, match_pts_of_prev_obj[match_index].first, ave_depth, match_pts_of_prev_obj[match_index].second, true, true, 
                                        P_12, R_12, dep_match_pts_of_prev_obj[match_index].first, dep_match_pts_of_prev_obj[match_index].second, cal_ave_depth_prev_pts);
        // 如果当前背景点和上一帧的物体点一样比较集中(2D以及3D），且速度模型的预测点与匹配点的平均3D距离小于阈值
        // 2D点的分布方差会跟成像距离有关，而3D的分布方差不受距离影响。
        //if (dist[1] < 1.4 * dist[0] && dist[3] < 1.25 * dist[2] && dist[2] < 6 && dist[4] < AVE_DIST_3D_PTS_THRES)
        // TODO: AVE_DIST_3D_PTS_THRES这个变量的值需要仔细设置
        
        if(dist.empty()) return false;
        
        bool valid = false;
        if(dist.size() == 2)
        {
            if(min(dist[1],dist[0]) > 0 && max(dist[1],dist[0])/min(dist[1],dist[0]) < factor_fea*1.35) valid = true;
        }
        else if(dist.size() == 5)
        {
            if(min(dist[1],dist[0]) > 0 && min(dist[3],dist[2]) > 0)
                if(max(dist[1],dist[0])/min(dist[1],dist[0]) < factor_fea*1.5 && max(dist[3],dist[2])/min(dist[3],dist[2]) < factor_fea*1.4 && dist[4] < factor_fea*AVE_DIST_3D_PTS_THRES)
                    valid = true;
        }

        if(valid)
        {
            // 这里的-1是一种特殊情况，是当前帧多个物体要融合时，验证它们的合集与上一帧物体的匹配关系，并不要求加入匹配结果
            if (id_checked_cur_obj != -1) 
            {
                // debug
                // if(id_checked_cur_obj >= final_assign_id_cur_objs.size())
                // {
                //     cout << "id_checked_cur_obj: " << id_checked_cur_obj << endl;
                //     cout << "id_checked_prev_obj: " << id_checked_prev_obj << endl;
                //     cout << "size of vec: " << final_assign_id_cur_objs.size() << endl;
                //     cout << "capacity of vec: " << final_assign_id_cur_objs[id_checked_cur_obj].capacity() << endl;

                //     for(auto pt: fea_id)
                //     {
                //         int index = gl_id_index_map[pt];
                //         cout << "index of fea in gl_id_index_map: " << index << endl;
                //     }
                // }
                
                final_assign_id_cur_objs[id_checked_cur_obj].push_back(id_checked_prev_obj);
            }

            // 如果上一帧该物体是静态，且验证的是关联点，则将其中的异常关联点去除
            if (prev_sta_obj && use_fea) 
            {
                // 如果上一帧的静态物体通过了3D重投影验证，则认为它在当前帧仍然为静态
                cur_sta_obj = true;
                // 如果静态物体的匹配点中有极端异常值(其基于预测的运动变换后的预测点与匹配点的距离过大，说明要么匹配错误，要么点的深度估计有问题），则直接把该点的status置为false
                if (!outliers.empty())
                {
                    for(auto &pt_index: outliers)
                    {
                        int index = gl_id_index_map[fea_id[pt_index]];
                        if(index > 0)
                        {
                            statusLeftRIght[index-1] = 0;
                        }
                        else
                        {
                            status_sift[-1*index] = 0;
                        }
                        // 是否要直接从给定的物体点集map中删除该跟踪点。当此函数运行在多线程时，应该避免改变多线程共享的map，否则会导致其重新分配地址而出现内存错误
                        if (direct_erase_pts) fea_cur_obj.erase(fea_id[pt_index]);
                    }
                }
            }
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        bool is_pixel = !use_fea;
        // 赋值为-1.0则不会计算物体在当前帧的平均深度
        if(!cal_ave_depth_prev_pts)
            ave_depth = -1.0;
        
        // 新物体或上一帧的漏检物体无法进行3D变换预测，则只计算2D和3D匹配点集合的聚集性的相似程度
        outliers = calcu_2d_3d_pts_dist(dist, is_pixel, match_pts_of_prev_obj[match_index].first, ave_depth, match_pts_of_prev_obj[match_index].second, true, false, 
                            P_12, R_12, dep_match_pts_of_prev_obj[match_index].first, dep_match_pts_of_prev_obj[match_index].second, cal_ave_depth_prev_pts);
        //if (dist[1] < 1.4 * dist[0] && dist[3] < 1.25 * dist[2] && dist[2] < 6)

        if(dist.empty()) return false;
        bool succ = false;
        if(dist.size() == 2 && max(dist[1],dist[0])/min(dist[1],dist[0]) < factor_fea*1.35) succ = true;

        if (dist.size() == 4 && max(dist[1],dist[0])/min(dist[1],dist[0]) < factor_fea*1.5 && max(dist[3],dist[2])/min(dist[3],dist[2]) < factor_fea*1.4) succ = true;
        if(succ)
        {
            // if的原因同上
            if (id_checked_cur_obj != -1) final_assign_id_cur_objs[id_checked_cur_obj].push_back(id_checked_prev_obj);
            // 如果是上一帧的漏检点 或 当前帧的漏检点 的完成了匹配，则要把其中的异常点去除！
            if(use_fea && (id_checked_prev_obj == 0 || id_checked_cur_obj == 0) && !outliers.empty())
            {
                for(auto &pt_index: outliers)
                {
                    int index = gl_id_index_map[fea_id[pt_index]];
                    if(index > 0)
                    {
                        statusLeftRIght[index-1] = 0;
                    }
                    else
                    {
                        status_sift[-1*index] = 0;
                    }

                    if (direct_erase_pts) fea_cur_obj.erase(fea_id[pt_index]);
                }
            }
            return true;
        }
        else
            return false;
        
        // else
        // {
        //     if (final_assign_id_cur_objs[id_checked_cur_obj].size() > 0 && final_assign_id_cur_objs[id_checked_cur_obj].back() != -1)
        //         // 如果不满足，则放弃该物体
        //         final_assign_id_cur_objs[id_checked_cur_obj].push_back(-1);
        // }
    }
}

// 检查关联的物体是不是静态的，如果是，还要排除其中的异常点
bool FeatureTracker::check_stat_obj(bool use_fea, map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &fea_cur_obj, 
                                    const int id_checked_prev_obj, const vector<int> &matched_cur_objs, const Vector3d &P_12, const Matrix3d &R_12, 
                                    float &ave_depth_cur, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map_2, bool direct_erase_pts)               
{
    bool is_stat = false;
    float factor_fea = 1.0;
    vector<Vec2f> match_pts_of_prev_obj,match_pts_of_cur_obj;
    vector<float> dep_pts_of_prev_obj, dep_pts_of_cur_obj;
    vector<int> fea_id;
    if(use_fea)
    {
        factor_fea = factor_fea * 0.9;
        float depth_prev, depth_cur;
        for(auto &iter: fea_cur_obj)
        {
            int index = gl_id_index_map[iter.first];
            if (index > 0 && statusLeftRIght[index-1] == 0) continue;
            if (index <= 0 && status_sift[-1*index] == 0) continue;

            depth_cur  = iter.second[0].second(2,0);
            if (depth_cur <= 0) continue;

            if (iter.second.size() == 2)
            {
                depth_prev = iter.second[1].second(2,0);
            }
            else
            {
                if(index > 0)
                    depth_prev = prev_FAST_dep[index-1];
                else
                    depth_prev = prev_sift_dep[-index];
            }
            
            if (depth_prev <= 0) continue;

            dep_pts_of_prev_obj.push_back(depth_prev);
            dep_pts_of_cur_obj.push_back(depth_cur);

            // 给定左图像中匹配点的去畸变归一化平面坐标!!!
            float cur_x = iter.second[0].second(0,0);
            float cur_y = iter.second[0].second(1,0);
            
            float prev_x = cur_x - iter.second[0].second(5,0) * (cur_time - prev_time);
            float prev_y = cur_y - iter.second[0].second(6,0) * (cur_time - prev_time);

            match_pts_of_prev_obj.emplace_back(prev_x,prev_y);
            match_pts_of_cur_obj.emplace_back(cur_x,cur_y);

            fea_id.push_back(iter.first);
        }
    }
    // 如果关联的特征点不够多，则使用上一帧采样的物体像素点进行关联程度计算
    else
    {
        // factor_fea = factor_fea * 0.95;
        map<int, pair<vector<Vec2f>,vector<Vec2f>>> map_match_pts_of_prev_obj;
        map<int, pair<vector<float>,vector<float>>> map_dep_match_pts_of_prev_obj;

        // 这时先要寻找指定的两个物体之间的配对像素点，要注意这时是采用当前帧的物体临时id作为map的key
        int num_prev_pixel = track_pixels_one_prev_obj(id_checked_prev_obj, seg_map, flow_map, depth_map_2, map_match_pts_of_prev_obj, map_dep_match_pts_of_prev_obj);
        for (auto &pt_id:matched_cur_objs)
        {
            if (map_match_pts_of_prev_obj.find(pt_id) == map_match_pts_of_prev_obj.end()) 
            {
                //final_assign_id_cur_objs[id_checked_cur_obj].push_back(-1);
                return false;
            }
            else if(map_match_pts_of_prev_obj[pt_id].second.size() < 0.4 * num_prev_pixel || num_prev_pixel < 20)
            {
                //final_assign_id_cur_objs[id_checked_cur_obj].push_back(-1);
                return false;
            }
            vector<Vec2f> &pts_prev = map_match_pts_of_prev_obj[pt_id].first;
            vector<Vec2f> &pts_cur  = map_match_pts_of_prev_obj[pt_id].second;
            vector<float> &dep_prev = map_dep_match_pts_of_prev_obj[pt_id].first;
            vector<float> &dep_cur  = map_dep_match_pts_of_prev_obj[pt_id].second;
            // vector的insert函数调用格式之一为vec.insert(pos, iter_begin, iter_end)，注意，一定要指定vec所要插入的位置pos
            // vector的insert是值复制传递，因此如果vector中的元素类型是自定义class，则必须保证该class拥有复制构造函数！
            // 其实STL中所有容器的内容填充都是value-copy！！如果想实现内存共用，只能用std::move进行内存转移（即insert等函数或者operator=实现了右值参数的实现！）！
            match_pts_of_prev_obj.insert(match_pts_of_prev_obj.end(),pts_prev.begin(),pts_prev.end());
            match_pts_of_cur_obj.insert(match_pts_of_cur_obj.end(),pts_cur.begin(),pts_cur.end());
            dep_pts_of_prev_obj.insert(dep_pts_of_prev_obj.end(),dep_prev.begin(),dep_prev.end());
            dep_pts_of_cur_obj.insert(dep_pts_of_cur_obj.end(),dep_cur.begin(),dep_cur.end());
        }
        //map_match_pts_of_prev_obj.clear();
        //map_dep_match_pts_of_prev_obj.clear();
    }

    vector<float> dist;
    set<int> outliers_3D_reproj;
    // 默认如果使用特征点匹配进行关联，则给定的都是特征点的归一化平面坐标
    bool is_pixel = !use_fea;

    outliers_3D_reproj = calcu_2d_3d_pts_dist(dist, is_pixel, match_pts_of_prev_obj, ave_depth_cur, match_pts_of_cur_obj, true, true, 
                        P_12, R_12, dep_pts_of_prev_obj, dep_pts_of_cur_obj);
    // 如果当前背景点和上一帧的物体点一样比较集中(2D以及3D），且速度模型的预测点与匹配点的平均3D距离小于阈值
    // 2D点的分布方差会跟成像距离有关，而3D的分布方差不受距离影响。
    //if (dist[1] < 1.4 * dist[0] && dist[3] < 1.25 * dist[2] && dist[2] < 6 && dist[4] < AVE_DIST_3D_PTS_THRES)
    // TODO: AVE_DIST_3D_PTS_THRES这个变量的值需要仔细设置

    bool valid = false;
    // 如果没有3D重投影距离误差，则不认为是静态物体
    if(dist.size() == 5)
    {
        if(max(dist[1],dist[0])/min(dist[1],dist[0]) < factor_fea*1.5 && max(dist[3],dist[2])/min(dist[3],dist[2]) < factor_fea*1.35 && dist[4] < factor_fea*AVE_DIST_3D_PTS_THRES)
            valid = true;
    }
    
    if(valid)
    {
        if(use_fea && !outliers_3D_reproj.empty())
        {
            for(auto &index_pt: outliers_3D_reproj)
            {
                int index = gl_id_index_map[fea_id[index_pt]];
                if(index > 0)
                {
                    statusLeftRIght[index-1] = 0;
                }
                else
                {
                    status_sift[-1*index] = 0;
                }
                if (direct_erase_pts) fea_cur_obj.erase(fea_id[index_pt]);
            }
        }
        return true;
    }
    else
        return false;
}


// 根据上一帧采样的像素点 和 两帧之间的flow_map图，找到当前帧中的匹配点。
// 有一种情况会导致前后两帧图像中某物体的特征点或像素点的匹配都比较少：目标汽车在前后两帧中发生较大的旋转，比如路口的掉头。
// 这时就需要相机的频率足够高！或者在运动模型中考虑到物体在具体场景下可能的运动（根据汽车的转向灯），比如物体可能会调转方向等。本工作中暂时不考虑这些，如果匹配不到，就当作新物体出现吧！
int FeatureTracker::track_pixels_one_prev_obj(int prev_glob_obj_id, const cv::Mat &seg_map, const cv::Mat &flow_map, const cv::Mat &depth_map, 
                                        map<int, pair<vector<Vec2f>, vector<Vec2f>>> &match_pts, map<int, pair<vector<float>,vector<float>>> &match_pts_depth)
{
    int num_prev_pixel;
    map<int, float*>::iterator iter = pixel_objs_prev.find(prev_glob_obj_id);

    if (iter == pixel_objs_prev.end())
        num_prev_pixel = 0;
    else
    {
        float* ptr_pixel = iter->second;
        num_prev_pixel = (int)ptr_pixel[-1];
        int total_pixel = num_prev_pixel;
        cv::Point2f tmp_undist_pt, pt_match;
        float x, y, x_match, y_match;
        Vec2b pt_info;
        // 注意，上一帧保存的三维像素点中，x和y是像素坐标，而z为深度值（注意不是视差值！）
        Point3f pts1;

        int obj_id;
        for (int i = 0; i < total_pixel; ++i)
        {   
            pts1.x = ptr_pixel[i*3];
            pts1.y = ptr_pixel[i*3+1];
            pts1.z = ptr_pixel[i*3+2];
            x = pts1.x;
            y = pts1.y;
            // flow_map表示为 coord_img2 - coord_img1。但是0和1分别表示x还是y方向的光流值？
            x_match = x + flow_map.at<Vec2f>(y,x)[0];
            y_match = y + flow_map.at<Vec2f>(y,x)[1];
            // x_match = x + flow_map.at<Vec2f>(y,x)[1];
            // y_match = y + flow_map.at<Vec2f>(y,x)[0];

            // cout << "flow_x: " << flow_map.at<Vec2f>(y,x)[0] << endl;
            // cout << "flow_y: " << flow_map.at<Vec2f>(y,x)[1] << endl;
            pt_match.x = x_match;
            pt_match.y = y_match;
            // 位置或深度不符合要求的像素点不算
            if (!inBorder(pt_match)) 
            {
                --num_prev_pixel;
                continue;
            }
            pt_info = seg_map.at<Vec2b>(y_match, x_match);
            uchar cls_label = pt_info[0];
            if (cls_label == 1 || cls_label == 2 || cls_label == 4 || cls_label == 7) continue;

            float disp = depth_map.at<float>(y_match, x_match);
            if (disp <= 0)
            {
                --num_prev_pixel;
                continue;
            }

            float depth = mbf/disp;

            if (cls_label == 0)
            {
                if(depth >= mThDepthBg || depth < mMinDepthPt || x_match <= bg_left_border_left_img)
                {
                    --num_prev_pixel;
                    continue;
                }
                // undistortedPts(pt_match, tmp_undist_pt, m_camera[0]);
                // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                // if (err <= 0) 
                // {
                //     --num_prev_pixel;
                //     continue;
                // }
            }
            else
            {
                if(depth >= mThDepthObj || depth < mMinDepthPt || x_match <= obj_left_border_left_img) 
                {
                    --num_prev_pixel;
                    continue;
                }
                // undistortedPts(pt_match, tmp_undist_pt, m_camera[0]);
                // float err = (r_cam_3D_plane[0]*tmp_undist_pt.x+r_cam_3D_plane[1]*tmp_undist_pt.y+r_cam_3D_plane[2])*depth+r_cam_3D_plane[3];
                // if (err <= 0) 
                // {
                //     --num_prev_pixel;
                //     continue;
                // }
            }
            
            int obj_id = (int)pt_info[1];
            // cout << "find matched pixel for obj " << obj_id <<endl;
            // 此时加入的点为像素坐标，后续如果使用这些点，需要先去畸变并转换到归一化平面坐标
            
            // match_pts[obj_id].first.emplace_back(x, y);
            // match_pts[obj_id].second.emplace_back(x_match, y_match);
            // match_pts_depth[obj_id].first.push_back(pts1.z);
            // match_pts_depth[obj_id].second.push_back(depth);

            if(match_pts.find(obj_id) == match_pts.end())
            {
                vector<Vec2f> temp_fea_prev, temp_fea_cur;
                temp_fea_prev.emplace_back(x, y);
                temp_fea_cur.emplace_back(x_match, y_match);
                pair<vector<Vec2f>,vector<Vec2f>> temp_pair_fea(temp_fea_prev,temp_fea_cur);

                vector<float> temp_dep_prev, temp_dep_cur;
                temp_dep_prev.push_back(pts1.z);
                temp_dep_cur.push_back(depth);
                pair<vector<float>,vector<float>> temp_pair_dep(temp_dep_prev,temp_dep_cur);

                match_pts_depth.insert(make_pair(obj_id,temp_pair_dep));
                match_pts.insert(make_pair(obj_id,temp_pair_fea));
            }
            else
            {
                match_pts[obj_id].first.emplace_back(x, y);
                match_pts[obj_id].second.emplace_back(x_match, y_match);
                match_pts_depth[obj_id].first.push_back(pts1.z);
                match_pts_depth[obj_id].second.push_back(depth);
            }
        }
    }
    return num_prev_pixel;
}

// 用于分配带权二分图匹配任务中的关联矩阵所需的内存
void FeatureTracker::UpdateCosts(const std::vector<std::vector<float>>& association_mat, SecureMat<float>* costs) 
{
    size_t rows_size = association_mat.size();
    size_t cols_size = rows_size > 0 ? association_mat.at(0).size() : 0;

    costs->Resize(rows_size, cols_size);

    for (size_t row_idx = 0; row_idx < rows_size; ++row_idx) 
    {
        for (size_t col_idx = 0; col_idx < cols_size; ++col_idx) 
        {
            (*costs)(row_idx, col_idx) = association_mat.at(row_idx).at(col_idx);
        }
    }
}

// 计算二分图匹配中的关联矩阵中某两个物体的关联分数
bool FeatureTracker::match_score_two_objs(const int id_checked_prev_obj, const vector<int> &id_checked_cur_objs, vector<float> &score_match, const cv::Mat &seg_map, 
                                            const cv::Mat &flow_map, const cv::Mat &depth_map_2, const vector<Vector3d> &Ps, const vector<Matrix3d> &Rs)
{
    // 两帧的相机全局位姿
    Vector3d Ps_c1, Ps_c2;
    Matrix3d Rs_c1, Rs_c2;
    bool has_motion_cam = false;
    if(!Ps.empty() && !Rs.empty())
    {
        Ps_c1 = Ps[0];
        Ps_c2 = Ps[1];
        Rs_c1 = Rs[0];
        Rs_c2 = Rs[1];
        has_motion_cam = true;
    }

    map<int, pair<vector<Vec2f>,vector<Vec2f>>> match_pts_of_obj;
    map<int, pair<vector<float>,vector<float>>> dep_match_pts_of_obj;
    int num_prev_pixel = track_pixels_one_prev_obj(id_checked_prev_obj, seg_map, flow_map, depth_map_2, match_pts_of_obj, dep_match_pts_of_obj);
    // cout << "Num of tracked pixel from prev obj: " << num_prev_pixel << endl;

    // 如果上一帧的物体在当前帧中的像素点大部分都没有深度估计（说明已经快要离开右相机的左侧视野了），则抛弃到对该物体的跟踪
    if (num_prev_pixel < 10) return false;
    for (int i = 0; i < id_checked_cur_objs.size(); ++i)
    {
        int cur_obj_id = id_checked_cur_objs[i];
        if (match_pts_of_obj.find(cur_obj_id) == match_pts_of_obj.end()) 
        {
            // cout << "can not find tracked pixel between 2 objs!" << endl;
            score_match[i] = 0;
            continue;
        }
        else if(match_pts_of_obj[cur_obj_id].second.size() > 0.25 * num_prev_pixel || match_pts_of_obj[cur_obj_id].second.size() > 7)
        {
            // 注意，这里是要得到两个数量的比值，需要为float，则需要先把其中一个数转为float！否则这里两个int相除，且前者小于后者，则结果恒为0！
            score_match[i] = (float)match_pts_of_obj[cur_obj_id].second.size() / (float)num_prev_pixel;
            // cout << "Num of tracked pixel between 2 objs: " << match_pts_of_obj[cur_obj_id].second.size() << endl;
        }
        else
        {
            score_match[i] = 0;
            // continue;
        }
    }

    Vector3d P_12 = Vector3d::Zero();
    Matrix3d R_12 = Matrix3d::Identity();
    // 计算上一帧某个obj与当前帧中可能有关联的物体之间的关联程度
    for (int i = 0; i < score_match.size(); ++i)
    {
        if (score_match[i] == 0) 
        {
            // cout << "Has no pixel matching!" << endl;
            continue;
        }
        int cur_obj_id = id_checked_cur_objs[i];
        bool has_pred_motion = true;
        // bool prev_sta_obj = false;
        
        // 给定相机坐标下的 该物体的帧间运动，即将该物体在上一帧相机坐标系下的点 变换到 当前帧相机坐标下的该对应点
        auto iter = RP_objs_pred.find(id_checked_prev_obj);
        // 如果该物体在上一帧是动态物体
        if (iter != RP_objs_pred.end())
        {
            // R_12 = Rs_c2.transpose()*RP_objs_pred[id_checked_prev_obj].first*Rs_c1;
            // P_12 = Rs_c2.transpose()*(RP_objs_pred[id_checked_prev_obj].first*Ps_c1 + RP_objs_pred[id_checked_prev_obj].second - Ps_c2);
            R_12 = RP_objs_pred[id_checked_prev_obj].first;
            P_12 = RP_objs_pred[id_checked_prev_obj].second;
        }
        // 如果该物体上一帧为静态物体或新物体
        else 
        {
            auto iter_ = status_objs_prev.find(id_checked_prev_obj);
            assert(iter_ != status_objs_prev.end());
            // 上一帧新出现的物体（如果当前帧为系统第二帧，例如首帧出现的所有物体都是新物体）。status_objs_prev中为0的物体一定在RP_objs_prev中有运动模型
            // 或者当前为系统第2帧，则没有相机运动预测
            if (!has_motion_cam || iter_->second == 2)
            {
                has_pred_motion = false;
                //if(status_objs_prev[id_index] == 1) prev_sta_obj = true;
            }
            // 如果上一帧对应的物体为已有的静态物体(1)或者背景点（可能是两帧间的背景点关联；也可能是上一帧背景中的漏检物体的一些散落在背景中的特征点关联到了当前帧的某物体，但是我们无法得知该物体是否运动，因此需要此函数调用着给定has_pred_motion的值）
            else if(has_motion_cam)
            {
                R_12 = Rs_c2.transpose()*Rs_c1;
                P_12 = Rs_c2.transpose()*(Ps_c1 - Ps_c2);
            }
        }
        vector<float> dist;
        float score_2d_rate_var_err = 0.0;
        // 赋值为-1.0，表示不需要计算第二个物体的平均深度
        float ave_depth_cur = -1.0;
        // cout << "matched prev obj: " << id_checked_prev_obj << endl;
        // 如果能进行3D位置变换，则还要计算变换后的3D距离平均值；否则，只计算2D和3D点集的分布方差的相似性
        if (has_pred_motion)
        {
            calcu_2d_3d_pts_dist(dist, true, match_pts_of_obj[cur_obj_id].first, ave_depth_cur, match_pts_of_obj[cur_obj_id].second, true, true, 
                            P_12, R_12, dep_match_pts_of_obj[cur_obj_id].first, dep_match_pts_of_obj[cur_obj_id].second);
            
            // 关联像素点不一定有有效深度值，但是最起码有2d点的var
            if(dist.size() == 2) score_2d_rate_var_err = (1.1 - score_match[i]) * max(dist[1],dist[0])/min(dist[1],dist[0]) * 10;
            // 如果有3D点关联，则综合考虑了 基于光流的关联点数比例的反值 和 基于运动模型的关联点与预测点的平均距离，两者的乘积越小，则说明两个物体的匹配程度越高！
            if(dist.size() >= 4) score_2d_rate_var_err = (1.1 - score_match[i]) * max(dist[1],dist[0])/min(dist[1],dist[0]) * max(dist[3],dist[2])/min(dist[3],dist[2]) * 10;
            // TODO：对于有运动预测的匹配，额外进行奖罚；如果平均距离小于阈值，则越小奖励越大（整体匹配系数越小）；如果平均距离大于阈值，则距离越大惩罚越大！
            // 注意AVE_DIST_3D_PTS_THRES是米为单位，暂时设为0.3m
            if(dist.size() == 5) score_2d_rate_var_err = score_2d_rate_var_err * exp(dist[4]/AVE_DIST_3D_PTS_THRES - 1.0);

            // cout << "matched cur obj: " << id_checked_cur_objs[i] << endl;

            // 如果3D转换距离过大，则认为两者基本是不匹配的！
            // if (dist[4]/AVE_DIST_3D_PTS_THRES - 1 >= 0.5) score_2d_rate_var_err = 0;
        }
        else
        {
            // cout << "matched cur obj: " << id_checked_cur_objs[i] << endl;
            calcu_2d_3d_pts_dist(dist, true, match_pts_of_obj[cur_obj_id].first, ave_depth_cur, match_pts_of_obj[cur_obj_id].second, true, false, 
                            P_12, R_12, dep_match_pts_of_obj[cur_obj_id].first, dep_match_pts_of_obj[cur_obj_id].second);
            
            if(dist.size() == 2) score_2d_rate_var_err = (1.1 - score_match[i]) * max(dist[1],dist[0])/min(dist[1],dist[0]) * 10;
            // 两个点集的2D和3D的离散程度的相似性，相似程度越高需要比值越接近于1
            if(dist.size() == 4) score_2d_rate_var_err = (1.1 - score_match[i]) * max(dist[1],dist[0])/min(dist[1],dist[0]) * max(dist[3],dist[2])/min(dist[3],dist[2]) * 10;
        }
        // 是否需要考虑检测的类别是否相同？
        Vec2f &pt = match_pts_of_obj[cur_obj_id].second[0];
        uchar cls_label_cur = seg_map.at<Vec2b>(pt(1), pt(0))[0];
        auto iter_gl_obj = find(glob_obj_id_prev.begin(), glob_obj_id_prev.end(), id_checked_prev_obj);
        int index = std::distance(glob_obj_id_prev.begin(),iter_gl_obj);
        // 如果前后两帧的关联物体的cls不一样，则关联程度需要再乘以一个大于1的系数；由于存在错检的可能，因此这个系数与两帧的像素点关联数比例成反比。
        if (cls_label_cur != obj_cls_prev[index])
        {
            score_match[i] = score_2d_rate_var_err * 2/score_match[i];
        }
    }
    // 判断上一帧的该物体在当前帧中是否存在潜在的关联物体
    float thres = 5.0;
    // ！！！！！这里是个隐藏很深的bug！！min()和max()函数需要直接给定进行比较的两个或多个变量，根据给定或默认的比较函数来获取其中最小或最大的变量！！其返回量是找到的变量的复制而不是该变量在序列中的指针或迭代器！！
    // 而min_element()和max_element()给定的是序列容器中的要比较的变量序列的 起始迭代器 和 终止迭代器（左开右闭）！！返回值也是找到的变量的迭代器（如果有多个相同值，则返回找到的第一个）
    // float min_socre = *(min(score_match.begin(),score_match.end()));
    // float max_socre = *(max(score_match.begin(),score_match.end()));
    float min_socre = *(min_element(score_match.begin(),score_match.end()));
    float max_socre = *(max_element(score_match.begin(),score_match.end()));
    // if (min_socre >= thres || max_socre == 0) 
    //     return false;
    // else
        return true;
}

// 给定图像平面上的2d像素点集，计算这些点之间的距离均方差。如果是给定两个互相配对的点集以及它们的深度值和刚体转换矩阵，则计算变换后的平均3D点距离。
// 默认depth_map中保存的是disparity值，需要转换成深度值。pts_depth_中保存的则是深度值
set<int> FeatureTracker::calcu_2d_3d_pts_dist(vector<float> &dist, bool pixel_pt, vector<Vec2f> &pts1, float &ave_depth_cur, vector<Vec2f> &pts2, 
                                            bool cal_pts_var, bool cal_ave_3d_dist, const Vector3d &p_obj12, const Matrix3d &r_obj12, 
                                            const vector<float> &pts_depth_1, const vector<float> &pts_depth_2, const bool &cal_ave_depth_prev_pts)
{
    if (!dist.empty()) dist.clear();
    int num_1 = pts1.size();
    int num_2 = pts2.size();
    vector<float> dist_pts;
    // 如果有pts2，则主要是为了计算pts2中的明显异常点
    set<int> outliers_pts;
    float normalized_stderr;
    // 注意，在这里对于2D和3D点集最简单的假设是正态分布，计算的是其距离的均值和均方差。我们可以将（均值+/-三倍标准差）的点作为外点去除。
    // 但是注意，这种方式对数据集中的极端异常值较为敏感（即异常值的影响会比较大）。而且点到中心的距离的分布通常不会是正态分布。
    // ！！！而如果不考虑去除异常值，则含有少数极端异常匹配的匹配点集就很难被当作合格的物体关联；甚至可能别的不正确的物体关联的整体方差反而比较小，导致了物体关联的错误！！
    // 使用中位数绝对偏差MAD来代替标准差（这对于任何分布均适用，可尽量发现极端异常值）以排除异常值！
    // http://www.990755.com/stock/5211.html
    // https://blog.csdn.net/zty0104/article/details/122376349 
    // 把基于MAD的合理范围外的匹配点给去除后，再重新计算中心点和各点距离，然后计算距离标准差和均值，“两者的比值“（针对2D像素点距离）或者只是标准差（针对3D点距离）作为该点集的归一化离散程度度量
    // TODO：之所以用两者的比值，当物体与相机的深度距离有变化时，其成像区域会变化，则像素点的距离的均值方差肯定都会变化！而当物体纯旋转时，情况会比较复杂，但点的距离的平均和方差似乎不会变化很大？猜测此时用两者比值还是较为合适的？
    // 另外，被识别出来的异常点，在计算出点集的属性后，对于动态物体不用专门将其从两个物体的匹配点集中erase，因为这里只是确保两个物体可以根据正确的点匹配而被关联，后续这些异常点在位姿估计中是可以被排除的！但对于静态物体应该需要？
    if (cal_pts_var)
    {
        Vec2f cent_pts(0,0);
        float factor_ = 1.0;
        // 如果给定的2d点是归一化平面上的点，则在计算2d方差时需要转化为像素点坐标的尺度，这样计算出来的方差值不会太小
        if (!pixel_pt) factor_ = factor_ * FOCAL_LENGTH_X;
        
        if (num_1 >= 3)
        {
            // ！是否要直接使用所有点的均值作为中心点？这样如果有某个点是极端偏离的，那么基于它形成的中心点会对所有点的“中心距离”都是有影响的呀？
            // 是否在计算中心点时就把这个极端的点去除？例如利用mean_shift来寻找中心点？
            // 应该不需要，当绝大部分点比较聚集，而异常点数量很少且其偏离程度足够大时，即使使用它来计算出中心点，后续也还是可以用MAD来去除这些点！但是假如本身点集的大部分点不够集中，则也不会出现某些所谓的异常值！
            // cal_centre_and_dist_pts(pts1, outliers_pts, dist_pts1, factor_, false, cent_pts);
            normalized_stderr = cal_rubust_norm_stderr_pts_dist(dist_pts, true, pts1, factor_, outliers_pts, false, cent_pts);
            dist.push_back(normalized_stderr);
        }
        
        if (num_2 >= 3)
        {
            // cal_centre_and_dist_pts(pts2, outliers_pts, dist_pts1, factor_, false, cent_pts);
            normalized_stderr = cal_rubust_norm_stderr_pts_dist(dist_pts, true, pts2, factor_, outliers_pts, false, cent_pts);
            dist.push_back(normalized_stderr);
        }
    }

    // 计算3D点的属性
    if (num_1 >= 3 && pts_depth_1.size() == num_1 && (cal_pts_var || cal_ave_3d_dist))
    {
        // 默认如果pts2如果不为空，则其必须与pts1是点数相同且一一配对的。因为如果不是匹配点对的话，那这里用于比较两个点集的属性是没有意义的！
        bool has_pts2 = (num_1 == num_2) && (num_2 == pts_depth_2.size());
        if (ave_depth_cur != -1.0)
        {
            float sum;
            if (!cal_ave_depth_prev_pts)
            {
                // 默认pts2是当前帧的点，而pts1是上一帧的点
                sum = std::accumulate(pts_depth_2.begin(), pts_depth_2.end(), 0.0); 
                ave_depth_cur = sum/pts_depth_2.size();
            }
            else
            {
                sum = std::accumulate(pts_depth_1.begin(), pts_depth_1.end(), 0.0); 
                ave_depth_cur = sum/pts_depth_1.size();
            }
        }

        // vector<cv::Point2f> undist_pts1, undist_pts2;
        // 如果给定的是像素点
        if (pixel_pt)
        {
            undistortedPts(pts1, m_camera[0]);
            undistortedPts(pts2, m_camera[0]);
        }
        
        vector<Vec3f> tmp_pts1, tmp_pts2;
        vector<float> dist_transf;
        Vec3d cent_pts_1(0,0,0), cent_pts_2(0,0,0);
        float depth_1, depth_2;
        float var_pts_1 = 0, var_pts_2 = 0;

        bool cal_3D_repro_err = has_pts2 && cal_ave_3d_dist;
        Vector3d pt1, pt2;
        int num_valid_dep = 0;
        // 需要给定两个点集之间的刚体位姿变换矩阵！
        for (int i = 0; i < pts1.size(); ++i)
        {
            depth_1 = pts_depth_1[i];
            depth_2 = pts_depth_2[i];
            
            // 最大距离统一采用物体的限制距离
            if (depth_1 > mMinDepthPt && depth_1 <= mThDepthObj && depth_2 > mMinDepthPt && depth_2 <= mThDepthObj)
            {
                ++num_valid_dep;
                float X_1 = pts1[i](0) * depth_1;
                float Y_1 = pts1[i](1) * depth_1;
                Vec3d Pts1(X_1, Y_1, depth_1);
                
                tmp_pts1.emplace_back(X_1, Y_1, depth_1);
                if(cal_pts_var)
                {
                    cent_pts_1 += Pts1;
                    if(has_pts2)
                    {
                        float X_2 = pts2[i](0) * depth_2;
                        float Y_2 = pts2[i](1) * depth_2;
                        Vec3d Pts2(X_2,Y_2,depth_2);
                        tmp_pts2.emplace_back(X_2,Y_2,depth_2);
                        cent_pts_2 += Pts2;

                        // 3D点运动转换后的平均距离
                        if (cal_3D_repro_err)
                        {
                            cv::cv2eigen(Pts1,pt1);
                            cv::cv2eigen(Pts2,pt2);
                            float dist_tranff_3d = ((r_obj12 * pt1 + p_obj12) - pt2).norm();
                            dist_transf.push_back(dist_tranff_3d);
                        }
                    }
                }
            }
        }

        // 3D点集的分布方差
        if (cal_pts_var)
        {
            if(num_valid_dep > 0)
            {
                // outliers_pts.clear();
                // cent_pts_1 = cent_pts_1 / num_1;
                cent_pts_1 = cent_pts_1 / num_valid_dep;
                Vec3f cent1(cent_pts_1[0],cent_pts_1[1],cent_pts_1[2]);

                // cal_centre_and_dist_pts(tmp_pts1, outliers_pts, dist_pts1, 1.0, true, cent);
                // 注意3D距离均为米制单位，其均值与方差都应该不会很大！
                // 3D距离的方差不需要进行归一化（即除以距离的均值），因为刚体上的相同两点的距离是不会改变的！
                normalized_stderr = cal_rubust_norm_stderr_pts_dist(dist_pts, true, tmp_pts1, 1.0, outliers_pts, true, cent1, false);

                dist.push_back(normalized_stderr);

                if(has_pts2) 
                {   
                    cent_pts_2 = cent_pts_2 / num_valid_dep;
                    Vec3f cent2(cent_pts_2[0],cent_pts_2[1],cent_pts_2[2]);
                    normalized_stderr = cal_rubust_norm_stderr_pts_dist(dist_pts, true, tmp_pts2, 1.0, outliers_pts, true, cent2, false);
                    dist.push_back(normalized_stderr);
                }
            }
        }

        if (cal_3D_repro_err) 
        {
            if(num_valid_dep > 0)
            {
                // 记得初始化这些变量
                Vec3f cent(0.0,0.0,0.0);
                float ave_3d_dist = 0.0;
                int num_valid_3d = 0;
                // 3D距离的方差不需要进行归一化（即除以距离的均值）
                normalized_stderr = cal_rubust_norm_stderr_pts_dist(dist_transf, false, tmp_pts1, 1.0, outliers_pts, false, cent, false, false);
                if(outliers_pts.empty())
                {
                    ave_3d_dist = std::accumulate(dist_transf.begin(), dist_transf.end(), 0.0);
                    num_valid_3d = num_valid_dep;
                }
                else
                {

                    for(int n = 0; n < num_valid_dep; ++n)
                    {
                        if(outliers_pts.find(n) != outliers_pts.end()) continue;
                        ave_3d_dist += dist_transf[n];
                        ++num_valid_3d;
                    }
                }
                if(num_valid_3d > 0)
                    dist.push_back(ave_3d_dist/num_valid_3d);
            }
        }
    }
    return outliers_pts;
}

int FeatureTracker::check_flow_with_F(const Matrix3d &cam_F, const Point2f &pt1, const Point2f &pt2)
{
    // 静态点需要离预测的极线不能太远
    // 极线的参数
    float a = cam_F(0,0) * pt1.x + cam_F(0,1) * pt1.y + cam_F(0,2);
    float b = cam_F(1,0) * pt1.x + cam_F(1,1) * pt1.y + cam_F(1,2);
    float c = cam_F(2,0) * pt1.x + cam_F(2,1) * pt1.y + cam_F(2,2);

    float num = a * pt2.x + b * pt2.y + c;

    float den = a * a + b * b;

    // F矩阵无效，一般是因为相机位移t为0? 也有可能是误差造成的
    if(den != 0) 
    {
        // 点到线的距离 的平方
        float dsqr = num*num/den;
        
        // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
        // 由于这是预测的运动构成的极线，精度不高，所以不应该限制太大
        if(dsqr > Th_epipolar_con * Th_epipolar_con) 
        {
            return -1;
        }
        else
        {
            return 1;
        }
    }
    else
        return 0;
}

// 分解E矩阵来得到两帧之间的R和t。其中t不具有真实尺度，且其取正或负得到的E是等价的（即E和-E都可以满足极线约束）；而R也有2个解。
// 因此一组匹配点通过极线约束和E的分解可以得到4组（R，t)，通过一个有深度的匹配点可以确定其中有效的一组。
// 最后通过最小化所有3D-2D匹配的重投影误差来得到t的尺度
void FeatureTracker::DecomposeE(const Mat &E, Mat &R1, Mat &R2, Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

// 用极线约束来筛选匹配外点
// bool FeatureTracker::epipolarConstrain(const vector<Point2f> &kpts1, const vector<Point2f> &kpts2, const Eigen::Matrix3d& Mat_F, vector<uchar> &is_inlier)
// {
//     if(kpts1.size() != kpts2.size()) return false;
//     int num_pt = kpts1.size();
//     if(is_inlier.size() != num_pt)
//     {
//         is_inlier.clear();
//         is_inlier.resize(num_pt,0);
//     }

//     bool valid_F = true;
//     int num_invalid = 0;
//     Point2f pt1, pt2;
//     float Th_dist = 2.5;
//     for(int i = 0; i < kpts1.size(); ++i)
//     {
//         pt1 = kpts1[i];
//         pt2 = kpts2[i];

//         const float a = pt1.x*Mat_F(0,0)+pt1.y*Mat_F(1,0)+Mat_F(2,0);
//         const float b = pt1.x*Mat_F(0,1)+pt1.y*Mat_F(1,1)+Mat_F(2,1);
//         const float c = pt1.x*Mat_F(0,2)+pt1.y*Mat_F(1,2)+Mat_F(2,2);

//         const float num = a*pt2.x+b*pt2.y+c;

//         const float den = a*a+b*b;

//         // F矩阵无效，一般是因为相机位移t为0?
//         if(den==0) 
//         {
//             // valid_F = false;
//             // break;

//             ++num_invalid;
//             continue;
//         }

//         // 点到线的距离 的平方
//         float dsqr = num*num/den;
        
//         // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
//         if(dsqr <= Th_dist*Th_dist) is_inlier[i] = 1;
//     }

//     if(num_invalid > 1.0/2 * num_pt) valid_F = false;

//     return valid_F;
// }


// 用H矩阵排除不是该平面或者视差太大的点
bool FeatureTracker::HomographyConstrain(vector<Point2f> &kp1, vector<Point2f> &kp2, const Eigen::Matrix3d& Mat_H, const Eigen::Matrix3d& Mat_H_inv, 
                                            vector<uchar> &is_inlier, const float &Th_score, float &score, const float &Th_dist)
{
    int num_pts = kp1.size();
    if(is_inlier.empty()) is_inlier.resize(num_pts, 0);

    float proj_x, proj_y, proj_z;
    float h_00 = Mat_H(0,0);
    float h_01 = Mat_H(0,1);
    float h_02 = Mat_H(0,2);
    float h_10 = Mat_H(1,0);
    float h_11 = Mat_H(1,1);
    float h_12 = Mat_H(1,2);
    float h_20 = Mat_H(2,0);
    float h_21 = Mat_H(2,1);
    float h_22 = Mat_H(2,2);

    // todo: 要计算当前帧投影到上一帧的误差
    float inv_proj_x, inv_proj_y, inv_proj_z;
    float inv_h_00 = Mat_H_inv(0,0);
    float inv_h_01 = Mat_H_inv(0,1);
    float inv_h_02 = Mat_H_inv(0,2);
    float inv_h_10 = Mat_H_inv(1,0);
    float inv_h_11 = Mat_H_inv(1,1);
    float inv_h_12 = Mat_H_inv(1,2);
    float inv_h_20 = Mat_H_inv(2,0);
    float inv_h_21 = Mat_H_inv(2,1);
    float inv_h_22 = Mat_H_inv(2,2);

    // float Th_dist = 2.0;
    for(int i = 0; i < num_pts; ++i)
    {
        Point2f &prev_pt = kp1[i];
        Point2f &cur_pt = kp2[i];

        // prev -> cur_pt
        proj_x = prev_pt.x * h_00 + prev_pt.y * h_01 + h_02;
        proj_y = prev_pt.x * h_10 + prev_pt.y * h_11 + h_12;
        proj_z = prev_pt.x * h_20 + prev_pt.y * h_21 + h_22;
        if(proj_z == 0)
        {
            // is_inlier[i] = 0;
            continue;
        }
        proj_x = proj_x/proj_z;
        proj_y = proj_y/proj_z;

        float dist = (proj_x - cur_pt.x) * (proj_x - cur_pt.x) + (proj_y - cur_pt.y) * (proj_y - cur_pt.y);

        // cur_pt -> prev_pt
        inv_proj_x = cur_pt.x * inv_h_00 + cur_pt.y * inv_h_01 + inv_h_02;
        inv_proj_y = cur_pt.x * inv_h_10 + cur_pt.y * inv_h_11 + inv_h_12;
        inv_proj_z = cur_pt.x * inv_h_20 + cur_pt.y * inv_h_21 + inv_h_22;
        if(inv_proj_z == 0)
        {
            // is_inlier[i] = 0;
            continue;
        }
        inv_proj_x = inv_proj_x/inv_proj_z;
        inv_proj_y = inv_proj_y/inv_proj_z;

        float inv_dist = (inv_proj_x - prev_pt.x) * (inv_proj_x - prev_pt.x) + (inv_proj_y - prev_pt.y) * (inv_proj_y - prev_pt.y);

        // 这里Th_dist其实是允许距离误差阈值的平方
        if(dist < Th_dist && inv_dist < Th_dist) 
        {
            is_inlier[i] = 1;
            score += (Th_score - dist);
            score += (Th_score - inv_dist);
        }
    }
    
    return true;
}

// 根据对极约束估计F矩阵，并筛除其中的异常匹配点。
// 如果是已知相机大概的运动变换T，则可以用已知的F来排除原理极线的当前匹配点
void FeatureTracker::rejectWithFV1(bool for_sift, bool init_succ_IMU, const set<int> &pts_for_F)
{
    if(for_sift)
        printf("FM ransac for sift begins\n");
    else
        printf("FM ransac for FAST begins\n");
    
    vector<Point2f> *cur_fea;
    vector<Point2f> *prev_fea;
    vector<int> *id_bg_pts;
    vector<uchar> *status;
    vector<float> *depth;
    vector<int> *ids;
    if(for_sift)
    {
        cal_Mat_F_H = false;
        cur_fea = &cur_sift;
        prev_fea = &prev_sift;
        id_bg_pts = &id_bg_track_sift;
        status = &status_sift;
        depth = &prev_sift_dep;
        ids = &ids_sift;
    }
    else
    {
        cur_fea = &cur_FAST;
        prev_fea = &prev_FAST;
        id_bg_pts = &id_bg_track_FAST;
        status = &status_FAST;
        depth = &prev_FAST_dep;
        ids = &ids_FAST;
    }
    
    // 引用变量只能在声明时直接赋值
    vector<Point2f> &cur_Fea = (*cur_fea);
    vector<int> &id_bg_Pts = (*id_bg_pts);
    vector<Point2f> &prev_Fea = (*prev_fea);
    vector<uchar> &status_Fea = (*status);
    vector<float> &depth_Fea = (*depth);
    vector<int> &ids_Fea = (*ids);

    int num_track_for_F = 0, num_track;
    bool spec_pts_for_F = false;

    num_track = id_bg_Pts.size();
    if(for_sift)
    {
        if(!pts_for_F.empty())
        {
            spec_pts_for_F = true;
            num_track_for_F = pts_for_F.size();
        }
        else
            num_track_for_F = num_track;
    }
    
    int valid_match = num_track;
    
    TicToc t_f;
    if (num_track > 0)
    {
        // ROS_DEBUG("FM ransac begins");
        vector<uchar> status_fea;
        vector<cv::Point2f> prev_pts_match, cur_pts_match;
        vector<cv::Point2f> prev_pts_for_F, cur_pts_for_F;
        Matrix3d F_cam;

        vector<int> id_pt_for_epi;
        for(unsigned int i = 0; i < num_track; ++i)
        {
            int id = id_bg_Pts[i];

            if(spec_pts_for_F && pts_for_F.find(id) != pts_for_F.end()) continue;

            cur_pts_match.push_back(cur_Fea[id]);
            prev_pts_match.push_back(prev_Fea[id]);
            if(spec_pts_for_F) id_pt_for_epi.push_back(id);
        }
        
        // 系统第2帧如果能有相机的运动预测，则此处P_cam_motion也已经被赋值了
        // bool has_T = (P_cam_motion.norm() >= 0.05) && (frame_count > 1 || (has_motion_pred_first_two_frame));
        bool has_T = true;
        // 注意，对于纯双目，还是要判断当前帧是否有位移的，否则下面无法估计F和利用F。纯双目可以使用恒速模型
        if(!USE_IMU || init_succ_IMU)
            has_T = (P_cam_motion.norm() >= 0.05);

        bool use_ransac = EASI_RANSAC_FH && has_T;

        // 考虑旋转较大的情况，此时需要使用sift匹配来获取较好的F矩阵估计
        Quaterniond delta_Q(R_cam_motion);
        double delta_angle = acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0;
        if (delta_angle >= 2.0)
        {
            printf(" big delta_angle!\n");
            // ROS_INFO(" big delta_angle ");
            //return true;
        }
        
        bool filter_fea = true;
        // 如果旋转和位移均很小，则默认不对sift和FAST进行筛选
        // 其实就算没有IMU，也可以使用恒速模型来预估当前帧相机的运动
        if(abs(delta_angle) < 0.2 && P_cam_motion.norm() < 0.05) 
        {
            filter_fea = false;
            cout << "camera in current frame is nearly static!" << endl;
        }
        // 其实如果IMU的标定结果 和 LBA优化有bias的结果都比较好，则这里直接使用IMU的积分来进行筛选也可以
        if(filter_fea)
        {
            has_valid_F = false;
            
            // sift点匹配的精度比较高，可以用来估计基础矩阵
            // 匹配点数不能太少，不然findFundamentalMat无法得到满足要求的结果(会直接abort())
            if(use_ransac && for_sift && num_track_for_F >= 10)
            {
                Mat F_tmp;
                vector<int> id_pt_for_F;
                float score = 0.0;
                if(spec_pts_for_F)
                {
                    for(auto id: pts_for_F)
                    {
                        prev_pts_for_F.push_back(prev_sift[id]);
                        cur_pts_for_F.push_back(cur_sift[id]);
                        id_pt_for_F.push_back(id);
                    }
                    // 计算F矩阵用的应该是像素匹配，而不是归一化平面点
                    F_tmp = cv::findFundamentalMat(prev_pts_for_F, cur_pts_for_F, cv::FM_RANSAC, F_THRESHOLD, 0.99, status_fea);
                }
                else
                {
                    F_tmp = cv::findFundamentalMat(prev_pts_match, cur_pts_match, cv::FM_RANSAC, F_THRESHOLD, 0.99, status_fea);
                }
                
                cout << "Find matrix F succeeded!" << endl;
                Mat_F = F_tmp.clone();
                cal_Mat_F_H = true;

                for(int i = 0; i < num_track_for_F; ++i)
                {
                    if(status_fea[i] == 0)
                    {
                        int id;
                        if(spec_pts_for_F)
                            id = id_pt_for_F[i];
                        else
                            id = id_bg_Pts[i];
                        // 如果是上一帧有深度值（这种情况一般是该点从初始观测帧开始就有深度值了），是否保留该点？
                        // 算了不保留，因为不准确的匹配其实会导致相机的位姿估计很差
                        // 如果当前帧不够3D点来进行PnP,则不进行视觉运动估计，而是直接使用IMU的积分（这对于汽车这种近似地平面的运动是可以不用初始化g的，当然最好是让汽车在静止中估计初始时刻的位姿）
                        // if(prev_sift_dep[id] > 0) continue;

                        status_Fea[id] = 0;
                        --valid_match;
                    }
                }

                cv2eigen(Mat_F, F_cam);
                has_valid_F = true;
            }
            // 如果有运动的预测，则可以使用它来构造F矩阵
            else
            {
                // 使用相机运动预测 或 IMU积分 来作为当前帧的F
                if(!cal_Mat_F_H && has_T)
                {
                    // 如果相机的平移很小(这里5厘米的规定是否合理？），即要么纯旋转，要么完全静止，则无法构建本质矩阵
                    Matrix3d t_up;
                    t_up << 0.0, -P_cam_motion(2), P_cam_motion(1), P_cam_motion(2), 0.0, -P_cam_motion(0), -P_cam_motion(1), P_cam_motion(0), 0.0;
                    // 本质矩阵到关键矩阵
                    F_cam = K_trans_inv * t_up * R_cam_motion * K_inv;
                    has_valid_F = true;
                }
            }
            
            // 用F矩阵基于极线约束筛选2D-2D匹配
            int id;
            if(!for_sift || (!cal_Mat_F_H || spec_pts_for_F))
            {
                // 如果有F矩阵，则用极线约束来排除异常匹配点
                if(has_valid_F)
                {
                    float score;
                    int num_pt = cur_pts_match.size();
                    vector<uchar> is_inlier(num_pt, 0);
                    bool valid_F = epipolarConstrain(prev_pts_match, cur_pts_match, F_cam, is_inlier, 5.0, score);
                    if(valid_F)
                    {
                        for(int i = 0; i < num_pt; ++i)
                        {
                            if(is_inlier[i] == 0) 
                            {
                                if(spec_pts_for_F) 
                                    id = id_pt_for_epi[i];
                                else
                                    id = id_bg_Pts[i];
                                
                                status_Fea[id] = 0;
                                --valid_match;
                            }
                        }
                    }
                    // 发现还是没有有效的F矩阵。可以通过极线方程的参数来判断F矩阵是否有效吗？?
                    else
                    {
                        has_valid_F = false;
                    }
                }
            }
        }
        else
        {
            has_valid_F = false;
        }
        
        // 如果没有F矩阵，则用预测的R和t来排除异常3d-2d匹配点（只排除在上一帧有深度的点）
        // 这意味着，如果系统没法使用IMU(例如纯双目，或者使用IMU且需要优化g且还未VI初始化）来为系统前2帧间的相机运动提供预测值，则系统第2帧没法在此排除任何匹配外点（因为不知道系统首帧是不是静止或者纯旋转，所以不能利用2d-2d的极线约束来估计基础矩阵）！
        // 如果没有较为准确的相机运动预测，则只能靠PnP阶段的RANSAC的结果排除3d-2d匹配外点！
        // 最后，可以选择系统一开始总是使用双目+IMU，如果第2帧发现相机是静止的，则将系统切换回纯双目（防止IMU标定不准确对LBA的影响）；等到相机开始运动，又可以切换为双目+IMU（但这可能需要重新VI初始化（如果要优化g的话），使得前后2段轨迹的参考坐标系不一致）
        // if(!has_valid_F && (USE_IMU || frame_cnt > 1))
        // 除了上面可能进行基于F的2d-2d匹配筛选，这里还可以进行3D-2D的匹配筛选（目的是去除上一帧深度明显错误的点）。但是如果IMU的参数（尤其是静态bias估计错误）对预测的最新2帧间运动的影响很大，如果此预测值不准确，则可能会排除大多数3D点！
        bool filter_3D_2D = false;
        // 尤其要对sift点进行滤除，因为sift的大多数跟踪点在上一帧中并没有立体匹配，其深度值是从depth_map中获取的？
        if(filter_3D_2D && (has_motion_pred_first_two_frame || frame_cnt > 1))
        {
            Vector3d pt_1, pt_2, pt_pixel;
            float u_proj, v_proj, u_est, v_est;
            int pt_id;
            float depth;
            // TODO：重投影的距离阈值怎么取？由于此时R和t不是很精确（如果是用IMU的积分值，还取决于IMU的参数估计是否准确），此值是否应该取得大一些
            float Th_dist = 1.5;
            // 如果没有F矩阵，则用R来筛选上一帧有深度值的点在当前帧的匹配
            for(int i = 0; i < num_track; ++i)
            {
                int id = id_bg_Pts[i];
                // 如果上面已经不满足极线约束，则不考虑
                if(status_Fea[id] == 0) continue;
                depth = depth_Fea[id];
                if(depth > 0)
                {
                    pt_id = ids_Fea[id];
                    
                    // 被跟踪的上一帧的背景点不一定有立体匹配，其深度值可能是通过运动变换的投影得到的
                    // assert(prev_un_Fea_map.find(pt_id) != prev_un_Fea_map.end() && "Weired!");
                    pt_1(2) = depth;
                    pt_1(0) = prev_Fea[id].x * depth;
                    pt_1(1) = prev_Fea[id].y * depth;
                    
                    pt_2 = R_cam_motion * pt_1 + P_cam_motion;
                    pt_2(0) = pt_2(0)/pt_2(2);
                    pt_2(1) = pt_2(1)/pt_2(2);
                    pt_2(2) = 1.0;
                    pt_pixel = K * pt_2;
                    u_proj = pt_pixel(0);
                    v_proj = pt_pixel(1);

                    u_est = cur_Fea[id].x;
                    v_est = cur_Fea[id].y;
                    if((u_proj-u_est)*(u_proj-u_est) + (v_proj-v_est)*(v_proj-v_est) > Th_dist * Th_dist)
                    {
                        status_Fea[id] = 0;
                        --valid_match;
                    }   
                }
            }
        }
        
        // 如果根据sift匹配点获得了F矩阵，则尝试通过分解E矩阵来得到R和t。用某个有深度值的匹配点可以确定最终的R和t，并且通过最小化所有有深度值的匹配点的重投影误差来求解t的尺度
        if(for_sift && cal_Mat_F_H)
        {
            // 分解得到 R 和 t的方向
            // Mat R_E_1, R_E_2, t_E;
            // 自己写分解E求解（R,t)时还需要三角化特征点以选择唯一正确的一组解。
            // DecomposeE(Mat_F, R_E_1, R_E_2, t_E);

            Point2d pp(SHIFT_X,SHIFT_Y);
            Matrix3d Mat_E = K_trans * F_cam * K;
            Mat E, R_E, t_E;
            eigen2cv(Mat_E,E);
            // 此函数从E分解得到R和t时 会得到四组可能的解（R，t），其中t是尺度归一化的（这相当于固定了整个场景的尺度），
            // 并利用给定的2d-2d匹配点对，使用三角化来恢复给定尺度下的点深度值，只有正确的一对(R,t)才会使得所有3D点的深度值为正。
            if(spec_pts_for_F)
                recoverPose(E, prev_pts_for_F, cur_pts_for_F, R_E, t_E, FOCAL_LENGTH_X, pp);
            
            cv2eigen(R_E,R_from_E);
            cv2eigen(t_E,t_from_E);
            
            // 后续使用有深度值的匹配点来恢复t的尺度，基于最小化投影误差
            // 这里应该选取那些深度值比较小的点，因为深度越大的点，其深度估计受点匹配精度的影响越大，从而对这里尺度的估计影响也大。
            // 根据SOFT2中的说法，深度值在3-4倍基线距离的点是较为合适的（KITTI的基线长度大约0.6m)
            // 把尺度估计放在上一帧的立体匹配点三角化完成之后。
            // if(pts_stereo_for_F.size() >= 5);
            // {
            //     // 
            //     for(int i =0; i < pts_stereo_for_F.size(); ++i)
            //     {
            //         int id = pts_stereo_for_F[i];
            //         Point2f &prev_pt = prev_Fea[id];
            //         int g_id = ids_Fea[id];


            //     }
            // }
        }
    }
    
    if(for_sift)
    {
        // 删除无效的sift跟踪点先不在这里进行，因为有些sift跟踪点和新点可能会被FAST跟踪点所覆盖（根据点标注的先后顺序而定)
        reduce_invalid_fea(true);
        printf("FM ransac for sift: %d -> %lu: %f\n", num_track, valid_match, (1.0 * valid_match) / num_track);
    }
    else
    {
        // 删除无效的FAST跟踪点不在这里进行
        // reduce_invalid_fea(false);
        printf("FM ransac for FAST: %d -> %lu: %f\n", num_track, valid_match, (1.0 * valid_match) / num_track);
    }
    
    printf("FM ransac costs: %fms\n", t_f.toc());
}

// sift和FAST跟踪点一起用来估计F矩阵，因为sift跟踪点中近处点很少，如果只用sift点估计F矩阵，后续会排除掉很多近处的FAST跟踪点
void FeatureTracker::rejectWithFV2(bool init_succ_IMU, const set<int> &pts_for_F)
{   
    TicToc t_f;
    
    has_valid_F = false;
    has_valid_H = false;

    vector<cv::Point2f> prev_pts_match, cur_pts_match;
    vector<cv::Point2f> prev_pts_for_FH, cur_pts_for_FH;
    Matrix3d F_cam, H_cam;

    int num_track_sift = id_bg_track_sift.size();
    int num_track_FAST = id_bg_track_FAST.size();
    int valid_match_sift = num_track_sift;
    int valid_match_FAST = num_track_FAST;

    vector<int> pt_id_for_F;
    set<int> pt_id_in_sift, pt_id_in_FAST;
    for(auto pt_id: pts_for_F)
    {
        if(pt_id <= 0)
        {
            int id = -1 * pt_id;
            if(status_sift[id] == 0) continue;
            prev_pts_for_FH.push_back(prev_sift[id]);
            cur_pts_for_FH.push_back(cur_sift[id]);
            pt_id_in_sift.insert(id);

            // 静态物体跟踪点可能在pts_for_F中，但是不会在id_bg_track_中
            if(obj_cls_id_sift[id].first > 0) ++num_track_sift;
        }
        else
        {
            int id = pt_id - 1;
            if(status_FAST[id] == 0) continue;
            prev_pts_for_FH.push_back(prev_FAST[id]);
            cur_pts_for_FH.push_back(cur_FAST[id]);
            pt_id_in_FAST.insert(id);

            if(obj_cls_id_FAST[id].first > 0) ++num_track_FAST;
        }
        pt_id_for_F.push_back(pt_id);
    }

    int valid_sift_F = num_track_sift;
    int valid_sift_H = num_track_sift;
    int valid_FAST_F = num_track_FAST;
    int valid_FAST_H = num_track_FAST;

    int num_track_for_F = prev_pts_for_FH.size();

    bool has_T = true;
    double delta_angle = 0;
    bool filter_fea = true;
    // 判断当前帧是否有位移，否则下面无法估计F和利用F。纯双目可以使用恒速模型;使用IMU，在VI初始化成功之前用双目的恒速模型，之后用IMU的积分
    if(frame_cnt > 1)
    {
        cout << "norm of predicted P_cam_motion: " << P_cam_motion.norm() << endl;
        // 预测的位移至少要5cm
        has_T = (P_cam_motion.norm() >= 0.05);

        // 考虑旋转较大的情况
        Quaterniond delta_Q(R_cam_motion);
        delta_angle = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

        if(delta_angle >= 1.5)
        {
            printf(" big delta_angle!\n");
        }

        if(delta_angle < 0.3 && !has_T) 
        {
            // 如果旋转和位移均很小，即接近静止，则默认不对sift和FAST进行筛选（认为绝大多数跟踪是正确的）
            // 其实就算没有IMU，也可以使用恒速模型来预估当前帧相机的运动
            // 如果相机静止，则也无法无需估计F或H矩阵
            // filter_fea = false;
            cout << "camera in current frame is nearly static!" << endl;
            return;
        }
        
        if(num_track_for_F < 8)
        {
            cout << "Not enough tracked fea in current frame!" << endl;
            return;
        }
    }
    else if(frame_cnt == 1)
    {
        // 系统第2帧，只能选择估计H或F矩阵。如果点数太少不足以估计H或F矩阵
        if(num_track_for_F < 8)
        {
            // filter_fea = false;
            cout << "Not enough tracked fea in the second frame!" << endl;
            return;
        }
    }
    else
    {
        // filter_fea = false;
        return;
    }

    bool filter_by_F_from_predict_motion = false;
    bool need_PnP = true;
    if(filter_fea)
    {
        // 如果使用IMU，且已经完成IMU初始化，且当前帧预测的相机位移不接近0，则可以直接使用IMU的数据构建F矩阵，而不需要进行基于视觉的F或H矩阵的估计
        if(USE_IMU && init_succ_IMU)
        {
            // todo: IMU初始化之后，如果这里采用某种方式过滤了匹配点，则当前帧后续不再需要进行PnP？
            // 即此处估计H或F，只是为了过滤外点，而不是为了估计R和t的方向？当前帧的相机位姿初值直接用IMU积分来获取
            // need_PnP = false;
            
            // 发现用IMU预测的运动所构建的F矩阵很不精确，应该主要是由于加速度计的零偏导致的误差较大
            if(filter_by_F_from_predict_motion && has_T)
                EASI_RANSAC_FH = false;
            else
            {
                // 就算没有位移，仍可以估计H矩阵
                EASI_RANSAC_FH = true;   
            }
        }
        else
            EASI_RANSAC_FH = true;
    }
    
    vector<uchar> status_fea_F, status_fea_H;
    int num_inliers_F = 0, num_inliers_H = 0;

    vector<int> invalid_pt_id_F, invalid_pt_id_H;
    // 其实如果IMU的标定结果 和 LBA优化有bias的结果都比较好，则这里直接使用IMU的积分来进行筛选也可以
    if(filter_fea)
    {
        Mat F_tmp, H_tmp;
        Matrix3d H_cam_inv;

        float score_F = 0.0, score_H = 0.0;
        vector<Point2f> inliers_prev_F, inliers_cur_F, inliers_prev_H, inliers_cur_H;
        
        cout << "num_track_for_F: " << num_track_for_F << endl;

        if(EASI_RANSAC_FH)
        {
            bool est_F = true;
            bool est_H = true;

            // cout << "Start estimate fundamental matrix with fea matching!" << endl;
            // cv::FM_RANSAC方法至少需要15对匹配点
            if(num_track_for_F >= 18)
                F_tmp = cv::findFundamentalMat(prev_pts_for_FH, cur_pts_for_FH, cv::FM_RANSAC, F_THRESHOLD, 0.99, status_fea_F);
            else if(num_track_for_F >= 8)
                // 基于七点法的最小二乘
                F_tmp = cv::findFundamentalMat(prev_pts_for_FH, cur_pts_for_FH, 4, F_THRESHOLD, 0.99, status_fea_F);
            else 
                est_F = false;
            
            // cout << "Start estimate homography matrix with fea matching!" << endl;
            if(num_track_for_F >= 12)
            {
                // H_tmp = cv::findHomography(prev_pts_for_FH, cur_pts_for_FH, RANSAC, 3.0, status_fea_H, 250, 0.99);
                H_tmp = cv::findHomography(prev_pts_for_FH, cur_pts_for_FH, LMEDS, 3.0, status_fea_H, 250, 0.99);
            }
            else
            {
                // H_tmp = cv::findHomography(prev_pts_for_FH, cur_pts_for_FH, 0, F_THRESHOLD, status_fea_H, 250, 0.99);
                H_tmp = cv::findHomography(prev_pts_for_FH, cur_pts_for_FH, 0, 3.0, status_fea_H, 250, 0.99);
            }
            
            if(est_F || est_H)
            {
                for(int i = 0; i < num_track_for_F; ++i)
                {
                    int id = pt_id_for_F[i];

                    if(est_F)
                    {
                        if(status_fea_F[i] == 0)
                        {
                            if(id <= 0)
                            {
                                // id = -1 * id;
                                // status_sift[id] = 0;
                                
                                // 注意，参与F或H估计的静态点中，部分是属于上一帧的静态物体，这部分跟踪外点不能被删除！
                                // 此外，这部分物体跟踪点不在id_bg_track_sift或id_bg_track_FAST中！
                                invalid_pt_id_F.push_back(id);
                                --valid_sift_F;
                            }
                            else
                            {
                                // id -= 1;
                                // status_FAST[id] = 0;

                                invalid_pt_id_F.push_back(id);
                                --valid_FAST_F;
                            }
                        }
                        else
                        {
                            ++num_inliers_F;
                            inliers_prev_F.push_back(prev_pts_for_FH[i]);
                            inliers_cur_F.push_back(cur_pts_for_FH[i]);
                        }
                    }
                    
                    if(est_H)
                    {
                        if(status_fea_H[i] == 0)
                        {
                            if(id <= 0)
                            {
                                invalid_pt_id_H.push_back(id);

                                --valid_sift_H;
                            }
                            else
                            {
                                invalid_pt_id_H.push_back(id);

                                --valid_FAST_H;
                            }
                        }
                        else
                        {
                            ++num_inliers_H;
                            inliers_prev_H.push_back(prev_pts_for_FH[i]);
                            inliers_cur_H.push_back(cur_pts_for_FH[i]);  
                        }
                    }
                }
            }

            vector<uchar> is_inlier;
            // 如果内点数太少，则放弃该F估计
            if(num_inliers_F < 0.5 * num_track_for_F || num_inliers_F < 7)
            {
                has_valid_F = false;
            }
            else
            {
                Mat_F = F_tmp.clone();
                
                // Mat_F的数据类型为double
                // auto type = Mat_F.type();
                // cout << "type of element in t is " << type << endl;

                cv2eigen(Mat_F, F_cam);
                // 计算内点的分数
                // 是否会再次有外点？其实上面计算出的F矩阵不一定有效，因为有可能相机无位移，这里还是要用跟踪点来验证该F矩阵是否有效
                // todo:是否要计算双向投影误差？
                bool valid = epipolarConstrain(inliers_prev_F, inliers_cur_F, F_cam, is_inlier, Th_score, score_F);
                if(!valid)
                {
                    has_valid_F = false;
                    cout << "Invalid matrix F! Camere maybe has no translation!" << endl;
                }
                else
                {
                    has_valid_F = true;
                    cal_Mat_F_H = true;
                    cout << "Find matrix F succeeded!" << endl;
                }
            }

            // 检查该H矩阵
            if(num_inliers_H <= 0.5 * num_track_for_F || num_inliers_H < 6)
            {
                has_valid_H = false;
            }
            else
            {
                Mat_H = H_tmp.clone();

                // auto type = Mat_H.type();
                // cout << "type of element in t is " << type << endl;

                cv2eigen(Mat_H, H_cam);
                H_cam_inv = H_cam.inverse();
                // 注意，这里记得clear，因为在HomographyConstrain内部不会对其进行清除
                if(!is_inlier.empty()) is_inlier.clear();
                // todo::对于H矩阵是否需要根据匹配内点的score来判断该H的有效性？
                HomographyConstrain(inliers_prev_H, inliers_cur_H, H_cam, H_cam_inv, is_inlier, Th_score, score_H);

                has_valid_H = true;
                cal_Mat_F_H = true;
                cout << "Find matrix H succeeded!" << endl;
            }
        }
        else
        {
            {
                // 只有使用IMU并初始化成功后，且相机有位移，才会考虑有预测的运动值来构建F矩阵
                Matrix3d t_up;
                t_up << 0.0, -P_cam_motion(2), P_cam_motion(1), P_cam_motion(2), 0.0, -P_cam_motion(0), -P_cam_motion(1), P_cam_motion(0), 0.0;
                // 本质矩阵到关键矩阵
                F_cam = K_trans_inv * t_up * R_cam_motion * K_inv;
                has_valid_F = true;
            }
            
            // todo:如果使用了IMU，当前帧就算没有位移，只要有旋转R，也可以根据地平面的近似参数 和 R 来构造H矩阵，用以筛选近似地面上的跟踪点？

        }
        
        // 如果估计了H或F矩阵
        if(has_valid_F || has_valid_H)
        {
            vector<int> temp_pt_id;
            vector<uchar> is_inlier_F, is_inlier_H;

            // pt_id_in_sift中的部分静态跟踪点（即静态物体跟踪点）不在id_bg_track_sift中
            // if(pt_id_in_sift.size() < num_track_sift)
            {
                // 寻找没有参与F和H矩阵估计的剩余背景跟踪点
                for(auto pt: id_bg_track_sift)
                {
                    if(status_sift[pt] == 0) 
                    {
                        // 去掉这部分无效的背景点
                        --num_track_sift;
                        --valid_sift_F;
                        --valid_sift_H;
                        continue;
                    }

                    if(EASI_RANSAC_FH)
                    {
                        if(pt_id_in_sift.find(pt) == pt_id_in_sift.end())
                        {
                            cur_pts_match.push_back(cur_sift[pt]);
                            prev_pts_match.push_back(prev_sift[pt]);
                            temp_pt_id.push_back(pt);
                        }
                    }
                    else
                    {
                        cur_pts_match.push_back(cur_sift[pt]);
                        prev_pts_match.push_back(prev_sift[pt]);
                        temp_pt_id.push_back(pt);
                    }
                }

                int num_pt = cur_pts_match.size();
                bool valid;

                if(has_valid_F && num_pt > 0)
                {
                    is_inlier_F.resize(num_pt, 0);
                    valid = epipolarConstrain(prev_pts_match, cur_pts_match, F_cam, is_inlier_F, Th_score, score_F);
                    if(valid)
                    {
                        for(int i = 0; i < num_pt; ++i)
                        {
                            if(is_inlier_F[i] == 0) 
                            {
                                int id = temp_pt_id[i];
                                // status_sift[id] = 0;

                                invalid_pt_id_F.push_back(-id);
                                --valid_sift_F;
                                
                                // --valid_match_sift;
                            }
                        }
                    }
                    else
                    {
                        has_valid_F = false;
                    }
                }
                
                if(has_valid_H && num_pt > 0)
                {
                    is_inlier_H.resize(num_pt,0);
                    // 使用估计的H矩阵对剩下的跟踪点进行过滤
                    // 使用H矩阵，会留下平面上的绝大部分点，以及深度值足够大（相对于位移绝对值，即两个匹配点形成的视差足够大）的点，但对于近处的非平面上的点会直接排除（是否要保留这些点呢？）
                    // 这里还是要暂时用H来排除一些外点并保留尽可能多该平面上的跟踪点，以便之后有足够多的有深度值的跟踪内点来恢复t的尺度值
                    // 如果还要保留这些近处的非平面点，则需要后续估计出带尺度的t之后，形成F矩阵并用极线约束来检验这些点。但是这些近处的点真的有必要保留吗？下一帧还会被跟踪到吗？
                    valid = HomographyConstrain(prev_pts_match, cur_pts_match, H_cam, H_cam_inv, is_inlier_H, Th_score, score_H);
                    
                    if(valid)
                    {
                        for(int i = 0; i < num_pt; ++i)
                        {
                            if(is_inlier_H[i] == 0) 
                            {
                                int id = temp_pt_id[i];
                                
                                // 记录这些被排除在平面外的点
                                invalid_pt_id_H.push_back(-id);
                                --valid_sift_H;
                                
                                // --valid_match_sift;
                            }
                        }
                    }
                    // 发现还是没有有效的F矩阵。可以通过极线方程的参数来判断F矩阵是否有效吗？?
                    else
                    {
                        has_valid_H = false;
                    }
                }
            }

            temp_pt_id.clear();
            prev_pts_match.clear();
            cur_pts_match.clear();

            // if(pt_id_in_FAST.size() < num_track_FAST)
            {
                for(auto pt: id_bg_track_FAST)
                {
                    if(status_FAST[pt] == 0) 
                    {
                        --num_track_FAST;
                        --valid_FAST_F;
                        --valid_FAST_H;
                        continue;
                    }

                    if(EASI_RANSAC_FH)
                    {
                        if(pt_id_in_FAST.find(pt) == pt_id_in_FAST.end())
                        {
                            prev_pts_match.push_back(prev_FAST[pt]);
                            cur_pts_match.push_back(cur_FAST[pt]);
                            temp_pt_id.push_back(pt);
                        }
                    }
                    else
                    {
                        prev_pts_match.push_back(prev_FAST[pt]);
                        cur_pts_match.push_back(cur_FAST[pt]);
                        temp_pt_id.push_back(pt);
                    }
                }

                int num_pt = cur_pts_match.size();
                
                bool valid;
                if(has_valid_F && num_pt > 0)
                {
                    if(!is_inlier_F.empty()) is_inlier_F.clear();
                    is_inlier_F.resize(num_pt, 0);
                    valid = epipolarConstrain(prev_pts_match, cur_pts_match, F_cam, is_inlier_F, Th_score, score_F);
                    if(valid)
                    {
                        for(int i = 0; i < num_pt; ++i)
                        {
                            if(is_inlier_F[i] == 0) 
                            {
                                int id = temp_pt_id[i];
                                // status_FAST[id] = 0;

                                --valid_FAST_F;
                                invalid_pt_id_F.push_back(id+1);
                                
                                // --valid_match_FAST;
                            }
                        }
                    }
                    // 发现还是没有有效的F矩阵。可以通过极线方程的参数来判断F矩阵是否有效吗？?
                    else
                    {
                        has_valid_F = false;
                    }
                }
                
                if(has_valid_H && num_pt > 0)
                {
                    if(!is_inlier_H.empty()) is_inlier_H.clear();
                    is_inlier_H.resize(num_pt, 0);
                    valid = HomographyConstrain(prev_pts_match, cur_pts_match, H_cam, H_cam_inv, is_inlier_H, Th_score, score_H);
                    if(valid)
                    {
                        for(int i = 0; i < num_pt; ++i)
                        {
                            if(is_inlier_H[i] == 0) 
                            {
                                int id = temp_pt_id[i];
                                // status_FAST[id] = 0;

                                --valid_FAST_H;
                                invalid_pt_id_H.push_back(id+1);
                                
                                // --valid_match_FAST;
                            }
                        }
                    }
                    else
                    {
                        has_valid_H = false;
                    }
                }
            }
            
            if(EASI_RANSAC_FH)
            {
                // 如果F和H矩阵都有估计结果，则查看两者的内点数哪个更多
                if(has_valid_F && has_valid_H)
                {
                    // todo: 如何决定应该选择哪个结果？？
                    // H矩阵的精度要求需要比F的要低一些。同时，由于重投影误差的阈值的设置强烈依赖于经验和方法的误差难易程度，因此比较两种方法的内点数不太可靠，应该设置误差阈值较大，比较两者的总误差大小
                    
                    // if(score_H > 2.0/3 * score_F)
                    if(score_H > 2.0/3 * score_F && (valid_FAST_H+valid_sift_H) >= 0.75*(valid_FAST_F+valid_sift_F) && valid_FAST_H+valid_sift_H > 10)
                    // if((valid_FAST_H+valid_sift_H) > 0.9*(valid_FAST_F+valid_sift_F))
                    // if(score_H > 0.9 * score_F || (valid_FAST_H+valid_sift_H) >= 1.0*(valid_FAST_F+valid_sift_F))
                    {
                        has_valid_F = false;
                        cout << "Try to get R and normalized t from matrix H!" << endl;
                    }
                    else
                    {
                        has_valid_H = false;
                        cout << "Try to get R and normalized t from matrix F!" << endl;
                    }
                }
                else if(!has_valid_F && !has_valid_H)
                {
                    cout << "Failed to get R and normalized t from matrix F or H!" << endl;
                    cal_Mat_F_H = false;
                }
                else if(has_valid_F)
                {
                    if(valid_FAST_F+valid_sift_F >= 8)
                        cout << "Try to get R and normalized t from matrix F!" << endl;
                    else
                    {
                        cout << "Failed to get R and normalized t from matrix F or H!" << endl;
                        has_valid_F = false;
                        cal_Mat_F_H = false;
                    }
                }
                else
                {
                    if(valid_FAST_H+valid_sift_H >= 8)
                        cout << "Try to get R and normalized t from matrix H!" << endl;
                    else
                    {
                        cout << "Failed to get R and normalized t from matrix F or H!" << endl;
                        has_valid_H = false;
                        cal_Mat_F_H = false;
                    }
                }
            }
            else
            {
                cout << "Succeed filter bg tracked fea using matrix F from predicted camera motion!" << endl;
            }
        }
    }
    else
    {
        has_valid_F = false;
        has_valid_H = false;
        fea_filtered = false;
    }

    // todo
    // bool filter_3D_2D = false;
    // 根据预测的运动对3D-2D匹配进行滤除
    // if(filter_3D_2D && (has_motion_pred_first_two_frame || frame_cnt > 1))
    // {

    // }

    if(filter_fea)
    {
        // 如果是通过IMU预测的运动构建的F矩阵过滤了匹配点
        if(!EASI_RANSAC_FH)
        {
            if(has_valid_F)
            {
                for(auto id: invalid_pt_id_F)
                {
                    if(id <= 0)
                    {
                        // 只删除纯背景跟踪点中的外点
                        if(obj_cls_id_sift[(-id)].first == 0)
                            status_sift[(-id)] = 0;
                    }
                    else
                    {
                        // 只删除纯背景跟踪点中的外点
                        if(obj_cls_id_FAST[(id-1)].first == 0)
                            status_FAST[(id-1)] = 0;
                    }

                    valid_match_sift = valid_sift_F;
                    valid_match_FAST = valid_FAST_F;
                }
                fea_filtered = true;
            }
            else
                fea_filtered = false;
        }
        // 如果是根据匹配点获得了F或H矩阵，则尝试通过分解该矩阵来得到R和t。
        else if(cal_Mat_F_H)
        {
            // 是否要保留近处的非平面点
            bool reserve_non_planar_pt = true;
            
            if(!need_PnP) reserve_non_planar_pt = false;

            if(has_valid_F)
            {
                // 分解得到 R 和 t的方向
                // Mat R_E_1, R_E_2, t_E;
                // 自己写分解E求解（R,t)时还需要三角化特征点以选择唯一正确的一组解。
                // DecomposeE(Mat_F, R_E_1, R_E_2, t_E);

                Point2d pp(SHIFT_X,SHIFT_Y);
                Matrix3d Mat_E = K_trans * F_cam * K;
                Mat E, R_E, t_E;
                eigen2cv(Mat_E,E);
                // 输入recoverPose的mask需要是Mat类型，单通道的数据。复制数据，而不仅仅是复制数据的指针
                Mat mat_temp = Mat(status_fea_F, true);
                
                Mat mat_status = mat_temp.reshape(1,status_fea_F.size());

                // // 此函数从E分解得到R和t时 会得到四组可能的解（R，t），其中t是尺度归一化的（这相当于固定了整个场景的尺度），
                // // 并利用给定的2d-2d匹配点对，使用三角化来恢复给定尺度下的点深度值，只有正确的一对(R,t)才会使得所有3D点的深度值为正。
                int num_inliers = recoverPose(E, prev_pts_for_FH, cur_pts_for_FH, R_E, t_E, FOCAL_LENGTH_X, pp, mat_status);
                
                if(num_inliers > 0.5*num_inliers_F)
                {
                    cv2eigen(R_E,R_from_E);
                    
                    Quaterniond delta_Q(R_from_E);
                    double delta_ang = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

                    // 前后2帧之间汽车的旋转角不应该太大吧？0.1s的时间，最大应该是多少度？如果真的是快速转弯的话，视觉跟踪应该很难完成？
                    // 注意只有系统第3帧开始才会有相机运动的预测值
                    if (frame_cnt > 1 && (delta_ang > 3.2 && delta_ang > 2.5 * delta_angle))
                    {
                        cout << "Wrong estimate of rotation from matrix F!" << endl;
                        cout << "predicted delta_angle: " << delta_angle << ", delta_angle from matrix F: " << delta_ang << endl;
                        has_valid_F = false;
                        fea_filtered = false;
                    }
                    else if(frame_cnt == 1 && delta_ang >= 3.2)
                    {
                        // 1s内转弯大于25度认为不可能
                        cout << "Unbelievable estimate of rotation from matrix F in between the first two frames!" << endl;
                        cout << "delta_angle from matrix F: " << delta_ang << endl;
                        has_valid_F = false;
                        fea_filtered = false;
                    }
                    else
                    {
                        cv2eigen(t_E,t_from_E);
                        cout << "Succeed in getting R and t direction with F matrix!" << endl;

                        // 删除F排除的跟踪外点
                        for(auto id: invalid_pt_id_F)
                        {
                            if(id <= 0)
                            {
                                // 只删除纯背景跟踪点中的外点
                                if(obj_cls_id_sift[(-id)].first == 0)
                                    status_sift[(-id)] = 0;
                            }
                            else
                            {
                                // 只删除纯背景跟踪点中的外点
                                if(obj_cls_id_FAST[(id-1)].first == 0)
                                    status_FAST[(id-1)] = 0;
                            }
                        }

                        valid_match_sift = valid_sift_F;
                        valid_match_FAST = valid_FAST_F;

                        fea_filtered = true;

                        // 如果当前帧是IMU初始化之后，则后续可能不需要PnP,则这里只排除跟踪外点，后续不单独恢复估计的位移的尺度
                        // 这种情况下后续是否需要PnP取决于变量fea_filtered
                        if(!need_PnP) has_valid_F = false;
                    }
                }
                else
                {
                    cout << "Failed to get R and t direction with F matrix!" << endl;
                    has_valid_F = false;
                    fea_filtered = false;
                }
            }
            else
            {
                vector<Mat> R_H, t_H, p_normal_H;
                Mat K_Mat, R_H_valid, t_H_valid;
                eigen2cv(K, K_Mat);
                decomposeHomographyMat(Mat_H, K_Mat, R_H, t_H, p_normal_H);

                // 需要对得到的多个R和t进行检验
                // int num_cand = R_H.size();
                // cout << "Got " << num_cand << " candidate R and t" << endl;
                
                // t的格式为3*1的Mat，元素形式为6，即CV_64F，即double形单通道
                // auto SIZE = t_H[0].size();
                // cout << "row of t is " << SIZE.height << " col of t is " << SIZE.width << endl;
                // auto type = t_H[0].type();
                // cout << "type of element in t is " << type << endl;
                
                // int k = 0;
                // for(auto &iter: t_H)
                // {
                //     cout << "No." << k++ << " translation is " << iter << endl;
                // }
                
                vector<int> status_RT;

                // 这个函数一般会返回一个或2个解。但是也有可能没找到任何有效解，因为这个函数要求给定的“所有”检验点在三角化后“均有”正的深度值，但是实际上可能有少数的点会不符合要求（通常是这些点的视差太小，甚至小于匹配误差）
                // 因此，采用将每个点逐一进行检验的策略，这样可以统计对于每个候选解其有效点的比例，最后选择比例最高的解
                // filterHomographyDecompByVisibleRefpoints(R_H, p_normal_H, prev_pts_for_F, cur_pts_for_F, status_RT, status_fea_H);
                
                vector<Point2f> prev_Pt, cur_Pt;
                vector<float> valid_solu_ratio;
                // vector<uchar> status(1,1);
                int num_pt_check = 0;
                int num_cand_solu = R_H.size();
                
                if(num_cand_solu > 0)
                {
                    vector<int> num_valid_pt_for_solu(num_cand_solu,0);
                    for(int i = 0; i < num_track_for_F; ++i)
                    {
                        if(status_fea_H[i] == 0) continue;
                        prev_Pt.clear();
                        cur_Pt.clear();
                        status_RT.clear();
                        ++num_pt_check;
                        prev_Pt.push_back(prev_pts_for_FH[i]);
                        cur_Pt.push_back(cur_pts_for_FH[i]);
                        // todo:这里是否需要开启多线程进行验证？
                        filterHomographyDecompByVisibleRefpoints(R_H, p_normal_H, prev_Pt, cur_Pt, status_RT);
                        for(auto &id:status_RT)
                        {
                            num_valid_pt_for_solu[id] += 1;
                        }
                    }
                    
                    status_RT.clear();
                    
                    if(num_pt_check > 0)
                    {
                        map<float,int,greater<float>> ratio_valid;
                        float ratio;
                        for(int i = 0; i < num_cand_solu; ++i)
                        {
                            ratio = num_valid_pt_for_solu[i]*1.0/num_pt_check;
                            if(ratio > 0)
                                ratio_valid[ratio] = i;
                        }

                        // for(auto &iter: ratio_valid)
                        // {
                        //     cout << "ratio of valid check pt for No." << iter.second << " solution is: " << iter.first << endl;
                        // }
                        
                        int count = 0;
                        for(auto &iter: ratio_valid)
                        {
                            float ratio_thres = 0.4;
                            if(iter.first >= ratio_thres)
                            {
                                status_RT.push_back(iter.second);
                                valid_solu_ratio.push_back(iter.first);
                                // cout << "z component of translation is " << t_H[iter.second].at<double>(2,0) << endl;
                            }
                            ++count;
                        }
                    }
                }

                if(status_RT.empty())
                {
                    // cout << "num of valid solution: " << status_RT.size() << endl;
                    // assert(false && "Weired! There are no valid solution after decomposition of H!");

                    has_valid_H = false;
                    fea_filtered = false;
                }
                else
                {
                    int valid_id = -1;
                    bool has_valid = false;
                    if(status_RT.size() > 1)
                    {
                        bool has_best = false, has_second_best = false;
                        float best_ratio, second_best_ratio;
                        for(int j = 0; j < status_RT.size(); ++j)
                        {
                            int id = status_RT[j];
                            // todo: 假设汽车是往前走的，至少静止，不会是倒退
                            if(t_H[id].at<double>(2,0) > 0) continue;
                            
                            if(!has_best) 
                            {
                                valid_id = id;
                                has_best = true;
                                best_ratio = valid_solu_ratio[j];
                                has_valid = true;
                            }
                            else
                            {
                                has_second_best = true;
                                second_best_ratio = valid_solu_ratio[j];
                                break;
                            }
                        }

                        // todo:是否需要比较第一和第二好结果的内点比例？
                        if(has_valid)
                        {
                            if(has_best && has_second_best)
                            {
                                // 在ORB-SLAM2中，不仅要比较第一高分和第二高分的比例，还要内点的平均视差大于最小阈值，成功三角化的点（三维坐标没有哪个值是inf或nan）数大于最小阈值，且最高分的结果的内点比例大于最小阈值（0.9）
                                // 上面我们是借助opencv的函数对每个点进行三角化，无法统计每个点的视差角，因此内点中无法包含那些较远而导致三角化深度为负的点，会导致正确解的内点数减少
                                if(second_best_ratio > 0.75*best_ratio)
                                {
                                    cout << "inliers ratio of best sloution: " << best_ratio << endl;
                                    cout << "inliers ratio of second best sloution: " << second_best_ratio << endl;
                                    has_valid = false;
                                //     assert(false && "Best and second best solution has little difference!");
                                }
                            }
                        }
                    }
                    else
                    {
                        int id = status_RT[0];
                        if(t_H[id].at<double>(2,0) < 0)
                        {
                            valid_id = id;
                            has_valid = true;
                        }
                    }
                    
                    if(has_valid)
                    {
                        R_H_valid = R_H[valid_id].clone();
                        
                        cv2eigen(R_H_valid,R_from_E);
                        Quaterniond delta_Q(R_from_E);
                        double delta_ang = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

                        // 前后2帧之间汽车的旋转角不应该太大吧？0.1s的时间，最大应该是多少度？如果真的是快速转弯的话，视觉跟踪应该很难完成？
                        if (frame_cnt > 1 && (delta_ang >= 3.2 && delta_ang > 2.5 * delta_angle))
                        {
                            cout << "Wrong estimate of rotation from matrix H!" << endl;
                            cout << "predicted delta_angle: " << delta_angle << ", delta_angle from matrix H: " << delta_ang << endl;
                            has_valid_H = false;
                            fea_filtered = false;
                        }
                        else if(frame_cnt == 1 && delta_ang > 3.2)
                        {
                            // 1s内转弯大于20度认为是不可能的
                            cout << "Unbelievable estimate of rotation from matrix H in between the first two frames!" << endl;
                            cout << "delta_angle from matrix H: " << delta_ang << endl;
                            has_valid_H = false;
                            fea_filtered = false;
                        }
                        else
                        {
                            valid_match_sift = valid_sift_H;
                            valid_match_FAST = valid_FAST_H;

                            t_H_valid = t_H[valid_id].clone();
                            cv2eigen(t_H_valid,t_from_E);

                            for(auto id: invalid_pt_id_H)
                            {
                                if(id <= 0)
                                {
                                    if(!reserve_non_planar_pt)
                                    {
                                        // 只删除纯背景跟踪点中的外点
                                        if(obj_cls_id_sift[(-id)].first == 0)
                                            status_sift[(-id)] = 0;
                                    }
                                    else
                                    {
                                        // 其中部分跟踪点是静态物体的跟踪点，而这些物体跟踪点中有一部分可能是新的跟踪点，还未在地图中
                                        reserve_bg_track_pt_id.insert(ids_sift[(-id)]);
                                    }
                                }
                                else
                                {
                                    if(!reserve_non_planar_pt)
                                    {
                                        // 只删除纯背景跟踪点中的外点
                                        if(obj_cls_id_FAST[(id-1)].first == 0)
                                            status_FAST[(id-1)] = 0;
                                    }
                                    else
                                    {
                                        reserve_bg_track_pt_id.insert(ids_FAST[(id-1)]);
                                    }
                                }
                            }

                            fea_filtered = true;

                            if(!need_PnP) has_valid_H = false;

                            cout << "Succeed got R and t direction with H matrix!" << endl;
                        }
                    }
                    else
                    {
                        cout << "Failed to get R and t direction with H matrix!" << endl;
                        has_valid_H = false;
                        fea_filtered = false;

                        // assert(false && "Weired! There are no valid solution of t after decomposition of H!");
                    }
                }
            }
            // cout << "translation direction is " << t_from_E.transpose() << endl;

            if(!fea_filtered)
                cal_Mat_F_H = false;
        }
    }

    // 暂时不再这里删除prev_sift中的跟踪丢失点
    // reduce_invalid_fea(true);

    if(fea_filtered)
    {
        printf("FM ransac for sift: %d -> %lu: %f\n", num_track_sift, valid_match_sift, (1.0 * valid_match_sift) / num_track_sift);
        
        // 删除无效的FAST跟踪点不在这里进行
        // reduce_invalid_fea(false);
        printf("FM ransac for FAST: %d -> %lu: %f\n", num_track_FAST, valid_match_FAST, (1.0 * valid_match_FAST) / num_track_FAST);
    }
    
    printf("FM ransac costs: %fms\n", t_f.toc());
}

// 读取相机的内参
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); ++i)
    {
        printf("reading paramerter of camera %s\n", calib_file[i].c_str());
        // ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        // instance()是创建一个相机工厂类对象的指针，而其成员函数generateCameraFromYamlFile是创建具体类型的相机类对象指针（返回时转化为父类指针）
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; ++i)
        for (int j = 0; j < row; ++j)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); ++i)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        // pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH_X + col / 2;
        // pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH_Y + row / 2;
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH_X + SHIFT_X;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH_Y + SHIFT_Y;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            assert(false && "Problem here!");
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    
    // turn the following code on if you need
    
    // while(true)
    // {
    //     cv::imshow(name, undistortedImg);
    //     // 一直等待用户按下ESC键（ASCI码为27）
    //     if(waitKey(0) == 27)
    //     {
    //         break;
    //     }
    // }

    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

// 将像素坐标去畸变并变换到归一化平面坐标(x,y,1.0).
// 这个函数和下面的第2个函数的函数签名是相同的（返回类型不作为签名的标志）！！！则想调用此函数时其实会调用下面的第2个函数，从而导致返回类型不匹配！
// vector<cv::Point2f> FeatureTracker::undistortedPts(const vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
// {
//     vector<cv::Point2f> un_pts;
//     for (unsigned int i = 0; i < pts.size(); ++i)
//     {
//         Eigen::Vector2d a(pts[i].x, pts[i].y);
//         Eigen::Vector3d b;
//         cam->liftProjective(a, b);
//         un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
//     }
//     return un_pts;
// }

void FeatureTracker::undistortedPts(const vector<cv::Point2f> &pts, vector<cv::Point2f> &un_pts, camodocal::CameraPtr cam, const vector<uchar> &status_pts, bool for_rigth_pts)
{
    if(!un_pts.empty()) un_pts.clear();
    bool has_status = false;
    has_status = (!status_pts.empty());
    for (unsigned int i = 0; i < pts.size(); ++i)
    {
        if(has_status)
        {
            if(status_pts[i] == 0) 
            {
                un_pts.emplace_back(0.0,0.0);
                continue;
            }
            else if (for_rigth_pts && status_pts[i] > 1)
            {
                un_pts.emplace_back(0.0,0.0);
                continue;
            }
        }
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        // 将像素点转换为归一化平面上的点（z为1）或者其他投影平面上的点
        cam->liftProjective(a, b);
        // 最终返回的就是归一化平面上的点坐标
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
        // cout << " Tracked undistored sift: " << b.x() << " " << b.y() << " " << b.z() << endl;
    }
}

void FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    for (unsigned int i = 0; i < pts.size(); ++i)
    {
        Eigen::Vector2d a((double)pts[i].x, (double)pts[i].y);
        Eigen::Vector3d b;
        // 将像素点转换为归一化平面上的点（z为1）或者其他投影平面上的点
        cam->liftProjective(a, b);
        // 最终返回的就是归一化平面上的点坐标
        pts[i].x = b.x() / b.z();
        pts[i].y = b.y() / b.z();
    }
}

void FeatureTracker::undistortedPts(vector<cv::Vec2f> &pts, camodocal::CameraPtr cam)
{
    for (unsigned int i = 0; i < pts.size(); ++i)
    {
        Eigen::Vector2d a(pts[i](0), pts[i](1));
        Eigen::Vector3d b;
        // 将像素点转换为归一化平面上的点（z为1）或者其他投影平面上的点
        cam->liftProjective(a, b);
        // 最终返回的就是归一化平面上的点坐标
        pts[i](0) = b.x() / b.z();
        pts[i](1) = b.y() / b.z();
    }
}

// 允许单个像素点进行去畸变
void FeatureTracker::undistortedPts(const cv::Point2f &pts, cv::Point2f &un_pts, camodocal::CameraPtr cam)
{
    Eigen::Vector2d a(pts.x, pts.y);
    Eigen::Vector3d b;
    cam->liftProjective(a, b);
    un_pts.x = b.x() / b.z();
    un_pts.y = b.y() / b.z();
}

void FeatureTracker::spaceToPlane(const Eigen::Vector3d& P, Point2f& p, camodocal::CameraPtr cam)
{
    Eigen::Vector2d a;
    cam->spaceToPlane(P, a);
    p.x = a(0);
    p.y = a(1);
}

// 使用两帧配对点的去畸变的归一化平面坐标，计算平面坐标的速度
void FeatureTracker::ptsVelocity(vector<cv::Point2f> &vel_pts, vector<int> &ids_pts, vector<cv::Point2f> &un_pts,
                                map<int, cv::Vec4f> &prev_id_un_pts, bool cal_vel, bool cal_map_un, const vector<uchar> &status_pts, bool for_right_pts)
{
    // 是否要在此函数内形成当前帧的map
    if (cal_map_un)
    {
        bool has_status = (!status_pts.empty());
        // prev_id_un_pts.clear();
        for (unsigned int i = 0; i < ids_pts.size(); ++i)
        {
            // 如果是无效点
            if(has_status) 
            {
                if(status_pts[i] == 0)
                    continue;

                // status不等于1，则是2或者3，即当前帧没有立体匹配的跟踪点或者新点
                if(for_right_pts && status_pts[i] > 1)
                    continue;
            }

            // 保留上一帧各个点的归一化平面的坐标和速度
            prev_id_un_pts[ids_pts[i]] = cv::Vec4f(un_pts[i].x, un_pts[i].y, vel_pts[i].x, vel_pts[i].y);
        }
    }

    // caculate points velocity
    if(cal_vel)
    {
        bool has_status = (!status_pts.empty());
        vel_pts.clear();
        if (!prev_id_un_pts.empty())
        {
            double dt = cur_time - prev_time;
            std::map<int, cv::Vec4f>::iterator it;
            for (unsigned int i = 0; i < un_pts.size(); ++i)
            {
                if(has_status)
                {
                    // 无效的点
                    if(status_pts[i] == 0)
                    {
                        vel_pts.emplace_back(0.0f, 0.0f);
                        continue;
                    }
                    // 当前帧的点可能没有右观测，则直接赋值0
                    else if(for_right_pts && status_pts[i] > 1)
                    {
                        vel_pts.emplace_back(0.0f, 0.0f);
                        continue;
                    }
                }

                it = prev_id_un_pts.find(ids_pts[i]);
                // 注意，该点在上一帧中不一定有立体匹配！！！
                if (it != prev_id_un_pts.end())
                {
                    float v_x = (un_pts[i].x - it->second[0]) / dt;
                    float v_y = (un_pts[i].y - it->second[1]) / dt;
                    // vel_pts.push_back(cv::Point2f(v_x, v_y));
                    vel_pts.emplace_back(v_x, v_y);
                }
                // 如果当前帧的某特征地图点在上一帧中没有被观测和匹配到，则该点在两帧图像中的速度为0。即当前帧新点
                else
                    vel_pts.emplace_back(0.0f, 0.0f);
            }
        }
        else
        {
            // for (unsigned int i = 0; i < ids_pts.size(); ++i)
            // {
            //     vel_pts.push_back(cv::Point2f(0, 0));
            // }
            vel_pts.resize(ids_pts.size(),cv::Point2f(0, 0));
        }
    }
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftFASTMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); ++j)
    {
        double len = std::min(1.0, 1.0 * track_cnt_FAST[j] / 20.0);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); ++i)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); ++i)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftFASTMap.find(id);
        if(mapIt != prevLeftFASTMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); ++i)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

// VINS版本的预测下一帧各个特征点的2d像素坐标，这里是将预测的3D点投影到2D图像上
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    // hasPrediction = true;
    predict_FAST.clear();
    //predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids_FAST.size(); ++i)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids_FAST[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            // 预测默认是用3D运动模型，而不是光流的预测。不用检查该预测点是否在图像区域之外吗？
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_FAST.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            //predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            // 注意，这里的prev_FAST实际上应该是当前帧的特征点像素坐标，即已经实施 prev_FAST = cur_FAST
            predict_FAST.push_back(prev_FAST[i]);
    }
}

// 当前滑窗内静态地图点的一部分要么在初始位姿估计（这只在VI没有完成初始化时会进行）中成为外点；要么在LBA完成后（VI初始化之后）重投影误差大于阈值。
// 这些静态点已经被作为外点从滑窗的地图点集中去除了，这里需要在跟踪线程的变量中去除与其中某些点相关的信息
// 而在MVIO中，需要去除的点集中还包含各物体上的FAST匹配外点(sift回收为新特征点)、位姿估计中的FAST外点（sift回收为新特征点），以及先前特征点跟踪时没有立体深度估计的点
void FeatureTracker::removeOutliers(set<int> &removePtsIds, const vector<int> &reserve_sift, const vector<int> &reserve_FAST)
{
    // std::set<int>::iterator itSet;
    // vector<uchar> status_FAST;
    // // ids中保存的是当前帧中保留的特征点（包括从上一帧跟踪到的和当前帧中检测和新添的）所对应的(滑窗内的）特征地图点的全局id。
    // for (size_t i = 0; i < ids_FAST.size(); ++i)
    // {
    //     itSet = removePtsIds.find(ids_FAST[i]);
    //     if(itSet != removePtsIds.end())
    //         status_FAST.push_back(0);
    //     else
    //         status_FAST.push_back(1);
    // }

    // // 因为当前帧中某些观测点是跟踪自上一帧的，因此在上一帧中也要去除这些点
    // reduceVector(prev_FAST, status_FAST);
    // reduceVector(ids_FAST, status_FAST);
    // // track_cnt保存的是对应ids的特征地图点在滑窗内被观测了几帧（包括当前帧的观测）
    // reduceVector(track_cnt_FAST, status_FAST);

    // 在VINS中，这里是直接去除点集中的外点；而在MVIO中，这里暂时只是先设置该点的清除标志
    auto iter_sift_end = reserve_sift.end();
    auto iter_FAST_end = reserve_FAST.end();
    for(auto &pt_id: removePtsIds)
    {
        // removePtsIds中的点不一定都是在当前帧中被跟踪点，因为它是LBA后总的重投影误差大于阈值的所有点
        if(gl_id_index_map.find(pt_id) == gl_id_index_map.end())
            continue;
        
        // 最新帧的sift跟踪外点需要保留为新点
        int index = gl_id_index_map[pt_id];
        if (index <= 0)
        {
            index = -1 * index;
            // 如果该点被保留为新点了，则不在这里修改其status
            auto iter = std::find(reserve_sift.begin(),reserve_sift.end(),pt_id);
            if(iter == iter_sift_end)
                status_sift[index] = 0;
        }
        else
        {
            index -= 1;
            auto iter = std::find(reserve_FAST.begin(),reserve_FAST.end(),pt_id);
            if(iter == iter_FAST_end)
                statusLeftRIght[index] = 0;
        }
    }
}

void FeatureTracker::RemoveOutliers()
{
    cout << "Start remove fea outliers!" << endl;

    // statusLeftRIght在物体关联之后还会进行修改，因此下面部分要放在全部reduceVector完成之后
    // ----------------------------------------------------------------------------
    int j = 0;
    // int temp_num_track_FAST = 0;
    // 只用一次遍历，对所有变量统一进行删减操作
    for (int i = 0; i < ids_FAST.size(); ++i)
    {
        if(statusLeftRIght[i] > 0)
        {
            // 如果之前都没有出现需要删除的元素，则不用重新赋值
            if (j < i)
            {
                cur_FAST[j] = cur_FAST[i];
                ids_FAST[j] = ids_FAST[i];
                cur_FAST_dep[j] = cur_FAST_dep[i];
                track_cnt_FAST[j] = track_cnt_FAST[i];
                obj_cls_id_FAST[j] = obj_cls_id_FAST[i];
                cur_un_FAST[j] = cur_un_FAST[i];
                FAST_velocity[j] = FAST_velocity[i];
                prev_FAST_global_obj_id[j] = prev_FAST_global_obj_id[i];

                cur_right_FAST[j] = cur_right_FAST[i];
                cur_un_right_FAST[j] = cur_un_right_FAST[i];
                right_FAST_velocity[j] = right_FAST_velocity[i];
                
                statusLeftRIght[j] = statusLeftRIght[i];
                // if (i < num_track_FAST && ids_FAST[i])
                // {
                //     prev_FAST_global_obj_id[j] = prev_FAST_global_obj_id[i];
                //     prev_FAST_dep[j] = prev_FAST_dep[i];
                // }
            }

            // if (i < num_track_FAST && ids_FAST[i] <= last_id_track_FAST_cur) ++temp_num_track_FAST;
            ++j;
        }
    }
    
    // 最终有效的跟踪点数量
    // num_track_FAST = temp_num_track_FAST;
    
    // resize会去除前j个元素之后的所有元素
    if(j < ids_FAST.size())
    {
        statusLeftRIght.resize(j);
        cur_FAST.resize(j);
        ids_FAST.resize(j);
        cur_FAST_dep.resize(j);
        track_cnt_FAST.resize(j);
        obj_cls_id_FAST.resize(j);
        cur_un_FAST.resize(j);
        FAST_velocity.resize(j);
        cur_right_FAST.resize(j);
        cur_un_right_FAST.resize(j);
        right_FAST_velocity.resize(j);
        prev_FAST_global_obj_id.resize(j);
    }
    
    // cout << "Succeeded remove FAST outliers!" << endl;

    // 删除无效的sift点
    // int temp_num_track_sift = 0;
    j = 0;
    // int num_sift_bg_cur_ = num_sift_bg_cur;
    for (int i = 0; i < ids_sift.size(); ++i)
    {
        if (status_sift[i] > 0)
        {
            // 如果之前都没有出现需要删除的元素，则不用重新赋值
            if (j < i)
            {
                cur_sift_index[j] = cur_sift_index[i];
                cur_sift[j] = cur_sift[i];
                ids_sift[j] = ids_sift[i];
                cur_sift_dep[j] = cur_sift_dep[i];
                track_cnt_sift[j] = track_cnt_sift[i];
                obj_cls_id_sift[j] = obj_cls_id_sift[i];
                cur_un_sift[j] = cur_un_sift[i];
                sift_velocity[j] = sift_velocity[i];
                prev_sift_global_obj_id[j] = prev_sift_global_obj_id[i];

                cur_right_sift[j] = cur_right_sift[i];
                cur_un_right_sift[j] = cur_un_right_sift[i];
                right_sift_velocity[j] = right_sift_velocity[i];

                status_sift[j] = status_sift[i];
            }
            // 这里需要加上id的范围限制，因为有些跟踪点在匹配或者运动估计时被当作外点，可能转而被当作新特征点，但它们在vector中的排序仍然没有改变！
            // if (i < num_track_sift && ids_sift[i] <= last_id_track_fea_cur) ++temp_num_track_sift;
            // if (i < num_track_sift) ++temp_num_track_sift;
            ++j;
        }
        else
        {
            // 目的是给下一帧masked_img寻找sift跟踪点时减小搜寻范围
            // if(i < num_sift_bg_cur_)
            //     --num_sift_bg_cur;
        }
    }

    // 最终有效的跟踪点数量
    // num_track_sift = temp_num_track_sift;

    // num_track_fea_prev = temp_num_track_FAST + temp_num_track_sift;

    // resize会去除前j个元素之后的所有元素
    if(j < ids_sift.size())
    {
        status_sift.resize(j);
        cur_sift_index.resize(j);
        cur_sift.resize(j);
        ids_sift.resize(j);
        cur_sift_dep.resize(j);
        track_cnt_sift.resize(j);
        obj_cls_id_sift.resize(j);
        cur_un_sift.resize(j);
        sift_velocity.resize(j);
        cur_right_sift.resize(j);
        cur_un_right_sift.resize(j);
        right_sift_velocity.resize(j);
        prev_sift_global_obj_id.resize(j);
    }

    // cout << "Succeeded remove sift outliers!" << endl;
    
    // 更新prev_un_Fea_map 和 prev_un_r_Fea_map
    // 在这里进行clear而不在ptsVelocity()函数内进行clear，不然第一次调用时写入的数据会被第2次调用所删除！
    prev_un_Fea_map.clear();
    // 上一帧图像点的速度也需要保存，因为有些点其上一帧观测没加入地图（如上一滑窗marg了次新帧），而且其在更之前的观测已经加入。当前帧需要一次性加入两帧观测，这时就需要上一帧点的速度
    ptsVelocity(FAST_velocity, ids_FAST, cur_un_FAST, prev_un_Fea_map, false, true);
    ptsVelocity(sift_velocity, ids_sift, cur_un_sift, prev_un_Fea_map, false, true);
    // 对于没有stereo match的点，则不记录其右特征点（因为都是无效信息）
    prev_un_r_Fea_map.clear();
    ptsVelocity(right_FAST_velocity, ids_FAST, cur_un_right_FAST, prev_un_r_Fea_map, false, true, statusLeftRIght, true);
    ptsVelocity(right_sift_velocity, ids_sift, cur_un_right_sift, prev_un_r_Fea_map, false, true, status_sift, true);
    
    // 赋值prevLeftFASTMap和prevRightFeaMap，保存的当前帧的像素点。一方面用于下一帧的图像中画track点示例，另一方面是下一帧可能还要将当前帧的某些新特征点的观测加入静态地图
    // 为什么不把这些map和上面的prev_un_Fea_map等直接合并？使用一个Vec6f就行了呀？
    prevLeftFeaMap.clear();
    prevRightFeaMap.clear();
    for(int i = 0; i < ids_FAST.size(); ++i)
    {
        prevLeftFeaMap[ids_FAST[i]] = cur_FAST[i];
        // 右图像的匹配像素点有需要吗？需要
        if (statusLeftRIght[i] == 1) 
            prevRightFeaMap[ids_FAST[i]] = cur_right_FAST[i];
    }
    for (int i = 0; i < ids_sift.size(); ++i)
    {
        prevLeftFeaMap[ids_sift[i]] = cur_sift[i];
        if (status_sift[i] == 1)
            prevRightFeaMap[ids_sift[i]] = cur_right_sift[i];
    }
    
    // cout << "Succeeded build prev_map!" << endl;

    // ------------------------------------------------------------------------------------
    // 将所有的当前帧参数设置为prev
    // prev_prev_time = prev_time;
    prev_time = cur_time;
    prev_img = cur_img.clone();
    prev_img_r = cur_img_r.clone();
    // 这里可否使用std::move()转移这些内存？但是cur_FAST这个变量下一帧还要用到，到时为其分配新的内存吗？上面用map保存这些变量了，方便索引
    if(!prev_FAST.empty()) prev_FAST.clear();
    prev_FAST = cur_FAST;

    if(!prev_FAST_dep.empty()) prev_FAST_dep.clear();
    prev_FAST_dep = cur_FAST_dep;
    
    if(!prev_sift_index.empty()) prev_sift_index.clear();
    prev_sift_index = cur_sift_index;

    if(!prev_sift.empty()) prev_sift.clear();
    prev_sift = cur_sift;

    if(!prev_sift_dep.empty()) prev_sift_dep.clear();
    prev_sift_dep = cur_sift_dep;

    last_id_track_fea_prev = last_id_track_fea_cur;

    // num_sift_bg_prev = num_sift_bg_cur;

    // 每一帧点跟踪结束后就将此值设为false，直到后续根据当前帧的位姿和恒速假设计算出下一帧的点的预测才重新设为true
    hasPrediction = false;
    
    ave_dep_bg_prev_frame = 0.0;
    if(num_bg_with_dep > 0)
        ave_dep_bg_prev_frame = ave_dep_bg_cur_frame/num_bg_with_dep;
    
    // -------------------------------------------------------------------

    prev_mask_solid_objs = mask_solid_objs.clone();

    cout << "Susseeded remove fea outliers and build map of fea in prev frame!!" << endl;
}

void FeatureTracker::clear_var()
{
    // 将上一帧的变量及时清空
    cur_FAST.clear();
    cur_un_FAST.clear();
    cur_FAST_dep.clear();
    FAST_velocity.clear();
    cur_right_FAST.clear();
    cur_un_right_FAST.clear();
    right_FAST_velocity.clear();

    cur_sift_index.clear();
    cur_sift.clear();
    cur_un_sift.clear();
    cur_sift_dep.clear();
    sift_velocity.clear();
    cur_right_sift.clear();
    cur_un_right_sift.clear();
    right_sift_velocity.clear();

    id_FAST_no_depth.clear();
    id_sift_no_depth.clear();
    FAST_no_stereo_bg.clear();
    sift_no_stereo_bg.clear();
    status_FAST.clear();
    statusLeftRIght.clear();
    status_sift.clear();
    id_bg_track_sift.clear();
    id_bg_track_FAST.clear();

    gl_id_index_map.clear();

    obj_fea_disp_num.clear();

    predict_dep_FAST.clear();
    FAST_new_objs.clear();

    pts_for_cal_F.clear();
    temp_pts_for_F.clear();

    // 用不到
    // prev_un_FAST.clear();
    // prev_un_sift.clear();

    valid_detect_obj.clear();

    TrackObjFeaFrame.clear();

    NewObjFeaFrame.clear();

    num_obj_sift.clear();

    num_obj_FAST.clear();

    TrackBgFea.clear();

    FinalTrackObjFea.clear();

    FinalTrackObj.clear();

    FinalTrackCurObj.clear();

    ParLostObjPrevBg.clear();

    TotalLostObjPrevBg.clear();

    detect_lost_objs_cur.clear();

    FinalLostObjPrev.clear();

    new_objs_cur.clear();

    cur_stat_objs.clear();

    cur_dyn_objs.clear();

    lose_objs_cur_bg.clear();

    fea_cur_lose_objs.clear();

    invalid_stat_objs.clear();

    reserve_bg_track_pt_id.clear();

    // 此变量需要在下一帧中使用到，暂不清除
    // sta_obj_fea_in_map.clear();
    
    sta_obj_fea_in_map_cur.clear();

    // added_short_track_Fea.clear();

    ave_dep_bg_cur_frame = 0.0;
    num_bg_with_dep = 0;

    last_id_track_FAST_cur = 0;
    last_id_track_fea_cur = 0;
    num_sift_bg_cur = 0;
    ave_disp_obj = 0.0;
    ave_disp_bg = 0.0;
    // 这些变量在下一帧检测sift之前可能需要用到，此处先不重置
    // num_new_sift_bg = 0;
    // num_bg_sift_with_dep = 0;
    num_track_fea_static = 0;
    num_track_FAST = 0;
    num_track_sift = 0;

    num_long_track_fea_stat = 0;

    cal_Mat_F_H = false;
    fea_filtered = false;
    appro_num_track_stat_fea = 0;
    use_prev_fea = false;

    has_lost_obj_prev = false;

    num_old_track_fea = 0;
    num_track_fea_with_dep_prev = 0;

    num_up_half = 0;
    
    // has_motion_pred_first_two_frame = false;
}

cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
