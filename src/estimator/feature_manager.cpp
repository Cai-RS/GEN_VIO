/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    // 根据起始帧id和被观测帧数相加就得到最后一个被观测帧，这说明了每个特征点从起始帧到最新的被观测帧应该是被持续观测到的！
    // 一个点在窗口内最大的观测帧数就是窗口的长度，即WINDOW_SIZE+1，而frame_count的起始值是0，因此其最大值为WINDOW_SIZE
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
    
    has_stereo_rectified = true;
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= TH_NUM_FRAME_FOR_LBA)
        {
            cnt++;
        }
    }
    // 参与LBA的点数不应该超过特征点深度的预分配内存大小
    cnt = cnt < NUM_OF_F ? cnt : NUM_OF_F;
    return cnt;
}

int FeatureManager::getFeatureCountLBAcur(int frameCnt)
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= TH_NUM_FRAME_FOR_LBA && it.estimated_depth > 0 && it.endFrame() == frameCnt)
        {
            cnt++;
        }
    }
    
    // 参与LBA的点数不应该超过特征点深度的预分配内存大小
    cnt = cnt < NUM_OF_F ? cnt : NUM_OF_F;
    return cnt;
}

// td为相机和IMU之间的时间戳差异，IMU真实对应时间戳=相机时间戳+td
// 添加新的特征地图点和增加某地图点的观测帧记录的操作都是在FeatureManager::addFeatureCheckParallax()函数中
// 另外根据次新帧和次次新帧之间匹配点的平均视差，来决定是否将次新帧作为关键帧，从而决定是要marg掉次新帧还是最老帧（前提是当前滑窗帧数已满）
// VINS-Mono中此函数代码的标注 https://blog.csdn.net/weixin_42846216/article/details/106614794 
bool FeatureManager::addFeatureCheckParallax(int frame_count, double prev_td, double td, FeatureTracker &tracker, int &num_track_add, bool use_fea_no_depth)
{
    // if(frame_count == 0) return true;
    
    // const vector<int> &new_objs_cur = tracker.new_objs_cur;
    FeaObjFrame &NewObjFeaFrame = tracker.NewObjFeaFrame;
    map<int, float> &cur_stat_objs = tracker.cur_stat_objs;
    const set<int> &invalid_stat_objs = tracker.invalid_stat_objs;
    
    //ROS_DEBUG("input feature: %d", (int)image.size());
    //ROS_DEBUG("num of feature: %d", getFeatureCount());
    new_feature_num = 0;

    // 是否要添加当前帧新检测的静态特征点，包括背景点和已知静态物体中的（后续参与位姿估计的动态物体中如果有确定为静态的，则也会添加进来）？
    // 要注意，当前帧中添加的所有背景中的FAST点，以及静态物体的特征点，在下一帧的跟踪中即使有找到匹配，其匹配也可能不会被放进下一帧的匹配集中（因为静态匹配点有数量限制），因此如果此时加入新点，这些只有一帧观测的地图点后续还要清除！
    // 这就会使得后续维护静态地图（从其中删除最新帧没有跟踪到，且历史跟踪帧数比较少的点）的时间会稍长一点。不如此处暂时不加入新地图点，等到下一帧其被跟踪到时，再和下一帧观测一起加入地图（某个FAST点一旦之前有了两帧观测，则第三帧起每次跟踪必定加入静态点集）！
    if(0)
    {
        for (auto &obj: NewObjFeaFrame)
        {
            if (obj.first == 0 || cur_stat_objs.find(obj.first) != cur_stat_objs.end())
            {
                if (invalid_stat_objs.find(obj.first) != invalid_stat_objs.end()) continue;
                
                for (auto &pt: obj.second)
                {
                    // 每一帧暂时不把新静态特征点加入地图！!而是等到下一帧看它是否被跟踪到
                    
                    {
                        feature.push_back(FeaturePerId(pt.first, frame_count));
                        feature.back().feature_per_frame.emplace_back(pt.second[0].second, td);
                        if(pt.second.size() == 2)
                        {
                            feature.back().feature_per_frame.back().rightObservation(pt.second[1].second);
                            // 如果有对原图像对进行立体校正，则在特征点跟踪中是有对当前特征点的深度估计。在本项目中，考虑到动态物体的特征点需要每一帧都进行深度估计，因此默认进行了图像的立体校正。
                            // 但是也可以不对原图进行立体校正，这样每一帧都要使用triangulatePoint()函数来估计左相机中该点的深度（此时只能每次都假设左相机为临时的全局坐标系，计算得到的是左相机中该点的3维坐标）
                            if (has_stereo_rectified)
                                feature.back().estimated_depth = pt.second[0].second(2);
                        }
                    }
                }
            
                new_feature_num += obj.second.size();
            }
        }
    }
    
    if(frame_count == 0) return true;

    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    long_track_num = 0;

    // 对于map变量，如果要使用[]对其元素进行索引，则不能用const修饰它！因为map的[]会在索引key不存在时自动创建pair，而这与const相违背！
    FeaFrame &sta_match_fea = tracker.TrackBgFea;
    const vector<float> &prev_FAST_dep = tracker.prev_FAST_dep;
    const vector<float> &prev_sift_dep = tracker.prev_sift_dep;
    const vector<float> &cur_FAST_dep = tracker.cur_FAST_dep;
    const vector<float> &cur_sift_dep = tracker.cur_sift_dep;
    map<int,int> &gl_id_index_map = tracker.gl_id_index_map;

    // 新物体的特征点也在这里添加进静态地图？不，下一帧确定该物体是否为静态物体之后再加入，以免动态点对位姿估计造成影响，另外后续要删除的话也麻烦

    // 上一帧该特征点在左相机归一化平面的点坐标和速度
    auto &un_pts_prev = tracker.prev_un_Fea_map;
    // 上一帧该特征点的左图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_prev = tracker.prevLeftFeaMap;
    // 上一帧该特征点在右相机归一化平面的点坐标和速度
    auto &un_pts_r_prev = tracker.prev_un_r_Fea_map;
    // 上一帧该特征点的右图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_r_prev = tracker.prevRightFeaMap;
    
    int index;
    float dep_prev, dep_cur;
    // 添加静态的跟踪点的信息，这些点如果在后续的位姿估计中有被当作外点的，要如何删除？在地图点的外点移除函数中会实现
    if(!sta_match_fea.empty())
    {
        for (auto &id_pts: sta_match_fea)
        {
            // ！！下面注释掉的部分为原VINS-FUSION代码
            // // 左图像跟踪点(每次都要构造一个临时的f_per_fra变量太浪费内存和时间了，直接emplace_back就行)
            // FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
            // // 为什么要assert?image的中vector的int表示的是该特征点的观测是在左相机(0)还是右相机(1)上
            // // 该帧相机帧中对某特征地图点的观测，必须首先要有在左图像中的观测（可以是从上一帧中跟踪到，也可以是在当前帧左图像中检测的新特征点），如果左图像的点在右图像中有匹配点，则再作为观测插入
            // assert(id_pts.second[0].first == 0);
            // if(id_pts.second.size() == 2)
            // {
            //     // FeaturePerFrame类中的这个函数被调用时，说明该地图点在此帧的左右图像中均有观测（左图像中的特征点用光流在右图像中匹配到了相关点）！则会设置对象中的is_stereo为true！
            //     // 该地图点在右图像中的观测放在当前帧左图像该点的FeaturePerFrame中
            //     f_per_fra.rightObservation(id_pts.second[1].second);
            //     assert(id_pts.second[1].first == 1);
            // }

            int feature_id = id_pts.first;
            // 给定lambda函数体来完成搜寻，其中使用了捕获
            auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &iter)
                            {
                                return iter.feature_id == feature_id;
                            });
            // 添加当前滑窗内的新的地图点。如果是系统的首帧，则该帧所有的观测特征点都是新检测点，全部添加进滑窗的地图点集合feature中
            // if (it == feature.end())
            // {   
            //     // 新特征地图点在当前滑窗内的全局ID其实是在视觉前端（即当前帧对上一帧的跟踪，以及检测新特征点）中就创建的；当前镇为该点的首个被观测帧
            //     feature.push_back(FeaturePerId(feature_id, frame_count));
            //     feature.back().feature_per_frame.push_back(f_per_fra);
            //     new_feature_num++;
            // }
            // else
            // {
            //     // 已有的地图点添加观测记录
            //     it->feature_per_frame.push_back(f_per_fra);
            //     last_track_num++;
            //     // 统计被当前帧跟踪到的点中，之前已经至少被连续跟踪过3次（即连续观察到4帧） 的点数
            //     if( it-> feature_per_frame.size() >= 4)
            //         long_track_num++;
            // }

            // 如果sta_match_fea中的跟踪点在之前的帧就已经加入静态地图了
            if (it != feature.end())
            {
                // 如果该点在上一帧的观测已加入，说明该点在上一帧不是新点
                if(it->endFrame() == frame_count - 1)
                {
                    // assert(id_pts.second[0].first == 0);
                    // 已有的地图点添加观测记录，此记录中不需要有帧序号信息，因为点的跟踪是连续的，只需要记录初始观测帧的序号即可
                    it->feature_per_frame.emplace_back(id_pts.second[0].second, td);
                    if(id_pts.second.size() == 2) 
                    {
                        // assert(id_pts.second[1].first == 1);
                        it->feature_per_frame.back().rightObservation(id_pts.second[1].second);

                        int index = gl_id_index_map[feature_id];
                        float dep_cur;
                        // 如果是静态物体点，则其深度值已经直接计算出来了
                        if(index > 0)
                        {
                            dep_cur = cur_FAST_dep[index-1];
                        }
                        else
                        {
                            dep_cur = cur_sift_dep[-index];
                        }

                        if(dep_cur > 0)
                            it->feature_per_frame.back().depth = dep_cur;
                    }
                }
                // 如果上一帧的观测还没加入，则将两帧观测一起加入。这种情况其实不会发生？！
                else
                {
                    if(it->endFrame() != frame_count - 2)
                        assert(false);

                    Eigen::Matrix<double, 8, 1> pts_l, pts_r;
                    if(un_pts_prev.find(feature_id) == un_pts_prev.end())
                        assert(false && "Something wrong with tracker.prev_un_Fea_map!");
                    
                    // 由于上一帧不是首帧，因此上一帧点的归一化平面的速度需要给出，这在LBA时需要使用！
                    pts_l << un_pts_prev[feature_id][0], un_pts_prev[feature_id][1], 1.0, pts_prev[feature_id].x, pts_prev[feature_id].y, un_pts_prev[feature_id][2], un_pts_prev[feature_id][3], 0;
                    
                    it->feature_per_frame.emplace_back(pts_l,prev_td);

                    // bool has_depth_prev = false;
                    // index = gl_id_index_map[feature_id];
                    // if(index > 0)
                    // {
                    //     index -= 1;
                    //     dep_prev = prev_FAST_dep[index];
                    //     if(dep_prev > 0)
                    //         has_depth_prev = true;
                    // }
                    // else
                    // {
                    //     index = -1 * index;
                    //     dep_prev = prev_sift_dep[index];
                    //     if(prev_sift_dep[index] > 0)
                    //         has_depth_prev = true;
                    // }

                    int index = gl_id_index_map[feature_id];
                    
                    // if(has_depth_prev && pts_r_prev.find(feature_id) != pts_r_prev.end())
                    if(pts_r_prev.find(feature_id) != pts_r_prev.end())
                    {
                        pts_r << un_pts_r_prev[feature_id][0], un_pts_r_prev[feature_id][1], 1.0, pts_r_prev[feature_id].x, pts_r_prev[feature_id].y, un_pts_r_prev[feature_id][2], un_pts_r_prev[feature_id][3], 0;
                        it->feature_per_frame.back().rightObservation(pts_r);

                        float dep_prev;
                        // 如果是静态物体点，则其深度值已经直接计算出来了
                        if(index > 0)
                        {
                            dep_prev = prev_FAST_dep[index-1];
                        }
                        else
                        {
                            dep_prev = prev_sift_dep[-index];
                        }

                        if(dep_prev > 0)
                            it->feature_per_frame.back().depth = dep_prev;
                    }

                    // 再添加当前帧观测
                    it->feature_per_frame.emplace_back(id_pts.second[0].second,td);
                    if(id_pts.second.size() == 2)
                    {
                        it->feature_per_frame.back().rightObservation(id_pts.second[1].second);
                        float dep_cur;
                        // 如果是静态物体点，则其深度值已经直接计算出来了
                        if(index > 0)
                        {
                            dep_cur = cur_FAST_dep[index-1];
                        }
                        else
                        {
                            dep_cur = cur_sift_dep[-index];
                        }

                        if(dep_cur > 0)
                            it->feature_per_frame.back().depth = dep_cur;
                    }
                }
                // ++last_track_num;
                // 统计被当前帧跟踪到的点中，之前已经至少被连续跟踪过3次（即连续观察到4帧） 的点数
                if(it-> feature_per_frame.size() >= TH_NUM_FRAME_FOR_LBA)
                    ++long_track_num;
            }
            // 如果该静态跟踪特征点还未出现在地图中。
            // 那么该点是该静态物体（或背景）在上一帧的新点
            // 因此当前帧一个点被加入地图，则它至少已经被上一帧所观测到！
            else
            {
                // ！！形成观测首帧的信息，即只有归一化平面坐标、像素坐标和全局物体id（这个可以随便写，用不到），速度值都是0？
                // 不需要使用td来对该点的观测首帧的图像预测与IMU时刻对齐的虚拟像素点吗？？原VINS项目确实是这样的，即td值是给了的，但是每帧的新特征点肯定没有速度值，因此其实每个点在其观测首帧是没法被用来对齐该帧图像与IMU的！!但是可以借助其他观测首帧不是该帧的点来对齐该帧！
                // 那么问题来了，如何保证每一帧都有点可以用来对齐图像与IMU的时间戳？
                // feature.push_back(FeaturePerId(feature_id, frame_count-1));
                feature.emplace_back(feature_id, frame_count-1);
                Eigen::Matrix<double, 8, 1> pts_l, pts_r;

                if(un_pts_prev.find(feature_id) == un_pts_prev.end())
                    assert(false);

                // 这里上一帧点没加入当前帧还有一种可能，即上一帧该物体在运动估计后才确认为静态物体，而在那之前为动态物体，恰好上一个滑窗是marg次新帧，则上一帧该新静态物体的点还未加入，等待当前帧该点的跟踪结果。但是该点上一帧是已经有二维平面速度的计算值了！！
                // pts_l << un_pts_prev[feature_id][0], un_pts_prev[feature_id][1], 1.0, pts_prev[feature_id].x, pts_prev[feature_id].y, 0.0, 0.0, 0;
                // pts_r << tracker.prev_un_r_Fea_map[feature_id].x, tracker.prev_un_r_Fea_map[feature_id].y, 1.0, tracker.prevRightFeaMap[feature_id].x, tracker.prevRightFeaMap[feature_id].y, 0.0, 0.0, 0;
                pts_l << un_pts_prev[feature_id][0], un_pts_prev[feature_id][1], 1.0, pts_prev[feature_id].x, pts_prev[feature_id].y, un_pts_prev[feature_id][2], un_pts_prev[feature_id][3], 0;
                
                feature.back().feature_per_frame.emplace_back(pts_l, prev_td);
                // assert(pts[0].first == 0);
                feature.back().feature_per_frame.emplace_back(id_pts.second[0].second, td);
                
                bool has_stereo_prev = false;
                bool has_stereo_cur = false;
                // 每个点在该帧下有没有深度值，与其在该帧中有没有右图像观测不一定相关，例如没有立体校正的情况下，该点的深度值需要后续三角化才能恢复
                if(pts_r_prev.find(feature_id) != pts_r_prev.end())
                {
                    pts_r << un_pts_r_prev[feature_id][0], un_pts_r_prev[feature_id][1], 1.0, pts_r_prev[feature_id].x, pts_r_prev[feature_id].y, un_pts_r_prev[feature_id][2], un_pts_r_prev[feature_id][3], 0;
                    feature.back().feature_per_frame[0].rightObservation(pts_r);
                    has_stereo_prev = true;
                }

                // 在这个项目中其实默认每个参与估计的点要有右观测。但是其实也可以没有，这样在后续的三角化测量函数中可以进行多视角下的深度估计！
                if(id_pts.second.size() == 2)
                {
                    // assert(pts[1].first == 0);
                    feature.back().feature_per_frame[1].rightObservation(id_pts.second[1].second);
                    // 如果有对原图像对进行立体校正，则在特征点跟踪中是有对当前特征点的深度估计。在本项目中，考虑到动态物体的特征点需要每一帧都进行深度估计，因此默认进行了图像的立体校正。
                    // 但是也可以不对原图进行立体校正，这样每一帧都要使用triangulatePoint()函数来估计左相机中该点的深度（此时只能每次都假设左相机为临时的全局坐标系，计算得到的是左相机中该点的3维坐标）

                    has_stereo_cur = true;
                }

                bool has_dep_prev = false;
                bool has_dep_cur = false;
                // 对于跟踪点在上一帧的深度，如果其在上一帧有立体匹配，其在prev_FAST_dep或prev_sift_dep中此时还没有效深度。
                // 被跟踪点在上一帧如果有立体匹配，则留到当前帧在进行PnP前再完成立体三角化
                // 这里有上一帧深度值的跟踪点 要么是在上一帧没有立体匹配，要么是静态物体的跟踪点
                index = gl_id_index_map[feature_id];
                if(index > 0)
                {
                    index -= 1;
                    
                    dep_prev = prev_FAST_dep[index];
                    if(dep_prev > 0)
                        has_dep_prev = true;

                    dep_cur = cur_FAST_dep[index];
                    if(dep_cur > 0)
                        has_dep_cur = true;
                }
                else
                {
                    index = -1 * index;
                    dep_prev = prev_sift_dep[index];
                    if(dep_prev > 0)
                        has_dep_prev = true;
                    
                    dep_cur = cur_sift_dep[index];
                    if(dep_cur > 0)
                        has_dep_cur = true;
                }

                if(has_dep_prev)
                {
                    feature.back().estimated_depth = dep_prev;
                    // 首帧下是否有立体匹配。后续不用再对该立体匹配进行立体三角化
                    if(has_stereo_prev) feature.back().feature_per_frame[0].depth = dep_prev;
                }
                else
                {
                    feature.back().estimated_depth = -1.0;
                }

                if(has_stereo_cur && has_dep_cur)
                {
                    feature.back().feature_per_frame[1].depth = dep_cur;
                }
                
                // ++last_track_num;
            }
        }
    }

    int num_new_ = 0;
    // 当前帧背景点中（如果是跟踪点，则是纯背景跟踪点）没有立体匹配深度的点（跟踪点或新点）之前是没有加入TrackObjFeaFrame和NewObjFeaFrame中的，因此它们还未被加入地图中
    if(use_fea_no_depth)
    {   
        uchar status;
        int pt_id;
        Eigen::Matrix<double, 8, 1> pts_l, pts_r, pts_l_1;
        pts_l(7) = 0;
        pts_r(7) = 0;
        pts_l_1(7) = 0;
        double depth;
        
        vector<uchar> *status_fea;
        vector<int> *ids_fea, *fea_no_stereo_bg;
        vector<Point2f> *cur_un_fea, *cur_fea, *fea_velocity;
        vector<float> *prev_fea_dep;
        for(int k = 0; k < 2; ++k)
        {
            if(k == 0)
            {
                status_fea = &(tracker.statusLeftRIght);
                ids_fea = &(tracker.ids_FAST);
                fea_no_stereo_bg = &(tracker.FAST_no_stereo_bg);
                cur_un_fea = &(tracker.cur_un_FAST);
                cur_fea = &(tracker.cur_FAST);
                fea_velocity = &(tracker.FAST_velocity);
                prev_fea_dep = &(tracker.prev_FAST_dep);
            }
            else
            {
                status_fea = &(tracker.status_sift);
                ids_fea = &(tracker.ids_sift);
                fea_no_stereo_bg = &(tracker.sift_no_stereo_bg);
                cur_un_fea = &(tracker.cur_un_sift);
                cur_fea = &(tracker.cur_sift);
                fea_velocity = &(tracker.sift_velocity);
                prev_fea_dep = &(tracker.prev_sift_dep);
            }

            for(auto &id: (*fea_no_stereo_bg))
            {
                status = (*status_fea)[id];
                // 跟踪点。
                if(status == 2)
                {
                    // 这里不用添加该点观测了，这些当前帧没有深度值的背景跟踪点在上面已经加入地图了（即跟踪点都在TrackObjFeaFrame中）
                    // continue;
                    ++last_track_num;
                    pts_l_1(0) = (*cur_un_fea)[id].x;
                    pts_l_1(1) = (*cur_un_fea)[id].y;
                    pts_l_1(2) = 1.0;
                    pts_l_1(3) = (*cur_fea)[id].x;
                    pts_l_1(4) = (*cur_fea)[id].y;
                    pts_l_1(5) = (*fea_velocity)[id].x;
                    pts_l_1(6) = (*fea_velocity)[id].y;

                    pt_id  = (*ids_fea)[id];
                    auto it = find_if(feature.begin(), feature.end(), [pt_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == pt_id;
                                    });
                    // 该跟踪点还未加入地图，说明是上一帧的新点
                    if(it == feature.end())
                    {
                        feature.emplace_back(pt_id, frame_count-1);

                        if(un_pts_prev.find(pt_id) == un_pts_prev.end())
                            assert(false);
                        
                        pts_l(0) = un_pts_prev[pt_id][0];
                        pts_l(1) = un_pts_prev[pt_id][1];
                        pts_l(2) = 1.0;
                        pts_l(3) = pts_prev[pt_id].x;
                        pts_l(4) = pts_prev[pt_id].y;
                        pts_l(5) = un_pts_prev[pt_id][2];
                        pts_l(6) = un_pts_prev[pt_id][3];
                        // pts_l(7) = 0;
                        feature.back().feature_per_frame.emplace_back(pts_l, prev_td);

                        bool has_stereo_prev = false;
                        if(pts_r_prev.find(pt_id) != pts_r_prev.end())
                        {
                            pts_r(0) = un_pts_r_prev[pt_id][0];
                            pts_r(1) = un_pts_r_prev[pt_id][1];
                            pts_r(2) = 1.0;
                            pts_r(3) = pts_r_prev[pt_id].x;
                            pts_r(4) = pts_r_prev[pt_id].y;
                            pts_r(5) = un_pts_r_prev[pt_id][2];
                            pts_r(6) = un_pts_r_prev[pt_id][3];
                            feature.back().feature_per_frame[0].rightObservation(pts_r);

                            has_stereo_prev = true;
                        }

                        // 再添加当前帧左图像的观测
                        feature.back().feature_per_frame.emplace_back(pts_l_1,td);

                        // 上一帧该点是否有有效的深度
                        depth = (*prev_fea_dep)[id];
                        if (depth > 0)
                        {
                            feature.back().estimated_depth = depth;

                            // 特征点的每一帧观测中的深度表示其是否有立体匹配且三角化成功
                            if(has_stereo_prev)
                                feature.back().feature_per_frame[0].depth = depth;
                        }
                        else
                        {
                            feature.back().estimated_depth = -1.0;
                        }
                    }
                    else if(it->endFrame() != frame_count)
                    {
                        // debug.该跟踪点的上一帧观测应该已经在地图中
                        if(it->endFrame() != frame_count-1)
                            assert(false);
                        
                        it->feature_per_frame.emplace_back(pts_l_1,td);

                        if(it->feature_per_frame.size() >= TH_NUM_FRAME_FOR_LBA)
                            ++long_track_num;
                    }
                    
                    // 没有立体匹配的背景跟踪点后续还要更新其在当前相机坐标系下的深度值，这里暂时不修改其status（==2），以便后续寻找这些点
                    // (*status_fea)[id] = 1;
                }
                // 没有深度估计的新点暂时不加入
                else if(status == 3)
                {
                    ++num_new_;
                    // 这里对这些没有stereo match的new bg点也不修改status
                    // (*status_fea)[id] = 1;
                    continue;
                }
            }
        }
    }
    
    // 这里使用tracker中计数的所有已确认的静态的物体上的跟踪点总数，而不是这里要加入地图的跟踪点数。
    // 因为部分仅上一帧和当前帧观测到的背景FAST点和静态物体FAST点可能并没有要加入地图！然而上面在计算新的特征点数时却是所有静态点均计算，这会影响到下面marg哪一帧的判断！！
    // last_track_num = tracker.appro_num_track_stat_fea + num_track_;
    last_track_num += sta_match_fea.size();
    num_track_add = last_track_num;
    new_feature_num += num_new_;
    
    cout << "Num of static tracked fea (with or without depth) in map of current frame: " << last_track_num << endl;
    
    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    // 如果滑窗内当前只有2帧，或者当一帧于上一帧之间的跟踪点数小于20，或者当前所跟踪点中属于长久跟踪的点数少于40个，或者当前帧新增特征点数大于跟踪点数的一半，则次新帧被认为是关键帧，需要保留
    // 因为系统首帧一定是关键帧，所以当前帧是系统第2帧，则无需下面的判断。
    // 这个判断逻辑是基于VINS的特征点跟踪和检测函数，但是此项目中没有在当前帧检测新的背景点，因此这个判断在此项目中无法使用，只能使用平均视差来判断
    // if (frame_count < 2 || last_track_num < 35 || long_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    //     return true;
    
    // 如果非初始前两帧并且当前帧和上一帧之间的跟踪结果比较好（即跟踪点数较多，新检测点数相比较而言较少，且连续跟踪次数大于3的点数较多），则进一步计算视差以决定marg哪一帧
    for (auto &it_per_id : feature)
    {
        // 找到次新帧和次次新帧之间的共视地图点
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    // 如果两帧没有共视点，则次新帧为关键帧（此时不就是上一帧已经跟丢了吗？是的，如果IMU已经初始化了，则应该可以靠IMU延续轨迹；否则，上一帧就应该重新初始化了？）
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        //ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        //ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH_X);
        //TODO： 乘以焦距的目的是什么？平时两帧之间点的平均深度吗？MIN_PARALLAX这个参数对于不同场景应该要有不同的设置（例如室内场景和室外场景）？
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH_X;
        // 如果次新帧和次次新帧之间关联特征点的平均视差大于阈值，则认为次新帧为关键帧，需要被保留。如果当前滑窗帧数已满，则会marg掉最老的一帧；否则，marg掉最新帧
        // 首个滑窗满帧之前，每一帧都被保留；随着首个滑窗满帧之后，随着每次插入新的关键帧（即确定次新帧为关键帧），则会逐渐将最老的帧给marg掉，直到最后滑窗内剩下的都是真正的关键帧！
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

// 当前帧的某些静态物体（物体关联时期为静态，但仅有2帧观测，在VI初始化后需要等待是否marg次新帧再决定是否加入地图）
// 或动态物体（指的是在物体关联阶段没法确定是否为静态的，在完成位姿估计后被确定为静态），使用此函数将这些静态物体的跟踪特征点观测（上一帧以及当前帧的）加入到静态地图中
// 如果当前帧marg了次新帧，则暂时不把当前帧该物体的观测加入地图（除非加上当前这一帧观测，该特征点在map中的观测数达到了参与LBA的最低帧数）
// 由于这里只会添加物体点，所以这些点在每一帧都是有depth的（物体点必须是有stereo match）
void FeatureManager::addStaticFeature(int frame_count, int prev_td, double td, FeatureTracker &tracker, const vector<pair<int,int>> &id_fea, const int id_obj_cur, 
                                        const vector<int> &reserve_new_sift, const vector<int> &ignore_pts, bool add_new_fea, int global_cls)
{
    // 上一帧该特征点在左相机归一化平面的点坐标和速度
    auto &un_pts_prev = tracker.prev_un_Fea_map;
    // 上一帧该特征点的左图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_prev = tracker.prevLeftFeaMap;
    // 上一帧该特征点在右相机归一化平面的点坐标和速度
    auto &un_pts_r_prev = tracker.prev_un_r_Fea_map;
    // 上一帧该特征点的右图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_r_prev = tracker.prevRightFeaMap;

    for (auto &pt_id: id_fea)
    {
        int old_pt_id = pt_id.first;
        auto it = find_if(feature.begin(), feature.end(), [old_pt_id](const FeaturePerId &it)
                        {
                            return it.feature_id == old_pt_id;
                        });
        
        // 由于不添加新点，只添加跟踪点，因此这里前后2个点的id应该是一样的，不然就是代码逻辑有bug
        if(old_pt_id != pt_id.second)
        {
            cout << "added original pt id: " << old_pt_id << endl;
            cout << "added new pt id: " << pt_id.second << endl;
            assert(false);
        }
        
        auto &pts = tracker.TrackObjFeaFrame[id_obj_cur][old_pt_id];
        // 该点之前还未加入地图
        // 此种情况可以是 该物体在上一帧为动态物体（则pt_id的first和second不会相同）或者是新物体，则其所有的特征点都不会在地图中。
        // 也可能是上上帧的新点，在上一帧跟踪到该点并确认为静态物体，但是上一滑窗marg了次新帧，则上一帧的跟踪暂未加入，则此时该点不在地图中。
        // 此时pt_id.first和pt_id.second是一样的值，即该点在前后两帧的全局id是不变的
        if (it == feature.end())
        {
            // 如果某些跟踪点（即运动估计外点）是无效的，但是被保留为新点（当前帧有深度估计），则这里暂时不添加
            if (find(reserve_new_sift.begin(),reserve_new_sift.end(),old_pt_id) != reserve_new_sift.end()) continue;
            // 如果是已经被删除了的点（即运动估计外点且不能保留为新点）
            if (find(ignore_pts.begin(),ignore_pts.end(),old_pt_id) != ignore_pts.end()) continue;
            // 形成观测首帧的信息，即只有归一化平面坐标、像素坐标和全局物体id（这个可以随便写，用不到），速度值都是0。
            // 注意观测首帧是上一帧，即frame_count-1
            // feature.push_back(FeaturePerId(pt_id.second, frame_count-1));
            feature.push_back(FeaturePerId(old_pt_id, frame_count-1));
            Eigen::Matrix<double, 8, 1> pts_l, pts_r;

            if(un_pts_prev.find(old_pt_id) == un_pts_prev.end())
            {
                cout << "old_pt_id" << old_pt_id << endl;
                cout << "new_pt_id" << pt_id.second << endl;
                assert(false && "Something wrong with tracker.prev_un_Fea_map!");
            }
            
            pts_l << un_pts_prev[old_pt_id][0], un_pts_prev[old_pt_id][1], 1.0, pts_prev[old_pt_id].x, pts_prev[old_pt_id].y, 0.0, 0.0, 0;
            
            // list并不会进行排序
            feature.back().feature_per_frame.emplace_back(pts_l, prev_td);
            // assert(pts[0].first == 0);
            feature.back().feature_per_frame.emplace_back(pts[0].second, td);
            
            bool has_stereo_prev = false;
            if(un_pts_r_prev.find(old_pt_id) != un_pts_r_prev.end())
            {
                pts_r << un_pts_r_prev[old_pt_id][0], un_pts_r_prev[old_pt_id][1], 1.0, pts_r_prev[old_pt_id].x, pts_r_prev[old_pt_id].y, 0.0, 0.0, 0;
                feature.back().feature_per_frame[0].rightObservation(pts_r);
                has_stereo_prev = true;
            }

            int index = tracker.gl_id_index_map[old_pt_id];
            float dep_prev;
            uchar cls;
            // FAST点
            if (index > 0)
            {
                dep_prev = tracker.prev_FAST_dep[index-1];
                cls = tracker.obj_cls_id_FAST[(index-1)].first;
            }
            else
            {
                dep_prev = tracker.prev_sift_dep[-1*index];
                cls = tracker.obj_cls_id_sift[(-1*index)].first;
            }

            // 如果时物体点，则该物体在上一帧和当前帧都应该有深度值才对，不论其深度值是否来自立体匹配
            if(dep_prev > 0)
            {
                feature.back().estimated_depth = dep_prev;
                // 如果上一帧的深度值来自于立体匹配
                if (has_stereo_rectified && has_stereo_prev)
                {
                    feature.back().feature_per_frame[0].depth = dep_prev;
                }
            }
            else
            {
                if(cls > 0)
                {
                    assert(false && "why obj fea has no depth?");
                }
            }

            bool has_stereo_cur = false;
            float dep_cur; 
            // 如果当前帧有右观测
            if(pts.size() == 2)
            {
                // assert(pts[1].first == 0);
                feature.back().feature_per_frame[1].rightObservation(pts[1].second);
                has_stereo_cur = true;
                // 如果有对原图像对进行立体校正，则在特征点跟踪中是有对当前特征点的深度估计。在本项目中，考虑到动态物体的特征点需要每一帧都进行深度估计，因此默认进行了图像的立体校正。
                // 但是也可以不对原图进行立体校正，这样每一帧都要使用triangulatePoint()函数来估计左相机中该点的深度（此时只能每次都假设左相机为临时的全局坐标系，计算得到的是左相机中该点的3维坐标）
            }

            if (index > 0)
            {
                dep_cur = tracker.cur_FAST_dep[index-1];
            }
            else
            {
                dep_cur = tracker.cur_sift_dep[-1*index];
            }
            
            if(dep_cur <= 0)
            {
                if(cls > 0)
                    assert(false && "why obj fea has no depth?");
            }
            else
            {
                if(has_stereo_rectified && has_stereo_cur)
                    feature.back().feature_per_frame[1].depth = dep_cur;
            }
        }

        // 该静态点在之前就已经加入了地图。但是该点上一帧的观测不一定已经在地图中。无论如何，该静态“点”在上一帧之前就已经加入了地图。
        // 这种情况可以是 该物体一直是静态物体，而且该点是在上上上帧或更早之前就加入地图且一直跟踪，但在上一帧marg了次新帧，则上一帧观测暂时不在该点的记录中
        // 也可以是该静态物体上一帧有部分漏检或全部漏检，而且上一帧这些漏检的背景点也是跟踪自上上帧的背景点
        //（这意味着这部分物体漏检了至少两帧，但是一直都作为背景点被跟踪着。如果该点上一帧观测还未加入地图，则是该点观测在上一帧没有加入一开始的静态点集，且上一帧marg了次新帧）。
        else
        {

            int index = tracker.gl_id_index_map[old_pt_id];
            float dep_prev, dep_cur;
            uchar cls;
            // FAST点
            if (index > 0)
            {
                dep_prev = tracker.prev_FAST_dep[index-1];
                dep_cur = tracker.cur_FAST_dep[index-1];
                cls = tracker.obj_cls_id_FAST[(index-1)].first;
            }
            else
            {
                dep_prev = tracker.prev_sift_dep[-1*index];
                dep_cur = tracker.cur_sift_dep[-1*index];
                cls = tracker.obj_cls_id_sift[(-1*index)].first;
            }

            // 如果该点上一帧的观测也还未加入。则先加入上一帧的观测。
            // 这有2种情况： 1. 只存在于上一帧滑窗已满且marg了次新帧，则该静态物体跟踪点在上一帧的观测暂时还没加入当前帧; 2. 该点为静态物体点，上一帧该点及之前该点已连续加入，但是当前帧该物体直到运动估计后才确定为静态。
            if(it->endFrame() < frame_count - 1)
            {
                Eigen::Matrix<double, 8, 1> pts_l, pts_r;
                if(un_pts_prev.find(old_pt_id) == un_pts_prev.end())
                    assert(false && "Something wrong with tracker.prev_un_Fea_map!");
                
                // 由于上一帧不是首帧，因此上一帧点的归一化平面的速度需要给出，这在LBA时需要使用！
                pts_l << un_pts_prev[old_pt_id][0], un_pts_prev[old_pt_id][1], 1.0, pts_prev[old_pt_id].x, pts_prev[old_pt_id].y, un_pts_prev[old_pt_id][2], un_pts_prev[old_pt_id][3], 0;

                it->feature_per_frame.emplace_back(pts_l,prev_td);

                bool has_stereo_prev = false;
                if(un_pts_r_prev.find(old_pt_id) != un_pts_r_prev.end())
                {
                    pts_r << un_pts_r_prev[old_pt_id][0], un_pts_r_prev[old_pt_id][1], 1.0, pts_r_prev[old_pt_id].x, pts_r_prev[old_pt_id].y, un_pts_r_prev[old_pt_id][2], un_pts_r_prev[old_pt_id][3], 0;
                    it->feature_per_frame.back().rightObservation(pts_r);
                    has_stereo_prev = true;
                }
                
                // 如果时物体点，则该物体在上一帧和当前帧都应该有深度值才对，不论其深度值是否来自立体匹配
                if(dep_prev > 0)
                {
                    // 如果上一帧的深度值来自于立体匹配
                    if (has_stereo_rectified && has_stereo_prev)
                    {
                        it->feature_per_frame.back().depth = dep_prev;
                    }
                }
                else
                {
                    if(cls > 0)
                    {
                        assert(false && "why obj fea has no depth?");
                    }
                }
            }

            // 再添加当前帧观测
            it->feature_per_frame.emplace_back(pts[0].second,td);
            bool has_stereo_cur = false;
            if(pts.size() == 2)
            {
                it->feature_per_frame.back().rightObservation(pts[1].second);
                has_stereo_cur = true;
            }

            if(dep_cur > 0)
            {
                // 如果上一帧的深度值来自于立体匹配
                if (has_stereo_rectified && has_stereo_cur)
                {
                    it->feature_per_frame.back().depth = dep_cur;
                }
            }
            else
            {
                if(cls > 0)
                {
                    assert(false && "why obj fea has no depth?");
                }
            }
        }
    }
    
    // 将该静态物体在特征跟踪阶段的新检测点也加入到地图中。此功能暂时不使用，新特征点要等到下一帧中被跟踪到时再一起加入静态地图！
    // 这部分功能不在此实现，而是直接在物体运动估计函数中完成了
    if(0)
    {
        bool has_to_change_cls = true;
        if(add_new_fea && id_obj_cur != 0)
        {
            for(const auto &pt:tracker.NewObjFeaFrame[id_obj_cur])
            {
                feature.push_back(FeaturePerId(pt.first, frame_count));
                feature.back().feature_per_frame.emplace_back(pt.second[0].second, td);
                if(pt.second.size() == 2)
                {
                    feature.back().feature_per_frame.back().rightObservation(pt.second[1].second);
                    // 如果有对原图像对进行立体校正，则在特征点跟踪中是有对当前特征点的深度估计。在本项目中，考虑到动态物体的特征点需要每一帧都进行深度估计，因此默认进行了图像的立体校正。
                    // 但是也可以不对原图进行立体校正，这样每一帧都要使用triangulatePoint()函数来估计左相机中该点的深度（此时只能每次都假设左相机为临时的全局坐标系，计算得到的是左相机中该点的3维坐标）
                    if (has_stereo_rectified)
                        feature.back().estimated_depth = pt.second[0].second(2);
                }
                
                if(global_cls != 0 && has_to_change_cls) 
                {
                    int index = tracker.gl_id_index_map[pt.first];
                    if (index > 0)
                    {
                        index = index - 1;
                        if (tracker.obj_cls_id_FAST[index].first == global_cls)
                        {
                            has_to_change_cls = false;
                            continue;
                        }
                        tracker.obj_cls_id_FAST[index].first = global_cls; 
                    }
                    else
                    {
                        index = -1 * index;
                        if (tracker.obj_cls_id_sift[index].first == global_cls)
                        {
                            has_to_change_cls = false;
                            continue;
                        }
                        tracker.obj_cls_id_sift[index].first = global_cls; 
                    }
                }
            }
        }
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < TH_NUM_FRAME_FOR_LBA || it_per_id.estimated_depth <= 0)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0 || it_per_id.estimated_depth > mThDepthBg)
        {
            it_per_id.solve_flag = 2;
        }
        else
            // solve_flag为该点是成功点的标志
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
        // 对于自始自终solve_flag一直是0的点，该怎么处理？因为有些点自从被添加之后，就没有参与过LBA呀？这些点既不参与LBA，也不会参与marg形成先验信息，只会在slide操作时逐帧地直接被删除相关信息！
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    // 
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < TH_NUM_FRAME_FOR_LBA)
            continue;
        // 地图中不是所有点的首观测帧均有有效的深度估计值（即还未完成三角化，那么该点就不应该参与LBA估计）
        if(it_per_id.estimated_depth <= 0)
            continue;
        // LBA优化时默认优化的是逆深度？对于自驾场景，是否应该使用正深度，因为场景中大部分点的深度应该都是大于1m的
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

bool FeatureManager::solvePoseByPnP(int frame_count, Eigen::Matrix3d &R, Eigen::Vector3d &P, vector<cv::Point2d> &pts2D, 
                                      vector<cv::Point3d> &pts3D, vector<int> &pts_id_vec, bool initial_succ)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r, inliers;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    // 内参矩阵设置为单位帧，则说明pts2D不是像素点坐标，而是归一化平面坐标
    cv::Mat M_K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    bool use_RANSAC = false;

    // 在未完成初始化之前，这里每一帧的相机位姿变换估计都用RANSAC，这样得到的相机估计会比较准确一点（则初始化期间的物体的帧间运动估计也会比较准确）。然而，在VI初始化之后，就不会再使用视觉匹配来估计相机的初始位姿了，而是直接用IMU！
    {
        // 特征点数比较少，因此使用较高的置信度
        // 需要迭代的次数和 选择的估计方法（即每次计算需要选取的点数，P3P为3个点）、要达到内点集的置信度 和 正确匹配点占点集的比例 相关。这里如果选择P3P，0.95置信度，0.7的内点比例，则只需要7次迭代...
        pnp_succ = cv::solvePnPRansac(pts3D, pts2D, M_K, D, rvec, t, true, 200, 4.0 / FOCAL_LENGTH_X, 0.99, inliers); // AP3P(5) EPNP(1) ITERATIVE DLS
        if(!pnp_succ)
        {
            printf("pnp using RANSAC with high confidence and accuracy failed ! Try again with lower conf and accu! \n");
            cv::Mat inliers_loose;
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_initial, t);
            // 此函数内应该是不会改变非空Mat的inliers的size的，如果直接使用上面的inliers，则两次估计时如果内点数不一样，则数量差异无法体现在inliers中！所以这里使用新的inliers_loose
            pnp_succ = cv::solvePnPRansac(pts3D, pts2D, M_K, D, rvec, t, true, 100, 7.0 / FOCAL_LENGTH_X, 0.95, inliers_loose);
            
            if(pnp_succ)
            {
                use_RANSAC = true;
                inliers_loose.copyTo(inliers);
            }
            else
            {
                printf("pnp using RANSAC failed ! Try using PnP without RANSAC! \n");
                cv::eigen2cv(R_initial, tmp_r);
                cv::Rodrigues(tmp_r, rvec);
                cv::eigen2cv(P_initial, t);
                pnp_succ = cv::solvePnP(pts3D, pts2D, M_K, D, rvec, t, 1);
            }
        }
        else
        {
            use_RANSAC = true;
        }
    }
    
    // VINS-Fusion项目中，此处只是用匹配点集估计两帧间的相机位姿，而没有将外点跟踪点剔除。如果总的跟踪长度小于4帧的点，在滑窗满了之后不会参与LBA，会被逐渐逐帧地删除观测；而大于3帧观测的点后续则会参与LBA，如果其总的投影误差仍大于阈值，则会被删除。
    // 标记运动估计的内点为负数。这里每一帧都将运动估计外点删除，以便下一帧添加新的特征点！但这样子会不会使得长跟踪的点数变少？
    // 
    if(use_RANSAC && !inliers.empty())
    {
        cout << "Get inliers of camera PnP!" << endl;
        for(int j = 0; j < inliers.rows; ++j)
        {
            // 使得该点的标记一定小于0
            pts_id_vec[inliers.at<int>(j)] = -1 * (pts_id_vec[inliers.at<int>(j)]+1);
        }
    }
    else if(!use_RANSAC)
    {
        pts_id_vec.clear();
    }
    
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

// 当系统估计相机位姿只能使用纯双目时才会调用此函数（这包含了在完成VI初始化之前的双目-IMU系统，因此IMU在初始化和对齐之前无法使用）
int FeatureManager::initFramePoseByPnP(int frameCnt, FeatureTracker &tracker, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], 
                                         Matrix3d &pred_R, Vector3d &pred_P, const Matrix3d &prev_cam_R, const Vector3d &prev_cam_P, 
                                         vector<int> &reserve_new_sift, int &num_track_cur_bg, bool use_IMU, bool initial_succ)
{
    int num_inlier_fea_PnP  = 0;
    // int num_track_fea_stat;
    // 注意，最开始进入这里处理的是系统的第二帧图像，此时有深度的特征点既包括上一帧和当前帧都被观测到的点，也包括当前帧新增加的特征点
    if(frameCnt > 0)
    {
        Matrix3d orig_pred_cam_R = pred_R;
        Vector3d orig_pred_cam_P = pred_P;

        vector<cv::Point2d> pts2D;
        vector<cv::Point3d> pts3D;
        vector<int> pt_id_vec;
        vector<list<FeaturePerId>::iterator> fea_iters;
        double dep;
        int num_frame, start_frame;
        set<int> selected_pt;
        for (list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
        {
            int num_frame = it_per_id->feature_per_frame.size();
            if(num_frame < 2 || it_per_id->endFrame() != frameCnt) continue;
            // 应该从离当前帧最近 还是 最远的帧开始遍历该点的观测。
            // 这有好有坏，越早的帧，其全局位姿误差会越小，但是和当前帧的特征点匹配准度越低（毕竟是间接地匹配）！
            // 所以SOFT-SLAM中的设计是将某个跟踪点在当前帧的投影点与之前的帧中都分别进行匹配优化，即当前帧与之前的所有帧的匹配都是独立的，而且它是估计之前各帧与当前帧之间的相对运动，用以平滑当前帧与上一帧的运动！
            
            for(int i = (num_frame-2); i >= 0; --i)
            // for(int i = 0; i <= (num_frame-2); ++i)
            {
                // dep = it_per_id->estimated_depth;

                // 首先考察该点在每一帧下是否具有来自立体匹配的有效深度
                dep = it_per_id->feature_per_frame[i].depth;
                int index = (num_frame -1);
                if (dep > 0)
                {
                    // int index = frameCnt - it_per_id->start_frame;
                    // num_frame = it_per_id->feature_per_frame.size();
                    // 不可能大于，最多就是等于（即从首个被观测帧开始一直持续被跟踪到当前帧）
                    // =号其实还包括了当前帧中添加的新特征地图点！！！因为此时index=0
                    // 因此此处在原代码的基础上是否应该添加多一个条件，即只选择那些跟踪自上一帧的特征点的观测？跟踪点数应该是足够完成PnP估计的（在追踪线程中有阈值条件来判断是否跟踪成功）
                    // if((int)it_per_id.feature_per_frame.size() >= index + 1)
                    // if(num_frame >= index + 1 && num_frame > 1)

                    {
                        // start_frame = it_per_id->start_frame;

                        start_frame = frameCnt - (num_frame - 1 - i);

                        // 计算当前帧所观测到的特征地图点在其被观测首帧的相机坐标系下的3D坐标，并且转换到该帧的IMU坐标系下。此时应该要保证外参ric和tic是较为准确的。
                        // 应该是ptsInIMU吧？这里就是把IMU初始帧的位姿跟全局位姿绑定了，只不过优化g之后需要校正其roll和pitch角
                        Vector3d ptsInCam = ric[0] * (it_per_id->feature_per_frame[i].point * dep) + tic[0];
                        // 根据地图点被观测首帧图像所对应的IMU坐标系的位姿，将所有地图点投影到统一的当前假定的世界坐标系下！
                        // 因为Rs[0]是初始帧IMU坐标系相对于“当前世界坐标系（通过取加速度测量值的平均来作为首帧IMU坐标系中的g，从而得到一个与东北天坐标系的相对姿态）“的姿态（这个姿态无法使首帧IMU变换到准确的东北天坐标系）；
                        // 而在estimator.cpp文件的processImage()函数中，在完成VI初始化之前，对于每一帧处理结束前都将当前帧的Rs和Ps赋值为为下一帧的Rs和Ps，
                        // 因此之后每一帧的此处的Rs和Ps是用 前一帧可靠的视觉估计的位姿(加入上一帧在下面的solvePoseByPnP()视觉估计成功了) 和 其与前一帧之间的IMU的积分 来推断得到该帧IMU在假定世界坐标系下的位姿。这是不准确的估计！
                        // 如果上面if中不添加第二个条件，则对于当前帧中的新特征地图点，此时Rs是用预积分得到的当前帧IMU的全局位姿估计，用它来投影相机坐标系中的点是不准确的！！所以要避免使用当前帧中的新地图点！
                        // 所以上面if应该添加第二个条件，则此时所有要参与当前帧与上一帧的相机相对位姿估计的点都是用之前帧可靠的视觉估计位姿来投影到当前的世界坐标系中！
                        Vector3d ptsInWorld = Rs[start_frame] * ptsInCam + Ps[start_frame];

                        cv::Point3d point3d(ptsInWorld(0), ptsInWorld(1), ptsInWorld(2));
                        // 注意，这里给的2D点是当前帧归一化平面上的坐标
                        cv::Point2d point2d(it_per_id->feature_per_frame[index].point(0), it_per_id->feature_per_frame[index].point(1));
                        pts3D.push_back(point3d);
                        pts2D.push_back(point2d); 
                        int id = it_per_id->feature_id;
                        pt_id_vec.push_back(id);
                        fea_iters.push_back(it_per_id);

                        selected_pt.insert(id);
                    }
                    break;
                }
            }
        }
        
        num_inlier_fea_PnP = pts3D.size();

        cout << "Num of tracked fea (witn depth from stereo match) of bg for PnP in map: " << pts3D.size() << endl;
        if(num_inlier_fea_PnP <= 8)
        {
            float dep;
            // 如果之前有立体匹配的跟踪点太少，则使用上一帧有深度值的跟踪点（来自于运动更新得到的深度）
            for (list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
            {
                int gl_id = it_per_id->feature_id;
                if(selected_pt.find(gl_id) != selected_pt.end())
                    continue;
                
                int num_frame = it_per_id->feature_per_frame.size();
                if(num_frame < 2 || it_per_id->endFrame() != frameCnt) continue;
                
                int index = tracker.gl_id_index_map[gl_id];
                
                if(index > 0)
                {
                    dep = tracker.prev_FAST_dep[(index-1)];
                }
                else
                {
                    dep = tracker.prev_sift_dep[(-index)];
                }
                
                if(dep > 0)
                {
                    start_frame = frameCnt - 1;

                    Vector3d ptsInCam = ric[0] * (it_per_id->feature_per_frame[(num_frame-2)].point * dep) + tic[0];

                    Vector3d ptsInWorld = Rs[start_frame] * ptsInCam + Ps[start_frame];

                    cv::Point3d point3d(ptsInWorld(0), ptsInWorld(1), ptsInWorld(2));
                    // 当前帧归一化平面上的坐标
                    cv::Point2d point2d(it_per_id->feature_per_frame[(num_frame-1)].point(0), it_per_id->feature_per_frame[(num_frame-1)].point(1));
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                    pt_id_vec.push_back(gl_id);

                    fea_iters.push_back(it_per_id);
                }
            }

            num_inlier_fea_PnP = pts3D.size();
            // 如果最终当前帧的3D-2D跟踪点还是不够最低数量，则放弃PnP
            if(num_inlier_fea_PnP < 6)
            {
                cout << "Not enough 3D tracked feature for PnP of camera! Stereo camera will degenerate to mono case!" << endl;
                // PnP失败，认为跟踪点数量不足或质量不好，后续如果有新的静态物体，则添加其中质量较好的点到地图
                num_inlier_fea_PnP = 0;
                return num_inlier_fea_PnP;
            }
        }

        // 这里是使用上一帧相机在当前世界坐标下的位姿来作为当前帧相机位姿的初始值。这里是否改变为使用恒速模型下的预测位姿？
        // 两帧图像过后应该可以使用恒速运动模型，或者VI初始化之后可以使用IMU推测，来获得当前帧位姿的初始值！（但是VI初始化之后此函数就不会再被调用了）
        // 如果求解不成功（跟踪点少于4个）呢？那就不用设置Rs和Ps了吗？没有给系统传递任何信息吗？如果失败了，Rs和Ps中仍然只是预积分呀？！
        // 对于双目IMU系统而言，此处应该要保证相机的位姿估计是成功的！！否则当前帧与后续的Rs和Ps就都不再是根据视觉计算出来的各帧IMU坐标系相对于假定世界坐标系的位姿！！
        // 而根据视觉计算出来的相对姿态是恨重要的，被认为是准确的，之后IMU的预积分值要与这些视觉的结果进行对齐以估计IMU的参数和状态！！
        if(solvePoseByPnP(frameCnt, pred_R, pred_P, pts2D, pts3D, pt_id_vec, initial_succ))
        {
            Matrix3d R_cam_motion = pred_R.transpose() * prev_cam_R;
            Vector3d P_cam_motion = pred_R.transpose() * (prev_cam_P - pred_P);

            Quaterniond delta_Q(R_cam_motion);
            double delta_angle = acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0;
            if (fabs(delta_angle) >= 3.5 || P_cam_motion.norm() >= (2.5*orig_pred_cam_P.norm()))
            {
                // trans to w_T_imu
                // Rs[frameCnt] = orig_pred_cam_R * ric[0].transpose(); 
                // Ps[frameCnt] = -Rs[frameCnt] * tic[0] + orig_pred_cam_P;

                // 外点太多，则去除2/3的有深度的跟踪点，以便后续可以添加静态物体点
                num_track_cur_bg = num_track_cur_bg - 2.0 * num_inlier_fea_PnP / 3;

                num_inlier_fea_PnP = 0;

                return num_inlier_fea_PnP;
            }
            
            // trans to w_T_imu
            Rs[frameCnt] = pred_R * ric[0].transpose(); 
            Ps[frameCnt] = -Rs[frameCnt] * tic[0] + pred_P;
            
            // 更新参与PnP的特征点在当前帧的深度值，并去除外点。这只在使用cv::solvePnPRansac()时才会进行，如果使用cv::solvePnP()，则在此处无法排除外点！
            if(!pt_id_vec.empty())
            {
                Matrix3d motion_R = pred_R.transpose() * prev_cam_R;
                Vector3d motion_P = pred_R.transpose() * (prev_cam_P - pred_P);

                // Eigen::Quaterniond Q(Rs[frameCnt]);
                //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
                //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;

                std::map<int,int> &gl_id_index_map = tracker.gl_id_index_map;
                vector<int> &id_sift_no_depth = tracker.id_sift_no_depth;
                vector<int> &id_FAST_no_depth = tracker.id_FAST_no_depth;
                vector<int>::iterator iter_sift_end = id_sift_no_depth.end();
                vector<int>::iterator iter_FAST_end = id_FAST_no_depth.end();
                map<int, Vec4f> &prev_un_Fea_map = tracker.prev_un_Fea_map;
                vector<float> &cur_dep_FAST = tracker.cur_FAST_dep;
                vector<float> &cur_dep_sift = tracker.cur_sift_dep;
                vector<float> &prev_dep_FAST = tracker.prev_FAST_dep;
                vector<float> &prev_dep_sift = tracker.prev_sift_dep;
                vector<uchar> &statusLeftRIght = tracker.statusLeftRIght;
                vector<uchar> &status_sift = tracker.status_sift;
                vector<pair<uchar,int>> &obj_cls_id_sift = tracker.obj_cls_id_sift;
                vector<pair<uchar,int>> &obj_cls_id_FAST = tracker.obj_cls_id_FAST;
                
                int id, id_pt, index;
                Vector3d pt_prev, pt_cur;
                float cur_z, max_depth;
                
                // int num_inliers = 0;
                
                uchar cls_pt;
                int l_obj_id;
                for(int i = 0; i < pt_id_vec.size(); ++i)
                {
                    id = pt_id_vec[i];
                    // id为负的是运动估计内点。内点是否保留取决于其更新后的深度估计是否可靠。
                    // 当前帧点的深度值更新不在这里进行，而是在三角化测量函数中
                    if (id < 0)
                    {
                        // 更新跟踪内点的当前帧深度值不在这里进行，而是后续在三角化测量函数中进行
                        continue;

                        id_pt = -1 *id - 1;
                        index = gl_id_index_map[id_pt];
                        if(index > 0)
                        {
                            index -= 1;
                            cls_pt = obj_cls_id_FAST[index].first;
                            // 修改之后的背景点不会采用depth_map获取立体深度估计；但是静态物体的特征点是可能通过depth_map获取深度估计的，因此这里更新的其实是这些静态物体点的深度估计！
                            if (cls_pt == 0 || std::find(id_FAST_no_depth.begin(), iter_FAST_end, index) == iter_FAST_end) continue;
                            pt_prev(2) = prev_dep_FAST[index];
                            pt_prev(0) = prev_un_Fea_map[id_pt](0) * pt_prev(2);
                            pt_prev(1) = prev_un_Fea_map[id_pt](1) * pt_prev(2);

                            cur_z = (motion_R * pt_prev + motion_P)(2);

                            if(cls_pt == 0)
                                max_depth = mThDepthBg;
                            else
                                max_depth = mThDepthObj;

                            if (cur_z < max_depth && cur_z > mMinDepthPt)
                            {
                                // ++num_inliers;
                                cur_dep_FAST[index] = cur_z;
                            }
                            else
                            {
                                statusLeftRIght[index] = 0;
                                // 如果深度值为负（会有这种情况发生吗？？），则删除该点当前帧观测在地图中的记录（或者直接将该点全部记录从地图中删除？）
                                if(cur_z <= 0)
                                {
                                    fea_iters[i]->feature_per_frame.pop_back();
                                }
                            }
                        }
                        else
                        {
                            index = -1 * index;
                            cls_pt = obj_cls_id_sift[index].first;
                            if (cls_pt == 0 || std::find(id_sift_no_depth.begin(), iter_sift_end, index) == iter_sift_end) continue;
                            pt_prev(2) = prev_dep_sift[index];
                            pt_prev(0) = prev_un_Fea_map[id_pt](0) * pt_prev(2);
                            pt_prev(1) = prev_un_Fea_map[id_pt](1) * pt_prev(2);

                            cur_z = (motion_R * pt_prev + motion_P)(2);

                            if(tracker.obj_cls_id_sift[index].first == 0)
                                max_depth = mThDepthBg;
                            else
                                max_depth = mThDepthObj;

                            if (cur_z < max_depth && cur_z > mMinDepthPt)
                            {
                                // ++num_inliers;
                                cur_dep_sift[index] = cur_z;
                            }
                            else
                            {
                                status_sift[index] = 0;
                                // 如果深度值为负（会有这种情况发生吗？？），则删除该点当前帧观测在地图中的记录（或者直接将该点全部记录从地图中删除？）
                                if(cur_z <= 0)
                                {
                                    fea_iters[i]->feature_per_frame.pop_back();
                                }
                            }
                        }

                        continue;
                    }

                    // 否则为外点
                    --num_inlier_fea_PnP;
                    // 当前帧在地图中的跟踪点数量减1
                    --num_track_cur_bg;
                    // assert(gl_id_index_map.find(id) != gl_id_index_map.end() && "There must be something wrong with var gl_id_index_map!");
                    index = gl_id_index_map[id];
                    // index小于等于0的是sift，则保留当前帧该点为新检测点
                    if (index <= 0)
                    {
                        index = -1*index;
                        l_obj_id = obj_cls_id_sift[index].second;
                        // 是否要将当前帧该点转为新点
                        // 如果是静态物体点，且有右图像匹配，则保留

                        // 对于物体的sift点，只要该点在当前帧中有深度值(不管是否来自depth_map），则可以保留该点。首先保留该点可以使得下一帧容易跟踪该物体，其次（尤其是VI初始化之前）如果该点有深度值，则可以为下一帧相机的PnP提供点
                        // 对于物体点，status == 1代表有立体匹配的跟踪点或新点，== 2代表深度值来自depth_map的跟踪点, ==3代表深度值来自depth_map的新点
                        // if(status_sift[index] == 1 && l_obj_id > 0)
                        if(status_sift[index] > 0 && l_obj_id > 0)
                        {
                            reserve_new_sift.push_back(id); 
                        }
                        else
                        {
                            // 如果要在当前帧保留背景新点
                            if(tracker.add_new_sift_in_next_frame)
                            {
                                // 当前帧有立体匹配的背景区域跟踪点
                                if(l_obj_id == 0 && status_sift[index] == 1)
                                {
                                    reserve_new_sift.push_back(id); 
                                }
                                else
                                    status_sift[index] = 0;
                            }
                            else
                                status_sift[index] = 0;
                        }   
                    }
                    // FAST点直接删除该点的跟踪和地图中的最新记录。因为FAST点只有被检测时比较有区分度，给跟踪的点不一定具有区分度
                    else
                    {
                        statusLeftRIght[index-1] = 0;
                    }

                    // 删除该点在地图中的当前帧观测。另存为新点的物体点在当前帧不会加入地图
                    fea_iters[i]->feature_per_frame.pop_back();
                    // 删除该点在地图中的当前帧观测。因为该点已经跟丢了，如果剩下的观测帧数小于LBA所需的最小观察帧数，则这里其实可以直接删除以尽早减小地图的体积！
                    // 如果该点在地图中的观测帧数小于LBA规定数目-1，则当前帧跟踪无效后，该点就已经不会再有任何作用了（后续也不会参与LBA），则直接从地图中删除。
                    // 当然了，如果刚好是5帧，又去掉当前帧，如果恰好当前滑窗又要marg次新帧，则该点只剩下3帧观测，后续也不足以参与LBA了（但是此处函数是在VI初始化之前，一般是在滑窗未满之前，则不会进行marg）
                    if(fea_iters[i]->feature_per_frame.size() < TH_NUM_FRAME_FOR_LBA)
                    {
                        // 要注意！！！
                        // 对于序列式容器(如vector,deque)，删除当前的iterator会使后面所有元素的iterator都失效。这是因为vetor,deque使用了连续分配的内存，删除一个元素导致后面所有的元素会向前移动一个位置！所以对此类容器进行erase是要特别注意！
                        // https://blog.csdn.net/Strengthennn/article/details/97645912
                        // 对于链表式容器(如list)，删除当前的iterator，仅仅会使当前的iterator失效，这是因为list 之类的容器，使用了链表来实现，插入、删除一个结点不会对其他结点造成影响（但是如果不想删除某元素之后再重新遍历，则需要提前保存被删除元素之后其他的iterator）。
                        // 这里feature是list，因此可以删除任一个iterator而不对已经提前存储的其他iterator产生影响！
                        // todo:但还得注意此操作对多线程的影响，别的线程是否也在使用feature变量？建议对feature加锁，避免此类错误
                        feature.erase(fea_iters[i]);
                    }
                }
            }
            else
            {
                // 由于Ransac不成功，则意味着内点比例不高，这里应该要适当减小内点的数量，以便后续可以添加静态物体跟踪点
                // todo:减小的规模应该按照Ransac设置的置信度来大致推断？
                num_inlier_fea_PnP = 2.0 * num_inlier_fea_PnP / 3;

                num_track_cur_bg = num_track_cur_bg - 1.0 * num_inlier_fea_PnP / 3;
            }
            // num_track_fea_stat = num_inliers;
        }
        // 双目+IMU不允许在初始化成功之前就视觉跟踪丢失，否则就需要重新启动！
        // 上面使用的PnP方式使得不会出现失败的情况
        else if(use_IMU)
        {
            assert(false && "solvePoseByPnP for Stereo+IMU system failed! Should restart the system!");
        }
    }
    return num_inlier_fea_PnP;
}

void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    // 对该3D地图点的每个观测可以提供2个等式，待估计量是点在参考帧下的齐次3D坐标（X，Y，Z，1），因此需要至少两个图像的观测！
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    // 用奇异值分解来求得四维变量之后，需要变为齐次坐标
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    // cout << "no.4 element of esti pt: " << triangulated_point(3) << endl;
    // cout << "no.3 element of esti pt: " << triangulated_point(2) << endl;

    // cv::Mat A(4,4,CV_64F), P1(4,4,CV_64F), P2(4,4,CV_64F), x3D;
    // cv::Vec2d pt1, pt2;
    // cv::eigen2cv(Pose0, P1);
    // cv::eigen2cv(Pose1, P2);
    // cv::eigen2cv(point0, pt1);
    // cv::eigen2cv(point1, pt2);
    // A.row(0) = pt1(0) * P1.row(2) - P1.row(0);
    // A.row(1) = pt1(1) * P1.row(2) - P1.row(1);
    // A.row(2) = pt2(0) * P2.row(2) - P2.row(0);
    // A.row(3) = pt2(1) * P2.row(2) - P2.row(1);
    // cv::Mat u,w,vt;
    // cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    // x3D = vt.row(3).t();
    // x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

    // point_3d(0) = x3D.at<double>(0,0);
    // point_3d(1) = x3D.at<double>(1,0);
    // point_3d(2) = x3D.at<double>(2,0);

    // cout << "no.4 element of esti pt: " << x3D.at<float>(3) << endl;
    // cout << "no.3 element of esti pt: " << point_3d(2) << endl;
    // cout << "no.2 element of esti pt: " << point_3d(1) << endl;
    // cout << "no.1 element of esti pt: " << point_3d(0) << endl;
}

// 根据估计的各帧IMU坐标系的全局位姿（在设定的世界坐标系，可能还没对齐到东北天坐标系）和外参，来估计还没有深度估计的地图点的深度
// 对于双目而言，各帧左图像新添加的特征点中，可能部分在右图象中是没有匹配的，因此没有深度（这些点即使被下一帧所跟踪到，在前面估计相机位姿时我们也不使用这些点，因为没法用PnP。而是在此处进行三角化）；
// 对于单目-IMU而言，会在VI初始化成功之后，根据估计的各帧全局位姿，调用此函数来重新三角化所有地图点（因为VI优化前后尺度改变了）！
// 给定的当前帧序号frameCnt并没有被使用，因为默认是对当前滑窗内所有还没有深度值的地图点进行估计
// 注意！此函数中的所有三角化操作，都要求已知左相机（和右相机）的全局位姿，因此在进行IMU-视觉的初始化之后，由于各帧相机的全局位姿改了，因此需要重新对所有的点都进行三角化！！
bool FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], FeatureTracker &tracker, Vector3d Ps_cam_pred[], Matrix3d Rs_cam_pred[],
                                 bool before_PnP, const Matrix3d &R_from_E, Vector3d *norm_t, double scale, float pred_dist_t, const set<int> &reserve_bg_track_pt_id)
{
    cout << "start triangulate points!" << endl;

    // 当某跟踪点在其被观测的最前2帧 的 三角测量失败时，如果其还被连续观测多帧，是否要同时使用多帧的观测来对其三角化？
    bool try_tria_using_multi_frame = false;

    vector<float> &cur_FAST_dep = tracker.cur_FAST_dep;
    vector<float> &cur_sift_dep = tracker.cur_sift_dep;
    vector<float> &prev_FAST_dep = tracker.prev_FAST_dep;
    vector<float> &prev_sift_dep = tracker.prev_sift_dep;
    vector<uchar> &status_sift = tracker.status_sift;
    vector<uchar> &status_FAST = tracker.statusLeftRIght;
    map<int,Vec4f> &prev_un_Fea_map = tracker.prev_un_Fea_map;
    map<int,Point2f> &prevLeftFeaMap = tracker.prevLeftFeaMap;
    map<int,Point2f> &prevRightFeaMap = tracker.prevRightFeaMap;
    vector<int> &ids_FAST = tracker.ids_FAST;
    vector<int> &ids_sift = tracker.ids_sift;
    
    vector<Point2f> &prev_FAST = tracker.prev_FAST;
    vector<Point2f> &prev_sift = tracker.prev_sift;
    vector<Point2f> &cur_FAST = tracker.cur_FAST;
    vector<Point2f> &cur_sift = tracker.cur_sift;
    map<int,int> &gl_id_index_map = tracker.gl_id_index_map;
    vector<pair<uchar,int>> &obj_id_cls_FAST = tracker.obj_cls_id_FAST;
    vector<pair<uchar,int>> &obj_id_cls_sift = tracker.obj_cls_id_sift;
    
    // float ave_dep_bg_prev_frame = tracker.ave_dep_bg_prev_frame;
    double &ave_dep_bg_cur_frame = tracker.ave_dep_bg_cur_frame;
    int &num_bg_with_dep = tracker.num_bg_with_dep;

    float ave_dep_bg_cur_frame_;

    if(num_bg_with_dep >= 6) 
    {
        ave_dep_bg_cur_frame_ = ave_dep_bg_cur_frame/num_bg_with_dep;
        if (ave_dep_bg_cur_frame_ < mMinDepthPt || ave_dep_bg_cur_frame_ > mThDepthBg) ave_dep_bg_cur_frame_ = INIT_DEPTH;
    }
    else
        ave_dep_bg_cur_frame_ = INIT_DEPTH;

    int num_frame, index, f_start;
    Eigen::Matrix3d R_start, R_cur, R_motion;
    Eigen::Vector3d P_start, P_cur, P_motion, pt_start, pt_cur;
    double prev_dep, cur_depth;
    bool good_dep;
    int num_tri_fail = 0;
    
    vector<Vector3d> prev_3d_pts;
    vector<Vector2d> cur_2d_un_pts;

    // 保存需要被删除的地图点
    vector<int> pt_need_to_erase;
    // 如果左右相机之间的外参需要被在线估计，则这里暂时不删除那些首帧立体匹配三角化失败的地图点？
    bool need_estimate_ex_param_stereo = (ESTIMATE_EXTRINSIC != 0);

    // 在当前帧之前系统是否进行过LBA。其中，纯双目是从第2帧开始都进行LBA；而双目+IMU是在第一个滑窗满时才开始进行LBA；
    // 进行LBA意味这会改变之前的某些帧的位姿；另外，如果LBA中还需要优化双目相机的外参，其值也会改变。
    // 这意味着前面三角化失败的某些点，在此时可能可以三角化成功（会有很大的改变吗？）
    bool has_LBA = STEREO && ((!USE_IMU && frameCnt > 1) || (USE_IMU && frameCnt == WINDOW_SIZE));

    // 保存图像下半部分 每个横向bloc中的点 深度和id，不同行的bloc中的深度值会有较大差别
    vector<pair<float,int>> row_1, row_2, row_3;
    // 保存图像上半部分中的点，其深度值一般比较大
    map<double,int,less<double>> temp_dep_id;
    
    // 保存所有有深度的3D-2D匹配点
    vector<Vector3d> temp_3d_pt;
    vector<Vector2d> temp_2d_pt;
    vector<int> temp_pt_in_col;
    int num_temp = 0;

    // 遍历滑窗内所有地图点
    for (auto &it_per_id : feature)
    {
        good_dep = false;
        num_frame = it_per_id.feature_per_frame.size();

        float dep_ = it_per_id.estimated_depth;

        int gl_id = it_per_id.feature_id;

        bool cond_1 = (dep_ > 0);
        bool cond_2 = (it_per_id.endFrame() != frameCnt);
        bool cond_3 = it_per_id.feature_per_frame[0].is_stereo;

        bool has_stereo_prev = false;
        // 判断当前帧的跟踪点在上一帧中是否有立体匹配，应该直接查看该点在地图中的记录，因为有些点在before_PnP的立体三角化失败后就被取消了右观测记录
        // has_stereo_prev = (prevRightFeaMap.find(gl_id) != prevRightFeaMap.end());
        if(num_frame > 1 && !cond_2) has_stereo_prev = it_per_id.feature_per_frame[(num_frame-2)].is_stereo;

        // 如果该点之前已经三角化成功，且 该点不是当前帧跟踪点（则不需要更新当前点的深度），或者是当前帧跟踪点且在上一帧没有立体匹配，则其使用上一帧的运动估计更新后的深度值
        if(cond_1 && (cond_2 || (!has_stereo_prev || !before_PnP)))
        {
            // if(!before_PnP && it_per_id.endFrame() == frameCnt && num_frame >= TH_NUM_FRAME_FOR_LBA) ++num_LBA_fea_cur_frame;
            good_dep = true;
        }
        else if(before_PnP)
        {
            // 该特征点被观测首帧下的左右图像之间/
            // 如果该点在当前帧被跟踪到
            if(STEREO && !cond_2)
            {
                Eigen::Vector2d point0, point1;
                Eigen::Vector3d point3d, localPoint;
                double depth;

                // 如果该点观测首帧(不是上一帧）下有立体匹配，但是还没有成功立体三角化，则再次尝试（只在需要在线估计双目相机外参时才进行）
                // 前提是立体相机外参会参与LBA并大幅改变其值，且该点的连续跟踪帧数大于阈值
                // if(!cond_1 && cond_3 && num_frame > 2 && has_LBA)
                // {
                //     // 通过IMU坐标系的全局位姿和左右相机与IMU的外参，计算左右相机的全局位姿
                //     int imu_i = it_per_id.start_frame;
                //     // 跟相机全局位姿相关的3*4矩阵，用于三角化测量
                //     Eigen::Matrix<double, 3, 4> leftPose;
                //     Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
                //     Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
                //     leftPose.leftCols<3>() = R0.transpose();
                //     leftPose.rightCols<1>() = -R0.transpose() * t0;
                    
                //     Eigen::Matrix<double, 3, 4> rightPose;
                //     Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
                //     Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
                //     rightPose.leftCols<3>() = R1.transpose();
                //     rightPose.rightCols<1>() = -R1.transpose() * t1;

                //     point0 = it_per_id.feature_per_frame[0].point.head(2);
                //     point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
                    
                //     triangulatePoint(leftPose, rightPose, point0, point1, point3d);
                    
                //     // 再投影到左相机坐标系得到点的深度
                //     localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
                //     depth = localPoint.z();

                //     // if (depth >= mMinDepthPt && depth < mThDepthBg)
                //     if (depth > 1.5 && depth < mThDepthBg)
                //     {
                //         it_per_id.estimated_depth = depth;
                //         it_per_id.feature_per_frame[0].depth = depth;

                //         cout << "Succeed triangulate a tracked point with stereo match in its first frame!" << endl;
                //         cout << "depth of point by triangulation of stereo match: " << depth << endl;

                //         good_dep = true;
                //     }
                //     else
                //     {
                //         good_dep = false;
                //         ++num_tri_fail;
                //         // 这里不用删除该点的该首帧观测，只需要它一直无法立体三角化，则使用
                //     }
                // }

                // 如果该跟踪点在上一帧中有立体匹配，则计算该点在上一帧中的深度
                // 只有立体三角化失败，才使用该点首观测帧下的深度来直接计算该点在上一帧的深度（但这样运动估计的误差就会累积下来）
                if(has_stereo_prev)
                {
                    depth = it_per_id.feature_per_frame[num_frame-2].depth;
                    // 如果上一帧该点还没有从立体匹配中得到深度值（即为背景点）
                    if(depth <= 0)
                    {
                        Eigen::Matrix<double, 3, 4> leftPose;
                        leftPose.leftCols<3>() = Eigen::Matrix<double, 3, 3>::Identity();
                        leftPose.rightCols<1>() = Eigen::Matrix<double, 3, 1>::Zero();

                        Eigen::Matrix<double, 3, 4> rightPose;
                        rightPose.rightCols<1>() = ric[1].transpose() * (tic[0] - tic[1]);
                        rightPose.leftCols<3>() = ric[1].transpose() * ric[0];
                        
                        // point是观测点的归一化平面上的坐标，即该点在相机坐标系下的z为1
                        point0 = it_per_id.feature_per_frame[(num_frame-2)].point.head(2);
                        point1 = it_per_id.feature_per_frame[(num_frame-2)].pointRight.head(2);

                        // 用三角化测量的公式求解点的全局坐标。已知某个全局点在左右相机中的归一化坐标，以及两个相机的全局位姿，则4个方程(左右相机归一化平面中的x和y坐标）可以求解三个未知量（点的全局3维坐标）
                        triangulatePoint(leftPose, rightPose, point0, point1, point3d);
                        depth = point3d.z();
                        
                        // if (depth >= mMinDepthPt && depth < mThDepthBg)
                        if (depth > 1.5 && depth < mThDepthBg)
                        {
                            good_dep = true;

                            it_per_id.feature_per_frame[(num_frame-2)].depth = depth;

                            // cout << "Succeed triangulate a tracked point with stereo match in prev frame!" << endl;
                            // cout << "depth of point by triangulation of stereo match: " << depth << endl; 
                            
                            int index = gl_id_index_map[gl_id];
                            if(index > 0)
                            {
                                prev_FAST_dep[(index-1)] = depth;
                            }
                            else
                            {
                                prev_sift_dep[(-index)] = depth;
                            }

                            // 如果上一帧是该点的首观测帧
                            if(num_frame == 2) 
                            {
                                it_per_id.estimated_depth = depth;
                            }
                            
                        }
                        else
                        {
                            // cout << "failed triangulate a point with stereo match in prev frame!" << endl;
                            ++num_tri_fail;
                            good_dep = false;
                            // 初始值是负的吗？不是，等于5，为什么？那样三角化没有成功的点也赋予5.0的深度值？那之后不久不再对其进行三角化了？
                            // 是的，VINS是寄希望于该点能连续被观测4帧，然后使用多帧三角化 以及 参与LBA 来优化该点的深度....
                            // 这里改变这个做法，特征点的观测首帧深度值只能通过三角化来完成（无论是两帧还是多帧（4帧）），如果三角化始终未成功，则不将其加入LBA！
                            if(num_frame == 2) 
                            {
                                it_per_id.estimated_depth = INIT_DEPTH;
                                // it_per_id.estimated_depth = -1.0;
                                // it_per_id.estimated_depth = ave_dep_bg_cur_frame_;
                            }

                            // 放弃该点的右观测，后续即使该点继续被跟踪，且双目相机外参被更新估计，也不会对其进行该帧下的立体三角化测量
                            // 因为本项目假定立体相机的外参的标定结果还不错，即使需要在线优化该参数，也只是因为该参数会在运动过程中有很小的变化，不应该对立体三角化有太大影响
                            // todo: 是否可以允许外参有较大变化，而在这里保留这些右图像观测？
                            // if(ESTIMATE_EXTRINSIC == 0)
                            {
                                it_per_id.feature_per_frame[(num_frame-2)].is_stereo = false;
                            }

                            // continue;
                        }
                        /*
                        Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                        printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                        ptsGt.x(), ptsGt.y(), ptsGt.z());
                        */
                    }
                    else
                    {
                        good_dep = true;
                    }
                }
            }
            // continue;
        }
        else if(!cond_1)
        {    
            // 如果是双目相机，并且该点在其初始观测帧的左右图像中均有观测。知道左右图像中的匹配，不就可以求解点在左图像中的深度了吗？
            // 能通过立体匹配的视差得到深度估计的前提是左右相机已经经过立体校正！如果没有经过立体校正，则需要一般的三角化测量！
            // 在addFeatureCheckParallax()函数中往滑窗中添加地图点的观测帧时，会设置该点观测帧中的is_stereo变量。对于双目相机，优先使用左右相机的位姿来三角化新地图点！
            // 优先使用该点首观测帧的左右图像中的匹配来完成该点的三角化和深度估计！!
            // 对于MVIO而言，如果某个点在观测首帧中有立体匹配，则其深度估计不会在这里进行，而是在特征点跟踪阶段就完成

            // 会有这样的点吗？双目的三角测量如果第一次不成功，后续也不会成功了吧？
            // 除非是左右相机的外参需要在线被估计（即need_estimate_ex_param_stereo == true)且有较大的变化(即has_LBA)，则会一直保留首观测帧的立体匹配
            // 这里只对那些已经跟踪丢失且保留下来（跟踪帧数够多）的点再进行立体三角化
            if(STEREO && need_estimate_ex_param_stereo && cond_3 && cond_2 && has_LBA && it_per_id.feature_per_frame.size() >= TH_NUM_FRAME_FOR_LBA)
            {
                // 通过IMU坐标系的全局位姿和左右相机与IMU的外参，计算左右相机的全局位姿
                int imu_i = it_per_id.start_frame;
                // 跟相机全局位姿相关的3*4矩阵，用于三角化测量
                Eigen::Matrix<double, 3, 4> leftPose;
                Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
                Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
                leftPose.leftCols<3>() = R0.transpose();
                leftPose.rightCols<1>() = -R0.transpose() * t0;
                //cout << "left pose " << leftPose << endl;
                
                Eigen::Matrix<double, 3, 4> rightPose;
                Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
                Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;
                //cout << "right pose " << rightPose << endl;
                
                Eigen::Vector2d point0, point1;
                Eigen::Vector3d point3d;
                // point是观测点的归一化平面上的坐标，即该点在相机坐标系下的z为1
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
                //cout << "point0 " << point0.transpose() << endl;
                //cout << "point1 " << point1.transpose() << endl;

                // 用三角化测量的公式求解点的全局坐标。已知某个全局点在左右相机中的归一化坐标，以及两个相机的全局位姿，则4个方程(左右相机归一化平面中的x和y坐标）可以求解三个未知量（点的全局3维坐标）
                triangulatePoint(leftPose, rightPose, point0, point1, point3d);
                Eigen::Vector3d localPoint;
                // 再投影到左相机坐标系得到点的深度
                localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
                double depth = localPoint.z();
                
                if (depth >= mMinDepthPt && depth < mThDepthBg)
                {
                    good_dep = true;
                    it_per_id.estimated_depth = depth;
                    it_per_id.feature_per_frame[0].depth = depth;
                    // cout << "Succeed triangulate a point with stereo match in its first frame!" << endl;

                    // if(it_per_id.endFrame() == frameCnt && num_frame >= TH_NUM_FRAME_FOR_LBA) ++num_LBA_fea_cur_frame;
                }
                else
                {
                    // cout << "Failed triangulate a point!" << endl;
                    ++num_tri_fail;
                    // cout << "failed triangulate a point with stereo match in its first frame!" << endl;
                    
                    // 如果两相机外参需要实时估计，则这里暂时不删除该地图点
                    // 最后决定就算外参有需要被优化，这些之前三角化失败的立体点也只再尝试一次三角化
                    // if(ESTIMATE_EXTRINSIC == 0) 
                        pt_need_to_erase.push_back(gl_id);

                    // VINS中这里初始值是负的吗？不是，等于5，为什么？那样三角化没有成功的点也赋予5.0的深度值？那之后不久不再对其进行三角化了？
                    // 是的，VINS是寄希望于该点能连续被观测4帧，然后使用多帧三角化 以及 参与LBA 来优化该点的深度....
                    // 这里改变这个做法，特征点的观测首帧深度值只能通过三角化来完成（无论是两帧还是多帧（4帧）），如果三角化始终未成功，则不将其加入LBA！
                    it_per_id.estimated_depth = INIT_DEPTH;
                    // it_per_id.estimated_depth = -1.0;
                    // it_per_id.estimated_depth = ave_dep_bg_cur_frame_;
                    good_dep = false;
                    // 如果这里没法估计出深度值，则认为该点有问题
                    // continue;
                }
                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
                // continue;
            }

            // 如果首帧下深度根据立体三角化还是没有得到，则进行观测首2帧的三角化
            
            // 如果是单目相机，或者是双目相机但该点在首观测帧的右图像没有匹配，并且该点至少被某两帧相机所观测到，使用其被观测的最早两帧相机的全局姿态来三角化该地图点。
            // 是否要求该点是被连续地跟踪到？如果不需要，是否应该要有合并地图点的操作？不然某个地图点在不连续的两帧中很可能被当作不同的地图点？（理论上来说同一个地图点被更多的帧所观测到，会提供更多有效约束）
            // 合并地图点这个操作对于VINS系列比较困难，因为地图点的观测虽然是特征点，但并不像ORB那样有描述子（这是考虑了该点周围的信息），因此无法较好地确定两个距离很近的地图点的观测是否较一致（这是合并这两个点的条件）！
            // 这里是否可以修改为该点首观测帧时没有右图像观测，且当前帧为该点的第2个观测帧（如果此次三角化失败，则必须等到该点有4帧观测后，再尝试用多帧观测对该点进行三角化）？
            // else if(num_frame == 2 && it_per_id.endFrame() == frameCnt)
            if(!good_dep && num_frame > 1 && (!cond_2 || (has_LBA && num_frame >= TH_NUM_FRAME_FOR_LBA)))
            {
                int imu_i = it_per_id.start_frame;
                
                Eigen::Vector3d t0, t1;
                Eigen::Matrix3d R0, R1;
                // if(imu_i < (frameCnt -1))
                // {
                //     t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
                //     R0 = Rs[imu_i] * ric[0];
                //     imu_i++;
                //     t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
                //     R1 = Rs[imu_i] * ric[0];
                // }
                // else
                // {
                //     t0 = Ps_cam_pred[0];
                //     R0 = Rs_cam_pred[0];
                //     t1 = Ps_cam_pred[1];
                //     R1 = Rs_cam_pred[1];
                // }

                t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
                R0 = Rs[imu_i] * ric[0];
                imu_i++;
                t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
                R1 = Rs[imu_i] * ric[0];

                Eigen::Matrix<double, 3, 4> leftPose;
                leftPose.leftCols<3>() = R0.transpose();
                leftPose.rightCols<1>() = -R0.transpose() * t0;

                // 只用其被观测的前两帧来估计该点的坐标吗？是的，这里只是计算出该点坐标的初始估计值，另外LBA时也会对地图点位置进行优化！
                // 如果是多帧相机一起进来用于所有地图点的三角化（即所有地图点此前均没有深度估计，这种情况出现在单目-IMU完成VI对齐之后，会重新三角化滑窗内所有地图点！），则后面还会使用多帧的观测来三角化该地图点。
                
                Eigen::Matrix<double, 3, 4> rightPose;
                
                rightPose.leftCols<3>() = R1.transpose();
                rightPose.rightCols<1>() = -R1.transpose() * t1;

                Eigen::Vector2d point0, point1;
                Eigen::Vector3d point3d;
                // 归一化平面的点坐标
                point0 = it_per_id.feature_per_frame[0].point.head(2);
                point1 = it_per_id.feature_per_frame[1].point.head(2);
                triangulatePoint(leftPose, rightPose, point0, point1, point3d);
                Eigen::Vector3d localPoint;
                localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
                double depth = localPoint.z();
                // if (depth > mMinDepthPt && depth < mThDepthBg)
                if (depth >= 1.5 && depth < mThDepthBg)
                {
                    // cout << "Succeed triangulate a point with 2 frame measurement!" << endl;
                    // cout << "depth of point: " << depth << endl;

                    // 点的每一帧观测中的此深度值，只能来自首帧的立体三角化
                    // it_per_id.feature_per_frame[0].depth = depth;
                    it_per_id.estimated_depth = depth;
                    good_dep = true;
                    
                }
                else
                {
                    // cout << "failed triangulate a point with first 2 frame measurement" << endl;
                    // cout << "Failed triangulate a point!" << endl;
                    ++num_tri_fail;
                    
                    // 这种已经跟丢的多帧观测点，只会在跟丢后再进行一次尝试三角化，如果失败，则直接删除（不能期待下一次LBA会大幅改变其首2个观测帧的位姿）
                    if(!try_tria_using_multi_frame || cond_2)
                    {
                        // 注意！！如果vector对象为空，则对其调用front()或end()函数会导致为定义行为，在linux的g++编译运行时会出现段错误的提示！！！
                        if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                            pt_need_to_erase.push_back(gl_id);
                    }
                    else
                    {
                        it_per_id.estimated_depth = INIT_DEPTH;
                        // it_per_id.estimated_depth = -1.0;
                        // it_per_id.estimated_depth = ave_dep_bg_prev_frame_;
                    }
                    
                    good_dep = false;
                    // continue;
                }
                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
                // 对于使用前后帧相机位姿来三角化点坐标的情况，此处应该不能要这个continue吧？否则就无法进行下面的用多帧的观测来三角化该点了呀！！
                // 其实只有在观测首帧没法通过左右观测来完成有效三角化，并且也没法用首2帧的匹配来完成有效三角化，才会等到该点被连续观测4帧时才采用下面的多帧联合的三角化！
                // 可是上面三角化失败时，又直接赋予该点初始观测帧下的深度值为5.0?!这样后续如何对其进行多帧的三角化？因此将上面三角化失败时应该仍然将深度设为负值
                // continue;
            }
        }

        // 上面只是用该点最早的两帧观测来三角化得到该点在观测首帧相机下的深度。如果该地图点被多帧观测到，可以联合多帧位姿和观测来三角化该点，使得其深度估计精度更高。
        // 地图点的feature_per_frame与其被观测到的不同时刻相机帧的个数有关，同一时刻左右相机中的观测是放在一个FeaturePerFrame中。所以这里是要求该点至少要被四个时刻的相机所观测到！
        // 上面的if和else中最后都用了continue，即该点三角化成功之后，才统计该点的观测帧数，这关系到该点后续是否会参与LBA！
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        
        // 记录上一帧有深度值的跟踪点,用于估计t的尺度
        if(before_PnP && norm_t != nullptr)
        {
            if(good_dep)
            {
                if(it_per_id.endFrame() == frameCnt)
                {
                    if(!reserve_bg_track_pt_id.empty() && reserve_bg_track_pt_id.find(gl_id) != reserve_bg_track_pt_id.end()) continue;

                    // int index = gl_id_index_map[gl_id];
                    
                    // if(index > 0)
                    // {
                    //     prev_dep = prev_FAST_dep[(index-1)];
                    // }
                    // else
                    // {
                    //     prev_dep = prev_sift_dep[(-index)];
                    // }

                    prev_dep = it_per_id.feature_per_frame[(num_frame-2)].depth;
                    
                    if(prev_dep < 1.5) continue;

                    // debug
                    // assert(prev_un_Fea_map.find(gl_id) != prev_un_Fea_map.end());
                    // Vec4f &prev_un_pt = prev_un_Fea_map[gl_id];
                    Vector3d &prev_un_pt = it_per_id.feature_per_frame[num_frame-2].point;

                    Vector3d &cur_ut_pt = it_per_id.feature_per_frame[num_frame-1].point;
                    
                    // // 注意，多选择那些深度值较小的点，它们受特征点匹配精度的影响要小一些
                    // if(prev_dep < 10.0)
                    // {
                    //     prev_3d_pts.emplace_back(prev_un_pt(0)*prev_dep,prev_un_pt(1)*prev_dep,prev_dep);
                    //     // 当前帧匹配点的归一化平面坐标
                    //     cur_2d_un_pts.emplace_back(cur_ut_pt(0),cur_ut_pt(1));
                    // }
                    // else
                    // {
                    //     temp_dep_id[prev_dep] = num_temp;
                    //     temp_3d_pt.emplace_back(prev_un_pt(0)*prev_dep,prev_un_pt(1)*prev_dep,prev_dep);
                    //     temp_2d_pt.emplace_back(cur_ut_pt(0),cur_ut_pt(1));
                    //     ++num_temp;
                    // }

                    // debug
                    assert(prevLeftFeaMap.find(gl_id) != prevLeftFeaMap.end());

                    Point2f &prev_pt = prevLeftFeaMap[gl_id];
                    int row = prev_pt.y/60;
                    if(row < 3)
                    {
                        temp_dep_id[prev_dep] = num_temp;
                    }
                    else
                    {
                        // 区分不同行，作为大致区分不同深度范围
                        if(row == 3)
                        {
                            row_3.emplace_back(prev_dep,num_temp);
                        }
                        else if(row == 4)
                        {
                            row_2.emplace_back(prev_dep,num_temp);
                        }
                        else
                        {
                            row_1.emplace_back(prev_dep,num_temp);
                        }
                    }

                    temp_3d_pt.emplace_back(prev_un_pt(0)*prev_dep,prev_un_pt(1)*prev_dep,prev_dep);
                    temp_2d_pt.emplace_back(cur_ut_pt(0),cur_ut_pt(1));

                    int col = prev_pt.x/200;
                    if(col == 6) col = 5;
                    temp_pt_in_col.push_back(col);
                    ++num_temp;
                }
            }
            continue;
        }

        // 如果上面基于2观测的三角化失败，则这里可以尝试多次观测联合的三角化
        
        // 如果该点当前帧仍被跟踪，且帧数大于参与LBA所与的观测帧数，则对其进行深度值优化。如果该点持续被跟踪，则每一帧都对齐进行更新深度值，这是为了给LBA提供较好的初值
        // 到达这里的前提是该点之前已经完成首观测帧下的深度值估计了，要么使用首帧的立体匹配，要么使用首2帧的三角测量
        // 即使是首观测帧下有深度的点，经过这里估计后的深度值很可能变为无效的，最终只能赋予原始深度值！
        if(try_tria_using_multi_frame && it_per_id.estimated_depth <= 0 && it_per_id.used_num >= TH_NUM_FRAME_FOR_LBA && ((it_per_id.endFrame() == frameCnt) || has_LBA))
        {
            // 对于长跟踪的点，在进行LBA之前每次都更新该点的深度值
            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

            // 该地图点的每帧观测提供2个等式
            Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
            int svd_idx = 0;

            Eigen::Matrix<double, 3, 4> P0;
            // 只使用左相机的观测
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            // 所以这个P0用来干嘛了？？
            P0.leftCols<3>() = Eigen::Matrix3d::Identity();
            P0.rightCols<1>() = Eigen::Vector3d::Zero();

            // 观测初始帧也会进行计算并形成svd_A的前两行，作用是什么？？
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;

                Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
                Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
                // 使用的是各观测帧相机与首观测帧相机之间的相对位姿，这样计算出来的点的3D坐标是在首观测帧相机坐标系下的深度
                Eigen::Vector3d t = R0.transpose() * (t1 - t0);
                Eigen::Matrix3d R = R0.transpose() * R1;
                Eigen::Matrix<double, 3, 4> P;
                P.leftCols<3>() = R.transpose();
                P.rightCols<1>() = -R.transpose() * t;
                // 为什么是使用该地图点在每帧相机坐标系下的坐标的方向向量？虽然这对求解之后的三角化求解的思路没有影响，但是这么做的原因是什么？数值稳定性吗？
                Eigen::Vector3d f = it_per_frame.point.normalized();
                // 对于非归一化平面的观测点的坐标，需要乘以f[2]这个量（如果使用的是观测的归一化平面坐标，则f[2]=1，就不需要写）
                svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

                // 上面为什么要计算初始帧和自己的相对位姿和观测？这个if是否要放在最上面？
                if (imu_i == imu_j)
                    continue;
            }
            // ROS_ASSERT(svd_idx == svd_A.rows());
            
            assert(svd_idx == svd_A.rows() && "Something wrong with svd decomposition of martix A!");
            // svd_A的维度是（2N，4），由于线性齐次方程AX=0，理想中的情况是A矩阵是不满秩的（即秩为3，正好对应了X是齐次坐标（x，y，z，1），此时变量X就在A矩阵的零空间中！
            // 但实际上由于测量噪声和估计误差，A矩阵极大可能是满秩的，即可能会有4个非0奇异值。
            // 此时我们选择最小的奇异值所对应的右奇异向量（奇异值分解后奇异值沿对角线从大到小排列，因此选择分解后V矩阵的第四列），因为最小的奇异值对应的奇异向量最可能接近理想解
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double svd_method = svd_V[2] / svd_V[3];
            //it_per_id->estimated_depth = -b / A;
            //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

            double ori_dep = it_per_id.estimated_depth;
            it_per_id.estimated_depth = svd_method;
            good_dep = true;
            //it_per_id->estimated_depth = INIT_DEPTH;

            // if (it_per_id.estimated_depth < 0.1)
            if (it_per_id.estimated_depth < mMinDepthPt || it_per_id.estimated_depth > mThDepthBg)
            {
                ++num_tri_fail;
                // it_per_id.estimated_depth = INIT_DEPTH;
                // it_per_id.estimated_depth = -1.0;
                it_per_id.estimated_depth = ori_dep;
                if(ori_dep <= 0)
                    good_dep = false;
                
                // continue;

                // if(it_per_id.endFrame() == frameCnt)
                // {
                    
                // }
            }
        }

        // 在通过PnP（或LBA）得到当前帧位姿估计之后，更新当前帧跟踪点的深度值.还未加入地图的静态点则是VI初始化之后的静态物体点，其深度值更新在物体运动估计线程中进行
        // 对于没法在当前帧获取有效深度值的点，直接放弃！
        if(it_per_id.endFrame() == frameCnt && num_frame > 1)
        {
            index = gl_id_index_map[gl_id];
            f_start = it_per_id.start_frame;
            bool has_stereo_cur = false;

            // 对于当前帧的背景跟踪点，如果其有立体匹配，则其深度估计留到下一帧其再次被跟踪时再对其进行立体三角化
            if(it_per_id.feature_per_frame.back().is_stereo) 
                has_stereo_cur = true;
            
            // if(index > 0)
            // {
            //     cur_depth = cur_FAST_dep[(index-1)];
            // }
            // else
            // {
            //     cur_depth = cur_sift_dep[(-1*index)];
            // }
            
            // 注意，当前帧的点没有给定深度值，不意味着该点没有立体匹配。即使有立体匹配，但是如果不默认立体校正足够准确，则需要进行立体三角化才会有深度值。
            // 而此系统是只对被跟踪到的点在上一帧进行立体三角化，或者跟踪已丢失但跟踪长度大于阈值的点在其首观测帧持续尝试进行立体三角化。对于最新帧的立体匹配点，暂不进行三角化（因为即使三角化了，下一帧也不一定会继续被跟踪到）
            // 虽然此处会用运动变换来更新所有被跟踪背景点在当前帧中的深度值，而不是用立体匹配来计算深度值，但是该立体匹配可以用于在LBA时优化该点的深度值！
            // 只有静态物体点才会在此处有正的depth值，而这些物体点的深度值需要更新吗？还是需要的，其立体匹配用于在LBA中参与该点深度值的优化
            
            // 只更新那些在当前帧中没有立体匹配的点的深度值。
            // 如果当前帧某个点有立体匹配，则其深度估计留到下一帧（如果其还被跟踪到）。实际上对于纯背景跟踪点，其立体匹配在当前帧是暂时不保留的
            // if(cur_depth <= 0)
            // if(1)
            if(!has_stereo_cur)
            {
                cur_depth = -1.0;
                bool has_dep_prev = false;
                int start_frame_id;
                float prev_dep;
                // 优先使用离当前帧较近的帧中,且该帧下的深度来自于立体匹配
                for(int i = (num_frame-2); i >= 0; --i)
                {
                    prev_dep = it_per_id.feature_per_frame[i].depth;
                    // 特征点每帧观测下的该变量代表该帧有有有效深度，如果该点立体匹配且立体三角化成功，则该深度值优先来自于三角化（较为可靠）
                    if(prev_dep > 0)
                    {
                        int frame_1 = frameCnt - (num_frame - i) + 1;
                        R_start = Rs[frame_1] * ric[0];
                        P_start = Ps[frame_1] + Rs[frame_1] * tic[0];
                        has_dep_prev = true;
                        start_frame_id = i;
                        break;
                    }
                }

                // 如果该点在之前每一帧均没有立体匹配，则使用其上一帧的深度（来自于每一帧的运动更新）
                if(!has_dep_prev)
                {
                    if(index > 0)
                    {
                        prev_dep = prev_FAST_dep[(index-1)];
                    }
                    else
                    {
                        prev_dep = prev_sift_dep[(-index)];
                    }

                    if(prev_dep > 0)
                    {
                        start_frame_id = num_frame-2;
                        has_dep_prev = true;

                        R_start = Rs[frameCnt-1] * ric[0];
                        P_start = Ps[frameCnt-1] + Rs[frameCnt-1] * tic[0];
                    }
                }

                if(has_dep_prev)
                {
                    R_cur = Rs[frameCnt] * ric[0];
                    P_cur = Ps[frameCnt] + Rs[frameCnt] * tic[0];
                    
                    R_motion = R_cur.transpose() * R_start;
                    P_motion = R_cur.transpose() * (P_start - P_cur);
                    pt_start = it_per_id.feature_per_frame[start_frame_id].point;
                    pt_start(2) = prev_dep;
                    pt_start(0) = pt_start(0) * prev_dep;
                    pt_start(1) = pt_start(1) * prev_dep;
                    pt_cur = R_motion * pt_start + P_motion;
                    cur_depth = pt_cur(2);
                    // if(cur_depth > mMinDepthPt && cur_depth < mThDepthBg)
                    if(cur_depth > 1.5 && cur_depth < mThDepthBg)
                    {
                        if(index > 0)
                        {
                            cur_FAST_dep[(index-1)] = cur_depth;

                            if(obj_id_cls_FAST[(index-1)].first == 0) 
                            {
                                ave_dep_bg_cur_frame += cur_depth;
                                num_bg_with_dep +=1;
                            }
                        }
                        else
                        {
                            cur_sift_dep[(-1*index)] = cur_depth;

                            if(obj_id_cls_sift[(-1*index)].first == 0) 
                            {
                                ave_dep_bg_cur_frame += cur_depth;
                                num_bg_with_dep +=1;
                            }
                        }
                        
                        // 如果某个点在某帧下的深度是通过之前帧该点深度和两帧间运动 计算得来的，那么认为不太可靠，不将该深度值记录到depth变量中
                        // depth变量只记录该点在某帧下由立体三角化得到的深度值，即前提是必须有立体匹配
                        // it_per_id.feature_per_frame[(num_frame-1)].depth = cur_depth;
                    }
                    else
                    {
                        // 不合格深度的点放在下面处理
                        // continue;
                    }
                }
                
                // 然后如果该点深度估计超过阈值，或者明显错误（远距离的点不应该比阈值大太多，因为默认车辆不是倒着走），则连其当前帧观测也从地图中删除
                if(cur_depth < 1.0 || cur_depth >= 1.2*mThDepthBg)
                {
                    // 如果无法更新或者更新后的深度值不符合要求，则后续不再跟踪该点；
                    if(index > 0)
                    {
                        status_FAST[(index-1)] = 0;
                    }
                    else
                    {
                        status_sift[(-1*index)] = 0;
                    }
                    
                    // 如果删除完当前帧观测后，该点在地图中的观测帧数小于LBA阈值，则直接删除该点。如果帧数大于LBA阈值，则看其首帧下深度是否已知，或者之后是否会再次尝试三角化
                    if(num_frame <= TH_NUM_FRAME_FOR_LBA || (it_per_id.estimated_depth <= 0 && !try_tria_using_multi_frame))
                    {
                        // 注意！！！如果vector对象为空，则对其调用front()或end()函数会导致为定义行为，在linux的g++编译运行时会出现段错误的提示！！！
                        if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                            pt_need_to_erase.push_back(gl_id);
                    }
                    else
                        it_per_id.feature_per_frame.pop_back();
                }
            }
        }
    }
    
    cout << "Num of fea that failed triangulate: " << num_tri_fail << endl;

    if(!pt_need_to_erase.empty())
    {
        // 删除无效的地图点
        list<FeaturePerId>::iterator it, next_it;
        for(it = feature.begin(), next_it = feature.begin(); it != feature.end(); it = next_it)
        {
            int gl_id = it->feature_id;
            // 提前保存下一个元素的iterator，防止当前元素被删除
            ++next_it;

            if(find(pt_need_to_erase.begin(),pt_need_to_erase.end(),gl_id) != pt_need_to_erase.end())
            {
                feature.erase(it);
            }
            else
                ++it;
        }
    }

    // 估计t的尺度
    if(before_PnP && norm_t != nullptr)
    {
        Vector3d Norm_t = *norm_t;
        // 需要有个较好的尺度初始估计值
        double H = 0;
        double b = 0;
        double delta_s;
        
        // 是否要使用恒速模型给出的预测位移的尺度
        // scale = 1.0;

        // 最大迭代次数和阈值跟初始值的设置有关，越接近真实值，所需的迭代次数越少
        double thres = 0.005;
        int max_iter = 500;
        Vector2d res, J;

        vector<vector<pair<float,int>>*> rows;
        // row_1 row_2 row_3分别是图像下半部分从下往上数的三行
        rows.push_back(&row_1);
        rows.push_back(&row_2);
        rows.push_back(&row_3);

        int num_1 = 15, num_2 = 9, num_3 = 5;

        for(int i = 0; i < 3; ++i)
        {
            vector<pair<float,int>> &row_ = *rows[i];
            // 为每一行的点根据深度值排序，然后在左右半区域内各选一个点
            if(!row_.empty())
            {
                sort(row_.begin(),row_.end(),[](const pair<float,int> &a, const pair<float,int> &b)
                {
                    return a.first < b.first;
                });

                bool left = false, middle = false, right = false;
                int left_col = 2, m_col_l = 1, m_col_r = 4, right_col = 3;

                int num_add = num_1, num_total = num_1;
                
                int num_l = num_1/3,  num_r = num_1/3, num_m = (num_1-num_l-num_r);
                int num_pt_in_prev_row = cur_2d_un_pts.size();
                // 理想情况情况下，第1 2 3 行中选取的点数分别是 15 9 5
                if(i == 1)
                {
                    num_total = num_2 + num_1;
                    if(num_pt_in_prev_row == num_1)
                    {
                        // 部分左中右，只分左右
                        middle = true;
                        left_col = 3;
                        right_col = 2;
                        num_add = num_2;

                        num_l = num_2/3;
                        num_r = num_2/3;
                        num_m = num_2-num_l-num_r;
                    }
                    else
                    {
                        num_add = num_total - num_pt_in_prev_row;
                        int ave = max(1, num_add/3);
                        // 需要分左中右
                        // if(num_add > 3)
                        {
                            num_l = ave;
                            num_m = ave;
                            num_r = ave;
                        }
                    }
                }
                else if(i == 2)
                {
                    num_total = num_3 + num_2 + num_1;
                    if(num_pt_in_prev_row == (num_2 + num_1))
                    {
                        middle = true;
                        right = true;
                        // 不再分左中右
                        left_col = 6;
                        num_add = num_3;

                        num_l = num_3/3;
                        num_r = num_3/3;
                        num_m = num_3-num_l-num_r;
                    }
                    else
                    {
                        num_add = num_total - num_pt_in_prev_row;
                        int ave = max(1, num_add/3);
                        // 需要分左中右
                        // if(num_add > 3)
                        {
                            num_l = ave;
                            num_m = ave;
                            num_r = ave;
                        }
                    }
                }
                
                // 左/中/右区域的点id
                // vector<int> pt_col_1, pt_col_2, pt_col_3;
                // 对下半部份的每一行均匀选择2个点，分别在左右部分
                for(auto &it: row_)
                {
                    if(!num_add) break;
                    int id = it.second;
                    int col = temp_pt_in_col[id];
                    if((!left && col < left_col && num_l) || (!right && col > right_col && num_r) || (!middle && col > m_col_l && col < m_col_r && num_m))
                    {
                        cur_2d_un_pts.push_back(temp_2d_pt[id]);
                        prev_3d_pts.push_back(temp_3d_pt[id]);

                        if(!left && col < left_col)
                        {
                            --num_l;
                            if(!num_l)
                                left = true;
                        }
                        else if(!right && col > right_col)
                        {
                            --num_r;
                            if(!num_r)
                                right = true;
                        }
                        else if(!middle)
                        {
                            --num_m;
                            if(!num_m)
                                middle = true;
                        }

                        --num_add;
                        // 标记该点已被加入
                        it.second = -(id+1);
                    }
                }

                if(cur_2d_un_pts.size() < num_total)
                {
                    int rest = num_total - cur_2d_un_pts.size();
                    for(auto &it: row_)
                    {
                        if(!rest) break;
                        int id = it.second;
                        if(id < 0) continue;

                        cur_2d_un_pts.push_back(temp_2d_pt[id]);
                        prev_3d_pts.push_back(temp_3d_pt[id]);

                        it.second = -(id+1);
                        --rest;
                    }
                }
            }
        }

        int total = (num_3 + num_2 + num_1);
        // 如果此时下半部分三行中选择的总点数小于规定值，则按照点深度从小到大(从最下行到最上行）从下半部分图的剩余点中遍历并补充
        if(cur_2d_un_pts.size() < total)
        {
            int rest = total - cur_2d_un_pts.size();
            for(int i = 0; i < 3; ++i)
            {
                if(!rest) break;
                vector<pair<float,int>> &row_ = *rows[i];
                for(auto &it: row_)
                {
                    if(it.first > 18) break;

                    if(!rest) break;
                    int id = it.second;
                    if(id < 0) continue;

                    cur_2d_un_pts.push_back(temp_2d_pt[id]);
                    prev_3d_pts.push_back(temp_3d_pt[id]);

                    // it.second = -(id+1);
                    --rest;
                }
            }
        }
        
        // 如果下半部分的点数实在太少，是否要添加图像上半部分的点
        // if(cur_2d_un_pts.size() < 8)
        {
            // int rest = 8 - cur_2d_un_pts.size();
            int rest = 8;
            for(auto &it: temp_dep_id)
            {
                // 上半部分的点的深度也不能太大。上半部分图像中其实可能会有比较近的点，例如杆或者路牌上的点
                if(it.first > 18) break;

                if(!rest) break;

                int l_id = it.second;
                cur_2d_un_pts.push_back(temp_2d_pt[l_id]);
                prev_3d_pts.push_back(temp_3d_pt[l_id]);
                --rest;
            }
        }
        
        float norm_orig = Norm_t.norm();
        // Vector3d prev_t = Norm_t;
        // 就算只有一个3D-2D匹配点，也可以用来估计尺度s
        if(cur_2d_un_pts.size() > 0)
        {
            H = 0;
            b = 0;

            for(int j = 0; j < max_iter; ++j)
            {
                H = 0;
                b = 0;
                
                int num_fit = 0;
                int num_all = 0;
                for(int k = 0; k < cur_2d_un_pts.size(); ++k)
                {
                    Vector3d P_rej = R_from_E * prev_3d_pts[k] + scale * Norm_t;
                    double dep_rej = P_rej(2);
                    // 深度值不能为负，也不能太小
                    if(dep_rej < 0.01) continue;
                    Vector2d P_norm(P_rej(0)/dep_rej, P_rej(1)/dep_rej);
                    res = P_norm - cur_2d_un_pts[k];
                    // 重投影误差0.01m足够小？
                    if(res.norm() < 0.01) ++num_fit;

                    // 用链式法则计算残差相对尺度s的雅可比
                    J(0) = Norm_t(0)/dep_rej - P_rej(0)/dep_rej/dep_rej*Norm_t(2);
                    J(1) = Norm_t(1)/dep_rej - P_rej(1)/dep_rej/dep_rej*Norm_t(2);
                    
                    H += J.transpose() * J;
                    b += (-J.transpose()*res);
                    ++num_all;
                }
                
                delta_s = b/H;
                
                // 认为给定的初始scale应该不会跟真实尺度相差太多，如果此处计算得到的尺度值太大，则可能是点的深度太远，不适合恢复尺度
                if(delta_s > 1e4 || isnanf(delta_s) == 1 || isinf(delta_s) == 1)
                {
                    cout << "invalid delta_s: " << delta_s << endl;
                    return false;
                }

                float delta_norm = abs(delta_s) * norm_orig;
                // todo: 如何限制高度方向上的运动值？是否应该将该约束直接加入优化问题中？
                // float delta_height = fabs(scale * Norm_t(1));

                // if(delta_norm < 0.025 && abs(delta_s) < thres && num_all > 0 && num_fit*1.0/num_all >= 0.90)
                // if(delta_norm < 0.025 && abs(delta_s) < thres && num_fit*1.0/num_all >= 0.90 && delta_height <= 0.1)
                if(delta_norm < 0.025 && abs(delta_s) < thres)
                // if(delta_norm < 0.025 && num_fit*1.0/num_all >= 0.95)
                // if(abs(delta_s) < thres && num_fit*1.0/num_all >= 0.95) 
                // if(abs(delta_s) < thres)
                {
                    scale += delta_s;
                    break;
                }
                else
                {
                    scale += delta_s;
                }
            }
        }
        
        Vector3d P = Norm_t * scale;

        // 如果tracker通过估计H矩阵得到R和norm_t，则其可能保留了相对于H的外点匹配(这些点远离H所属的平面，自驾场景中一般是指地平面）
        // 这些点在这里用极线约束来最终排除其中的外点，剩下的点可以进一步优化t的尺度
        if(!reserve_bg_track_pt_id.empty())
        {
            vector<int> erase_pt_gl_id;
            Matrix3d R_cam_motion, t_up, F_cam;
            R_cam_motion = (Rs[frameCnt]*ric[0]).transpose() * (Rs[(frameCnt-1)]*ric[0]);
            t_up << 0.0, -P(2), P(1), P(2), 0.0, -P(0), -P(1), P(0), 0.0;
            // 本质矩阵到关键矩阵
            F_cam = K_trans_inv * t_up * R_cam_motion * K_inv;
            float prev_x, prev_y, cur_x, cur_y;
            vector<Point2f> prev_pts, cur_pts;
            vector<int> g_pt_id;

            // 这里所有的H矩阵估计的外点中部分是纯背景跟踪点，一部分是静态物体的跟踪点（其中部分是新跟踪点，即还未加入地图）
            for(auto pt: reserve_bg_track_pt_id)
            {
                int l_id = gl_id_index_map[pt];
                if(l_id <= 0)
                {
                    l_id = -1 * l_id;
                    Point2f &pt_prev = prev_sift[l_id];
                    Point2f &pt_cur = cur_sift[l_id];
                    prev_pts.push_back(pt_prev);
                    cur_pts.push_back(pt_cur);
                }
                else
                {
                    l_id -= 1;
                    Point2f &pt_prev = prev_FAST[l_id];
                    Point2f &pt_cur = cur_FAST[l_id];
                    prev_pts.push_back(pt_prev);
                    cur_pts.push_back(pt_cur);
                }

                g_pt_id.push_back(pt);
            }

            // 通过极线约束验证的点，可以保留，因为它们不位于H所指定的3D平面上，且其深度值相对于相机的位移值不大，因为保留这些点可以优化相机的运动值（后续的PnP估计），尤其是高度方向的值！！！
            vector<uchar> is_inlier;
            float score;

            bool valid = epipolarConstrain(prev_pts, cur_pts, F_cam, is_inlier, Th_score, score, 4.0);
		
            vector<int> erase_pt;
            vector<int> re_invalid_pts;
            bool refine_scale_using_near_high_fea = true;
            if(valid)
            {   
                for(int k = 0; k < is_inlier.size(); ++k)
                {
                    if(is_inlier[k] == 0)
                    {
                        erase_pt.push_back(g_pt_id[k]);
                    }
                    else
                    {
                        int g_id = g_pt_id[k];
                        int index = gl_id_index_map[g_id];

                        // todo: 是否要求该点在上一帧中具有立体匹配？可以不要求
                        // if(prevRightFeaMap.find(g_id) == prevRightFeaMap.end())
                        // {
                        //     continue;
                        // }

                        // 对于重新认可的内点，使用其中离开地面的点来再度优化t的尺度
                        float dep_;
                        if(index > 0)
                        {
                            // 这些点在上一帧的深度值，是基于立体匹配的深度值
                            dep_ = prev_FAST_dep[(index-1)];
                        }
                        else
                        {
                            dep_ = prev_sift_dep[(-index)];
                        }
                        
                        // 即使是通过了极线约束的点，如果没有深度值，需要放弃吗？不，近处的点不一定都有立体匹配，尤其是sift点
                        // 有了上面的continue，这里的点应该都有有效深度值吧？
                        if(dep_ <= 0)
                        {
                            // erase_pt.push_back(g_id);
                        }
                        else
                        {
                            // 上一帧太远的点要保留吗？
                            if(dep_ >= 20)
                            {
                                // erase_pt.push_back(g_id);
                            }
                            else if(refine_scale_using_near_high_fea)
                            {
                                // 用来优化t的尺度
                                re_invalid_pts.push_back(g_id);
                            }
                        }
                    }
                }
            }
            else
            {
                // 如果发现得到的F矩阵是无效的
                erase_pt = g_pt_id;
            }

            // todo: 这里删除这些点，会对objs_matching的后续处理有影响吗？
            if(!erase_pt.empty())
            {
                for(int k = 0; k < erase_pt.size(); ++k)
                {
                    int g_id = erase_pt[k];
                    
                    // 可能有的跟踪点是静态物体的新跟踪点，可能还未加入地图
                    auto it = find_if(feature.begin(), feature.end(), [g_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == g_id;
                                    });
                    
                    // assert(it != feature.end());
                    // 不仅需要该点已经在地图中，还需要有当前帧的观测（针对的是静态物体点）
                    if(it != feature.end() && it->endFrame() == frameCnt)
                    {
                        // 删除该点的当前帧观测
                        if(it->feature_per_frame.size() <= TH_NUM_FRAME_FOR_LBA || (it->estimated_depth <= 0 && !try_tria_using_multi_frame))
                            feature.erase(it);
                        else
                        {
                            it->feature_per_frame.pop_back();
                        }   

                        // 对于已经在地图中且当前帧观测也已添加的点（全部纯背景跟踪点，和部分静态物体跟踪点），则确认是静态跟踪点，无论是不是物体的点，都可以删除该跟踪点
                        int index = gl_id_index_map[g_id];
                        if(index <= 0)
                        {
                            status_sift[(-index)] = 0;
                        }
                        else
                        {
                            status_FAST[(index-1)] = 0;
                        }
                    }
                }
            }

            if(refine_scale_using_near_high_fea && !re_invalid_pts.empty())
            {
                cur_2d_un_pts.clear();
                prev_3d_pts.clear();
                float dep;
                for(int k = 0; k < re_invalid_pts.size(); ++k)
                {
                    int g_id = re_invalid_pts[k];
                    
                    // 其中部分是静态物体的跟踪点，其中的新跟踪点可能还不在地图中
                    auto it = find_if(feature.begin(), feature.end(), [g_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == g_id;
                                    });
                    
                    // assert(it != feature.end());
                    if(it != feature.end() && it->endFrame() == frameCnt)
                    {
                        int num_frame = it->feature_per_frame.size();
                        if(num_frame > 1)
                        {
                            // todo:上一帧该点的深度是否一样要来自于立体匹配？？
                            // dep = it->feature_per_frame[(num_frame-2)].depth;

                            // 以下面这种情况来寻找上一帧该点的深度，则可以包含那些在上一帧中没有立体匹配的点，它们在上一帧的深度值来自于运动变换
                            int index = gl_id_index_map[g_id];
                            if(index > 0)
                            {
                                dep = prev_FAST_dep[(index-1)];
                            }
                            else
                            {
                                dep = prev_sift_dep[(-index)];
                            }

                            if(dep > 0 && dep <= 20)
                            {
                                Vector3d &cur_pt = it->feature_per_frame[(num_frame-1)].point;
                                Vector3d &prev_pt = it->feature_per_frame[(num_frame-2)].point;
                                float pos_y = prev_pt(1) * dep;
                                // todo:选取距离地面高度大于等于0.5m的点。
                                // 注意，KITTI的相机坐标系中竖直轴为y轴，且向下为正向，相机的高度大约为1.65m，则地面点的y坐标为+1.65。
                                // 但是这也有问题，如果当前道路是个海拔逐渐下降的斜坡，那么远处的离开地面一段距离的点可能比当前相机所在低平缅还要低.....
                                // 或者如果是个上坡，那么远处的地面上的点可能就变成比当前相机所在地面要高出不少...
                                // 所以最好是能估计地平面方程
                                // if(pos_y <= 1.15)
                                {
                                    cur_2d_un_pts.emplace_back(cur_pt(0),cur_pt(1));
                                    Vector3d prev_p = prev_pt * dep;
                                    prev_3d_pts.push_back(prev_p);
                                }
                            }
                        }
                    }
                }
                
                // 至少要4个满足要求的点
                if(cur_2d_un_pts.size() > 3)
                {
                    float norm_orig = Norm_t.norm();
                    int num_iter = 50;
                    H = 0;
                    b = 0;

                    // 就算只有一个3D-2D匹配点，也可以用来估计尺度s
                    for(int j = 0; j < num_iter; ++j)
                    {
                        H = 0;
                        b = 0;

                        int num_fit = 0;
                        int num_all = 0;
                        for(int k = 0; k < cur_2d_un_pts.size(); ++k)
                        {
                            Vector3d P_rej = R_from_E * prev_3d_pts[k] + scale * Norm_t;
                            double dep_rej = P_rej(2);
                            if(dep_rej < 0.01) continue;
                            Vector2d P_norm(P_rej(0)/dep_rej, P_rej(1)/dep_rej);
                            res = P_norm - cur_2d_un_pts[k];
                            // 重投影误差足够小？
                            if(res.norm() < 0.01) ++num_fit;

                            // 用链式法则计算残差相对尺度s的雅可比
                            J(0) = Norm_t(0)/dep_rej - P_rej(0)/dep_rej/dep_rej*Norm_t(2);
                            J(1) = Norm_t(1)/dep_rej - P_rej(1)/dep_rej/dep_rej*Norm_t(2);
                            
                            H += J.transpose() * J;
                            b += (-J.transpose()*res);
                            ++num_all;
                        }
                        
                        delta_s = b/H;
                        
                        // 认为给定的初始scale应该不会跟真实尺度相差太多，如果此处计算得到的尺度值太大，则可能是点的深度太远，不适合恢复尺度
                        if(delta_s > 1e4 || isnanf(delta_s) == 1 || isinf(delta_s) == 1)
                        {
                            cout << "invalid delta_s: " << delta_s << endl;
                            return false;
                        }

                        float delta_norm = fabs(delta_s) * norm_orig;
                        // float delta_height = fabs(scale * Norm_t(1));
                        
                        // if(delta_norm < 0.025 && abs(delta_s) < thres && num_all > 0 && num_fit*1.0/num_all >= 0.90)
                        // if(delta_norm < 0.025 && abs(delta_s) < thres && num_fit*1.0/num_all >= 0.90 && delta_height <= 0.1)
                        if(delta_norm < 0.025 && abs(delta_s) < thres)
                        // if(delta_norm < 0.025 && num_fit*1.0/num_all >= 0.95)
                        // if(abs(delta_s) < thres && num_fit*1.0/num_all >= 0.95) 
                        // if(abs(delta_s) < thres) 
                        {
                            scale += delta_s;
                            break;
                        }
                        else
                        {
                            scale += delta_s;
                        }
                    }
                    
                    P = Norm_t * scale;
                }
            }
        }
        
        // 认为0.1s内高度落差不应该大于0.25m。按照15m/s，坡度为8度,则0.1s内高度变化大约为0.21m。一般城市道路的坡度不应大于10度，而坡度越大，一般汽车的速度限制会越大
        // KITTI数据集是在德国采集的，其高速公路的坡度一般较小；但是城区内非标准化道路的坡度则比较大！！
        if(fabs(P(1)) < 0.25)
        {
            // if(pred_dist_t > 0 && P.norm() > 2.5*pred_dist_t)
            // {
            //     return false;
            // }
            
            cout << "Succeed estimate scale of P!" << endl;
            cout << "estimated scale is: " << scale << " final delta_s is: " << delta_s << endl;
            // cout << "estimated translation P is: " << P.transpose() << endl;
            
            *norm_t = P;
        }
        else
        {
            // 是要放弃该估计的t，还是直接将高度方向的运动值置为阈值
            return false;
            // P(1) = P(1)/fabs(P(1))*0.10;
            // *norm_t = P;
        }
    }
    
    return true;
}

// 去除参与LBA的点（在滑窗内至少有连续4帧观测）。
// 对于其中在当前帧被跟踪到的sift点，可以选择将其保留为新的sift检测点（但是仍将该点从地图中删除，至于是否将该点再加入地图，取决于它在下一帧是否被跟踪到）
void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    // 为什么要遍历feature，而不是遍历outlierIndex?？因为遍历后者时需要从list中寻找有对应点Id的元素，这种寻找效率其实没高多少
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        // 因为当前的iterator it可能会被从list删除，则该iterator会失效，之后如果再对it++则会导致崩溃！所以要提前先保存好下一个iterator it_next
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            // 直接消除该FeatureId对象，不需要先管理其中的指针吗？该类及其类成员变量中均没有指针成员，所以无需手动释放！
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

// 下面这个函数是VINS原版的函数，其逻辑有问题
// 此函数用于滑窗要marg最老帧时，将 初始观测帧为最老帧的特征点 的信息 全部转移到第2帧中，包括点的深度估计
// VINS这里的代码是很有问题的，即首观测帧在当前帧的点，其可能参与了当前滑窗的LBA和首帧的marg，那么这个点的所有观测就应该直接从地图中删除才对。另外，当前首帧观测都没有参与LBA和marg的点，后续肯定也不会再参与了！
// 对于参与当前首帧marg的点，像这里还保留其他帧的观测，余下的观测数小于4还好，不会再才与LBA了；要是余下观测还大于等于4，难道它下一个滑窗还要再参与LBA并且再次marg吗（LBA函数中只要是观测帧数大于等于4的点就拿来参与LBA和marg)？
// 保留marg的点的其他观测本身是没问题的，比如用于后续的可视化，但是得保证它不会再参与LBA和再次marg了！！

// ！上面的思考是有问题的，marg某一帧不会直接将其观测到的点从地图删除，而是仅marg该点在该帧的观测（并形成剩下各共视帧之间的位姿约束），但是这些点并没有从地图中删去，即它们可以被后续帧继续观测并提供新的优化约束
// 对点观测的marg形成的共视帧之间的残差信息 相当于 对下一滑窗中这些共视帧对该点观测的 优化的先验约束（即优化后的结果，包括该点在新的观测首帧中的深度值，不能偏离此约束下的值太远，正式因为有这层约束以及需要再次优化这些变量，才会有FEJ理论的出现）
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果该点参与了LBA，且估计得到的深度值变为负的，则删除该点
        if(it->estimated_depth <= 0 && it->has_LBA)
        {
            feature.erase(it);
            continue;
        }

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 剩下的该点的观测帧数小于4帧，则该点肯定不会再参与LBA了，那么就无需再保留该点了！
            if (it->feature_per_frame.size() < TH_NUM_FRAME_FOR_LBA)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                if(it->feature_per_frame[0].depth > 0)
                {
                    it->estimated_depth = it->feature_per_frame[0].depth;
                }
                else if(it->estimated_depth > 0)
                {
                    Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                    Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                    Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                    double dep_j = pts_j(2);
                    // if (dep_j > 0)
                    if(dep_j >= 1.5 && dep_j < mThDepthBg)
                        it->estimated_depth = dep_j;
                    else
                        it->estimated_depth = INIT_DEPTH;
                        // it->estimated_depth = -1.0;
                }
                else
                    continue;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

// 此函数用于滑窗要marg最老帧时，直接将最老帧中的点从地图中删除。因为无论它有没有参与当前滑窗最老帧的marg，它后续都不会再被使用了（如果在当前滑窗参与marg,则理论上就应该彻底删除其所有观测；如果没有参与，则后续也肯定不会参与）
// 原版VINS保留其他的记录可能是为了可视化，或者为了回环？但是其中没有参与过LBA（和marg）的点的估计会准确吗？
void FeatureManager::removeBackShiftDepth()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 初始观测帧不是当前滑窗首帧的特征点，将其初始观测帧的id减去1。那么地图点对象中的used_num是不是也要在这里-1呢？下面的feature_per_frame会删除元素，其元素个数其实就是sed_num
        if (it->start_frame != 0)
        {
            // 在此之前，当前帧的所有静态跟踪点的观测均已加入地图，因此那些最后观测帧不是当前帧的点，则说明跟丢了，这些跟丢的点如果在滑窗中的观测帧数还大于等于4，则之后它还能参与LBA，则保留；
            // 如果当前帧还有跟踪，则下一帧可能还被继续跟踪，最后可能能达到4帧观测参与LBA，则保留。否则就直接删除该点的所有记录
            if (it->feature_per_frame.size() < TH_NUM_FRAME_FOR_LBA && it->endFrame() < WINDOW_SIZE) 
            {
                feature.erase(it);
                continue;
            }
            it->start_frame--;
        }
        // 由于有上面if的情况，则下面else中的点就都是首观测帧在当前滑窗首帧，且连续观测帧数大于等于4。即该点肯定参与了当前滑窗的LBA和marg!那么marg完之后直接把该点从地图中删除了即可。
        else
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // 需要该点在下一帧中重新进行三角化测量吗？
            // it->estimated_depth = -1.0;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 调用此函数则说明系统还没成功VI初始化，则应该保留所有至少有两帧观测(前后2帧，或者单帧的左右观测）的点，这样可以用于VI初始化～
            if (it->feature_per_frame.size() == 0 || (it->feature_per_frame.size() == 1 && !it->feature_per_frame[0].is_stereo))
                feature.erase(it);
            else
            {
                if(it->feature_per_frame[0].is_stereo)
                    it->estimated_depth = it->feature_per_frame[0].depth;
            }
        }
    }
}

// 此函数只有在滑窗去掉次新帧时才被调用，因此frame_count == WINDOW_SIZE
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果该地图点是最新帧中新检测的特征点
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            // 当前帧滑窗次新帧的id - 该点的初始观测帧id
            int j = WINDOW_SIZE - 1 - it->start_frame;
            // 如果该点没有被次新帧观测到，则不处理
            if (it->endFrame() < frame_count - 1)
                continue;
            // 如果该点首观测帧是次新帧，则要求该点下一帧重新三角化测量（如果该点下一帧仍能被观测）
            if(it->start_frame == WINDOW_SIZE - 1) it->estimated_depth = -1.0;
            // 否则，去除该点在次新帧的观测记录
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            // 在这里直接减1，以防别处没有重新计算就直接用该点的used_num
            it->used_num--;
            // 同样地，如果该点只剩下一个观测帧，且不在最新帧上，是否要去除？
            // 否，在我们修改后的项目中，如果当前帧marg次新帧，则那些最后才确认为静态物体的跟踪点以及在物体关联阶段就已经确定为静态的物体上因点数过多而部分暂时被放弃加入的跟踪点观测，
            // 其当前帧观测暂时不加入地图（除了那些在当前帧之前就已经有4帧观测的点，当前帧加入后一定确保其在下一滑窗中仍然有4帧观测，可以参与LBA），而是看其下一帧是否再被跟踪后再两帧观测一起加入！
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
