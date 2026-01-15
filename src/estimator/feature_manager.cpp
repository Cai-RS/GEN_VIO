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
        // todo: 是否应该也要统计窗口内其他帧的观测点数量？
    }
    
    // 参与LBA的点数不应该超过特征点深度的预分配内存大小
    cnt = cnt < NUM_OF_F ? cnt : NUM_OF_F;
    return cnt;
}

// td为相机和IMU之间的时间戳差异，IMU真实对应时间戳=相机时间戳+td
// 添加新的特征地图点和增加某地图点的观测帧记录的操作都是在FeatureManager::addFeatureCheckParallax()函数中
// 另外根据次新帧和次次新帧之间匹配点的平均视差，来决定是否将次新帧作为关键帧，从而决定是要marg掉次新帧还是最老帧（前提是当前滑窗帧数已满）
// VINS-Mono中此函数代码的标注 https://blog.csdn.net/weixin_42846216/article/details/106614794 
bool FeatureManager::addFeatureCheckParallax(int frame_count, double Headers[], double prev_td, double td, FeatureTracker &tracker, int &num_track_add, int &num_3D2D_track, bool need_LBA)
{
    // if(frame_count == 0) return true;
    
    bool use_fea_no_depth = !tracker.add_stereo_for_bg_fea_cur_frame;

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
    int &num_old_track_fea = tracker.num_old_track_fea;
    num_old_track_fea = 0;
    
    int index;
    float dep_prev, dep_cur;
    // 添加静态的跟踪点的信息，这些点如果在后续的位姿估计中有被当作外点的，要如何删除？在地图点的外点移除函数中会实现
    if(!sta_match_fea.empty())
    {
        last_track_num += sta_match_fea.size();

        for(auto &id_pts: sta_match_fea)
        {
            int feature_id = id_pts.first;

            index = gl_id_index_map[feature_id];

            // 给定lambda函数体来完成搜寻，其中使用了捕获
            auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &iter)
                            {
                                return iter.feature_id == feature_id;
                            });
            
            // 如果sta_match_fea中的跟踪点在之前的帧就已经加入静态地图了
            if (it != feature.end())
            {
                int num_frame = it->feature_per_frame.size();
                // 如果该点在上一帧的观测已加入
                if(it->endFrame() == frame_count - 1)
                {
                    // 如果该跟踪点在当前帧才为其在上一帧寻找到立体匹配
                    if(!it->feature_per_frame.back().is_stereo)
                    {
                        if(pts_r_prev.find(feature_id) != pts_r_prev.end())
                        {
                            Eigen::Matrix<double, 8, 1> prev_pts_r;
                            prev_pts_r << un_pts_r_prev[feature_id][0], un_pts_r_prev[feature_id][1], 1.0, pts_r_prev[feature_id].x, pts_r_prev[feature_id].y, un_pts_r_prev[feature_id][2], un_pts_r_prev[feature_id][3], 0;

                            // 如果该点在上上帧中也有立体匹配，则这里需要计算该点在上一帧右图像中的速度
                            if(USE_IMU && ESTIMATE_TD)
                            {
                                if(num_frame > 1 && it->feature_per_frame[(num_frame-2)].is_stereo)
                                {
                                    Vector3d &p_p_r = it->feature_per_frame[(num_frame-2)].pointRight;
                                    // 注意，这里不能直接用tracker中的dt，因为tracker中的dt是实际连续2帧之间的时间间隔。
                                    // 而地图中某点前后2帧不一定是实际中连续的2帧，因为其中间可以连续被marg多帧（次新帧），因此这里要用该2帧的实际时间戳间隔
                                    // prev_pts_r(5) = (prev_pts_r(0) - p_p_r(0))/tracker.prev_dt;
                                    // prev_pts_r(6) = (prev_pts_r(1) - p_p_r(1))/tracker.prev_dt;
                                    int end_f = frame_count - 1;
                                    double dt = Headers[(end_f)] - Headers[(end_f-1)];
                                    prev_pts_r(5) = (prev_pts_r(0) - p_p_r(0))/dt;
                                    prev_pts_r(6) = (prev_pts_r(1) - p_p_r(1))/dt;
                                }
                            }
                            
                            it->feature_per_frame.back().rightObservation(prev_pts_r);
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

                            // 如果是物体点，则其深度直接根据立体匹配给出
                            if(dep_prev > 0)
                            {
                                // 这里其实要么两个深度值都不提前赋予（后续会对在上一帧有立体匹配的2d-2d点进行立体三角化），要么就两个都赋予
                                // 如果只执行第1行而不执行第2行，则后续该点不会进行立体三角化（判断条件为第1行的深度），从而导致该点在首帧下有立体匹配但estimated_depth却为-1.0
                                it->feature_per_frame.back().depth = dep_prev;
                                if(num_frame == 1) it->estimated_depth = dep_prev;
                            }
                        }
                    }
                    
                    // assert(id_pts.second[0].first == 0);
                    // 已有的地图点添加观测记录，此记录中不需要有帧序号信息，因为点的跟踪是连续的，只需要记录初始观测帧的序号即可
                    it->feature_per_frame.emplace_back(id_pts.second[0].second, td);
                    if(id_pts.second.size() == 2) 
                    {
                        // assert(id_pts.second[1].first == 1);
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

                    num_frame += 1;
                }
                // 如果上一帧的观测还没加入，则将两帧观测一起加入。
                else
                {
                    bool is_new_track = false;
                    uchar cls = 0;
                    float dep_prev;
                    if(index > 0)
                    {
                        cls = tracker.obj_cls_id_FAST[(index-1)].first;
                        dep_prev = prev_FAST_dep[index-1];
                    }
                    else
                    {
                        cls = tracker.obj_cls_id_sift[-index].first;
                        dep_prev = prev_sift_dep[-index];
                    }
                    
                    // 如果该点上上帧观测也还没加入？这种情况其实不会发生？
                    // 可能发生在静态物体的跟踪点上，该点在跟踪期间某一帧没有被加入地图
                    // 这样就会使得该点在滑窗内的观测帧不连续，后续一系列操作（入PnP时会出现错误）。
                    // 这里采取的策略是：如果系统需要LBA，则看该点在之前的点数数否足够参与当前帧的LBA，如果不足，则删除该点在之前的所有观测，再加入最新2帧的观测；否则，放弃往该点加入新观测
                    if(it->endFrame() < frame_count - 2)
                    {
                        if(cls == 0 || dep_prev <= 0)
                        {
                            // assert(false);
                            cout << "Weired! Line 281" << endl;
                            exit(-1);
                        }
                        // 保留该点之前的长观测用于参与LBA
                        if(need_LBA && num_frame >= TH_NUM_FRAME_FOR_LBA && it->estimated_depth > 0)
                        {
                            --last_track_num;
                            // if(tracker.sta_obj_fea_in_map_cur.find(feature_id) != tracker.sta_obj_fea_in_map_cur.end())
                                tracker.sta_obj_fea_in_map_cur.erase(feature_id);

                            continue;
                        }
                        else
                        {
                            // 如果该点在上一帧有立体匹配
                            if(pts_r_prev.find(feature_id) != pts_r_prev.end())
                            {
                                // 删除该点在之前的所有观测记录
                                it->feature_per_frame.clear();
                                it->start_frame = (frame_count-1);
                                it->has_LBA = false;
                                is_new_track = true;
                            }
                            else
                            {
                                --last_track_num;
                                // if(tracker.sta_obj_fea_in_map_cur.find(feature_id) != tracker.sta_obj_fea_in_map_cur.end())
                                    tracker.sta_obj_fea_in_map_cur.erase(feature_id);

                                // 是否还要从地图中删除该点？意味它不参与LBA没有作用？可以暂时放着不管
                                // feature.erase(it);
                                continue;
                            }
                        }
                    }

                    Eigen::Matrix<double, 8, 1> pts_l, pts_r;
                    if(un_pts_prev.find(feature_id) == un_pts_prev.end())
                    {
                        // assert(false && "Something wrong with tracker.prev_un_Fea_map!");
                        cout << "Weired! Line 312" << endl;
                        exit(-1);
                    }
                    
                    // 由于上一帧不是首帧，因此上一帧点的归一化平面的速度需要给出，这在LBA时需要使用！
                    pts_l << un_pts_prev[feature_id][0], un_pts_prev[feature_id][1], 1.0, pts_prev[feature_id].x, pts_prev[feature_id].y, un_pts_prev[feature_id][2], un_pts_prev[feature_id][3], 0;
                    if(is_new_track)
                    {
                        // 如果上一帧成为该点的观测首帧，则其二维速度应该为0
                        pts_l(5) = 0;
                        pts_l(6) = 0;
                    }

                    it->feature_per_frame.emplace_back(pts_l,prev_td);
                    
                    // if(has_depth_prev && pts_r_prev.find(feature_id) != pts_r_prev.end())
                    if(pts_r_prev.find(feature_id) != pts_r_prev.end())
                    {
                        pts_r << un_pts_r_prev[feature_id][0], un_pts_r_prev[feature_id][1], 1.0, pts_r_prev[feature_id].x, pts_r_prev[feature_id].y, un_pts_r_prev[feature_id][2], un_pts_r_prev[feature_id][3], 0;
                        if(is_new_track)
                        {
                            // 如果上一帧成为该点的观测首帧，则其二维速度应该为0
                            pts_r(5) = 0;
                            pts_r(6) = 0;
                        }
                        it->feature_per_frame.back().rightObservation(pts_r);

                        if(dep_prev > 0) it->feature_per_frame.back().depth = dep_prev;
                        
                        if(is_new_track) it->estimated_depth = dep_prev;
                    }

                    // 再添加当前帧观测
                    it->feature_per_frame.emplace_back(id_pts.second[0].second,td);
                    if(is_new_track) 
                    {
                        it->used_num = 2;
                        num_frame = 2;
                    }
                    else
                        num_frame += 2;

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

                // 统计被当前帧跟踪到的点中，之前已经至少被连续跟踪过3次（即连续观察到4帧） 的点数
                if(num_frame >= TH_NUM_FRAME_FOR_LBA)
                    ++long_track_num;

                if(num_frame == 2)
                {
                    if(it->estimated_depth > 0) 
                        ++num_3D2D_track;
                    else if(it->feature_per_frame[0].is_stereo)
                    {
                        // 背景点只要有立体匹配，就一定会立体三角化得到有效深度吗？
                        // todo: 建议将立体三角化的操作放在 tracker中找到立体匹配时立即进行（这样可以不要求进行立体校正且立即判断该深度是否有效）；要么将num_3D2D_track的统计放到三角化操作函数内！
                        ++num_3D2D_track;
                    }
                }
                else if(num_frame > 2)
                {
                    if(it->estimated_depth > 0) 
                    {
                        ++num_3D2D_track;
                        ++num_old_track_fea;
                    }
                }
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
                {
                    // assert(false);
                    cout << "Weired! Line 384" << endl;
                    exit(-1);
                }

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
                    // 新跟踪点在上一帧的深度要么来自立体匹配，要么来自后续的三角化。只要有了立体匹配，一般其深度值都是有效的（即使需要后面通过立体三角化）
                    // todo: 其实更通用的做法应该是在tracker线程中找到立体匹配后，就立即基于给定（或上一帧最新估计）的相机外参 来 立体三角化得到深度值并检查深度是否有效！而不是在这里假设立体三角化一定成功....
                    ++num_3D2D_track;
                }

                // 如果该点在当前帧有立体匹配
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
            }
        }
    }
    
    int num_new_ = 0;
    // 当前帧背景点中（如果是跟踪点，则是纯背景跟踪点）没有立体匹配深度的点（跟踪点或新点）之前是没有加入TrackObjFeaFrame和NewObjFeaFrame中的（因为非3D-3D点无法参与物体关联），因此它们还未被加入地图中
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
                        {
                            // assert(false);
                            cout << "Weired! Line 563" << endl;
                            exit(-1);
                        }
                        
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
                            // 新跟踪点其上一帧的深度值要么来自立体匹配，要么只能后续来自于成功三角化
                            ++num_3D2D_track;
                        }

                        // 再添加当前帧左图像的观测
                        feature.back().feature_per_frame.emplace_back(pts_l_1,td);

                        // 上一帧该点是否已经有了有效的深度
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
                        // 背景跟踪点如果已经在地图，则其上一帧观测应该已经在地图中
                        // 即每一帧的纯背景跟踪点都会加入地图，如果跟踪点无效，则会放弃后续继续跟踪该点
                        if(it->endFrame() != frame_count-1)
                        {
                            // assert(false);
                            cout << "Weired! Line 624" << endl;
                            exit(-1);
                        }

                        int num_frame = it->feature_per_frame.size();

                        // 如果该跟踪点在当前帧才临时为其在上一帧寻找到立体匹配
                        if(!it->feature_per_frame.back().is_stereo)
                        {
                            if(pts_r_prev.find(pt_id) != pts_r_prev.end())
                            {
                                Eigen::Matrix<double, 8, 1> prev_pts_r;
                                prev_pts_r << un_pts_r_prev[pt_id][0], un_pts_r_prev[pt_id][1], 1.0, pts_r_prev[pt_id].x, pts_r_prev[pt_id].y, un_pts_r_prev[pt_id][2], un_pts_r_prev[pt_id][3], 0;

                                // 如果该点在上上帧中也有立体匹配，则这里是否需要计算该点在上一帧右图像中的速度
                                // 二维速度只有在需要估计IMU和相机之间的时间戳差值时才会用到？
                                if(USE_IMU && ESTIMATE_TD)
                                {
                                    if(num_frame > 1 && it->feature_per_frame[(num_frame-2)].is_stereo)
                                    {
                                        Vector3d &p_p_r = it->feature_per_frame[(num_frame-2)].pointRight;
                                        // prev_pts_r(5) = (prev_pts_r(0) - p_p_r(0))/tracker.prev_dt;
                                        // prev_pts_r(6) = (prev_pts_r(1) - p_p_r(1))/tracker.prev_dt;
                                        int end_f = frame_count - 1;
                                        double dt = Headers[(end_f)] - Headers[(end_f-1)];
                                        prev_pts_r(5) = (prev_pts_r(0) - p_p_r(0))/dt;
                                        prev_pts_r(6) = (prev_pts_r(1) - p_p_r(1))/dt;
                                    }
                                }
                                it->feature_per_frame.back().rightObservation(prev_pts_r);

                                float dep_prev;
                                index = gl_id_index_map[pt_id];
                                // 如果是静态物体点，则其深度值已经直接计算出来了
                                if(index > 0)
                                {
                                    dep_prev = prev_FAST_dep[index-1];
                                }
                                else
                                {
                                    dep_prev = prev_sift_dep[-index];
                                }
                                
                                // 如果是物体点，则其深度直接根据立体匹配给出
                                if(dep_prev > 0)
                                {
                                    it->feature_per_frame.back().depth = dep_prev;
                                    if(num_frame == 1) it->estimated_depth = dep_prev;
                                }
                            }
                        }

                        // 加入当前帧的观测
                        it->feature_per_frame.emplace_back(pts_l_1,td);
                        ++num_frame;

                        if(num_frame >= TH_NUM_FRAME_FOR_LBA)
                            ++long_track_num;

                        if(num_frame == 2)
                        {
                            if(it->estimated_depth > 0 || it->feature_per_frame[0].is_stereo)
                                ++num_3D2D_track;
                        }
                        else if(num_frame > 2)
                        {
                            // 即使该点在上一帧没有来自立体匹配的深度，也可将其当作3D-2D点（只要其首帧下有深度值即可，后续每帧的深度至少可以通过运动变换来计算）
                            if(it->estimated_depth > 0)
                            {
                                ++num_old_track_fea;
                                ++num_3D2D_track;
                            }
                        }
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
        // 得到以像素距离为单位的两帧间点匹配的平均光流长度
        // last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH_X;

        // 如果次新帧和次次新帧之间关联特征点的平均视差大于阈值，则认为次新帧为关键帧，需要被保留。如果当前滑窗帧数已满，则会marg掉最老的一帧；否则，marg掉最新帧
        // 首个滑窗满帧之前，每一帧都被保留；随着首个滑窗满帧之后，随着每次插入新的关键帧（即确定次新帧为关键帧），则会逐渐将最老的帧给marg掉，直到最后滑窗内剩下的都是真正的关键帧！
        // TODO：MIN_PARALLAX这个参数对于不同场景应该要有不同的设置（例如室内场景和室外场景）？
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

// 当前帧的某些静态物体（物体关联时期为静态，但仅有2帧观测，在VI初始化后需要等待是否marg次新帧再决定是否加入地图）
// 或动态物体（指的是在物体关联阶段没法确定是否为静态的，在完成位姿估计后被确定为静态），使用此函数将这些静态物体的跟踪特征点观测（上一帧以及当前帧的）加入到静态地图中
// 如果当前帧marg了次新帧，则暂时不把当前帧该物体的观测加入地图（除非加上当前这一帧观测，该特征点在map中的观测数达到了参与LBA的最低帧数）
// 由于这里只会添加物体点，所以这些点在每一帧都是有depth的（物体点必须是有stereo match）
int FeatureManager::addStaticFeature(int frame_count, int prev_td, double td, FeatureTracker &tracker, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[],
                                        Matrix3d ric[], const vector<pair<int,int>> &id_fea, const int id_obj_cur, bool need_marg, bool marg_old, 
                                        const vector<int> &reserve_new_fea, const vector<int> &ignore_pts, bool add_new_fea, int global_cls)
{
    cout << "Add static obj fea track into map!" << endl;
    // 上一帧该特征点在左相机归一化平面的点坐标和速度
    auto &un_pts_prev = tracker.prev_un_Fea_map;
    // 上一帧该特征点的左图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_prev = tracker.prevLeftFeaMap;
    // 上一帧该特征点在右相机归一化平面的点坐标和速度
    auto &un_pts_r_prev = tracker.prev_un_r_Fea_map;
    // 上一帧该特征点的右图像像素坐标（那为何不把所有的元素都放在一个map中呢？）
    auto &pts_r_prev = tracker.prevRightFeaMap;

    int num_fail = 0;

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
            // assert(false);
            exit(-1);
        }

        auto &pts = tracker.TrackObjFeaFrame[id_obj_cur][old_pt_id];
        // 该点之前还未加入地图
        // 此种情况可以是 该物体在上一帧为动态物体（则pt_id的first和second不会相同）或者是新物体，则其所有的特征点都不会在地图中。
        // 也可能是上上帧的新点，在上一帧跟踪到该点并确认为静态物体，但是上一滑窗marg了次新帧，则上一帧的跟踪暂未加入，则此时该点不在地图中。
        // 此时pt_id.first和pt_id.second是一样的值，即该点在前后两帧的全局id是不变的
        if (it == feature.end())
        {
            if(need_marg && !marg_old) 
            {
                ++num_fail;
                continue;
            }

            // 如果某些跟踪点（即运动估计外点）是无效的，但是被保留为新点（当前帧有深度估计），则这里暂时不添加
            if (!reserve_new_fea.empty() && find(reserve_new_fea.begin(),reserve_new_fea.end(),old_pt_id) != reserve_new_fea.end()) 
            {
                ++num_fail;
                continue;
            }
            // 如果是已经被删除了的点（即运动估计外点且不能保留为新点）
            if (!ignore_pts.empty() && find(ignore_pts.begin(),ignore_pts.end(),old_pt_id) != ignore_pts.end()) 
            {
                ++num_fail;
                continue;
            }

            // 形成观测首帧的信息，即只有归一化平面坐标、像素坐标和全局物体id（这个可以随便写，用不到），速度值都是0。
            // 注意观测首帧是上一帧，即frame_count-1
            // feature.push_back(FeaturePerId(pt_id.second, frame_count-1));
            feature.push_back(FeaturePerId(old_pt_id, frame_count-1));
            Eigen::Matrix<double, 8, 1> pts_l, pts_r;

            if(un_pts_prev.find(old_pt_id) == un_pts_prev.end())
            {
                cout << "old_pt_id" << old_pt_id << endl;
                cout << "new_pt_id" << pt_id.second << endl;
                // assert(false && "Something wrong with tracker.prev_un_Fea_map!");
                exit(-1);
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
                    // assert(false && "why obj fea has no depth?");
                    cout << "Weired! why obj fea has no depth? Line 797" << endl;
                    exit(-1);
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
                {
                    // 物体跟踪点在当前帧可以没有立体匹配，即可以是3D-2D点？但如果是物体跟踪点，则应该已经通过运动更新获得有效深度值了
                    cout << "Weired! Line 828" << endl;
                    exit(-1);
                }
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

            bool new_track = false;
            int num_frame = it->feature_per_frame.size();

            // 如果该点上一帧的观测也还未加入。则先加入上一帧的观测。
            // 这有2种情况： 1. 只存在于上一帧滑窗已满且marg了次新帧，则该静态物体跟踪点在上一帧的观测暂时还没加入当前帧; 2. 该点为静态物体点，上一帧该点及之前该点已连续加入，但是当前帧该物体直到运动估计后才确定为静态。
            if(it->endFrame() < frame_count - 1)
            {
                // 如果该点在地图中的最新观测帧不是上上帧，则该点在地图中的观测是不连续的。则要么放弃该点的加入，要么删除该点在地图的原观测转而加入最新的2帧观测
                if(it->endFrame() != (frame_count - 2))
                {
                    int cnt = 0;
                    if(need_marg && marg_old && it->start_frame == 0) cnt = 1; 
                    // 如果该点不能参与下一个滑窗的LBA，则从地图删除该点的原记录，加入新的观测;否则，保留该点的旧观测，放弃加入该点的新观测
                    if(num_frame >= (TH_NUM_FRAME_FOR_LBA+cnt)) 
                    {
                        ++num_fail;
                        continue;
                    }
                    else
                    {
                        // 如果当前滑窗不需要marg次新帧
                        if(!need_marg || marg_old)
                        {
                            // 如果要使该点上一帧为首帧，则其必须有立体匹配
                            if(un_pts_r_prev.find(old_pt_id) != un_pts_r_prev.end())
                            {
                                it->feature_per_frame.clear();
                                new_track = true;
                            }
                            else
                            {
                                ++num_fail;
                                // 是否要在这里删除该点?可以不用
                                // feature.erase(it);
                                continue;
                            }
                        }
                        else
                        {
                            ++num_fail;
                            // 是否要在这里删除该点?可以不用
                            // feature.erase(it);
                            continue;
                        }
                    }
                }

                Eigen::Matrix<double, 8, 1> pts_l, pts_r;
                if(un_pts_prev.find(old_pt_id) == un_pts_prev.end())
                {
                    // assert(false && "Something wrong with tracker.prev_un_Fea_map!");
                    cout << "Weired! Line 980" << endl;
                    exit(-1);
                }
                
                // 如果上一帧不是首帧，则上一帧点的归一化平面的速度需要给出，这在LBA时需要使用！
                pts_l << un_pts_prev[old_pt_id][0], un_pts_prev[old_pt_id][1], 1.0, pts_prev[old_pt_id].x, pts_prev[old_pt_id].y, un_pts_prev[old_pt_id][2], un_pts_prev[old_pt_id][3], 0;
                if(new_track)
                {
                    pts_l(5) = 0;
                    pts_l(6) = 0;
                }
                it->feature_per_frame.emplace_back(pts_l,prev_td);

                bool has_stereo_prev = false;
                if(un_pts_r_prev.find(old_pt_id) != un_pts_r_prev.end())
                {
                    pts_r << un_pts_r_prev[old_pt_id][0], un_pts_r_prev[old_pt_id][1], 1.0, pts_r_prev[old_pt_id].x, pts_r_prev[old_pt_id].y, un_pts_r_prev[old_pt_id][2], un_pts_r_prev[old_pt_id][3], 0;
                    if(new_track)
                    {
                        pts_r(5) = 0;
                        pts_r(6) = 0;
                    }
                    it->feature_per_frame.back().rightObservation(pts_r);
                    has_stereo_prev = true;
                }
                
                // 如果是物体点，则该物体在上一帧和当前帧都应该有深度值才对，不论其深度值是否来自立体匹配（当前帧即使没有立体匹配，也应该有运动更新的深度）
                if(dep_prev > 0)
                {
                    // 如果上一帧的深度值来自于立体匹配
                    if (has_stereo_rectified && has_stereo_prev)
                    {
                        it->feature_per_frame.back().depth = dep_prev;
                    }
                    if(new_track) it->estimated_depth = dep_prev;
                }
                else
                {
                    if(cls > 0)
                    {
                        cout << "Weired! why obj fea has no depth? Line 1020" << endl;
                        exit(-1);
                    }
                }
            }
            else if(it->endFrame() == (frame_count - 1))
            {
                
                // 如果该点上一帧的观测已经加入地图。则查看该点是否在当前帧为上一帧临时添加了立体匹配
                if(!it->feature_per_frame.back().is_stereo)
                {
                    if(un_pts_r_prev.find(old_pt_id) != un_pts_r_prev.end())
                    {
                        Eigen::Matrix<double, 8, 1> pts_r;
                        pts_r << un_pts_r_prev[old_pt_id][0], un_pts_r_prev[old_pt_id][1], 1.0, pts_r_prev[old_pt_id].x, pts_r_prev[old_pt_id].y, un_pts_r_prev[old_pt_id][2], un_pts_r_prev[old_pt_id][3], 0;
                        // 如果上一帧为该点在地图的观测首帧，则将其2维速度置为0。左图像中的速度是否为要置为0？
                        // todo:那如果是这样，每次mmarg到某个点的首帧时，是否都要将其剩下的首帧的左右图像中的速度均置为0？这取决于LBA中如何使用某个点首观测帧下的速度变量！！！观测首帧应该就不会使用其速度变量了
                        if(num_frame == 1)
                        {
                            pts_r(5) = 0;
                            pts_r(6) = 0;
                        }
                        it->feature_per_frame.back().rightObservation(pts_r);
                        if(dep_prev > 0)
                        {
                            // 如果上一帧的深度值来自于立体匹配
                            if (has_stereo_rectified)
                            {
                                it->feature_per_frame.back().depth = dep_prev;
                            }
                            if(num_frame == 1) it->estimated_depth = dep_prev;
                        }
                    }
                    else
                    {
                        // 如果该点在地图中只有上一帧的观测，且没有有效深度值，则放弃该点
                        // 是否进一步要求在上一帧一定要有立体匹配的深度？可能本系统允许该观测首帧深度来自于运动更新（上上帧被marg时深度传递到上一帧）
                        if(num_frame == 1 && it->estimated_depth <= 0)
                        {
                            ++num_fail;
                            // 是否要在这里删除该点?可以不用
                            // feature.erase(it);
                            continue;
                        }
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
                    cout << "Weired! Line 1092" << endl;
                    exit(-1);
                }
            }

            if(new_track)
            {
                it->start_frame = (frame_count - 1);
                it->used_num = 2;
                it->has_LBA = false;
            }
        }

        // todo: 利用当前帧PnP或LBA后的位姿 对 加入的静态物体点进行重投影误差检验？如果需要，是否放在该点加入地图之前？
        // 还需要当前帧具有较好的PnP估计结果 或者 LBA估计结果？
        if(1)
        {
            double err = 0.0;
            int errCnt = 0;
            auto it_per_id = find_if(feature.begin(), feature.end(), [old_pt_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == old_pt_id;
                                    });
            int num_frame = it_per_id->feature_per_frame.size();
            if(num_frame > 1)
            {
                int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
                Vector3d pts_i = it_per_id->feature_per_frame[0].point;
                double depth = it_per_id->estimated_depth;
                for (auto &it_per_frame : it_per_id->feature_per_frame)
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

                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        // if(imu_i != imu_j)
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
                    }
                }

                double ave_err = err / errCnt;
                // ave_err是在归一化平面上的距离误差，乘以焦距以后就是在图像上的像素误差！
                if(ave_err * FOCAL_LENGTH_X >= 3.5)
                {
                    // 删除该点在地图中的记录
                    feature.erase(it_per_id);

                    // todo：是否要删除该物体跟踪点？如果删除了，是否会导致某个物体在当前帧完全没有了特征点？可能还有部分跟踪点和新店，极端情况下还能使用像素点跟踪进行物体的运动估计
                    // 是否有可能该物体的运动估计不准确，导致其被误认为静态物体？那是否要修改该物体为动态的？否则如果保留这些跟踪点，其下一帧是否会被用来参与F_H的估计？
                    int index = tracker.gl_id_index_map[old_pt_id];
                    if(index > 0)
                        tracker.statusLeftRIght[(index-1)] = 0;
                    else
                        tracker.status_sift[(-index)] = 0;
                    
                    ++num_fail;
                    continue;
                }
            }
        }
        // 记录当前帧观测加入地图的点
        tracker.sta_obj_fea_in_map_cur.insert(old_pt_id);
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

    return num_fail;
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

void FeatureManager::setDepth(int frameCnt, const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 没有参与当前LBA的点跳过
        if (it_per_id.used_num < TH_NUM_FRAME_FOR_LBA)
        {
            continue;
        }
        
        // 首帧深度为负的点也没有参与当前LBA
        if(it_per_id.estimated_depth <= 0)
        {
            // 将那些后续再也无法参与LBA的点及时删除
            // 因为当前系统不会为非最新跟踪点再进行2d-2d三角化，并且该点该地图点后续就算marg首帧后剩下的首帧有深度，其帧数也不足以参与LBA了
            if((it_per_id.endFrame() != frameCnt && it_per_id.used_num <= TH_NUM_FRAME_FOR_LBA) || (it_per_id.endFrame() == frameCnt && it_per_id.used_num > 2))
            {
                it_per_id.solve_flag = 2;
            }
            continue;
        }
        
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        // 该点参与过LBA
        it_per_id.has_LBA = true;
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth <= 0 || it_per_id.estimated_depth > mThDepthBg)
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

// 计算 给定的参与PnP的3D-2D点中的 内点们 到 估计的极线 的平均距离
float FeatureManager::cal_ave_epi_line_dist_pts(const Matrix3d &R_cam_motion, const Vector3d &P_cam_motion, const vector<list<FeaturePerId>::iterator> &fea_iters, const vector<uchar> &status)
{
    bool has_status = (!status.empty());
    int num_pt = fea_iters.size();

    int g_id, num_frame;
    double ave_dist = -1.0;

    Matrix3d t_up;
    t_up << 0.0, -P_cam_motion(2), P_cam_motion(1), P_cam_motion(2), 0.0, -P_cam_motion(0), -P_cam_motion(1), P_cam_motion(0), 0.0;
    // 本质矩阵到关键矩阵
    Matrix3d Mat_F = K_trans_inv * t_up * R_cam_motion * K_inv;

    float F00 = Mat_F(0,0);
    float F01 = Mat_F(0,1);
    float F02 = Mat_F(0,2);
    float F10 = Mat_F(1,0);
    float F11 = Mat_F(1,1);
    float F12 = Mat_F(1,2);
    float F20 = Mat_F(2,0);
    float F21 = Mat_F(2,1);
    float F22 = Mat_F(2,2);

    int num_valid = 0, num_invalid = 0;
    Vector3d pt1(0,0,1), pt2(0,0,1);

    for(int i = 0; i < num_pt; ++i)
    {
        if(has_status && status[i] == 0) continue;

        num_frame = fea_iters[i]->feature_per_frame.size();
        // 注意，对于基础矩阵的极线约束而言，要使用的是像素坐标！本质矩阵才使用归一化坐标！
        Vector2d &pix1 = fea_iters[i]->feature_per_frame[(num_frame-2)].uv;

        pt1(0) = pix1(0);
        pt1(1) = pix1(1);
        // 上一帧的点pt1在当前帧中的极线 l2 = Mat_F*x1 = (a, b, c)
        const float a = F00*pt1(0) + F01*pt1(1) + F02;
        const float b = F10*pt1(0) + F11*pt1(1) + F12;
        const float c = F20*pt1(0) + F21*pt1(1) + F22;

        const float den = a*a + b*b;
        // F矩阵无效，一般是因为相机位移t为0?
        if(den == 0) 
        {
            ++num_invalid;
            continue;
        }
        
        Vector2d &pix2 = fea_iters[i]->feature_per_frame.back().uv;
        pt2(0) = pix2(0);
        pt2(1) = pix2(1);
        // 当前帧的点pt2在上一帧中的极线 l1 = x2^t*Mat_F = (a_inv, a_inv, a_inv)
        const float a_inv = pt2(0)*F00 + pt2(1)*F10 + F20;
        const float b_inv = pt2(0)*F01 + pt2(1)*F11 + F21;
        const float c_inv = pt2(0)*F02 + pt2(1)*F12 + F22;
        
        const float den_inv = a_inv*a_inv + b_inv*b_inv;
        if(den_inv == 0) 
        {
            ++num_invalid;
            continue;
        }
        
        const float num = a*pt2(0) + b*pt2(1) + c;
        // 点到线的距离
        float dist = sqrt(num*num/den);

        const float num_inv = a_inv*pt1(0) + b_inv*pt1(1) + c_inv;
        float dist_inv = sqrt(num_inv*num_inv/den_inv);

        ave_dist += dist;
        ave_dist += dist_inv;
        ++num_valid;
    }
    
    if(num_valid > 0) ave_dist = ave_dist/2.0/num_valid;

    return ave_dist;
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
        int num_try_ransac = 0;
        float reprojErr = 1.5;
        double conf = 0.99;
        // 特征点数比较少，因此使用较高的置信度
        // 需要迭代的次数和 选择的估计方法（即每次计算需要选取的点数，P3P为3个点）、要达到内点集的置信度 和 正确匹配点占点集的比例 相关。这里如果选择P3P，0.95置信度，0.7的内点比例，则只需要7次迭代...
        pnp_succ = cv::solvePnPRansac(pts3D, pts2D, M_K, D, rvec, t, true, 300, reprojErr/FOCAL_LENGTH_X, conf, inliers); // AP3P(5) EPNP(1) ITERATIVE DLS
        ++num_try_ransac;
        
        // 如果估计失败，则适当放宽条件
        if(!pnp_succ)
        {
            while(!pnp_succ)
            {
                if(num_try_ransac >= 4) break;
                printf("pnp using RANSAC with high confidence and accuracy failed ! Try again with lower conf and accu!\n");
                cv::Mat inliers_loose;
                cv::Rodrigues(tmp_r, rvec);
                cv::eigen2cv(P_initial, t);

                reprojErr += 0.5;
                conf -= 0.01;
                // 此函数内应该是不会改变非空Mat的inliers的size的，如果直接使用上面的inliers，则两次估计时如果内点数不一样，则数量差异无法体现在inliers中！所以这里使用新的inliers_loose
                pnp_succ = cv::solvePnPRansac(pts3D, pts2D, M_K, D, rvec, t, true, 300, reprojErr/FOCAL_LENGTH_X, conf, inliers_loose);
                
                ++num_try_ransac;

                if(pnp_succ)
                {
                    use_RANSAC = true;
                    inliers_loose.copyTo(inliers);
                    printf("Threshold of reprojection error: %f, confidence of PnP: %f\n", reprojErr, conf);
                }
            }

            if(!pnp_succ)
            {
                printf("pnp using RANSAC failed ! Try using PnP without RANSAC!\n");
                // 如果仍然估计失败，则放弃使用RANSAC，这样的估计结果很难保证精度！！
                cv::eigen2cv(R_initial, tmp_r);
                cv::Rodrigues(tmp_r, rvec);
                cv::eigen2cv(P_initial, t);
                pnp_succ = cv::solvePnP(pts3D, pts2D, M_K, D, rvec, t, 1);
            }
        }
        else
        {
            use_RANSAC = true;
            printf("Threshold of reprojection error: %f, confidence of PnP: %f\n", reprojErr, conf);
        }
    }
    
    // VINS-Fusion项目中，此处只是用匹配点集估计两帧间的相机位姿，而没有将外点跟踪点剔除。如果总的跟踪长度小于4帧的点，在滑窗满了之后不会参与LBA，会被逐渐逐帧地删除观测；而大于3帧观测的点后续则会参与LBA，如果其总的投影误差仍大于阈值，则会被删除。
    // 标记运动估计的内点为负数。这里每一帧都将运动估计外点删除，以便下一帧添加新的特征点！但这样子会不会使得长跟踪的点数变少？
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
bool FeatureManager::initFramePoseByPnP(int frameCnt, FeatureTracker &tracker, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], Matrix3d &pred_R, Vector3d &pred_P, 
                                         const Matrix3d &prev_cam_R, const Vector3d &prev_cam_P, float &ave_epi_dist, vector<int> &reserve_new_sift, vector<int> &reserve_new_FAST,
                                         int &num_track_cur_bg, int &num_inlier_fea_PnP, bool comp_with_prev_esti, bool initial_succ)
{
    num_inlier_fea_PnP = 0;
    bool has_RANSAC = false;
    // int num_track_fea_stat;
    // 注意，最开始进入这里处理的是系统的第二帧图像，此时有深度的特征点既包括上一帧和当前帧都被观测到的点，也包括当前帧新增加的特征点
    if(frameCnt > 0)
    {
        int &num_old_track = tracker.num_old_track_fea;
        vector<cv::Point2d> pts2D;
        vector<cv::Point3d> pts3D;
        vector<int> pt_id_vec;
        vector<list<FeaturePerId>::iterator> fea_iters;
        double dep;
        int num_frame, start_frame, row_bloc, col_bloc, row_big_bloc, col_big_bloc, g_id, l_id;
        
        vector<int> g_id_pt_not_in_PnP;
        vector<list<FeaturePerId>::iterator> iters_fea_not_in_PnP;
        set<int> id_old_track;
        double pred_delta_ang = 0, pred_norm_p = 0;

        Matrix3d pred_R_cam_motion;
        Vector3d pred_P_cam_motion;

        // 要么是有来自IMU积分或恒速运动模型的运动预测，要么是已经进行了至少一次PnP而有了初始估计值
        if(frameCnt > 1 || comp_with_prev_esti)
        {
            pred_R_cam_motion = pred_R.transpose() * prev_cam_R;
            pred_P_cam_motion = pred_R.transpose() * (prev_cam_P - pred_P);

            Quaterniond delta_Q(pred_R_cam_motion);
            pred_delta_ang = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);

            pred_norm_p = pred_P_cam_motion.norm();
        }

        // 是否要均匀地选择3D-2D点。
        // 这里不是说不要求各个大bloc中的3D-2D点数大致相同，而是因为在tracker阶段就已经尽可能追求均匀化了，这里就不再特意进行均匀选取
        bool uniformly_select = false;
        
        if(uniformly_select)
        {
            int num_3D_2D = 0, num_stereo_prev = 0, num_in_bloc = 0;
            vector<int> num_pts_bloc(36,0);

            vector<int> &num_pts_in_big_bloc = tracker.num_fea_3D2D_big_bloc;
            auto iter = min_element(num_pts_in_big_bloc.begin(), num_pts_in_big_bloc.end());
            // 迭代器就像一个指针
            int min_num_big_bloc = *iter;
            auto iter_1 = max_element(num_pts_in_big_bloc.begin(), num_pts_in_big_bloc.end());
            int max_num_big_bloc = *iter_1;

            // 理想的情况是从每个大bloc中选取7个2D-2D点
            int Th_num_big_bloc = 7;
            // 如果每个bloc中的点数都比较多，则适当增加点数
            if(min_num_big_bloc > 10)
            {
                Th_num_big_bloc = min_num_big_bloc * 0.7;
            }
            else if(min_num_big_bloc == 0)
            {
                Th_num_big_bloc = 5;
            }
            
            // 从每个大bloc中选来进行PnP的最大点数
            Th_num_big_bloc = min(max_num_big_bloc, Th_num_big_bloc);
            
            int w_big_bloc = 200*3;
            int h_big_bloc = 60*3;
            uchar cls;
            // 由于在某些情况下，纯背景点的3D-2D点可能较少，因此会在物体关联阶段往地图中加入一些静态物体的跟踪点！这里找出这些点并加入各个大bloc的后面
            for(list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
            {
                num_frame = it_per_id->feature_per_frame.size();
                if(num_frame < 2 || it_per_id->endFrame() != frameCnt) continue;

                g_id = it_per_id->feature_id;
                int l_id = tracker.gl_id_index_map[g_id];
                if(l_id > 0)
                {
                    cls = tracker.obj_cls_id_FAST[(l_id-1)].first;
                    dep = tracker.prev_FAST_dep[(l_id-1)];
                }
                else
                {
                    cls = tracker.obj_cls_id_sift[(-l_id)].first;
                    dep = tracker.prev_sift_dep[(-l_id)];
                }
                
                // 静态物体的跟踪点
                if(cls != 0)
                {
                    if(dep <= 0)
                    {
                        cout << "Weired! Line 1426" << endl;
                        exit(-1);
                    }

                    row_big_bloc = it_per_id->feature_per_frame[(num_frame-2)].uv(1)/h_big_bloc;
                    col_big_bloc = it_per_id->feature_per_frame[(num_frame-2)].uv(0)/w_big_bloc;

                    if(row_big_bloc > 1) row_big_bloc = 1;
                    if(col_big_bloc > 1) col_big_bloc = 1;

                    int id_big_bloc = row_big_bloc * 2 + col_big_bloc;
                    num_in_bloc = num_pts_in_big_bloc[id_big_bloc];

                    if(num_in_bloc < NUM_FEA_IN_BIG_BLOC)
                    {
                        tracker.id_fea_3D2D_big_bloc[id_big_bloc][num_in_bloc] = g_id;
                        num_pts_in_big_bloc[id_big_bloc] += 1;
                    }
                    else
                    {
                        g_id_pt_not_in_PnP.push_back(g_id);
                        iters_fea_not_in_PnP.push_back(it_per_id);
                    }
                }
            }
            
            int sum = 0;
            // 地图中存在的总的当前帧的3D-2D点数
            for(int num_pt: num_pts_in_big_bloc)
            {
                cout << "num of 3D-2D in big_bloc: " << num_pt << endl;
                sum += num_pt;
            }

            num_inlier_fea_PnP = sum;

            // 直接使用所有的3D-2D点来参与PnP？
            // auto iter_2 = max_element(num_pts_in_big_bloc.begin(), num_pts_in_big_bloc.end());
            // Th_num_big_bloc = *iter_2;
            
            int g_id, l_id, num_check, num_select, total_select = 0;
            vector<int> num_select_big_bloc(4,0);
            vector<int> num_check_big_bloc(4,0);
            while(sum > 0)
            {
                for(int k = 0; k < 4; ++k)
                {
                    num_check = num_check_big_bloc[k];

                    if(num_check < num_pts_in_big_bloc[k])
                    {
                        --sum;
                        num_check_big_bloc[k] += 1;

                        g_id = tracker.id_fea_3D2D_big_bloc[k][num_check];

                        if(tracker.gl_id_index_map.find(g_id) == tracker.gl_id_index_map.end())
                        {
                            // 有效的3D-2D点减1
                            --num_inlier_fea_PnP;
                            continue;
                        }
                        
                        l_id = tracker.gl_id_index_map[g_id];

                        if(l_id > 0)
                        {
                            if(tracker.statusLeftRIght[(l_id-1)] == 0)
                            {
                                --num_inlier_fea_PnP;
                                continue;
                            }
                        }
                        else
                        {
                            if(tracker.status_sift[-l_id] == 0)
                            {
                                --num_inlier_fea_PnP;
                                continue;
                            }
                        }

                        auto it = find_if(feature.begin(), feature.end(), [g_id](const FeaturePerId &it)
                                        {
                                            return it.feature_id == g_id;
                                        });
                        
                        if(it == feature.end())
                        {
                            --num_inlier_fea_PnP;
                            continue;
                        }
                        else
                        {
                            if(it->endFrame() != frameCnt)
                            {
                                --num_inlier_fea_PnP;
                                continue;
                            }
                        }
                        
                        // 该大bloc中已被选择参与PnP的3D-2D点数
                        num_select = num_select_big_bloc[k];
                        
                        if(num_select < Th_num_big_bloc)
                        {
                            num_frame = it->feature_per_frame.size();
                            bool find_stereo = false, find_dep = false;
                            int index;
                            
                            // 应该从离当前帧最近 还是 最远的帧开始遍历该点的观测。
                            // 这有好有坏，越早的帧，其全局位姿误差会越小，但是和当前帧的特征点匹配准度越低（毕竟是间接地匹配）！
                            // 所以SOFT-SLAM中的设计是将某个跟踪点在当前帧的投影点与之前的帧中都分别进行匹配优化，即当前帧与之前的所有帧的匹配都是独立的，而且它是估计之前各帧与当前帧之间的相对运动，用以平滑当前帧与上一帧的运动！
                            for(int j = (num_frame-2); j >= 0; --j)
                            {
                                if(it->feature_per_frame[j].is_stereo)
                                {
                                    dep = it->feature_per_frame[j].depth;
                                    index = j;

                                    if(dep > 0)
                                    {
                                        start_frame = frameCnt - (num_frame - 1 - j);

                                        find_stereo = true;
                                        find_dep = true;

                                        break;
                                    }
                                }
                            }

                            // 如果该点在之前的任何帧中都没有立体匹配，则使用该点在上一帧的深度值(加入系统允许保留来自运动更新的深度值)
                            if(!find_stereo)
                            {
                                // 筛选过的点在上一帧一定有深度值?
                                if(l_id > 0)
                                {
                                    dep = tracker.prev_FAST_dep[(l_id-1)];
                                }
                                else
                                {
                                    dep = tracker.prev_sift_dep[(-l_id)];
                                }
                                
                                if(dep > 0)
                                {
                                    index = num_frame-2;
                                    start_frame = frameCnt-1;
                                    find_dep = true;
                                }
                            }

                            // 确实有些点可能是在上一帧没有深度值的，例如该点在上一帧有立体匹配，但立体三角测量后得到的深度不满足要求？
                            if(!find_dep)
                            {
                                // 这应该是该点在立体三角化失败
                                --num_inlier_fea_PnP;
                                // cout << "l_id of pt: " << tracker.gl_id_index_map[g_id] << endl;
                                // cout << "Weired! Line 1194" << endl;
                                // exit(-1);

                                continue;
                            }

                            // 计算当前帧所观测到的特征地图点在其被观测首帧的相机坐标系下的3D坐标，并且转换到该帧的IMU坐标系下。此时应该要保证外参ric和tic是较为准确的。
                            Vector3d ptsInIMU = ric[0] * (it->feature_per_frame[index].point * dep) + tic[0];
                            // 根据地图点被观测首帧图像所对应的IMU坐标系的位姿，将所有地图点投影到统一的当前假定的世界坐标系下！
                            // 因为Rs[0]是初始帧IMU坐标系相对于“当前世界坐标系（通过取加速度测量值的平均来作为首帧IMU坐标系中的g，从而得到一个与东北天坐标系的相对姿态）“的姿态（这个姿态无法使首帧IMU变换到准确的东北天坐标系）；
                            // 而在estimator.cpp文件的processImage()函数中，在完成VI初始化之前，对于每一帧处理结束前都将当前帧的Rs和Ps赋值为为下一帧的Rs和Ps，
                            // 因此之后每一帧的此处的Rs和Ps是用 前一帧可靠的视觉估计的位姿(加入上一帧在下面的solvePoseByPnP()视觉估计成功了) 和 其与前一帧之间的IMU的积分 来推断得到该帧IMU在假定世界坐标系下的位姿。这是不准确的估计！
                            Vector3d ptsInWorld = Rs[start_frame] * ptsInIMU + Ps[start_frame];

                            cv::Point3d point3d(ptsInWorld(0), ptsInWorld(1), ptsInWorld(2));
                            // 注意，这里给的2D点是当前帧归一化平面上的坐标
                            Vector3d &un_pt_cur = it->feature_per_frame[(num_frame-1)].point;
                            cv::Point2d point2d(un_pt_cur(0), un_pt_cur(1));
                            pts3D.push_back(point3d);
                            pts2D.push_back(point2d);

                            pt_id_vec.push_back(g_id);
                            fea_iters.push_back(it);

                            num_select_big_bloc[k] += 1;
                            ++total_select;
                        }
                        else
                        {
                            g_id_pt_not_in_PnP.push_back(g_id);
                            iters_fea_not_in_PnP.push_back(it);
                        }
                    }
                }
            }

            // todo: 如果此时选择的3D-2D点可能还不足，那么就需要从有足够点的大bloc中继续选择！
        }
        else
        {
            num_old_track = 0;
            num_track_cur_bg = 0;

            set<int> selected_pt;
            int num_select = 0;
            // 保证一定数量的远点
            // 如果当前帧有较为明显的旋转，则增大远跟踪点（它们的光流会因为旋转而较大，因此相对误差较小，对精确估计旋转R非常重要）
            // int num_track_far_pt = 5;
            // if(pred_delta_ang >= 0.3) num_track_far_pt = 10;

            for(list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
            {
                num_frame = it_per_id->feature_per_frame.size();

                if(it_per_id->endFrame() != frameCnt || num_frame < 2)
                    continue;
                
                g_id = it_per_id->feature_id;
                l_id = tracker.gl_id_index_map[g_id];

                if(l_id > 0)
                {
                    if(tracker.statusLeftRIght[(l_id-1)] == 0) continue;
                }
                else
                {
                    if(tracker.status_sift[(-l_id)] == 0) continue;
                }

                // 统计map中当前的跟踪点数量
                ++num_track_cur_bg;
                // 统计长3D-2D跟踪点的数量（长跟踪点的初始观测帧下一定有深度?）
                if(num_frame > 2 && it_per_id->estimated_depth > 0) 
                {
                    ++num_old_track;
                    id_old_track.insert(g_id);
                }
                
                // 优先选择所有那些在上一帧中具有立体匹配的 当前帧跟踪点
                if(it_per_id->feature_per_frame[(num_frame-2)].is_stereo)
                {
                    dep = it_per_id->feature_per_frame[(num_frame-2)].depth;
                    int index = (num_frame -1);

                    if(dep > 0)
                    {
                        start_frame = frameCnt-1;
                        Vector3d ptsInCam = ric[0] * (it_per_id->feature_per_frame[(num_frame-2)].point * dep) + tic[0];
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
                        ++num_select;
                        
                        // 远点可以是图像顶部1/3区域的点（其实这区域的点也不一定是远点），也可以是深度较大的点
                        // row_bloc = it_per_id->feature_per_frame[index].uv(1)/60;
                        // if(row_bloc < 3)
                        //     --num_track_far_pt;
                        // else if(dep >= 15)
                        //     --num_track_far_pt;
                    }
                    else
                    {
                        if(num_frame > 1 && it_per_id->estimated_depth > 0)
                            ++num_inlier_fea_PnP;
                    }
                }
                else
                {
                    if(num_frame > 1 && it_per_id->estimated_depth > 0)
                        ++num_inlier_fea_PnP;
                }
            }

            num_inlier_fea_PnP += num_select;

            // 如果点数不足，则再选择那些在更早的帧中有立体匹配的跟踪点(这其实就有点像LBA了，只是仅优化当前帧位姿)
            int num_th = Min_num_bg_track_with_dep_prev;
            // 如果当前帧与上一帧之间近乎静止，则不使用跟踪点在更早的帧的观测，因为这会使得更早之前的“运动“影响到当前帧位姿的估计？会吗？会的，因为需要用到点的全局3D坐标，这就需要用到先前帧的位姿，这其实隐含了先前帧之间的运动估计
            // 这里要判断最新2帧之间是否近乎静止，能否只使用comp_with_prev_esti来判断？这似乎是可以认为选择的选项，而不是客观的变量，应该用esitamtor中的no_initial_guess？
            if(frameCnt > 1 && !comp_with_prev_esti) num_th = 9;
            if(num_select < num_th)
            {
                for(list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
                {
                    // todo:是否要限制这部分点的数量？
                    // if(num_select >= 18) break;

                    g_id = it_per_id->feature_id;
                    if(selected_pt.find(g_id) != selected_pt.end()) continue;

                    if(it_per_id->endFrame() != frameCnt) continue;

                    num_frame = it_per_id->feature_per_frame.size();
                    if(num_frame <= 2) continue;
                    
                    int l_id = tracker.gl_id_index_map[g_id];

                    if(l_id > 0)
                    {
                        if(tracker.statusLeftRIght[(l_id-1)] == 0) continue;
                    }
                    else
                    {
                        if(tracker.status_sift[(-l_id)] == 0) continue;
                    }

                    // 从离当前时刻较早的帧开始遍历
                    for(int i = (num_frame-3); i >= 0; --i)
                    {
                        if(it_per_id->feature_per_frame[i].is_stereo)
                        {
                            dep = it_per_id->feature_per_frame[i].depth;
                            int index = (num_frame-1);

                            if(dep > 0)
                            {
                                start_frame = frameCnt - (num_frame - 1 - i);

                                Vector3d ptsInCam = ric[0] * (it_per_id->feature_per_frame[i].point * dep) + tic[0];

                                Vector3d ptsInWorld = Rs[start_frame] * ptsInCam + Ps[start_frame];

                                cv::Point3d point3d(ptsInWorld(0), ptsInWorld(1), ptsInWorld(2));
                                // 注意，这里给的2D点是当前帧归一化平面上的坐标
                                cv::Point2d point2d(it_per_id->feature_per_frame[index].point(0), it_per_id->feature_per_frame[index].point(1));
                                pts3D.push_back(point3d);
                                pts2D.push_back(point2d);

                                pt_id_vec.push_back(g_id);
                                fea_iters.push_back(it_per_id);

                                selected_pt.insert(g_id);
                                ++num_select;

                                // 远点可以是图像上1/3区域的点（其实这区域的点也不一定是远点），也可以是深度较大的点
                                // row_bloc = it_per_id->feature_per_frame[index].uv(1)/60;
                                // if(row_bloc < 3)
                                //     --num_track_far_pt;
                                // else if(dep >= 15)
                                //     --num_track_far_pt;

                                // todo:是否加入g_id_pt_not_in_PnP？即后续是否要用估计的运动检验这些点在前一帧和当前帧之间的3D-2D匹配的准确性？

                                break;
                            }
                        }
                    }
                }
            }

            cout << "Num of selected 3D-2D fea (witn stereo match in prev frames) of bg for PnP in map: " << pts3D.size() << endl;

            // 如果已有的3D-2D跟踪点太少，则使用上一帧有深度值而无立体匹配的跟踪点（假如允许基于估计的运动更新跟踪点在最新帧的深度，且信任并保留该深度值）
            if(Cal_cur_dep_by_motion && Trust_dep_from_motion)
            {
                bool not_enough_3D_2D = (num_select < 12);
                
                // 如果已有3D-2D点数较少，则减少远点的数量要求
                // if(not_enough_3D_2D) num_track_far_pt -= 3;
                
                for(list<FeaturePerId>::iterator it_per_id = feature.begin(); it_per_id != feature.end(); it_per_id++)
                {
                    g_id = it_per_id->feature_id;
                    if(selected_pt.find(g_id) != selected_pt.end()) continue;
                    
                    int num_frame = it_per_id->feature_per_frame.size();
                    if(num_frame < 2 || it_per_id->endFrame() != frameCnt) continue;
                    
                    l_id = tracker.gl_id_index_map[g_id];
                    
                    if(l_id > 0)
                    {
                        if(tracker.statusLeftRIght[(l_id-1)] == 0) continue;
                        dep = tracker.prev_FAST_dep[(l_id-1)];
                    }
                    else
                    {
                        if(tracker.status_sift[(-l_id)] == 0) continue;
                        dep = tracker.prev_sift_dep[(-l_id)];
                    }
                    
                    if(dep > 0)
                    {
                        // 如果之前有立体匹配的跟踪点太少，则使用上一帧有深度值而无立体匹配的跟踪点
                        // if((not_enough_3D_2D && num_select <= 16) || num_track_far_pt > 0)
                        if(not_enough_3D_2D && num_select <= 16)
                        {
                            // if(num_track_far_pt > 0)
                            // {
                            //     row_bloc = it_per_id->feature_per_frame[(num_frame-2)].uv(1)/60;
                            //     if(row_bloc >= 2 && dep < 15.0)
                            //     {
                            //         // 如果只是为了添加远点
                            //         if(!not_enough_3D_2D) 
                            //         {
                            //             g_id_pt_not_in_PnP.push_back(g_id);
                            //             iters_fea_not_in_PnP.push_back(it_per_id);
                            //             continue;
                            //         }
                            //     }
                            // }

                            start_frame = frameCnt - 1;

                            Vector3d ptsInCam = ric[0] * (it_per_id->feature_per_frame[(num_frame-2)].point * dep) + tic[0];

                            Vector3d ptsInWorld = Rs[start_frame] * ptsInCam + Ps[start_frame];

                            cv::Point3d point3d(ptsInWorld(0), ptsInWorld(1), ptsInWorld(2));

                            Vector3d &un_pt = it_per_id->feature_per_frame[(num_frame-1)].point;
                            // 当前帧归一化平面上的坐标
                            cv::Point2d point2d(un_pt(0), un_pt(1));
                            pts3D.push_back(point3d);
                            pts2D.push_back(point2d); 
                            pt_id_vec.push_back(g_id);
                            fea_iters.push_back(it_per_id);
                            ++num_select;
                            
                            // if(num_track_far_pt > 0)
                            // {
                            //     if(row_bloc < 2 || dep >= 15.0) --num_track_far_pt;
                            // }
                        }
                        else
                        {
                            // 否则将这些点记录下来，完成PnP估计后再对其进行检验
                            g_id_pt_not_in_PnP.push_back(g_id);
                            iters_fea_not_in_PnP.push_back(it_per_id);
                        }
                    }
                }
            }
        }

        // 当前帧地图中剩余的纯2D-2D跟踪点数(只有2帧观测)
        int num_2d_2d_cur = num_track_cur_bg - num_inlier_fea_PnP;

        // 如果最终当前帧的3D-2D跟踪点还是不够最低数量，则放弃PnP
        int num_select = pts2D.size();
        if(num_select < 6)
        {
            cout << "Not enough 3D tracked feature for PnP of camera! Stereo camera will degenerate to mono case!！！" << endl;
            cout << "Num of 3D-2D track in cur frame: " << num_inlier_fea_PnP << endl;
            // 没必要停止，某一帧PnP失败后就使用运动预测值计算该帧的位姿
            // exit(-1);
            
            return false;
        }
        else
        {
            cout << "Num of selected tracked static fea for PnP in map: " << num_select << endl;
        }
        
        int num_fail = 0;
        // 这里是使用上一帧相机在当前世界坐标下的位姿来作为当前帧相机位姿的初始值。这里是否改变为使用恒速模型下的预测位姿？
        // 两帧图像过后应该可以使用恒速运动模型，或者VI初始化之后可以使用IMU推测，来获得当前帧位姿的初始值！（但是VI初始化之后此函数就不会再被调用了）
        // 如果求解不成功（跟踪点少于4个）呢？那就不用设置Rs和Ps了吗？没有给系统传递任何信息吗？如果失败了，Rs和Ps中仍然只是预积分呀？！
        // 对于双目IMU系统而言，此处应该要保证相机的位姿估计是成功的！！否则当前帧与后续的Rs和Ps就都不再是根据视觉计算出来的各帧IMU坐标系相对于假定世界坐标系的位姿！！
        // 而根据视觉计算出来的相对姿态是恨重要的，被认为是准确的，之后IMU的预积分值要与这些视觉的结果进行对齐以估计IMU的参数和状态！！
        if(solvePoseByPnP(frameCnt, pred_R, pred_P, pts2D, pts3D, pt_id_vec, initial_succ))
        {
            Matrix3d new_R_cam_motion = pred_R.transpose() * prev_cam_R;
            Vector3d new_P_cam_motion = pred_R.transpose() * (prev_cam_P - pred_P);

            Quaterniond delta_Q(new_R_cam_motion);
            double delta_angle = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);
            cout << "delta_angle from PnP: " << delta_angle << endl;

            bool succ_pnp = true;
            // 是否要对相机的运动估计结果进行人为对错判断？
            if(frameCnt > 1 || comp_with_prev_esti)
            {
                if(pred_norm_p >= 0.15 && new_P_cam_motion.norm() > (2.5*pred_norm_p))
                {
                    succ_pnp = false;
                    cout << "Wrong delta_p from PnP: " << new_P_cam_motion.norm() << endl;
                }
                else
                {
                    float alpha = 2.5;
                    if(pred_delta_ang < 0.10)
                        alpha = 6.0;
                    else if(pred_delta_ang < 0.15)
                        alpha = 4.5;
                    else if(pred_delta_ang < 0.25)
                        alpha = 3.0;
                    else if(pred_delta_ang < 0.35)
                        alpha = 3.5;
                    else if (pred_delta_ang < 0.6)
                        alpha = 3.0;
                    
                    if(delta_angle >= 5.5)
                    {
                        succ_pnp = false;
                    }
                    else if((delta_angle > alpha * pred_delta_ang) || (pred_delta_ang > 0.50 && delta_angle < 0.2 * pred_delta_ang))
                    // else if((pred_delta_ang > 0.08 && delta_angle > alpha * pred_delta_ang || (pred_delta_ang > 0.6 && delta_angle < 1.0/alpha * pred_delta_ang)))
                    {
                        succ_pnp = false;
                    }
                    
                    if(!succ_pnp)
                    {
                        cout << "pred delta_angle: " << pred_delta_ang << endl;
                        cout << "Wrong delta_angle from PnP!" << endl;
                    }
                }
            }
            else
            {
                if(delta_angle >= 4.0) 
                {
                    succ_pnp = false;
                    cout << "pred delta_angle: " << pred_delta_ang << endl;
                    cout << "Wrong delta_angle from PnP!" << endl;
                }
            }

            if(!succ_pnp)
            {
                // todo: 外点太多导致估计结果被认为是错的，是否要去除1/2的有深度的跟踪点，以便后续可以添加静态物体点？
                num_inlier_fea_PnP = num_inlier_fea_PnP * 1.0/2;
                return false;
            }
            
            vector<uchar> status(num_select, 1);
            float dist_1;
            int num_1 = num_select, num_2 = num_select;
            // 是否与上一次PnP的结果进行比较，如果此次估计的平均重投影误差大于先前的估计，则放弃此次估计
            if(comp_with_prev_esti)
            {
                // 这里不应该 计算和比较 参与PnP的内点的重投影误差总和，因为PnP的目的本来就是为了减小重投影总和！！
                // 应该 计算和比较 所有内点对估计的 R/t的极线约束距离 的平均！
                // 是否只使用参与PnP的点来进行
                dist_1 = cal_ave_epi_line_dist_pts(pred_R_cam_motion, pred_P_cam_motion, fea_iters, status);
            }

            if(!pt_id_vec.empty())
            {
                for(int k = 0; k < num_select; ++k)
                {
                    if(pt_id_vec[k] >= 0)
                    {
                        status[k] = 0;
                        --num_2;
                    }
                }
            }
            
            float dist_2 = cal_ave_epi_line_dist_pts(new_R_cam_motion, new_P_cam_motion, fea_iters, status);

            if(comp_with_prev_esti)
            {
                cout << "ave_dist_epi of prev esti: " << dist_1 << ", ave_dist_epi of cur esti: " << dist_2 << endl;

                // if(dist_2 > dist_1 || (dist_2 >= 0.97*dist_1 && num_2 <= (num_1 - 4)))
                // {
                //     cout << "No or small improvement in epipolar constraint of this PnP compared with prev one!" << endl;
                //     return false;
                // }
                // else
                // {
                //     cout << "Accept this PnP!" << endl;
                // }

                // 如果与内点数相比，此次估计的平均重投影误差减小，或者增加得很少，则接受此次估计
                // if(dist_2 <= dist_1 || (dist_2 < 1.05*dist_1 && num_2 >= (num_1 - 2)))
                // {
                //     cout << "Accept this PnP!" << endl;
                // }
                // else
                // {
                //     cout << "No or small improvement in epipolar constraint of this PnP compared with prev one!" << endl;
                //     return false;
                // }

                // 极限约束的平均误差要增加较少（小于前者的10%)或者减少，且外点数不超过3个
                // if(dist_1 > 0 && dist_2 <= 1.05 * dist_1 && num_2 > (num_1 - 3))
                if(dist_1 > 0 && dist_2 <= 1.05 * dist_1)
                {
                    cout << "Accept this PnP!" << endl;
                    ave_epi_dist = dist_2;
                }
                else
                {
                    cout << "No or small improvement in epipolar constraint of this PnP compared with pred motion or prev PnP!" << endl;
                    if(dist_1 > 0) ave_epi_dist = dist_1;
                    return false;
                }
            }
            else
            {
                ave_epi_dist = dist_2;
            }
            
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

            vector<Point2f> &cur_un_FAST = tracker.cur_un_FAST;
            vector<Point2f> &cur_un_sift = tracker.cur_un_sift;

            // trans to w_T_imu
            Rs[frameCnt] = pred_R * ric[0].transpose(); 
            Ps[frameCnt] = -Rs[frameCnt] * tic[0] + pred_P;
            
            // Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;

            Matrix3d motion_R = pred_R.transpose() * prev_cam_R;
            Vector3d motion_P = pred_R.transpose() * (prev_cam_P - pred_P);

            int id, id_pt, index;
            Vector3d pt_prev, pt_cur;
            float cur_z, max_depth;
            uchar cls_pt;
            int l_obj_id;

            int total_pts = pt_id_vec.size();
            // 更新参与PnP的特征点在当前帧的深度值，并去除外点。
            // 这只在使用cv::solvePnPRansac()时才会进行，如果使用cv::solvePnP()，则在此处无法排除外点！
            if(!pt_id_vec.empty())
            {
                // int num_inliers = 0;
                has_RANSAC = true;
                
                for(int i = 0; i < pt_id_vec.size(); ++i)
                {
                    id = pt_id_vec[i];
                    // id为负的是运动估计内点。内点是否保留取决于其更新后的深度估计是否可靠。
                    // 当前帧点的深度值更新不在这里进行，而是在三角化测量函数中
                    if(id < 0)
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

                            if(prev_un_Fea_map.find(id_pt) == prev_un_Fea_map.end())
                            {
                                cout << "Weired! Line 2330" << endl;
                                exit(-1);
                            }

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

                            // if (cur_z < max_depth && cur_z > mMinDepthPt)
                            if (cur_z < max_depth && cur_z >= 1.5)
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

                    --total_pts;
                    ++num_fail;

                    // 否则为外点
                    --num_inlier_fea_PnP;
                    --num_track_cur_bg;
                    if(id_old_track.find(id) != id_old_track.end())
                    {
                        --num_old_track;
                    }

                    // assert(gl_id_index_map.find(id) != gl_id_index_map.end() && "There must be something wrong with var gl_id_index_map!");
                    index = gl_id_index_map[id];
                    // index小于等于0的是sift，则保留当前帧该点为新检测点
                    if(index <= 0)
                    {
                        index = -1*index;
                        // todo:这里的l_obj_id是否可能已经被物体运动估计线程所修改？因此最好是使用seg_map来查看该点的检测cls！
                        // （此处的执行应该是在物体运动估计之前）
                        l_obj_id = obj_cls_id_sift[index].second;
                        // 是否要将当前帧该点转为新点
                        // 如果是静态物体点，且有右图像匹配，则保留

                        // 对于物体的sift点，只要该点在当前帧中有深度值(不管是否来自depth_map），则可以保留该点。首先保留该点可以使得下一帧容易跟踪该物体，其次（尤其是VI初始化之前）如果该点有深度值，则可以为下一帧相机的PnP提供点
                        // 对于物体点，status == 1代表有立体匹配的跟踪点或新点，== 2代表深度值来自depth_map的跟踪点, ==3代表深度值来自depth_map的新点
                        // if(status_sift[index] == 1 && l_obj_id > 0)
                        if(l_obj_id > 0)
                        {
                            if(status_sift[index] > 0)
                                reserve_new_sift.push_back(id); 
                        }
                        else
                        {
                            // 如果要在当前帧保留背景新点
                            if(!tracker.add_new_fea_in_next_frame)
                            {
                                // 当前帧有立体匹配的背景区域跟踪点
                                if(status_sift[index] == 1)
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
                    // FAST点是否直接删除该点的跟踪和地图中的最新记录？因为FAST点只有被检测时比较有区分度，给跟踪的点不一定具有区分度
                    // 只保留其中的物体跟踪FAST点作为新点
                    else
                    {
                        index -= 1;
                        l_obj_id = obj_cls_id_FAST[index].second;
                        if(l_obj_id > 0)
                        {
                            if(statusLeftRIght[index] > 0)
                                reserve_new_FAST.push_back(id);
                        }
                        else
                        {
                            if(!tracker.add_new_fea_in_next_frame)
                            {
                                // 当前帧有立体匹配的背景区域跟踪点
                                if(statusLeftRIght[index] == 1)
                                {
                                    reserve_new_FAST.push_back(id); 
                                }
                                else
                                    statusLeftRIght[index] = 0;
                            }
                            else
                                statusLeftRIght[index] = 0;
                        }
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

                if(total_pts > 0) ave_epi_dist = ave_epi_dist * total_pts;
            }
            else
            {
                // 由于Ransac不成功，则意味着内点比例不高，这里应该要适当减小内点的数量，以便后续可以添加静态物体跟踪点
                // todo:减小的规模应该按照Ransac设置的置信度来大致推断？
                num_inlier_fea_PnP -= (pts3D.size() * 1.0/3);
            }

            // 如果还有未参与PnP估计的3D-2D跟踪点，则用位姿估计结果校验它们
            if(!g_id_pt_not_in_PnP.empty())
            {
                bool use_ransac = (!pt_id_vec.empty());

                float thres_err = 3.5;
                if(use_ransac) thres_err = 2.5;

                Vector3d pred_cur_p, prev_pix(0,0,1), cur_pix(0,0,1);    
                bool valid_p;
                int num = g_id_pt_not_in_PnP.size();
                float dep_prev;

                Matrix3d t_up;
                t_up << 0.0, -new_P_cam_motion(2), new_P_cam_motion(1), new_P_cam_motion(2), 0.0, -new_P_cam_motion(0), -new_P_cam_motion(1), new_P_cam_motion(0), 0.0;
                // 本质矩阵到关键矩阵
                Matrix3d Mat_F = K_trans_inv * t_up * new_R_cam_motion * K_inv;

                for(int i = 0; i < num; ++i)
                {
                    int g_pt_id = g_id_pt_not_in_PnP[i];
                    index = gl_id_index_map[g_pt_id];

                    valid_p = false;
                    
                    if(index > 0)
                    {
                        index -= 1;
                        dep_prev = prev_dep_FAST[index];
                        pt_prev(2) = dep_prev;
                        
                        if(prev_un_Fea_map.find(g_pt_id) == prev_un_Fea_map.end())
                        {
                            cout << "Weired! Line 2534" << endl;
                            exit(-1);
                        }
                        
                        pt_prev(0) = prev_un_Fea_map[g_pt_id](0) * dep_prev;
                        pt_prev(1) = prev_un_Fea_map[g_pt_id](1) * dep_prev;

                        pred_cur_p = (motion_R * pt_prev + motion_P);
                        float pred_z = pred_cur_p(2);

                        // 是否要区分背景或物体点？
                        if(pred_z >= 1.5 && pred_z < mThDepthBg)
                        {
                            pred_cur_p = pred_cur_p/pred_z;
                            Point2f &pt_cur = cur_un_FAST[index];
                            float err = (pred_cur_p(0) - pt_cur.x)*(pred_cur_p(0) - pt_cur.x) + (pred_cur_p(1) - pt_cur.y)*(pred_cur_p(1) - pt_cur.y);
                            // 这里是否需要根据点的深度大小来调整 重投影误差的阈值？越近的点，其flow越大，是否应该增大阈值？
                            if(pred_z < 12) thres_err += 0.5;
                            if(err < thres_err*thres_err/FOCAL_LENGTH_X/FOCAL_LENGTH_X)
                            {
                                valid_p = true;
                            }
                        }
                    }
                    else
                    {
                        index = -1 * index;
                        dep_prev = prev_dep_sift[index];
                        pt_prev(2) = dep_prev;
                        pt_prev(0) = prev_un_Fea_map[g_pt_id](0) * dep_prev;
                        pt_prev(1) = prev_un_Fea_map[g_pt_id](1) * dep_prev;
                        
                        pred_cur_p = (motion_R * pt_prev + motion_P);
                        float pred_z = pred_cur_p(2);

                        if(pred_z >= 1.5 && pred_z < mThDepthBg)
                        {
                            pred_cur_p = pred_cur_p/pred_z;
                            Point2f &pt_cur = cur_un_sift[index];
                            float err = (pred_cur_p(0) - pt_cur.x)*(pred_cur_p(0) - pt_cur.x) + (pred_cur_p(1) - pt_cur.y)*(pred_cur_p(1) - pt_cur.y);
                            
                            // 这里是否需要根据点的深度大小来调整 重投影误差的阈值？越近的点，其flow越大，是否应该增大阈值？
                            if(pred_z < 12) thres_err += 0.5;
                            if(err < thres_err*thres_err/FOCAL_LENGTH_X/FOCAL_LENGTH_X)
                            {
                                valid_p = true;
                            }
                        }
                    }
                    
                    if(!valid_p)
                    {
                        --num_inlier_fea_PnP;
                        -num_track_cur_bg;
                        if(id_old_track.find(g_pt_id) != id_old_track.end())
                        {
                            --num_old_track;
                        }

                        ++num_fail;
                        index = gl_id_index_map[g_pt_id];
                        uchar cls;
                        if (index <= 0)
                        {
                            index = -1*index;
                            // todo:其实这里的l_obj_id可能已经被物体运动估计线程所修改，因此最好是使用seg_map来查看该点的检测cls！!
                            l_obj_id = obj_cls_id_sift[index].second;
                            // cls =  obj_cls_id_sift[index].first;

                            // 如果在当前帧是物体的sift点，则保留
                            if(l_obj_id > 0)
                            {
                                if(status_sift[index] > 0)
                                    reserve_new_sift.push_back(g_pt_id); 
                            }
                            else
                            {
                                // 如果要在当前帧保留背景新点
                                if(!tracker.add_new_fea_in_next_frame)
                                {
                                    // 当前帧有立体匹配的背景区域跟踪点
                                    if(status_sift[index] == 1)
                                    {
                                        reserve_new_sift.push_back(g_pt_id); 
                                    }
                                    else
                                        status_sift[index] = 0;
                                }
                                else
                                    status_sift[index] = 0;
                            }   
                        }
                        else
                        {
                            index -= 1;
                            l_obj_id = obj_cls_id_FAST[index].second;
                            if(l_obj_id > 0)
                            {
                                if(statusLeftRIght[index] > 0)
                                    reserve_new_FAST.push_back(id);
                            }
                            else
                            {
                                if(!tracker.add_new_fea_in_next_frame)
                                {
                                    // 当前帧有立体匹配的背景区域跟踪点
                                    if(statusLeftRIght[index] == 1)
                                    {
                                        reserve_new_FAST.push_back(id); 
                                    }
                                    else
                                        statusLeftRIght[index] = 0;
                                }
                                else
                                    statusLeftRIght[index] = 0;
                            }
                        }
                        
                        // cout << "cls: " << (int)cls << ", g_pt_id:" << g_pt_id << ", l_id: " << gl_id_index_map[g_pt_id] << ", dep_prev: " << dep_prev << endl;

                        // 删除该点在地图中的当前帧观测。另存为新点的物体点在当前帧不会加入地图
                        iters_fea_not_in_PnP[i]->feature_per_frame.pop_back();

                        if(iters_fea_not_in_PnP[i]->feature_per_frame.size() < TH_NUM_FRAME_FOR_LBA)
                        {
                            feature.erase(iters_fea_not_in_PnP[i]);
                        }
                    }
                    else
                    {
                        float p_x, p_y;
                        if(index > 0)
                        {
                            Point2f &pt1 = tracker.prev_FAST[(index-1)];
                            prev_pix(0) = pt1.x;
                            prev_pix(1) = pt1.y;

                            Point2f &pt2 = tracker.cur_FAST[(index-1)];
                            cur_pix(0) = pt2.x;
                            cur_pix(1) = pt2.y;
                        }
                        else
                        {
                            Point2f &pt1 = tracker.prev_sift[(-index)];
                            prev_pix(0) = pt1.x;
                            prev_pix(1) = pt1.y;

                            Point2f &pt2 = tracker.cur_sift[(-index)];
                            cur_pix(0) = pt2.x;
                            cur_pix(1) = pt2.y;
                        }

                        float dist = cal_ave_epi_line_dist_pt(Mat_F, prev_pix, cur_pix);

                        if(dist >= 0) 
                        {
                            ave_epi_dist += dist;
                            ++total_pts;
                        }
                    }
                }
            }

            cout << "Num of failed fea tracking after PnP: " << num_fail << endl;
            if(total_pts > 0) ave_epi_dist = ave_epi_dist*1.0/total_pts;
        }
        // 双目+IMU不允许在初始化成功之前就视觉跟踪丢失，否则就需要重新启动！
        // 上面使用的PnP方式使得不会出现失败的情况
        else if(USE_IMU && !initial_succ)
        {
            // assert(false && "solvePoseByPnP for Stereo+IMU system failed! Should restart the system!");
            cout << "solvePoseByPnP for Stereo+IMU system failed! Should restart the system!" << endl;
            exit(-1);
            
            has_RANSAC = false;
        }
    }
    return has_RANSAC;
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
int FeatureManager::triangulate(int frameCnt, double Headers[], Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[], FeatureTracker &tracker, int &num_track_3D_2D, bool good_est_RT, bool marg_old, bool before_PnP, 
                                bool try_tria, bool try_update_dep, bool LBA_succ, const Matrix3d &R_from_E, Vector3d *norm_t, double scale, float pred_dist_t, const set<int> &reserve_bg_track_pt_id)
{
    if(frameCnt == 0) return 0;

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

    vector<int> &id_FAST_no_depth = tracker.id_FAST_no_depth;
    vector<int> &id_sift_no_depth = tracker.id_sift_no_depth;
    
    // float ave_dep_bg_prev_frame = tracker.ave_dep_bg_prev_frame;
    double &ave_dep_bg_cur_frame = tracker.ave_dep_bg_cur_frame;
    int &num_bg_with_dep = tracker.num_bg_with_dep;

    float ave_dep_bg_cur_frame_;

    if(use_motion_to_pred_fea_dep)
    {
        if(num_bg_with_dep >= 6) 
        {
            ave_dep_bg_cur_frame_ = ave_dep_bg_cur_frame/num_bg_with_dep;
            if (ave_dep_bg_cur_frame_ < mMinDepthPt || ave_dep_bg_cur_frame_ > mThDepthBg) ave_dep_bg_cur_frame_ = INIT_DEPTH;
        }
        else
            ave_dep_bg_cur_frame_ = INIT_DEPTH;
    }

    int num_frame, index, f_start;
    Eigen::Matrix3d R_start, R_cur, R_motion, R_motion_cur, Mat_F;
    Eigen::Vector3d P_start, P_cur, P_motion, P_motion_cur, pt_start, pt_cur;
    double prev_dep, cur_depth;
    bool good_dep;
    int num_tri_fail = 0;
    
    vector<Vector3d> prev_3d_pts;
    vector<Vector2d> cur_2d_un_pts;

    // 保存需要被删除的地图点
    vector<int> pt_need_to_erase;
    // 如果左右相机之间的外参需要被在线估计，则这里暂时不删除那些首帧立体匹配三角化失败的地图点？
    bool need_estimate_ex_param_stereo = (ESTIMATE_EXTRINSIC != 0);

    // 当前系统是否会进行LBA。第一个滑窗满时才开始进行LBA。
    // 进行LBA意味这会改变帧的位姿估计；另外，如果LBA中还需要优化双目相机的外参，其值也会改变。
    bool has_LBA = (STEREO && (USE_IMU || Use_LBA_for_puer_V) && frameCnt == WINDOW_SIZE);
    
    // 保存图像下半部分 每个横向bloc中的点 深度和id，不同行的bloc中的深度值会有较大差别
    vector<pair<float,int>> row_1, row_2, row_3;
    // 保存图像上半部分中的点，其深度值一般比较大
    map<double,int,less<double>> temp_dep_id;
    
    // 保存所有有深度的3D-2D匹配点
    vector<Vector3d> temp_3d_pt;
    vector<Vector2d> temp_2d_pt;
    vector<int> temp_pt_in_col;
    int num_temp = 0;

    float delta_height = 0.0;
    int num_delta_h = 0;

    bool has_p = false;
    
    Matrix3d cam_R_prev, cam_R_cur;
    Vector3d cam_P_prev, cam_P_cur;
    // 如果R_T的估计结果值得信赖（这里指的是要么有PnP，要么有LBA），则可以对2D-2D跟踪点进行三角化测量以恢复其为3D-2D点
    if(!before_PnP && good_est_RT)
    {
        cam_R_prev = Rs[(frameCnt-1)] * ric[0]; 
        cam_P_prev = Rs[(frameCnt-1)] * tic[0] + Ps[(frameCnt-1)];
        cam_R_cur = Rs[frameCnt] * ric[0]; 
        cam_P_cur = Rs[frameCnt] * tic[0] + Ps[frameCnt];

        R_motion_cur = cam_R_cur.transpose() * cam_R_prev;
        P_motion_cur = cam_R_cur.transpose() * (cam_P_prev - cam_P_cur);

        if(P_motion_cur.norm() >= 0.1)
        {
            has_p = true;

            Matrix3d t_up;
            t_up << 0.0, -P_motion_cur(2), P_motion_cur(1), P_motion_cur(2), 0.0, -P_motion_cur(0), -P_motion_cur(1), P_motion_cur(0), 0.0;
            // 本质矩阵到关键矩阵
            Mat_F = K_trans_inv * t_up * R_motion_cur * K_inv;
        }
    }

    float dep_;
    bool try_tria_meas = false;
    int invalid_track = 0, num_tria_succ = 0, num_tria_succ_far_pt = 0;
    
    // 遍历滑窗内所有地图点
    for(auto &it_per_id: feature)
    {
        good_dep = false;

        num_frame = it_per_id.feature_per_frame.size();

        int gl_id = it_per_id.feature_id;

        // 该点是否为当前帧的跟踪点
        bool cond_2 = (it_per_id.endFrame() != frameCnt);
        // 该点在首帧下是否为立体匹配
        bool cond_3 = it_per_id.feature_per_frame[0].is_stereo;

        try_tria_meas = false;

        if(cond_2)
        {
            dep_ = it_per_id.estimated_depth;
        }
        else
        {
            // 如果是当前帧的跟踪点，则优先考察上一帧的该点深度值
            if(num_frame > 1)
            {
                // int l_id = gl_id_index_map[gl_id];
                // if(l_id > 0)
                // {
                //     dep_ = prev_FAST_dep[(l_id-1)];
                // }
                // else
                // {
                //     dep_ = prev_sift_dep[(-l_id)];
                // }

                // 如果大于0，则该点在上一帧有立体匹配，且已经完成了立体三角化
                dep_ = it_per_id.feature_per_frame[(num_frame-2)].depth;
            }
            else
            {
                // 这种情况是该点为当前图像中新检测的背景点？本系统在地图中实际上不保留这样的点
                dep_ = -1.0;
            }
        }
        
        // 该点在之前的帧中是否已经有深度值
        bool cond_1 = (dep_ > 0);

        bool has_stereo_prev = false;
        // 判断当前帧的跟踪点在上一帧中是否有立体匹配，应该直接查看该点在地图中的记录，因为有些点在before_PnP的立体三角化失败后就被取消了右观测记录（认为不准确）
        // has_stereo_prev = (prevRightFeaMap.find(gl_id) != prevRightFeaMap.end());
        if(num_frame > 1 && !cond_2) has_stereo_prev = it_per_id.feature_per_frame[(num_frame-2)].is_stereo;
        
        // 如果该点的首帧下深度已知
        // 且 该点不是当前帧跟踪点（则不需要参与PnP前的立体三角化 或者 PnP之后的两帧三角化以及更新当前帧的深度），
        // 或者是当前帧的跟踪点，但该点 在上一帧没有立体匹配（则不需要在PnP前进行立体三角化），且不需要在PnP后进行两帧三角化（因为之前帧已经有深度值）
        if(cond_1 && (cond_2 || (!before_PnP || !has_stereo_prev)))
        {
            // if(!before_PnP && it_per_id.endFrame() == frameCnt && num_frame >= TH_NUM_FRAME_FOR_LBA) ++num_LBA_fea_cur_frame;
            good_dep = true;
        }
        else if(before_PnP)
        {
            // 该特征点被观测首帧下的左右图像之间
            // 如果该点在当前帧被跟踪到
            if(STEREO && !cond_2)
            {
                // 如果该点观测首帧(不是上一帧）下有立体匹配，但是还没有成功立体三角化，则再次尝试（只在需要在线估计双目相机外参时才进行）
                // 前提是立体相机外参会参与LBA并大幅改变其值，且该点的连续跟踪帧数大于阈值
                // if(!cond_1 && cond_3 && num_frame > 2 && has_LBA)
                // {
                //     Eigen::Vector2d point0, point1;
                //     Eigen::Vector3d point3d, localPoint;
                //     double depth;
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
                    Eigen::Vector2d point0, point1;
                    Eigen::Vector3d point3d, localPoint;
                    double depth = it_per_id.feature_per_frame[num_frame-2].depth;

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

                            // FeaturePerFrame类中的depth只用于记录该帧下来自于立体匹配的深度，如果没有立体深度，则该值为-1.0
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
                            // 如果立体匹配失败，则保留该2D-2D点，等待后续完成位姿估计后再进行两帧间的三角化测量（认为可能是立体匹配不够准确，放弃该立体匹配）

                            // cout << "failed triangulate a point with stereo match in prev frame!" << endl;
                            ++num_tri_fail;
                            good_dep = false;
                            // 初始值是负的吗？不是，等于5，为什么？那样三角化没有成功的点也赋予5.0的深度值？那之后不久不再对其进行三角化了？
                            // 是的，VINS是寄希望于该点能连续被观测4帧，然后使用多帧三角化 以及 参与LBA 来优化该点的深度....
                            // 这里改变这个做法，特征点的观测首帧深度值只能通过三角化来完成（无论是两帧还是多帧（4帧）），如果三角化始终未成功，则不将其加入LBA！
                            if(num_frame == 2) 
                            {
                                // it_per_id.estimated_depth = INIT_DEPTH;
                                it_per_id.estimated_depth = -1.0;
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
            // 合并地图点这个操作对于VINS系列比较困难，因为地图点的观测虽然是特征点，但并不像ORB那样有描述子（这是考虑了该点周围的信息），因此无法较好地确定两个距离很近的地图点的观测是否较一致（这是合并这两个点的必要条件）！
            // 这里是否可以修改为该点首观测帧时没有右图像观测，且当前帧为该点的第2个观测帧（如果此次三角化失败，则必须等到该点有足够多的帧观测后，再尝试用多帧观测对该点进行三角化）？
            if(!good_dep && num_frame > 1 && !cond_2)
            {
                // 尝试对当前帧的2D-2D点进行三角化测量
                // 这里以上一帧的dep值来判断其是否为2D-2D点，而不是根据点的跟踪数（因为可能允许某些旧跟踪点在上一帧没有深度值，例如要求所有PnP之前所有上一帧的点的深度值必须来自立体匹配？）
                // if(num_frame == 2 && Use_tria_for_2d2d && try_tria)
                if(dep_ <= 0 && Use_tria_for_2d2d && try_tria)
                {
                    Point2f prev_2d_pt, cur_2d_pt;
                    float pt_un_y;
                    // 这里引用其实挺危险，因为可能不注意就修改了原变量！因此最好加上const！
                    const Vector2d &pt1 = it_per_id.feature_per_frame[(num_frame-2)].uv;
                    prev_2d_pt.x = pt1(0);
                    prev_2d_pt.y = pt1(1);

                    bool pass_F_check = true;

                    // 如果R_T估计结果可靠，且有位移，则首先用估计的R和t通过极线约束来筛选异常的2D-2D匹配点
                    if(good_est_RT && has_p)
                    {
                        Vector2d &pt2 = it_per_id.feature_per_frame[(num_frame-1)].uv;
                        cur_2d_pt.x = pt2(0);
                        cur_2d_pt.y = pt2(1);

                        // 阈值应该设置为多少？
                        int succ_check = check_flow_with_F(Mat_F, prev_2d_pt, cur_2d_pt, 3.5);

                        if(succ_check > 0)
                        {
                            // 如果系统允许进行两帧间三角化
                            // if(Use_tria_for_2d2d && try_tria)
                                try_tria_meas = true;
                        }
                        else
                        {
                            ++invalid_track;
                            pass_F_check = false;
                        }
                    }
                    // else if(good_est_RT && has_p && Use_tria_for_2d2d && try_tria)
                    // {
                    //     // 运动很小时是否也可以进行两帧三角化测量？
                    //     // 三角测量的原理是： s1*x1^*x1 = 0 = s2*x1^*R*x2 + x1^*t，当相机运动很小，即R接近于单位矩阵，而t接近于0时，x1^*R*x2接近于0，因此s2的估计是极其数值不稳定的！
                    //     try_tria_meas = true;
                    // }
                    else
                    {
                        // 如果系统不允许两帧间三角化测量，或者在当前帧认为R_T估计不够好，则放弃为2D-2D进行三角化
                        try_tria_meas = false;
                    }

                    bool valid_pt = true;
                    double depth;

                    // 系统是否允许进行三角化测量，以及当前帧能否进行三角化
                    if(try_tria_meas)
                    {
                        // 这里还包括那些长跟踪的2D-2D点，因此不能用start_frame
                        // int imu_i = it_per_id.start_frame;
                        int imu_i = frameCnt - 1;
                        
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
                        // 最新2帧的归一化平面的点坐标
                        point0 = it_per_id.feature_per_frame[(num_frame-2)].point.head(2);
                        pt_un_y = point0(1);
                        point1 = it_per_id.feature_per_frame[(num_frame-1)].point.head(2);
                        triangulatePoint(leftPose, rightPose, point0, point1, point3d);
                        Eigen::Vector3d localPoint;
                        localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
                        depth = localPoint.z();
                        
                        // if (depth > mMinDepthPt && depth < mThDepthBg)
                        if(depth >= 1.5 && depth < mThDepthBg)
                        // if(depth >= 1.5 && depth < 26)
                        {
                            // 远点限制数量，还是不信任其立体匹配？
                            // if(pt1(1)/60.0 < 3 && depth > 21.0 && num_tria_succ_far_pt >= 4)
                            if(depth > 20.0 && num_tria_succ_far_pt >= 5)
                            {
                                valid_pt = false;
                            }
                            else
                                valid_pt = true;
                        }
                        else
                        {
                            // cout << "failed triangulate a tracking fea with two frames!" << endl;
                            valid_pt = false;
                        }
                    }
                    else
                    {
                        // 对于长期跟踪点应该要保存？
                        valid_pt = false;
                    }

                    index = gl_id_index_map[gl_id];
                    bool valid_dep = false;
                    
                    // 三角化成功的跟踪点
                    if(valid_pt)
                    {
                        // 可以尝试根据三角化得到的深度值，为该点在上一帧寻找立体匹配？如果没有找到有效的立体匹配，则认为该深度值不可靠
                        // todo: 是否只对仅2帧的新跟踪点在上一帧（即其初始观测帧）寻找立体匹配？
                        bool check_by_find_stereo_match = true;
                        // if(num_frame == 2 && check_by_find_stereo_match)
                        if(check_by_find_stereo_match)
                        {
                            float search_range = 5;
                            float Th_NCC = 0.985;
                            Point2f prev_r_pred = prev_2d_pt;
                            prev_r_pred.x -= mbf/depth;
                            // 如果是下1/2图像区域，则认为左右图像中点局部区域的差别会较大，放松NCC要求？
                            // 实际上应该是地面上的点才需要放宽要求，如果是近处的非地面点，其左右图像中的成像区别应该不大！
                            // 先统一降低该区域点的NCC要求，后续再根据其是否为地面点调整其NCC阈值
                            if(prev_2d_pt.y/60 > 3)
                            {
                                Th_NCC = 0.97;
                                search_range = 10;
                            }
                            
                            float val_NCC = tracker.find_stereo_match_by_best_NCC(prev_2d_pt, prev_r_pred, Th_NCC, search_range);

                            if(val_NCC > 0)
                            {
                                float disp_x = prev_2d_pt.x - prev_r_pred.x;
                                if(disp_x <= 0)
                                {
                                    // cout << "Weired! Line 3296" << endl;
                                    // exit(-1);
                                    depth = -1.0;
                                }
                                else
                                {
                                    depth = mbf/disp_x;

                                    // 如果该点是远离地面的点，则要提高其NCC阈值
                                    if(prev_2d_pt.y/60 > 4)
                                    {
                                        if(pt_un_y * depth <= (Cam_H - 0.35))
                                        {
                                            if(val_NCC < 0.98)
                                            {
                                                depth = -1.0;
                                            }
                                        }
                                    }
                                }
                                
                                if(depth >= 1.5 && depth < mThDepthBg)
                                {
                                    // 基于该深度值进行重投影误差检查！
                                    Vector3d pt_p = it_per_id.feature_per_frame[(num_frame-2)].point;
                                    pt_p = pt_p * depth;
                                    const Vector3d &pt_c = it_per_id.feature_per_frame[(num_frame-1)].point;

                                    Vector3d re_pt_c = R_motion_cur * pt_p + P_motion_cur;

                                    if(re_pt_c(2) <= 0)
                                    {
                                        valid_dep = false;
                                    }
                                    else
                                    {
                                        re_pt_c = re_pt_c/re_pt_c(2);
                                        float Th_err = 3.0;
                                        if(pt_p(1)/60.0 > 4) Th_err = 4.0;
                                        
                                        float re_err = (re_pt_c(0) - pt_c(0))*(re_pt_c(0) - pt_c(0)) + (re_pt_c(1) - pt_c(1))*(re_pt_c(1) - pt_c(1));
                                        
                                        if(re_err > Th_err*Th_err/FOCAL_LENGTH_X/FOCAL_LENGTH_X)
                                        {
                                            valid_dep = false;
                                            // int row_bloc = prev_2d_pt.y/60;
                                            // cout << "Got an invalid stereo match in prev image! row_bloc: " << row_bloc << ", NCC: " << val_NCC << ", dep: " << depth << endl;
                                        }
                                        else
                                        {
                                            if(depth > 20 && num_tria_succ_far_pt >= 5)
                                            {
                                                valid_dep = false;
                                            }
                                            else
                                            {
                                                // int row_bloc = prev_2d_pt.y/60;
                                                // cout << "Got a valid stereo match in prev image! row_bloc: " << row_bloc << ", NCC: " << val_NCC << ", dep: " << depth << endl;

                                                valid_dep = true;
                                                dep_ = depth;
                                                Point2f prev_un_r;
                                                tracker.undistortedPts(prev_r_pred, prev_un_r, tracker.m_camera[1]);
                                                Vector8d prev_r;
                                                prev_r << prev_un_r.x, prev_un_r.y, 1.0, prev_r_pred.x, prev_r_pred.y, 0.0, 0.0, 0;

                                                // 如果是长跟踪点，则查看右点是否有vel
                                                if(USE_IMU && ESTIMATE_TD)
                                                {
                                                    if(num_frame > 2)
                                                    {
                                                        if(it_per_id.feature_per_frame[(num_frame-3)].is_stereo)
                                                        {
                                                            Vector3d &p_r = it_per_id.feature_per_frame[(num_frame-3)].pointRight;
                                                            int end_f = frameCnt - 1;
                                                            double dt = Headers[(end_f)] - Headers[(end_f-1)];
                                                            prev_r(5) = (prev_un_r.x - p_r(0))/dt;
                                                            prev_r(6) = (prev_un_r.y - p_r(1))/dt;
                                                        }
                                                    }
                                                }

                                                it_per_id.feature_per_frame[(num_frame-2)].rightObservation(prev_r);

                                                if(USE_IMU && ESTIMATE_TD)
                                                {
                                                    if(it_per_id.feature_per_frame.back().is_stereo)
                                                    {
                                                        Vector3d &p_r = it_per_id.feature_per_frame.back().pointRight;
                                                        // 地图点如果有上一帧和当前帧的观测，则其总是实际上连续的2帧，因为上个滑窗marg次新帧只会影响上一帧与之前帧的dt
                                                        // double cur_dt = tracker.cur_dt;
                                                        int end_f = frameCnt;
                                                        double cur_dt = Headers[(end_f)] - Headers[(end_f-1)];
                                                        float vx = (p_r(0) - prev_un_r.x)/cur_dt;
                                                        float vy = (p_r(1) - prev_un_r.y)/cur_dt;
                                                        
                                                        it_per_id.feature_per_frame.back().velocityRight = Vector2d(vx,vy);
                                                    }
                                                }
                                                
                                                // 这里是否要在调用立体三角化函数？
                                                it_per_id.feature_per_frame[(num_frame-2)].depth = depth;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    valid_dep = false;
                                }
                            }
                            else
                            {
                                // if(prev_2d_pt.y/60 > 4)
                                // {
                                //     cout << "Got a stereo match with invalid NCC in lower 1/3 image!" << endl;
                                // }
                            }
                        }
                        else
                        {
                            valid_dep = true;
                            dep_ = depth;
                        }
                    }
                    
                    if(valid_dep)
                    {
                        // 点的每一帧观测中的此深度值，只能来该帧的三角化测量
                        if(num_frame == 2) it_per_id.estimated_depth = depth;

                        good_dep = true;

                        ++num_tria_succ;
                        // 新的3D-2D点，不包含那些长跟踪点（因为其在首帧必定有深度，不论其在上一帧是否有立体匹配深度，都已经将其计入当前帧的3D-2D点）
                        if(num_frame == 2) ++num_track_3D_2D;
                        if(depth > 20) ++num_tria_succ_far_pt;

                        if(index > 0)
                        {
                            prev_FAST_dep[(index-1)] = depth;
                        }
                        else
                        {
                            prev_sift_dep[(-index)] = depth;
                        }
                        // cout << "Succeed triangulate a point with 2 frame measurement!" << endl;
                        // cout << "depth of point: " << depth << endl;
                    }
                    else
                    {
                        ++num_tri_fail;

                        // 该2D-2D如果尝试了三角化，但深度值无效，或者没有尝试但仅有2帧观测（其首帧没有深度值，则后续无法再参与LBA），则放弃该点后续的跟踪
                        // 无论有没有尝试三角化，对于观测帧数大于2的点都不应该删除，它们在首帧下都是有深度值的，这些点可以参与LBA？？
                        // 三角化失败的原因可能是因为该点比较大（但没超过最大阈值）或者远点数达到阈值，不一定是因为该点的2d-2d跟踪很不准确？所以不应该以此放弃长跟踪点？
                        // if(try_tria_meas || num_frame == 2)
                        if(num_frame == 2 || !pass_F_check)
                        {
                            // 不再继续跟踪该点
                            if(index > 0)
                            {
                                status_FAST[(index-1)] = 0;
                            }
                            else
                            {
                                status_sift[(-1*index)] = 0;
                            }

                            // 至于是否从地图中删除该点，取决于其剩余的有效观测帧数是否足够进行之后窗口的LBA
                            if(num_frame == 2)
                            {
                                // 仅有2帧的跟踪点直接删除
                                if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                                    pt_need_to_erase.push_back(gl_id);
                            }
                            else
                            {
                                // 对于旧跟踪点，除了要去除当前帧的观测之外，还要考虑当前滑窗是否会marg掉某一帧，且该帧是否属于该点的观测帧之一？
                                // 只有确保剩下的帧数大于参与LBA的最小帧数要求，才保留该地图点
                                num_frame -= 1;

                                // 由于最后还是只使用PnP的结果来进行三角化，因此该点在三角化失败后，如果去除当前帧观测后仍有足够帧数参与LBA，则保留它
                                // todo:是否要在第二（或第N）次PnP之后再次选择三角化2d-2d点？
                                if(num_frame < TH_NUM_FRAME_FOR_LBA)
                                {
                                    if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                                        pt_need_to_erase.push_back(gl_id);
                                }
                                else
                                {
                                    it_per_id.feature_per_frame.pop_back();
                                }
                            }
                            continue;
                        }
                        else
                        {
                            // 这是没有进行三角化测量且有多帧跟踪的点。如果该点在初始帧有深度值，则暂时保留，后续需要进行重投影误差检测
                            if(it_per_id.estimated_depth <= 0)
                            {
                                if(index > 0)
                                {
                                    status_FAST[(index-1)] = 0;
                                }
                                else
                                {
                                    status_sift[(-1*index)] = 0;
                                }
                                
                                if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                                    pt_need_to_erase.push_back(gl_id);
                                
                                continue;
                            }
                        }
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
        }

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
                    if(prevLeftFeaMap.find(gl_id) == prevLeftFeaMap.end())
                    {
                        cout << "Weired!" << endl;
                        exit(-1);
                    }
                    
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

                    // 需要上一帧和当前帧该点均具有立体匹配
                    if(it_per_id.feature_per_frame[(num_frame-2)].is_stereo && it_per_id.feature_per_frame[(num_frame-1)].is_stereo)
                    {
                        float cur_x_l = it_per_id.feature_per_frame[(num_frame-1)].point(0);
                        float cur_x_r = it_per_id.feature_per_frame[(num_frame-1)].pointRight(0);

                        float dep_cur = mbf/(cur_x_l - cur_x_r)/FOCAL_LENGTH_X;

                        float prev_y_l = prev_un_pt(1);
                        float cur_y_l = it_per_id.feature_per_frame[(num_frame-1)].point(1);
                        delta_height += (dep_cur*cur_y_l - prev_dep*prev_y_l);
                        
                        ++num_delta_h;
                    }
                }
            }
            continue;
        }

        // 如果上面基于2观测的三角化失败，则这里可以尝试多次观测联合的三角化
        
        // 如果该点当前帧仍被跟踪，且帧数大于参与LBA所与的观测帧数，则对其进行深度值优化。如果该点持续被跟踪，则每一帧都对齐进行更新深度值，这是为了给LBA提供较好的初值
        // 到达这里的前提是该点之前已经完成首观测帧下的深度值估计了，要么使用首帧的立体匹配，要么使用首2帧的三角测量
        // 即使是首观测帧下有深度的点，经过这里估计后的深度值很可能变为无效的，最终只能赋予原始深度值！
        // 本系统不使用此功能
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

        // 在通过PnP或LBA得到当前帧位姿估计，并且三角化某个点（如果需要）之后，更新当前帧跟踪点的深度值.
        // 还未加入地图的静态点则是VI初始化之后的静态物体点，其深度值更新在物体运动估计线程中进行
        // 对于没法在当前帧获取有效深度值的点，直接放弃！
        if(!before_PnP && !cond_2 && num_frame > 0 && try_update_dep)
        { 
            // 到达此处的点都是当前帧的3D-2D点
            index = gl_id_index_map[gl_id];
            
            bool has_stereo_cur = false;

            bool valid_track = true;

            cur_depth = -1.0;

            // 对于当前帧的跟踪点，如果其有立体匹配，则其深度估计留到下一帧其再次被跟踪时再对其进行立体三角化
            // 这只会是物体点，因为背景跟踪点不在当前帧寻找立体匹配！
            if(it_per_id.feature_per_frame.back().is_stereo) 
            {
                has_stereo_cur = true;
            }

            // 检查当前帧的跟踪点的重投影误差（只在有LBA的情况下，因为PnP的内点是经过检验的了），且可以选择更新跟踪点在当前帧的深度值
            {
                bool has_dep_prev = false;
                bool valid_dep_cur = false;
                // 首先对所有有效跟踪点进行重投影误差检测！
                // 重投影时不需要有位移运动!
                // if(good_est_RT && has_p)
                if(good_est_RT)
                {
                    int start_frame_id;
                    float prev_dep;
                    // 优先使用离当前帧较近的帧中,且该帧下的深度来自于立体匹配
                    for(int i = (num_frame-2); i >= 0; --i)
                    {
                        prev_dep = it_per_id.feature_per_frame[i].depth;
                        // 特征点每帧观测下的该变量代表该帧有有有效深度，如果该点立体匹配且立体三角化成功，则该深度值优先来自于三角化（较为可靠）
                        if(prev_dep > 0)
                        {
                            int frame_1 = frameCnt - (num_frame-1 - i);
                            R_start = Rs[frame_1] * ric[0];
                            P_start = Ps[frame_1] + Rs[frame_1] * tic[0];
                            has_dep_prev = true;
                            start_frame_id = i;
                            break;
                        }
                    }

                    // 如果该点在之前每一帧均没有立体匹配，则使用其上一帧的深度（来自于每一帧的运动更新）
                    // 这种情况必须是系统使用两帧间的三角化测量，且首帧三角化成功
                    if(!has_dep_prev)
                    {   
                        if(Use_tria_for_2d2d)
                        {
                            // 首先使用上一帧的深度值。这来自于当前帧与上一帧的三角化，或者上上帧与上一帧之间的运动更新
                            // 这里继续查看的前提是允许上一帧的深度不是来源于立体匹配（即三角化成功之后不需要再寻找立体匹配）
                            if(index > 0)
                                prev_dep = prev_FAST_dep[(index-1)];
                            else
                                prev_dep = prev_sift_dep[(-index)];
                            
                            if(prev_dep > 0)
                            {
                                start_frame_id = num_frame-2;
                                has_dep_prev = true;

                                R_start = Rs[frameCnt-1] * ric[0];
                                P_start = Ps[frameCnt-1] + Rs[frameCnt-1] * tic[0];
                            }
                            else
                            {
                                // 如果当前帧该点三角化失败，则使用该点首帧下的深度
                                // 上一帧还没有深度值的情况可能是首帧的深度值始终未有有效估计，如果在当前帧估计成功，则直接使用其观测首帧下的深度
                                {
                                    prev_dep = it_per_id.estimated_depth;
                                    if(prev_dep > 0)
                                    {
                                        has_dep_prev = true;
                                        start_frame_id = 0;

                                        int start_frame_in_win = it_per_id.start_frame;
                                        R_start = Rs[start_frame_in_win] * ric[0];
                                        P_start = Ps[start_frame_in_win] + Rs[start_frame_in_win] * tic[0];
                                    }
                                }
                            }
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
                        if(cur_depth >= 1.5 && cur_depth < mThDepthBg)
                        {
                            // 是否先使用重投影误差对所有当前帧没有参与LBA的点进行3D-2D的检验？
                            bool check_by_err_reproj = true;

                            // 如果当前帧进行了LBA，则参与了LBA的点的重投影误差检验在别处进行，这里不再重复
                            // 这里只对没有参与LBA的点进行重投影误差检验（前提是有有效的LBA）
                            if(check_by_err_reproj && (LBA_succ && num_frame < TH_NUM_FRAME_FOR_LBA))
                            {
                                float pred_x = pt_cur(0)/cur_depth;
                                float pred_y = pt_cur(1)/cur_depth;
                                
                                Vector3d &cur_pt = it_per_id.feature_per_frame[(num_frame-1)].point;

                                float err = (cur_pt(0) - pred_x)*(cur_pt(0) - pred_x) + (cur_pt(1) - pred_y)*(cur_pt(1) - pred_y);

                                // 投影误差的阈值?4个像素的距离？
                                if(err <= (4.0*4.0/FOCAL_LENGTH_X/FOCAL_LENGTH_X))
                                {
                                    valid_dep_cur = true;
                                }
                            }
                            else
                            {
                                valid_dep_cur = true;
                            }

                            if(valid_dep_cur)
                            {
                                valid_track = true;
                                ++num_tria_succ;
                                // 只更新在当前帧没有立体匹配的点的深度值
                                if(!has_stereo_cur)
                                {
                                    if(index > 0)
                                    {
                                        // 是否要通过运动估计来更新跟踪点在当前帧的深度值
                                        if(Cal_cur_dep_by_motion)
                                            cur_FAST_dep[(index-1)] = cur_depth;

                                        if(use_motion_to_pred_fea_dep)
                                        {
                                            if(obj_id_cls_FAST[(index-1)].first == 0) 
                                            {
                                                ave_dep_bg_cur_frame += cur_depth;
                                                num_bg_with_dep +=1;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        // 是否要通过运动估计来更新跟踪点在当前帧的深度值
                                        if(Cal_cur_dep_by_motion)
                                            cur_sift_dep[(-1*index)] = cur_depth;

                                        if(use_motion_to_pred_fea_dep)
                                        {
                                            if(obj_id_cls_sift[(-1*index)].first == 0) 
                                            {
                                                ave_dep_bg_cur_frame += cur_depth;
                                                num_bg_with_dep +=1;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                valid_track = false;
                            }
                            
                            // 如果某个点在某帧下的深度是通过之前帧该点深度和两帧间运动 计算得来的，那么认为不太可靠，不将该深度值记录到depth变量中
                            // depth变量只记录该点在某帧下由立体三角化得到的深度值，即前提是必须有立体匹配
                            // it_per_id.feature_per_frame[(num_frame-1)].depth = cur_depth;
                        }
                        else
                        {
                            // valid_dep_cur = false;
                            valid_track = false;
                        }
                    }
                    else
                    {
                        if(!it_per_id.has_LBA)
                        {
                            cout << "Weired! Why long track fea has no depth in its first frame? Line 3912" << endl;
                            exit(-1);
                        }

                        valid_track = false;
                        // valid_dep_cur = false;
                    }
                }
                else
                {
                    if(index > 0)
                    {
                        // 如果是静态物体点且在当前帧中没有立体匹配，则放弃该物体跟踪点
                        if(find(id_FAST_no_depth.begin(), id_FAST_no_depth.end(), (index-1)) != id_FAST_no_depth.end())
                        {
                            valid_track = false;
                        }
                        else
                        {
                            // todo: 如果当前帧的运动估计不可靠（只有非RANSAC的PnP或甚至只有预测的运动值），那么是否要保留其他的跟踪内点（包括3D-2D和2D-2D点）？
                            // 选择保留，否则，当前帧的位姿后续就没办法再通过LBA进行优化了。
                            // 但是该点在当前帧没有深度值，因此下一帧如果其再被跟踪到，则在为其寻找立体匹配时需要使用depth_map来提供预测值！!
                            // valid_track = false;
                            valid_track = true;
                            ++num_tria_succ;
                        }
                    }
                    else
                    {
                        if(find(id_sift_no_depth.begin(), id_sift_no_depth.end(), (-index)) != id_sift_no_depth.end())
                        {
                            valid_track = false;
                        }
                        else
                        {
                            // valid_track = false;
                            valid_track = true;
                            ++num_tria_succ;
                        }
                    }
                    // valid_dep_cur = false;
                }
            }
            
            if(!valid_track)
            {
                ++num_tri_fail;
                // 放弃继续跟踪该点
                if(index > 0)
                {
                    status_FAST[(index-1)] = 0;
                }
                else
                {
                    status_sift[(-1*index)] = 0;
                }

                // 然后该点用在当前帧没有有效深度或者深度估计超过阈值
                if(cur_depth < 1.2 || cur_depth >= 1.1*mThDepthBg)
                {
                    num_frame -= 1;
                    if(frameCnt >= WINDOW_SIZE)
                    {
                        // 如果滑窗要marg次新帧
                        if(!marg_old)
                        {
                            num_frame -= 1;
                        }
                        else
                        {
                            if(it_per_id.start_frame == 0)
                            {
                                num_frame -= 1;
                            }
                        }
                    }

                    if(num_frame < TH_NUM_FRAME_FOR_LBA)
                    {
                        if(pt_need_to_erase.empty() || pt_need_to_erase.back() != gl_id)
                            pt_need_to_erase.push_back(gl_id);
                    }
                    else
                    {
                        it_per_id.feature_per_frame.pop_back();
                    }
                }
            }
        }
    }
    
    if(before_PnP)
    {
        cout << "Num of fea that failed triangulate with stereo match: " << num_tri_fail << endl;
    }
    else if(good_est_RT)
    {
        if(Use_tria_for_2d2d && try_tria)
            cout << "Finished triangulation tracking fea with two frames! Num of succeed fea: " << num_tria_succ << ", Num of failed fea : " << num_tri_fail << endl;
        else if(try_update_dep)
            cout << "Finished update depth of tracking fea in current frame! Num of succeed fea: " << num_tria_succ << ", Num of failed fea : " << num_tri_fail << endl;
    }
    
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
        // 查看静态地图点在前后2帧的相机坐标系下的高度变化
        if(num_delta_h > 0)
        {
            float ave_delta_h = delta_height/num_delta_h;
            
            cout << "average delta dep of map points is: " << ave_delta_h << endl;
        }
        
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

        int num_1 = 13, num_2 = 9, num_3 = 5;

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
                    if(it.first > 20) break;

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
        if(cur_2d_un_pts.size() < 10)
        {
            int rest = 10 - cur_2d_un_pts.size();
            // int rest = 8;
            for(auto &it: temp_dep_id)
            {
                // 上半部分的点的深度也不能太大。上半部分图像中其实可能会有比较近的点，例如杆或者路牌上的点
                if(it.first > 26) break;

                if(!rest) break;

                int l_id = it.second;
                cur_2d_un_pts.push_back(temp_2d_pt[l_id]);
                prev_3d_pts.push_back(temp_3d_pt[l_id]);
                --rest;
            }
        }
        
        float norm_orig = Norm_t.norm();
        // Vector3d prev_t = Norm_t;

        // 最少需要2个3D-2D匹配点来估计尺度s？
        if(cur_2d_un_pts.size() > 1)
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
                    if(abs(dep_rej) < 0.01) 
                    {
                        cout << "" << dep_rej << endl;
                        continue;
                    }
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
                if(delta_norm < 0.01 && abs(delta_s) < thres)
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
        else
        {
            return false;
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

            // 这里所有的H矩阵估计的外点中部分是纯背景跟踪点，一部分是静态物体的跟踪点
            // 要注意，如果是上一帧的静态物体点，其当前帧的观测甚至其所有的观测可能都还未加入地图！
            // 但是这里是否可以直接使用其3D-2D观测来优化t的尺度？
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
            int has_outlier = 0;
            // 这些点到已估计极线的距离限制如何取？
            float ratio_out = epipolarConstrain(prev_pts, cur_pts, F_cam, is_inlier, Th_score, score, has_outlier, F_THRESHOLD*F_THRESHOLD);
            
            vector<int> erase_pt;
            vector<int> re_valid_pts;

            // todo:: 是否要用重新验证为内点的 跟踪点来进一步优化尺度？
            bool refine_scale_using_near_high_fea = false;

            // 存在内点
            if(ratio_out < 1.0)
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
                            // 上一帧太远的点要保留吗？倾向于不保留（因为远点一般都会近似符合地面H矩阵约束，除非其匹配误差较大）
                            // 而且这里重新恢复这些外点的原因就是用其中的近点来优化尺度
                            if(dep_ > 12)
                            {
                                erase_pt.push_back(g_id);
                            }
                            else if(refine_scale_using_near_high_fea)
                            {
                                // 用来优化t的尺度
                                re_valid_pts.push_back(g_id);
                            }
                        }
                    }
                }
            }
            else
            {
                // 所有的点均为无效点
                erase_pt = g_pt_id;
            }
            
            // todo: 这里删除这些点，会对objs_matching的后续处理有影响吗？
            if(!erase_pt.empty())
            {
                for(int k = 0; k < erase_pt.size(); ++k)
                {
                    int g_id = erase_pt[k];
                    
                    // 可能有的跟踪点是上一帧的静态物体上的点？
                    // 虽然参与E或H矩阵估计的物体点必须是旧跟踪点，即该物体在上一帧就被确认为静态，但是其在上一帧不一定被加入地图，且在当前帧的物体匹配阶段未被选择加入地图，那么其当前帧的观测就还未在地图中！
                    auto it = find_if(feature.begin(), feature.end(), [g_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == g_id;
                                    });
                    
                    // assert(it != feature.end());
                    // 如果该点已经在地图中，并且有当前帧的观测（针对的是静态物体点）
                    // todo:如果该物体点在地图中，但是当前帧观测还未加入，那么如何防止这个外点的当前帧观测在后续会被加入（在物体运动估计阶段）？
                    // 后续如果其他静态物体的跟踪点想要加入地图，必须通过更精准的相机运动（来自PnP)的重投影检测！因此即使这里某个H的外点后续想要加入地图，也需要被检验！
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

            if(refine_scale_using_near_high_fea && !re_valid_pts.empty())
            {
                cur_2d_un_pts.clear();
                prev_3d_pts.clear();
                float dep;
                for(int k = 0; k < re_valid_pts.size(); ++k)
                {
                    int g_id = re_valid_pts[k];
                    
                    
                    auto it = find_if(feature.begin(), feature.end(), [g_id](const FeaturePerId &it)
                                    {
                                        return it.feature_id == g_id;
                                    });
                    
                    // assert(it != feature.end());
                    // 其中部分是上一帧静态物体的跟踪点，其在当前帧的观测可能还不在地图中
                    if(it != feature.end() && it->endFrame() == frameCnt)
                    {
                        int num_frame = it->feature_per_frame.size();
                        if(num_frame > 1)
                        {
                            // todo:上一帧该点的深度是否要来自于立体匹配？？
                            dep = it->feature_per_frame[(num_frame-2)].depth;

                            // 以下面这种情况来寻找上一帧该点的深度，则可以包含那些在上一帧中没有立体匹配的点，它们在上一帧的深度值来自于运动变换
                            // int index = gl_id_index_map[g_id];
                            // if(index > 0)
                            // {
                            //     dep = prev_FAST_dep[(index-1)];
                            // }
                            // else
                            // {
                            //     dep = prev_sift_dep[(-index)];
                            // }

                            if(dep > 1.0 && dep <= 12)
                            {
                                Vector3d &cur_pt = it->feature_per_frame[(num_frame-1)].point;
                                Vector3d &prev_pt = it->feature_per_frame[(num_frame-2)].point;

                                // float pos_y = prev_pt(1) * dep;
                                // todo:选取距离地面高度大于等于0.5m的点。KITTI数据集中相机的高度大约为1.65m
                                // 注意，在KITTI数据集中相机坐标系的竖直轴为y轴，且向下为正向，因此归一化平面上在中心横线以上的y值是负的，以下是正的，则地面点的y坐标为+1.65。
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
                    // 第二次优化应该在第一次的结果P上进行，而不是在原始的Norm_t上，因为此时的点数太少了！
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

                            if(abs(dep_rej) < 0.01) continue;

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
                        if(delta_norm < 0.0025 && abs(delta_s) < thres)
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
    
    return num_tria_succ;
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

// 遍历地图中的当前帧跟踪点，记录其中剩余观测帧数 >=2 和 >=3的点
void FeatureManager::find_long_track_fea_in_map(int frameCnt, set<int> &fea_with_more_frames_in_map, set<int> &fea_with_3_frames_in_map)
{
    if(frameCnt < 1) return;
    
    if(!fea_with_more_frames_in_map.empty()) fea_with_more_frames_in_map.clear();
    if(!fea_with_3_frames_in_map.empty()) fea_with_3_frames_in_map.clear();

    for(auto &fea: feature)
    {
        int num_frame = fea.feature_per_frame.size();
        if(num_frame >= 2 && fea.endFrame() == frameCnt)
        {
            fea_with_more_frames_in_map.insert(fea.feature_id);
            if(num_frame >= 3) fea_with_3_frames_in_map.insert(fea.feature_id);
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
void FeatureManager::removeBackShiftDepth(int frameCnt, Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P, FeatureTracker &tracker)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int id = it->feature_id;
        if(it->estimated_depth <= 0)
        {
            if(it->endFrame() == frameCnt)
            {
                if(tracker.gl_id_index_map.find(id) == tracker.gl_id_index_map.end())
                {
                    cout << "Weired! Line 4684" << endl;
                    exit(-1);
                }
                int l_id = tracker.gl_id_index_map[id];
                cout << "pt_l_id: " << l_id  << endl;
                float dep;
                int obj_cls, obj_status, cnt_track;
                int has_stereo = 0;
                if(it->feature_per_frame[0].is_stereo) has_stereo = 1;
                // 当前帧无法三角化测量的点跟踪应该在之前就已经从地图中删除了？
                if(l_id  > 0)
                {
                    dep = tracker.prev_FAST_dep[(l_id -1)];
                    obj_cls = tracker.obj_cls_id_FAST[(l_id -1)].first;
                    obj_status = tracker.obj_cls_id_FAST[(l_id -1)].second;
                    cnt_track = tracker.track_cnt_FAST[(l_id -1)];
                }
                else
                {
                    dep = tracker.prev_sift_dep[(-l_id )];
                    obj_cls = tracker.obj_cls_id_sift[(-l_id )].first;
                    obj_status = tracker.obj_cls_id_sift[(-l_id )].second;
                    cnt_track = tracker.track_cnt_sift[(-l_id )];
                }
                cout << "num of frame: " << it->feature_per_frame.size() << ", has stereo in first frame: " << has_stereo <<  ", estimated dep: " << it->estimated_depth << endl;
                cout << "dep of pt in prev frame: " << dep << ", obj_cls: " << obj_cls << ", obj_status: " << obj_status << ", cnt_track: " << cnt_track << endl;
                cout << "Weired! Line 4706" << endl;
                exit(-1);
            }
            feature.erase(it);
            continue;
        }

        // 如果观测首帧不是滑窗首帧
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            int num_frame = it->feature_per_frame.size();

            bool invalid_track = false;

            // 如果只有滑窗首帧这一观测帧
            if(num_frame == 1)
            {
                feature.erase(it);
                continue;
            }
            // 如果不是当前帧的跟踪点，且删除滑窗首帧后其剩余的观测帧数小于LBA所需帧数，则该点肯定不会再参与LBA了，那么就无需再保留该点了！
            else if(num_frame < (TH_NUM_FRAME_FOR_LBA+1) && it->endFrame() != frameCnt)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                // 如果该点在滑窗第二帧中有来自立体匹配的深度值
                if(it->feature_per_frame[1].depth > 0)
                {
                    it->estimated_depth = it->feature_per_frame[1].depth;
                }
                else if(it->estimated_depth > 0)
                {
                    Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                    Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                    Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                    double dep_j = pts_j(2);
                    // 是否要按照背景点/物体点来取最大深度值？
                    // if (dep_j > 0)
                    if(dep_j >= 1.5 && dep_j < mThDepthBg)
                        it->estimated_depth = dep_j;
                    else
                    {
                        
                        // it->estimated_depth = INIT_DEPTH;
                        // it->estimated_depth = -1.0;
                        // 由于此系统不再为上一帧之前的点寻找立体匹配或者三角化测量，因此如果该点去除首帧后剩余超过1帧或者不是当前帧的跟踪点，则该点后续首帧都不会再有深度值（即无法参与LBA），因此删除该点
                        if(num_frame > 2 || it->endFrame() != frameCnt)
                        {
                            invalid_track = true;
                        }
                        else
                        {
                            // 这取决于滑窗的长度（这种情况下滑窗长度只能为2）
                            if(it->feature_per_frame.back().depth > 0)
                            {
                                it->estimated_depth = it->feature_per_frame.back().depth;
                            }
                            else
                            {
                                invalid_track = true;
                            }
                        }
                    }
                }
                else
                {
                    // 如果该点删除首帧后只剩下当前帧观测（这取决于滑窗的长度是否为2），且允许2d-2d三角化以恢复深度，则保留该点在地图中的观测以待其在下一帧被跟踪到
                    if(num_frame == 2 && it->endFrame() == frameCnt && Use_tria_for_2d2d)
                    {
                        if(it->feature_per_frame[1].depth > 0)
                        {
                            it->estimated_depth = it->feature_per_frame[1].depth;
                        }
                    }
                    else
                    {
                        invalid_track = true;
                    }
                }
            }

            if(invalid_track)
            {
                if(it->endFrame() == frameCnt)
                {
                    int l_id = tracker.gl_id_index_map[id];
                    if(l_id  > 0)
                        tracker.statusLeftRIght[(l_id-1)] = 0;
                    else
                        tracker.status_sift[(-l_id)] = 0;
                }

                feature.erase(it);
                continue;
            }
            else
                it->feature_per_frame.erase(it->feature_per_frame.begin());
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

void FeatureManager::removeBack(int frame_cnt, FeatureTracker &tracker)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            bool invalid_fea = false;
            int end_frame = it->endFrame();
            int num_frame = it->feature_per_frame.size();
            // 调用此函数则说明系统还没成功VI初始化，则应该保留所有至少有两帧观测(前后2帧，或者单帧的左右观测）的点，这样可以用于VI初始化～
            if(num_frame < 2)
                invalid_fea = true;
            else if (!it->feature_per_frame[1].is_stereo)
            {
                invalid_fea = true;
            }
            else
            {
                if(end_frame != frame_cnt)
                {
                    if(num_frame < (TH_NUM_FRAME_FOR_LBA + 1))
                        invalid_fea = true;
                    else
                        it->estimated_depth = it->feature_per_frame[1].depth;
                }
                else
                    it->estimated_depth = it->feature_per_frame[1].depth;
            }

            if(invalid_fea)
            {
                if(end_frame == frame_cnt)
                {
                    int id = it->feature_id;
                    int l_id = tracker.gl_id_index_map[id];
                    if(l_id  > 0)
                        tracker.statusLeftRIght[(l_id-1)] = 0;
                    else
                        tracker.status_sift[(-l_id)] = 0;
                }

                feature.erase(it);
                continue;
            }
            else
                it->feature_per_frame.erase(it->feature_per_frame.begin());
        }
    }
}

// 此函数只有在滑窗去掉次新帧时才被调用
// 由于只有在滑窗满的时候才marg，因此这里frame_cnt就是等于WINDOW_SIZE
void FeatureManager::removeFront(int frame_cnt, FeatureTracker &tracker, double Headers[])
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果该地图点是最新帧中新检测的新特征点
        if (it->start_frame == frame_cnt)
        {
            it->start_frame--;
        }
        else
        {
            // 如果该点没有被次新帧观测到，则不处理
            if (it->endFrame() < frame_cnt - 1)
                continue;
            
            // 当前帧滑窗次新帧的id - 该点的初始观测帧id
            int j = frame_cnt - 1 - it->start_frame;
            
            int num_frame = it->feature_per_frame.size();
            // 如果该点首观测帧是次新帧,且在当前帧也被跟踪到，则修改该点的首帧下深度
            if(it->start_frame == WINDOW_SIZE-1 && num_frame > 1) 
            {
                // 如果其在当前帧有立体匹配,则该深度是其新的首帧深度
                if(it->feature_per_frame.back().is_stereo)
                {
                    float dep = it->feature_per_frame.back().depth;
                    if(dep > 0)
                        it->estimated_depth = dep;
                    else
                    {
                        // assert(false && "Why fea with stereo match has no depth?");
                        cout << "Why fea with stereo match has no depth?" << endl;
                        exit(-1);
                    }
                }
                // 否则要求该点下一帧可以重新三角化测量（如果该点下一帧仍能被观测）
                else
                {
                    if(Use_tria_for_2d2d)
                    {
                        // it->estimated_depth = INIT_DEPTH;
                        it->estimated_depth = -1.0;
                    }
                    else
                    {
                        feature.erase(it);
                        continue;
                    }
                }
            }

            // 如果系统需要LBA，则需要尽可能保证在每一帧有足够的长跟踪点
            // 如果该点是当前帧的背景跟踪点，且上一帧观测也在地图中，则由于marg次新帧而将其track_cnt减1，方便下一帧确认其是否为旧点（指的是添加该帧观测后该点在地图中的连续观测帧数是否大于2）
            // 最终决定不这样操作，而是建后续遍历所有地图点，将在地图中剩余观测帧数大于1的跟踪点记录下来！
            if(0 && (USE_IMU || Use_LBA_for_puer_V))
            {
                if(it->start_frame <= frame_cnt-1 && it->endFrame() == frame_cnt)
                {
                    int id = it->feature_id;
                    if(tracker.gl_id_index_map.find(id) == tracker.gl_id_index_map.end())
                    {
                        cout << "Weired! Line 4837" << endl;
                        exit(-1);
                    }

                    if(id > 0)
                    {
                        if(tracker.obj_cls_id_FAST[(id-1)].first == 0)
                        {
                            tracker.track_cnt_FAST[(id-1)] -= 1;
                        }
                    }
                    else
                    {
                        if(tracker.obj_cls_id_sift[(-id)].first == 0)
                        {
                            tracker.track_cnt_sift[(-id)] -= 1;
                        }
                    }
                }
            }
            
            // 当marg次新帧，是否要将marg后的最后一帧和倒数第2帧之间的二维速度进行修改（这对估计IMU和相机之间的dt很重要），因为该2帧实际上不是连续的。
            // marg次新帧意味着近期的视差很小，因此二维速度的帧间变化也很小？
            if(USE_IMU && ESTIMATE_TD)
            {
                if(it->endFrame() == frame_cnt)
                {
                    if(num_frame == 2)
                    {
                        // todo:如果最新的2帧观测，则marg次新帧后就只剩下1帧，是否要将剩下帧的左右二维点的速度均置为0？这取决于LBA时是否会使用点的观测首帧下的二维速度（似乎没使用）
                        it->feature_per_frame[(num_frame-1)].velocity = Vector2d(0,0);
                        if(it->feature_per_frame[(num_frame-1)].is_stereo) it->feature_per_frame[(num_frame-1)].velocityRight = Vector2d(0,0);
                    }
                    else if(num_frame > 2)
                    {
                        Vector3d &pt_l1 = it->feature_per_frame[(num_frame-1)].point;
                        Vector3d &pt_l2 = it->feature_per_frame[(num_frame-3)].point;
                        Vector3d new_vl = (pt_l1 - pt_l2)/(Headers[frame_cnt] - Headers[frame_cnt-2]);
                        it->feature_per_frame[(num_frame-1)].velocity = Vector2d(new_vl(0), new_vl(1));

                        if(it->feature_per_frame[(num_frame-1)].is_stereo && it->feature_per_frame[(num_frame-3)].is_stereo)
                        {
                            Vector3d &pt_r1 = it->feature_per_frame[(num_frame-1)].pointRight;
                            Vector3d &pt_r2 = it->feature_per_frame[(num_frame-3)].pointRight;
                            Vector3d new_vr = (pt_r1 - pt_r2)/(Headers[frame_cnt] - Headers[frame_cnt-2]);
                            it->feature_per_frame[(num_frame-1)].velocityRight = Vector2d(new_vr(0), new_vr(1));
                        }
                    }
                }
            }

            // 去除该点在次新帧的观测记录
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
    // 归一化平面的点坐标
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;

    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    // double dep_i_comp = p_i_comp(2);
    // double u_i_comp = p_i_comp(0) / dep_i_comp;
    // double v_i_comp = p_i_comp(1) / dep_i_comp;
    // double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
    ans = max(ans, sqrt(du * du + dv * dv));

    return ans;
}
