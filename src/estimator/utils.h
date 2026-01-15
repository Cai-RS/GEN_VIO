#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/types.h>    
#include <sys/stat.h>
#include <functional>
#include <map>
#include <limits.h>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "parameters.h"

struct RawImageData{
  size_t index;
  double time;
  cv::Mat image_left;
  cv::Mat image_right;

  RawImageData() {}
  RawImageData& operator =(RawImageData& other){
		index = other.index;
		time = other.time;
		image_left = other.image_left.clone();
		image_right = other.image_right.clone();
		return *this;
	}
};
typedef std::shared_ptr<RawImageData> RawImageDataPtr;

typedef std::shared_ptr<g2o::Line3D> Line3DPtr;
typedef std::shared_ptr<const g2o::Line3D> ConstLine3DPtr;

// Eigen type
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;

// temport memory for feature. These feature are used for 
// Matrix中前7个与VINS-Fusion中的一样，代表x, y, z, p_u, p_v, velocity_x, velocity_y；这里最后添加多一个double，表示该点是否为静态点（0是1否）！
// 所有点的map中第一个int均表示该特征点的全局id。pair中的第一个int表示观测是在左相机还是右相机。

typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> FeaFrame;

// 下面三种特征点在每一帧中应该要按顺序跟踪和添加，即先处理SIFT，然后是FAST，最后再处理普通像素。

// Sift点属于少数，因此需要FAST点来作为特征点的补充。
typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> SiftFrame;
// 如果是动态点，则FAST光流匹配时应该要加大搜索范围！检测和跟踪时是否为静态点是根据上一帧的该点的状态来决定的。
typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> FASTFrame;
// 如果特征点的数量实在不足，则使用普通像素点作为补充。这个主要是针对object的
typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> PixelFrame;

// 将当前帧跟踪到和新添的特征点分类为各个物体集中（第一个为背景），每个集合中背景点在前，新添加点在后
// 外层map的key为各个物体的id，内层map的key为属于该物体的全局特征点的id，vector中保存的是该特征点在左或者右图像中的坐标以及速度，pair中的int表示左（0）或者右(1)相机
typedef std::map<int, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>>> FeaObjFrame;

// 当前帧所检测和跟踪到的所有特征点集，包括了SIFT，FAST和普通像素点。
// vector指针类型的变量 https://www.cnblogs.com/CodeWorkerLiMing/p/11221367.html
// typedef std::vector<FeaObjFrame*> ObjFeaFrame;

// 背景中的所有特征点，包括Sift（排在前面）和FAST（排在後面）。极端情况下还可能包含普通像素点
// 第1个int表示该点是sift(0)，FAST(1)还是普通像素点（2）；第2个int表示该点在当前帧对应的特征点集中的索引序号
//typedef std::vector<std::pair<int, int>> StaFeatureFrame;
//typedef std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> StaFeatureFrame;

typedef std::map<int, std::pair<int, int>> NumFeaObjFrame;
typedef std::vector<NumFeaObjFrame*> ObjFeaNumFrame;

// 在帧间的物体关联 以及确定物体是否运动 之后，需要把ObjFeatureFrame中的点具体分配给每个全局动态物体，或者加入到StaFeatureFrame中。
typedef std::vector<std::vector<std::pair<int, int>>> DynObjFeatureFrame;
// 最后在后端线程中这些观测会被放进该全局地图点的FeaturePerId和相应的FeaturePerFrame中

template <template <typename, typename> class Container, typename Type>
using Aligned = Container<Type, Eigen::aligned_allocator<Type>>;

template <typename KeyType, typename ValueType>
using AlignedMap =
    std::map<KeyType, ValueType, std::less<KeyType>,
             Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMap = std::unordered_map<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMultimap = std::unordered_multimap<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename Type>
using AlignedUnorderedSet =
    std::unordered_set<Type, std::hash<Type>, std::equal_to<Type>,
                       Eigen::aligned_allocator<Type>>;

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1>& m, Eigen::Matrix3d& R, Eigen::Vector3d& t);
double DescriptorDistance(const Eigen::Matrix<double, 256, 1>& f1, const Eigen::Matrix<double, 256, 1>& f2);
cv::Scalar GenerateColor(int id);
void GenerateColor(int id, Eigen::Vector3d color);
cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<bool>& inliers, const std::vector<Eigen::Vector4d>& lines, 
    const std::vector<int>& line_track_ids, const std::vector<std::map<int, double>>& points_on_lines);

// files
void GetFileNames(std::string path, std::vector<std::string>& filenames);
bool FileExists(const std::string& file);
bool PathExists(const std::string& path);
void ConcatenateFolderAndFileName(const std::string& folder, const std::string& file_name, std::string* path);

std::string ConcatenateFolderAndFileName(const std::string& folder, const std::string& file_name);

void MakeDir(const std::string& path);

void ReadTxt(const std::string& file_path, std::vector<std::vector<std::string> >& lines, std::string seq);

void WriteTxt(const std::string file_path, std::vector<std::vector<std::string> >& lines, std::string seq);

int check_flow_with_H(const Eigen::Matrix3d &cam_H, const cv::Point2f &pt1, const cv::Point2f &pt2, float thres_dist = 0.0);

int check_flow_with_F(const Eigen::Matrix3d &cam_F, const cv::Point2f &pt1, const cv::Point2f &pt2, float thres_dist = 0.0);

float HomographyConstrain(const std::vector<cv::Point2f> &kp1, const std::vector<cv::Point2f> &kp2, const Eigen::Matrix3d& Mat_H, const Eigen::Matrix3d& Mat_H_inv,  
                            std::vector<uchar> &is_inlier, const float &Th_score, float &score, int &has_outlier, const float &Th_dist = 5.0);

float epipolarConstrain(const std::vector<cv::Point2f> &kp1, const std::vector<cv::Point2f> &kp2, const Eigen::Matrix3d& Mat_F, 
                        std::vector<uchar> &is_inlier, const float &Th_score, float &score, int &has_outlier, const float &Th_dist = 4.0);

double reprojectionError(const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi, const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
                        const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj, const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj, 
                        double depth, const Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj);

float cal_ave_epi_line_dist_pt(const Eigen::Matrix3d &Mat_F, const Eigen::Vector3d &pt1, const Eigen::Vector3d &pt2);

#endif  // UTILS_H_