
#include "utils.h"
#include <dirent.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1>& m, Eigen::Matrix3d& R, Eigen::Vector3d& t){
  Eigen::Quaterniond q(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
  R = q.matrix();
  t = m.block<3, 1>(4, 0);
}

// (f1 - f2) * (f1 - f2) = f1 * f1 + f2 * f2 - 2 * f1 *f2 = 2 - 2 * f1 * f2 -> [0, 4]
double DescriptorDistance(const Eigen::Matrix<double, 256, 1>& f1, const Eigen::Matrix<double, 256, 1>& f2){
  return 2 * (1.0 - f1.transpose() * f2);
}

cv::Scalar GenerateColor(int id){
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  return cv::Scalar(blue, green, red);
}

void GenerateColor(int id, Eigen::Vector3d color){
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  color << red, green, blue;
  color *= (1.0 / 255.0);
}

cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<bool>& inliers, const std::vector<Eigen::Vector4d>& lines, 
    const std::vector<int>& line_track_ids, const std::vector<std::map<int, double>>& points_on_lines){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  size_t point_num = keypoints.size();
  std::vector<cv::Scalar> colors(point_num, cv::Scalar(0, 255, 0));
  std::vector<int> radii(point_num, 2);

  // draw lines
  for(size_t i = 0; i < lines.size(); i++){
    if(line_track_ids[i] < 0) continue;
    cv::Scalar color = GenerateColor(line_track_ids[i]);
    Eigen::Vector4d line = lines[i];
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), color, 2);

    cv::putText(img_color, std::to_string(line_track_ids[i]), cv::Point((int)((line(0)+line(2))/2), 
        (int)((line(1)+line(3))/2)), cv::FONT_HERSHEY_DUPLEX, 1.0, color, 2);

    for(auto& kv : points_on_lines[i]){
      colors[kv.first] = color;
      radii[kv.first] *= 2;
    }
  }

  // draw points
  for(size_t j = 0; j < point_num; j++){
    cv::circle(img_color, keypoints[j].pt, radii[j], colors[j], 1, cv::LINE_AA);
  }
  return img_color;
}

void GetFileNames(std::string path, std::vector<std::string>& filenames){
  DIR *pDir;
  struct dirent* ptr;
  std::cout << "path = " << path << std::endl;
  if(!(pDir = opendir(path.c_str()))){
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while((ptr = readdir(pDir))!= 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
      filenames.push_back(ptr->d_name);
    }
  }
  closedir(pDir);
}

bool FileExists(const std::string& file) {
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFREG)) {
    return true;
  }
  return false;
}


bool PathExists(const std::string& path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

void ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name,
    std::string* path) {
  *path = folder;
  if (path->back() != '/') {
    *path += '/';
  }
  *path = *path + file_name;
}

std::string ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name) {
  std::string path;
  ConcatenateFolderAndFileName(folder, file_name, &path);
  return path;
}

void MakeDir(const std::string& path){
  if(!PathExists(path)){
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}

void ReadTxt(const std::string& file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq){
  if(!FileExists(file_path)){
    std::cout << "file: " << file_path << " dosen't exist" << std::endl;
    exit(0);
  }
  
  std::ifstream infile(file_path, std::ifstream::in);   
  if(!infile.is_open()){
    std::cout << "open file: " << file_path << " failure" << std::endl;
    exit(0);
  }

  std::string line;
  while (getline(infile, line)){ 
    std::string whitespaces(" \t\f\v\n\r");
    std::size_t found = line.find_last_not_of(whitespaces);
    if (found!=std::string::npos){
      line.erase(found+1);

      std::vector<std::string> line_data;
      while (true){
        int index = line.find(seq);
        std::string sub_str = line.substr(0, index);
        if (!sub_str.empty()){
          line_data.push_back(sub_str);
        }

        line.erase(0, index + seq.size());
        if (index == -1){
          break;
        }
      }
      lines.emplace_back(line_data);
    }
    else{
      line.clear();            // str is all whitespace
    }
  }
}

void WriteTxt(const std::string file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq)
{
  std::fstream file;
  file.open(file_path.c_str(), std::ios::out|std::ios::app);
  if(!file.good()){
    std::cout << "Error: cannot open file " << file_path << std::endl;
    exit(0);
  }
  for(std::vector<std::string>& line : lines){
    size_t num_in_line = line.size();
    if(num_in_line < 1) continue;
    std::string line_txt = line[0];
    for(size_t i = 1; i < num_in_line; ++i){
      line_txt = line_txt + seq + line[i];
    }
    line_txt += "\n";
    file << line_txt;
  }
  file.close();
}

int check_flow_with_H(const Eigen::Matrix3d &cam_H, const cv::Point2f &pt1, const cv::Point2f &pt2, float thres_dist)
{
  float proj_x, proj_y, proj_z;
  float h_00 = cam_H(0,0);
  float h_01 = cam_H(0,1);
  float h_02 = cam_H(0,2);
  float h_10 = cam_H(1,0);
  float h_11 = cam_H(1,1);
  float h_12 = cam_H(1,2);
  float h_20 = cam_H(2,0);
  float h_21 = cam_H(2,1);
  float h_22 = cam_H(2,2);

  proj_x = pt1.x * h_00 + pt1.y * h_01 + h_02;
  proj_y = pt1.x * h_10 + pt1.y * h_11 + h_12;
  proj_z = pt1.x * h_20 + pt1.y * h_21 + h_22;

  if(proj_z == 0)
  {
    return -1;
  }
  proj_x = proj_x/proj_z;
  proj_y = proj_y/proj_z;

  float dist = (proj_x - pt2.x) * (proj_x - pt2.x) + (proj_y - pt2.y) * (proj_y - pt2.y);
  
  if(thres_dist <= 0) thres_dist = Th_homography_con;

  if(dist > thres_dist*thres_dist)
  {
    return -1;
  }
  else
  {
    return 1;
  }
}

int check_flow_with_F(const Eigen::Matrix3d &cam_F, const cv::Point2f &pt1, const cv::Point2f &pt2, float thres_dist)
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
        
        if(thres_dist <= 0)
          thres_dist = Th_epipolar_con;
        
        // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
        // 由于这是预测的运动构成的极线，精度不高，所以不应该限制太大
        if(dsqr > thres_dist * thres_dist) 
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

// 用H矩阵排除不是该平面或者视差太大的点
float HomographyConstrain(const std::vector<cv::Point2f> &kp1, const std::vector<cv::Point2f> &kp2, const Eigen::Matrix3d& Mat_H, const Eigen::Matrix3d& Mat_H_inv,  
                            std::vector<uchar> &is_inlier, const float &Th_score, float &score, int &has_outlier, const float &Th_dist)
{
  int num_pts = kp1.size();
  if(is_inlier.empty()) is_inlier.resize(num_pts, 0);

  has_outlier = 0;
  int num_invalid = 0;

  float proj_x, proj_y, proj_z;
  cv::Point2f prev_pt, cur_pt;

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
  
  for(int i = 0; i < num_pts; ++i)
  {
      prev_pt = kp1[i];
      cur_pt = kp2[i];

      // prev -> cur_pt
      proj_x = prev_pt.x * h_00 + prev_pt.y * h_01 + h_02;
      proj_y = prev_pt.x * h_10 + prev_pt.y * h_11 + h_12;
      proj_z = prev_pt.x * h_20 + prev_pt.y * h_21 + h_22;
      if(proj_z == 0)
      {
          has_outlier = 1;
          ++num_invalid;

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
          has_outlier = 1;
          ++num_invalid;

          // is_inlier[i] = 0;
          continue;
      }
      inv_proj_x = inv_proj_x/inv_proj_z;
      inv_proj_y = inv_proj_y/inv_proj_z;

      float inv_dist = (inv_proj_x - prev_pt.x) * (inv_proj_x - prev_pt.x) + (inv_proj_y - prev_pt.y) * (inv_proj_y - prev_pt.y);

      // 这里Th_dist其实是允许距离误差阈值的平方
      // todo: 虽然计算H矩阵时用的点投影距离误差是sqrt(Th_dist)，然而后续发现H的内点不一定全都严格满足小于此距离误差！！可能有些点会稍微大于此误差！（那么在这里是否要将这些点重新变为外点？）
      // 在ORB-SLAM中对于H矩阵也是只统一使用一个Th_score来执行这里的判断，而没有另外使用Th_dist！为什么呢？？

      // if(dist <= Th_dist && inv_dist <= Th_dist) 
      if(dist <= Th_score && inv_dist <= Th_score) 
      {
          is_inlier[i] = 1;
          
          score += (Th_score - dist);
          score += (Th_score - inv_dist);
      }
      else
      {
          // // if(dist > Th_dist && inv_dist > Th_dist)
          // if(dist > Th_score && inv_dist > Th_score)
          //     cout << "both forward and inv dist invalid!" << endl;
          // // else if(dist > Th_dist)
          // else if(dist > Th_score)
          //     cout << "forward dist invalid!" << endl;
          // else
          //     cout << "inv dist invalid!" << endl;
          
          // cout << "forward dist: " << dist << " inv dist: " << inv_dist << endl;
          
          has_outlier = 1;
          ++num_invalid;
      }
  }
  
  // cout << "total checked pt for H: " << num_pts << " invalid pt: " << num_invalid << endl;

  return num_invalid*1.0/num_pts;  
}

float epipolarConstrain(const std::vector<cv::Point2f> &kpts1, const std::vector<cv::Point2f> &kpts2, const Eigen::Matrix3d& Mat_F, 
                        std::vector<uchar> &is_inlier, const float &Th_score, float &score, int &has_outlier, const float &Th_dist)
{
  if(kpts1.size() != kpts2.size()) 
    return 1.0;

  int num_pt = kpts1.size();
  if(is_inlier.size() != num_pt)
  {
      is_inlier.clear();
      is_inlier.resize(num_pt,0);
  }

  has_outlier = 0;
  bool valid_F = true;
  int num_invalid = 0;
  cv::Point2f pt1, pt2;
  
  float F00 = Mat_F(0,0);
  float F01 = Mat_F(0,1);
  float F02 = Mat_F(0,2);
  float F10 = Mat_F(1,0);
  float F11 = Mat_F(1,1);
  float F12 = Mat_F(1,2);
  float F20 = Mat_F(2,0);
  float F21 = Mat_F(2,1);
  float F22 = Mat_F(2,2);

  // 注意，Mat_F应该是由R12和t12构成的，其中R12表示从上一帧变换到当前帧的旋转矩阵。
  // 因此，Mat_F形成的极线约束为 x2^t*Mat_F*x1 = 0。
  for(int i = 0; i < kpts1.size(); ++i)
  {
      pt1 = kpts1[i];
      pt2 = kpts2[i];
      
      // 上一帧的点pt1在当前帧中的极线 l2 = Mat_F*x1 = (a, b, c)
      const float a = F00*pt1.x + F01*pt1.y + F02;
      const float b = F10*pt1.x + F11*pt1.y + F12;
      const float c = F20*pt1.x + F21*pt1.y + F22;

      const float den = a*a + b*b;

      // F矩阵无效，一般是因为相机位移t为0?
      if(den == 0) 
      {
          ++num_invalid;
          has_outlier = 1;
          continue;
      }
      
      // 当前帧的点pt2在上一帧中的极线 l1 = x2^t*Mat_F = (a_inv, a_inv, a_inv)
      const float a_inv = pt2.x*F00 + pt2.y*F10 + F20;
      const float b_inv = pt2.x*F01 + pt2.y*F11 + F21;
      const float c_inv = pt2.x*F02 + pt2.y*F12 + F22;
      
      const float den_inv = a_inv*a_inv + b_inv*b_inv;

      if(den_inv == 0) 
      {
        ++num_invalid;
        has_outlier = 1;
        continue;
      }
      
      const float num = a*pt2.x + b*pt2.y + c;
      // 点到线的距离 的平方
      float dsqr = num*num/den;

      const float num_inv = a_inv*pt1.x + b_inv*pt1.y + c_inv;
      float dsqr_inv = num_inv*num_inv/den_inv;
      
      // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
      // 由于估计的E也不是特别准确，因此这里对于剩下的点，适当放宽其到极线距离的要求。反向
      if(dsqr <= (1.05*1.05)*Th_dist && dsqr_inv <= (1.05*1.05)*Th_dist) 
      {
        is_inlier[i] = 1;
        // Th_score的值一般要比Th_dist大，即对对极约束的要求更高
        score += (Th_score - dsqr);
        score += (Th_score - dsqr_inv);
      }
      else
      {
        // if(dsqr > Th_dist && dsqr_inv > Th_dist)
        //   std::cout << "both forward and inv dist invalid!" << std::endl;
        // else if(dsqr > Th_dist)
        //   std::cout << "forward dist invalid!" << std::endl;
        // else
        //   std::cout << "inv dist invalid!" << std::endl;
        
        // std::cout << "forward dist: " << dsqr << " inv dist: " << dsqr_inv << std::endl;

        has_outlier = 1;

        ++num_invalid;
      }
  }

  // std::cout << "total checked pt for F: " << num_pt << " invalid pt: " << num_invalid << std::endl;
  
  return num_invalid*1.0/num_pt;
}

double reprojectionError(const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Pi, const Eigen::Matrix3d &rici, const Eigen::Vector3d &tici,
                        const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Pj, const Eigen::Matrix3d &ricj, const Eigen::Vector3d &ticj, 
                        double depth, const Eigen::Vector3d &uvi, const Eigen::Vector3d &uvj)
{
  Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
  Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
  Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}


float cal_ave_epi_line_dist_pt(const Eigen::Matrix3d &Mat_F, const Eigen::Vector3d &pt1, const Eigen::Vector3d &pt2)
{
    float ave_dist = 0.0;
    float F00 = Mat_F(0,0);
    float F01 = Mat_F(0,1);
    float F02 = Mat_F(0,2);
    float F10 = Mat_F(1,0);
    float F11 = Mat_F(1,1);
    float F12 = Mat_F(1,2);
    float F20 = Mat_F(2,0);
    float F21 = Mat_F(2,1);
    float F22 = Mat_F(2,2);

    const float a = F00*pt1(0) + F01*pt1(1) + F02;
    const float b = F10*pt1(0) + F11*pt1(1) + F12;
    const float c = F20*pt1(0) + F21*pt1(1) + F22;

    const float den = a*a + b*b;
    if(den == 0) 
    {
        return -1.0;
    }

    const float a_inv = pt2(0)*F00 + pt2(1)*F10 + F20;
    const float b_inv = pt2(0)*F01 + pt2(1)*F11 + F21;
    const float c_inv = pt2(0)*F02 + pt2(1)*F12 + F22;
    
    const float den_inv = a_inv*a_inv + b_inv*b_inv;
    if(den_inv == 0) 
    {
        return -1.0;
    }

    const float num = a*pt2(0) + b*pt2(1) + c;
    // 点到线的距离
    float dist = sqrt(num*num/den);
    ave_dist += dist;

    const float num_inv = a_inv*pt1(0) + b_inv*pt1(1) + c_inv;
    float dist_inv = sqrt(num_inv*num_inv/den_inv);
    ave_dist += dist_inv;

    ave_dist = ave_dist/2.0;
}