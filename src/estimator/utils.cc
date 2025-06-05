
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

bool epipolarConstrain(const std::vector<cv::Point2f> &kpts1, const std::vector<cv::Point2f> &kpts2, const Eigen::Matrix3d& Mat_F, 
                        std::vector<uchar> &is_inlier, const float &Th_score, float &score, const float &Th_dist)
{
  if(kpts1.size() != kpts2.size()) return false;
    int num_pt = kpts1.size();
    if(is_inlier.size() != num_pt)
    {
        is_inlier.clear();
        is_inlier.resize(num_pt,0);
    }

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

    // 注意，Mat_F应该是有R12和t12构成的，其中R12表示从上一帧变换到当前帧的旋转矩阵。
    // 因此，Mat_F形成的极线约束为 x2t*Mat_F*x1 = 0。
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
            continue;
        }

        // 当前帧的点pt2在上一帧中的极线 l1 = x2t*Mat_F = (a_inv, a_inv, a_inv)
        const float a_inv = pt2.x*F00 + pt2.y*F10 + F20;
        const float b_inv = pt2.x*F01 + pt2.y*F11 + F21;
        const float c_inv = pt2.x*F02 + pt2.y*F12 + F22;
        
        const float den_inv = a_inv*a_inv + b_inv*b_inv;

        if(den_inv == 0) 
        {
            ++num_invalid;
            continue;
        }

        const float num = a*pt2.x + b*pt2.y + c;
        // 点到线的距离 的平方
        float dsqr = num*num/den;

        const float num_inv = a_inv*pt1.x + b_inv*pt1.y + c_inv;
        float dsqr_inv = num_inv*num_inv/den_inv;
        
        // 如何设定这个距离阈值？越近的点，其绝对匹配误差就会越大，但是相对地，相同的绝对误差所造成的深度或位移的估计绝对误差越小
        if(dsqr <= Th_dist && dsqr_inv<= Th_dist) 
        {
          is_inlier[i] = 1;
          // Th_score的值一般要比Th_dist大，即对对极约束的要求更高
          score += (Th_score - dsqr);
          score += (Th_score - dsqr_inv);
        }
    }
    
    // 点数足够多，且无效的点太多，则认为F矩阵无效
    if(num_pt >= 8 && num_invalid > 1.0/2 * num_pt) valid_F = false;
    
    return valid_F;
}