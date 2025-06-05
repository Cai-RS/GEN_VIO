#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include <vector>

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  // 上面6个量是此点自身的信息，下面6个量是此点的匹配信息
  float score;
  // ambiguity表示此点的第二高分匹配点与最高分匹配点的匹配分数之比，即 sec_score / (max_score + 1e-6f)
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  // 每个点寻找匹配度最高的128个初步匹配点？还是说是描述子？
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Final number of detected Sift points, it is limited by maxPts
  int temp_numPts;    // to save the value of "numPts" of matched img, it will be assigned to numPts after post-process
  int actual_numPts;  // Actual number of detected Sift points, it might be larger than maxPtx
  int maxPts;         // Number of allocated Sift points
  int numValid_match; // Number of valid matching of Sift points within numPts
  bool malloc_host;
  std::vector<float> valid_disp_x;
  std::vector<float> valid_disp_y;
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

void InitCuda(int devNum = 0);
float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp = false);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false, float *tempMemory = 0, const cudaStream_t stream_ = nullptr);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
// 其定义位于matching.cu文件的第1118行！
double MatchSiftData(SiftData &data1, SiftData &data2, float *sort_pt_ambi = nullptr, int *sort_pt_id = nullptr, float *dev_sort_pt = nullptr);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
