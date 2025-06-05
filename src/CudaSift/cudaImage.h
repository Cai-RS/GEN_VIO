//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H
#include <cuda_tools.hpp>

class CudaImage {
public:
  bool left_img;
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  // 用于保存该img中的sift point中有与其他图像的进行匹配的点的序号
  int max_valid;
  int *h_matching_pts_flow;
  int *h_matching_pts_stereo;
  //void *d_matching_pts[2];
  bool d_internalAlloc;
  bool h_internalAlloc;
  bool h_Alloc_for_match;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL, int max_valid = 0, bool left_img_ = true, bool flow = true, bool with_dev = true);
  double Download(const cudaStream_t stream_cpy = nullptr);
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

#endif // CUDAIMAGE_H
