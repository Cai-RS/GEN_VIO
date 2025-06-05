//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

void CudaImage::Allocate(int w, int h, int p, bool with_host, float *devmem, float *hostmem, int max_valid_, bool left_img_, bool flow, bool with_dev) 
{
  width = w;
  height = h; 
  pitch = p; 
  d_data = devmem;
  h_data = hostmem; 
  t_data = NULL; 
  // 最终有效的匹配点的最大数量
  max_valid = max_valid_;
  if (devmem==NULL && with_dev) {
    // pitch是根据需要分配的矩阵的width(in bytes)来确定的，在这里，其实pitch = sizeof(float)*width，而这个赋值是在cudaMallocPitch函数中进行！
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==NULL) 
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (with_host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }

  if (max_valid_>0 && left_img_)
  {
    // 0 for flow and 1 for stereo
    // +1 (the first int) for the actual num of valid matching
    if(flow) 
    {
      if(h_matching_pts_flow == NULL)
      {
        h_matching_pts_flow = (int *)malloc(sizeof(int)*(max_valid+1));
        h_Alloc_for_match = true;
      }
      // h_matching_pts_stereo = nullptr;
    }
    //safeCall(cudaMalloc((void **)&d_matching_pts[0], (size_t)(sizeof(int)*(max_valid+1))));
    else
    {
      if(h_matching_pts_stereo == NULL)
      {
        h_matching_pts_stereo = (int *)malloc(sizeof(int)*(max_valid+1));
        h_Alloc_for_match = true;
      }
      // h_matching_pts_flow = nullptr;
    }
    //safeCall(cudaMalloc((void **)&d_matching_pts[1], (size_t)(sizeof(int)*(max_valid+1))));
  }
}

CudaImage::CudaImage() : 
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), 
  h_internalAlloc(false),h_Alloc_for_match(false),h_matching_pts_flow(NULL),h_matching_pts_stereo(NULL)
{
  
}

CudaImage::~CudaImage(){
  if (d_internalAlloc && d_data!=NULL) 
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL) 
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL) 
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
  if (h_Alloc_for_match && h_matching_pts_flow!=NULL)
    free(h_matching_pts_flow);
  if (h_Alloc_for_match && h_matching_pts_stereo!=NULL)
    free(h_matching_pts_stereo);
}

double CudaImage::Download(const cudaStream_t stream_cpy)  
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL) 
    if (stream_cpy != nullptr)
      safeCall(cudaMemcpy2DAsync(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice, stream_cpy));
    else 
      safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::Readback()
{
  TimerGPU timer(0);
  int p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::InitTexture()
{
  TimerGPU timer(0);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height)); 
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
 
double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(0);
  if (host)
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float)*pitch*dst.height, cudaMemcpyHostToDevice));
  else
    safeCall(cudaMemcpyToArray((cudaArray *)dst.t_data, 0, 0, d_data, sizeof(float)*pitch*dst.height, cudaMemcpyDeviceToDevice));
  safeCall(cudaDeviceSynchronize());
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
