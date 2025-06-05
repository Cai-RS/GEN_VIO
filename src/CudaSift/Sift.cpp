/*******************************************************
 * Copyright (C) 2023, BMSTU
 * 
 * This file is part of OVIO.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "Sift.h"

Sift::Sift(const SiftConfig &sift_config, cudaStream_t stream_common, cudaStream_t stream_cpy)
  :sift_config_(sift_config){
  
  if (stream_common){
    stream_  = stream_common;
    common_stream_infer_ = true;
  }

  if (use_multi_stream_ && stream_cpy){
    stream_cpy_ = stream_cpy;
    common_stream_cpy_ = true;
  }

  if (stream_ == nullptr)
  {
    //checkCudaRuntime(cudaStreamCreate(&stream_));
    checkCudaRuntime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    // 上面的checkCudaRuntime就会对结果进行检查和异常通知
    // if(stream_ == nullptr) 
    // {
    //   abort();
    // }
  }

  if (use_multi_stream_ && stream_cpy_ == nullptr){
    // checkCudaRuntime(cudaStreamCreate(&stream_cpy_));
    checkCudaRuntime(cudaStreamCreateWithFlags(&stream_cpy_, cudaStreamNonBlocking));
    // if(stream_cpy_ == nullptr) abort();
  }

  // 为了安全起见，这里其实不应该用sift_config_中给定的宽度和高度来赋值img_W和img_H，而是应该用当前数据集的图像的实际尺寸，这对于yolov8等模型也一样。
  img_W = sift_config_.inputW;
  img_H = sift_config_.inputH;
}

Sift::~Sift(){
  if (!common_stream_cpy_ && use_multi_stream_){
    cudaStreamDestroy(stream_cpy_);
  }
  else{
    stream_cpy_ = nullptr;
  }

  if (!common_stream_infer_){
    cudaStreamDestroy(stream_);
  }
  else{
    stream_ = nullptr;
  }

  if (!memory_sift_freed)
    deallocate();

}

void Sift::deallocate(){

  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
  FreeSiftData(siftData3);

  FreeSiftTempMemory(memoryTmp);
  memory_sift_freed = true;

  if(flow_dev_socre_id_match!=nullptr)
  {
    safeCall(cudaFree((void*)flow_dev_socre_id_match));
  }
  if(stereo_dev_socre_id_match!=nullptr)
  {
    safeCall(cudaFree((void*)stereo_dev_socre_id_match));
  }
  
  if(prev_stereo_match_info!=nullptr)
  {
    safeCall(cudaFreeHost((void*)prev_stereo_match_info));
  }
}

// 由于每两帧之间GPU都会有一定的空闲时间（用于CPU上的运动估计），则在这段时间内其实可以重新进行GPU上内存的分配！这样就可以节省一部分GPU内存！
void Sift::Allocate(int img_Width, int img_Height, bool host, int max_valid_num_match, bool use_masked_img)
{
  use_mask_img_for_sift_ = use_masked_img;
  // 当前数据集的图像真实尺寸
  img_W = img_Width;
  img_H = img_Height;
  // 需要在device上分配 三个 1280*384*sizeof(float)的内存来保存 预处理后的 上一时刻的左灰度图像、当前左灰度图像 和 当前右灰度图像。
  // 每一帧处理后的左右灰度图像 可以选择从 device上的DPStereo的预处理图像中得到（RGB三通道进行合并，但这需要在GPU上进行操作，会占用一定时间）；
  // 也可以先在CPU上进行灰度转换和padded，然后再传输到device上（使用不同的stream，这部分传输时间其实可以被网络推理所覆盖掉）！这里我们选择第二种！
  // pitch应该要乘以4（即float的字节数）？可以不用，它后续会被改变
  size_t width  = iAlignUp(img_W, 128);
  size_t height = iAlignUp(img_H, 32);
  // img1不需要开辟图像的device内存，它只需要device上的sift点检测和match信息的内存，以及host上的对应内存
  img1.Allocate(width, height, width*4, host, NULL, NULL, max_valid_num_match, true, true, false);
  img2.Allocate(width, height, width*4, host, NULL, NULL, max_valid_num_match, true, false);
  img3.Allocate(width, height, width*4, host, NULL, NULL, max_valid_num_match, false);
  
  // limg_padded.data = (uchar*)img2.h_data;
  // rimg_padded.data = (uchar*)img3.h_data;

  // 预分配host和device上检测到的SIFT点的内存
  int num_reserve = 8192; //32768  16384  8192
  // 只在device上分配用于保存检测的SIFT特征点的内存。对于右图像在host上不保存原始检测点，只在device上使用它们
  InitSiftData(siftData1, num_reserve, true, true); 
  InitSiftData(siftData2, num_reserve, true, true);
  InitSiftData(siftData3, num_reserve, false, true);

  if(use_masked_img)
  {
    img4.Allocate(width, height, width*4, host, NULL, NULL, 512, true, true, false);
    img5.Allocate(width, height, width*4, host, NULL, NULL, 512, true, false);
    // 都是左图像
    InitSiftData(siftData4, num_reserve/4, true, true);
    InitSiftData(siftData5, num_reserve/4, true, true);
  }

  // 这个中间变量的内存空间可以被多个检测和匹配所共用吧？
  // 这个内存是用于放置所有检测点，因为在检测的时候没有限制检测数量，那么最大的检测数量就是金字塔所有层的大小的总和，这里直接用5个原始图像的大小
  memoryTmp = AllocSiftTempMemory(width, height, 5, false);

  memory_sift_freed = false;
  // 用于保存前后2帧匹配点的score和相应点的id，这样在排序时就不需要每次都复制整个siftpoint的所有值
  safeCall(cudaMalloc((void**)&flow_dev_socre_id_match, sizeof(float)*2*num_reserve));
  safeCall(cudaMalloc((void**)&stereo_dev_socre_id_match, sizeof(float)*2*num_reserve));
  
  safeCall(cudaMallocHost((void**)&prev_stereo_match_info, sizeof(float)*5*num_reserve));
}

void Sift::preprocess_input(const Mat &image_left, const Mat &image_right, void* host_mask_objs, int mask_img_W, int mask_img_H, int left, int top)
{
  Mat raw_gray_img_L, raw_gray_img_R;

  // int raw_W = image_left.cols;
  // int raw_H = image_left.rows;

  // raw_gray_img_L = image_left.clone() * 1./255;
  // raw_gray_img_R = image_right.clone() * 1./255;
  // cvtColor(raw_gray_img_L, raw_gray_img_L, COLOR_BGR2GRAY, 1);
  // cvtColor(raw_gray_img_R, raw_gray_img_R, COLOR_BGR2GRAY, 1);

  // cvtColor(image_left, raw_gray_img_L, COLOR_BGR2GRAY);
  // cvtColor(image_right, raw_gray_img_R, COLOR_BGR2GRAY);

  // cout << "Succeeded convert raw image to gray!" << endl;
  
  raw_img_W = image_left.cols;
  raw_img_H = image_left.rows;
  // 如果只是改变数据类型的话，则目标Mat可以是原Mat,因为不涉及多个像素之间的线性变换；否则，目标矩阵和原矩阵应该不同
  image_left.convertTo(raw_gray_img_L, CV_32FC1);
  image_right.convertTo(raw_gray_img_R, CV_32FC1);
  if(raw_gray_img_R.type() != CV_32FC1) cout << "mat type is wrong!" << endl;
  
  // cout << "Succeeded convert raw image to CV_32FC1!" << endl;

  // cout << "Start pad raw image!" << endl;

  pad(raw_gray_img_L, limg_padded, Padder);

  pad(raw_gray_img_R, rimg_padded, Padder);

  // cout << "Succeeded pad raw image!" << endl;

  img2.h_data = (float*)limg_padded.data;
  img3.h_data = (float*)rimg_padded.data;

  // 也可以使用默认stream以阻塞host线程，因为sift的预处理之后紧接着就是gpu上sift的检测和匹配
  img2.Download(); 
  img3.Download(); 
  //img2.Download(stream_cpy_); 
  //img3.Download(stream_cpy_); 

  // safeCall(cudaDeviceSynchronize());
  siftData1.valid_disp_x.clear();
  siftData1.valid_disp_y.clear();
  siftData2.valid_disp_x.clear();
  siftData2.valid_disp_y.clear();

  if(use_mask_img_for_sift_ && host_mask_objs != nullptr)
  {
    int size_masked_img = mask_img_W * mask_img_H;
    uchar* ptr_mask = (uchar*)host_mask_objs + 3*size_masked_img;
    // cout << "Start copy data to padddd seg map!" << endl;
    Mat padded_full_mask_map = Mat(mask_img_H, mask_img_W, CV_8UC1, ptr_mask);

    if (left == 0 && top == 0 && mask_img_W == img_W && mask_img_H == img_H)
      raw_mask_map = padded_full_mask_map.clone();
    else
    {
      Range x_range(left, left + img_W);
      Range y_range(top, top + img_H);
      raw_mask_map = padded_full_mask_map(y_range, x_range).clone();
    }
    raw_gray_img_L.setTo(0, raw_mask_map);
    pad(raw_gray_img_L, masked_limg_padded, Padder);
    img5.h_data = (float*)masked_limg_padded.data;
    img5.Download(); 
    // safeCall(cudaDeviceSynchronize());

    siftData4.valid_disp_x.clear();
    siftData4.valid_disp_y.clear();
    siftData5.valid_disp_x.clear();
    siftData5.valid_disp_y.clear();
  }
}

void Sift::detect_match_sift(bool first_frame, bool use_masked_img)
{
  // 这个应该要放在预先分配内存的Allocate函数中？
  // float *memoryTmp = AllocSiftTempMemory(img_W, img_H, 5, false);
  // 多轮用于统计平均耗时
  int iter = 1;
  float time = 0;
  TicToc t_r; 

  for(int i=0; i<iter; i++) 
  {
    //cout << "start extract sift in image!" << endl;
    ExtractSift(siftData2, img2, 5, sift_config_.initBlur, sift_config_.thres_extract_flow, 0.0f, false, memoryTmp);
    //cout << "succeeded extract sift in left image!" << endl;
    ExtractSift(siftData3, img3, 5, sift_config_.initBlur, sift_config_.thres_extract_stereo, 0.0f, false, memoryTmp);
    //cout << "succeeded extract sift in right image!" << endl;

    if(use_mask_img_for_sift_)
    {
      ExtractSift(siftData5, img5, 5, sift_config_.initBlur, sift_config_.thres_extract_stereo, 0.0f, false, memoryTmp);
    }
  }
  // FreeSiftTempMemory(memoryTmp);

  // Match Sift features and find a homography
  // 多轮匹配会有利于提高匹配精度吗？没有，因为前后2轮匹配之间没有什么关系，这里多轮匹配只是用于统计平均每轮的时间
  for (int i=0; i<iter; i++)
  {
    if (!first_frame && siftData1.numPts > 0) 
    {
      // cout << "Start flow matching!" << endl;
      flow_sort_pt_ambi.resize(siftData1.numPts, 0);
      flow_sort_pt_id.resize(siftData1.numPts, 0);
      std::iota(flow_sort_pt_id.begin(),flow_sort_pt_id.end(),0);
      MatchSiftData(siftData1, siftData2, &flow_sort_pt_ambi[0], &flow_sort_pt_id[0], flow_dev_socre_id_match);
      
      // safeCall(cudaDeviceSynchronize());
      float *h1_ptr = &siftData1.h_data[0].xpos;
      float *d1_ptr = &siftData1.d_data[0].xpos;
      // 先将flow匹配结果从device复制到host，以供后处理使用。这里只需要复制x_pos和y_pos，其他需要的结果信息已经在MatchSiftData中完成了
      // 检测点数可能超过了
      safeCall(cudaMemcpy2D(h1_ptr, sizeof(SiftPoint), d1_ptr, sizeof(SiftPoint), 2*sizeof(float), siftData1.numPts, cudaMemcpyDeviceToHost));
      //safeCall(cudaDeviceSynchronize());
    }
    // 然后把siftData2中的检测结果复制到siftData1的device中，以便下一帧使用。这个无论是不是首帧都需要做
    safeCall(cudaMemcpy(&siftData1.d_data[0].xpos, &siftData2.d_data[0].xpos, sizeof(SiftPoint)*siftData2.numPts, cudaMemcpyDeviceToDevice));
    siftData1.temp_numPts = siftData2.numPts;
    // device到device的数据复制应该是默认不阻塞host的，因此要手动阻塞
    safeCall(cudaDeviceSynchronize());

    // 原始左右图像之间的match
    stereo_sort_pt_ambi.resize(siftData2.numPts, 0);
    stereo_sort_pt_id.resize(siftData2.numPts, 0);
    std::iota(stereo_sort_pt_id.begin(),stereo_sort_pt_id.end(),0);
    MatchSiftData(siftData2,siftData3,&stereo_sort_pt_ambi[0],&stereo_sort_pt_id[0],stereo_dev_socre_id_match);

    if (siftData2.h_data!=NULL) 
    {
      float *h2_ptr = &siftData2.h_data[0].xpos;
      float *d2_ptr = &siftData2.d_data[0].xpos;
      // 将stereo匹配结果从device复制到host
      safeCall(cudaMemcpy2D(h2_ptr, sizeof(SiftPoint), d2_ptr, sizeof(SiftPoint), 2*sizeof(float), siftData2.numPts, cudaMemcpyDeviceToHost));
      //cout << "num detected in left iamge: " << siftData2.numPts << endl;
    }
    
    if(use_mask_img_for_sift_)
    {
      if(!first_frame) 
      {
        MatchSiftData(siftData4, siftData5);
        // safeCall(cudaDeviceSynchronize());
        float *h1_ptr = &siftData5.h_data[0].xpos;
        float *d1_ptr = &siftData5.d_data[0].xpos;
        safeCall(cudaMemcpy2D(h1_ptr, sizeof(SiftPoint), d1_ptr, sizeof(SiftPoint), 2*sizeof(float), siftData4.numPts, cudaMemcpyDeviceToHost));
        //safeCall(cudaDeviceSynchronize());
      }

      // device到device的数据复制应该是默认不阻塞host的，因此下面要手动阻塞
      safeCall(cudaMemcpy(&siftData4.d_data[0].xpos, &siftData5.d_data[0].xpos, sizeof(SiftPoint)*siftData5.numPts, cudaMemcpyDeviceToDevice));
      siftData4.temp_numPts = siftData5.numPts;
      safeCall(cudaDeviceSynchronize());

      // padded的左图像与未pad的右图像之间的match
      MatchSiftData(siftData5, siftData3);
      if (siftData5.h_data!=NULL) 
      {
        float *h2_ptr = &siftData5.h_data[0].xpos;
        float *d2_ptr = &siftData5.d_data[0].xpos;
        // 将stereo匹配结果从device复制到host
        safeCall(cudaMemcpy2D(h2_ptr, sizeof(SiftPoint), d2_ptr, sizeof(SiftPoint), 2*sizeof(float), siftData5.numPts, cudaMemcpyDeviceToHost));
        //cout << "num detected in left iamge: " << siftData2.numPts << endl;
      }
    }
  }
  time = time + t_r.toc();
  std::cout << "Sift detecting and matching on GPU for " << iter << " iteration totally cost: " << time << "ms" << std::endl;
}

// SIFT的matching的过滤主要是在CPU上进行的，但是这部分的操作时间可以被flow的infer所覆盖
// max_disp_y是预计的当前帧所考虑的深度范围内的点 其左右两帧在v方向上的最大差值（l_v - r_v），这与立体校正后的参数有关，即左图像的点总是要比右图像的点要高一点（v坐标要小一点），因此这里的max_disp_y值总是为负的
// 是否也应给定各个跟踪点的flow_x的范围？以及各个当前帧左右点的disp_x？这部分留到CPU选择sift匹配时进行，因为这里暂时不好区分当前帧的跟踪点和新检测点
void Sift::postprocess(float max_disp_y, bool first_frame, bool use_masked_img)
{
  TicToc t_r;
  // 将ExtractSift和MatchSiftData中的host-device数据传输操作转移到上面的检测函数中完成了！

  // 似乎没必要将所有检测到的sift特征点传输到host中，只需要将符合match条件的sift点的信息传输会host即可！
  // if (siftData2.h_data)
  //   safeCall(cudaMemcpy(siftData2.h_data, siftData2.d_data, sizeof(SiftPoint)*siftData2.numPts, cudaMemcpyDeviceToHost));

  // if (siftData3.h_data)
  //   safeCall(cudaMemcpy(siftData3.h_data, siftData3.d_data, sizeof(SiftPoint)*siftData3.numPts, cudaMemcpyDeviceToHost));

  // 将所有device-host的数据传输操作都放在后处理中，这样就可以将这部分时间被别的gpu推理任务所覆盖！
  // if (!first_frame && siftData1.h_data!=NULL) 
  // {
  //   // 需要从device复制到host的每个点的信息是 xpos, ypos，然后(score, ambiguity, match, match_xpos, match_ypos)这五个为该点的匹配结果
  //   // 其中后面5个结果已经在Match函数中就复制到host对应位置了。这里只需要再复制xpos和ypos
  //   float *h1_ptr = &siftData1.h_data[0].xpos;
  //   // device上的内存也可以按自定义struct的指针来索引吗？
  //   float *d1_ptr = &siftData1.d_data[0].xpos;
  //   // 将每个点的这5个match结果复制到host中.此传输操作使用默认stream，会阻塞host线程。
    
  //   // 但是ypos到score之间还有4个float数，虽然这些不是需要的，但是也得一起复制！
  //   // 如果检测点数多的话，把检测点全部传输下来也是挺费时间的！
  //   safeCall(cudaMemcpy2D(h1_ptr, sizeof(SiftPoint), d1_ptr, sizeof(SiftPoint), 2*sizeof(float), siftData1.numPts, cudaMemcpyDeviceToHost));
  // }
  
  // 查看还有多少显存
  // size_t gpu_total_size, gpu_free_size;
  // cudaError_t cuda_status = cudaMemGetInfo(&gpu_free_size,&gpu_total_size);
  // if(cuda_status != cudaSuccess) cout << "Error: cudaMenGetInfo fails!" << endl;
  // cout << "剩余显存：" << double(gpu_free_size)/(1024.0*1024.0) << endl;

  int numPts = 0;
  float maxAmbiguity = 0.0;
  float thres_dist_match = 0.0;
  float thres_dist_match_x = 0.0;
  float thres_dist_match_y = 0.0;
  float max_disp_y_ = 0.0;
  int max_valid = 0;
  int numValid = 0;
  int *validPts;
  float minScore = sift_config_.minScore;
  float x_pos, y_pos, match_xpos, match_ypos;
  for(int k = 0; k < 2; k++)
  {
    SiftData* pts_data[3];
    if(k == 0)
    {
      pts_data[0] = &siftData1;
      pts_data[1] = &siftData2;
      pts_data[2] = &siftData3;
    }
    else if(use_masked_img)
    {
      pts_data[0] = &siftData4;
      pts_data[1] = &siftData5;
      pts_data[2] = &siftData3;
    }
    else
      break;
    
    int num_match_img = 2;
    if(first_frame) num_match_img = 1;

    for(int j=0; j<num_match_img; j++)
    {
      numValid = 0;
      
      //首帧没有前后帧的光流匹配
      if (j == 0 && !first_frame)
      {
        numPts = pts_data[j]->numPts;
        if(k == 0)
        {
          max_valid = img1.max_valid;
          // 这个host上的内存要提前分配
          validPts = img1.h_matching_pts_flow;
        }
        else
        {
          max_valid = img4.max_valid;
          validPts = img4.h_matching_pts_flow;
        }
        
        maxAmbiguity = sift_config_.thres_Ambiguity_flow;
        thres_dist_match = sift_config_.thres_dist_match_flow;
        thres_dist_match_x = sift_config_.thres_dist_match_x_flow;
        thres_dist_match_y = sift_config_.thres_dist_match_y_flow;
        max_disp_y_ = 0.0;
      }
      else if (first_frame || j!= 0)
      {
        // 对于首帧，只查看siftdata2与siftdata3的检测和匹配结果
        if (first_frame) ++j;
        numPts = pts_data[j]->numPts;
        if(k == 0)
        {
          max_valid = img2.max_valid;
          // 这个host上的内存要提前分配
          validPts = img2.h_matching_pts_stereo;
        }
        else
        {
          max_valid = img5.max_valid;
          validPts = img5.h_matching_pts_stereo;
        }
        
        maxAmbiguity = sift_config_.thres_Ambiguity_stereo;
        thres_dist_match = sift_config_.thres_dist_match_stereo;
        thres_dist_match_x = sift_config_.thres_dist_match_x_stereo;
        thres_dist_match_y = sift_config_.thres_dist_match_y_stereo;
        max_disp_y_ = max_disp_y;
      }

      //safeCall(cudaDeviceSynchronize());

      // 排除掉匹配score不满足要求的匹配。
      // 可以在GPU上完成此操作。但是点数不够多，而且数据传输本身也需要时间；另外GPU的使用需要排队（因为网络模型的推理比较占空间，实际上多stream任务无法并行）！
      
      for(int i=0; i<numPts && numValid < max_valid; ++i) 
      {
        // 这里视差和光流的定义都是 前一图像（或左图像）点坐标 - 后一图像（或右图像）点坐标，光流的定义与一般的光流网络的相反（后一帧点-前一帧点）
        x_pos = pts_data[j]->h_data[i].xpos;
        y_pos = pts_data[j]->h_data[i].ypos;
        match_xpos = pts_data[j]->h_data[i].match_xpos; 
        match_ypos = pts_data[j]->h_data[i].match_ypos;
        float disp_match_x = abs(x_pos - match_xpos);
        float disp_match_y = abs(y_pos - match_ypos);
        
        // cout << disp_match_x << " " << disp_match_y << endl;
        // float dist_match = sqrt(disp_match_x*disp_match_x + disp_match_y*disp_match_y);
        // if (pts_data[j].h_data[i].score>minScore && pts_data[j].h_data[i].ambiguity<maxAmbiguity && dist_match<thres_dist_match)
        // if(pts_data[j]->h_data[i].score>minScore && pts_data[j]->h_data[i].ambiguity<maxAmbiguity && disp_match_x < thres_dist_match_x && abs(disp_match_y - max_disp_y_) < thres_dist_match_y)
        if(pts_data[j]->h_data[i].score>minScore && pts_data[j]->h_data[i].ambiguity<maxAmbiguity && disp_match_x < thres_dist_match_x && disp_match_y < thres_dist_match_y)
        {
          if((j == 1 || first_frame) && disp_match_x <= 0) continue;
          if(k == 1 && j == 0 && (is_border_mask(x_pos, y_pos) || is_border_mask(match_xpos, match_ypos))) continue;
          if(k == 1 && (j == 1 || first_frame) && is_border_mask(x_pos, y_pos)) continue;
          validPts[++numValid] = i;
          pts_data[j]->valid_disp_x.push_back(disp_match_x);
          pts_data[j]->valid_disp_y.push_back(disp_match_y);
        }
      }
      
      // cout << "vaild matching num: " << numValid << endl;
      validPts[0] = numValid;
      
      std::cout << "Number of original detected features in two images: " <<  pts_data[j]->numPts << " " << pts_data[j+1]->numPts << std::endl;
      //std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;
      
      std::cout << "Number of valid matching features: " << numValid << " " << 100.0f*numValid/pts_data[j]->numPts << "% " << std::endl;
    }
  }
  siftData1.numPts = siftData1.temp_numPts;
  if(use_masked_img) siftData4.numPts = siftData4.temp_numPts;
  
  std::cout<<"Sift filtering on CPU totally cost:"<<  t_r.toc() << "ms" << std::endl;
}

// 当前帧中的立体匹配点。
// 筛选已经排序过的stereo match。其实这是只针对物体点的
void Sift::select_stereo_matching(float max_disp_y)
{
  int numPts = 0;
  float maxAmbiguity = 0.0;
  float thres_dist = 0.0;
  float min_disp_x = 0.0;
  float max_disp_x = 0.0;
  float thres_dist_match_y = 0.0;
  float max_disp_y_ = 0.0;
  int max_valid = 0;
  int numValid = 0;
  int *validPts;
  float minScore = sift_config_.minScore;
  float x_pos, y_pos, match_xpos, match_ypos;

  numPts = siftData2.numPts;

  max_valid = img2.max_valid;
  // 这个host上的内存要提前分配
  validPts = img2.h_matching_pts_stereo;

  maxAmbiguity = sift_config_.thres_Ambiguity_stereo;

  thres_dist = sift_config_.thres_dist_match_stereo;
  min_disp_x = mbf/max(mThDepthObj,mThDepthBg);
  // max_disp_x = sift_config_.thres_dist_match_x_stereo;
  max_disp_x = mbf/1.5;
  thres_dist_match_y = sift_config_.thres_dist_match_y_stereo;

  max_disp_y_ = max_disp_y;

  float ambi, score, disp_match_x, disp_match_y;

  // 注意，检测点数 可能比 允许保留的匹配数 要多
  for(int i=0; i<numPts && numValid < max_valid; ++i) 
  {
    int id = stereo_sort_pt_id[i];
    // 这里视差和光流的定义都是 前一图像（或左图像）点坐标 - 后一图像（或右图像）点坐标，光流的定义与一般的光流网络的相反（后一帧点-前一帧点）
    x_pos = siftData2.h_data[id].xpos;
    y_pos = siftData2.h_data[id].ypos;
    match_xpos = siftData2.h_data[id].match_xpos; 
    match_ypos = siftData2.h_data[id].match_ypos;
    disp_match_x = x_pos - match_xpos;
    disp_match_y = y_pos - match_ypos;
    
    ambi = siftData2.h_data[id].ambiguity;
    score = siftData2.h_data[id].score;

    // 是否要考虑立体校正后y方向的差距？
    // if(score > minScore && ambi <= maxAmbiguity && disp_match_x >= min_disp_x && disp_match_x <= max_disp_x && abs(disp_match_y) < max_disp_y_)
    if(score > minScore && ambi <= maxAmbiguity && disp_match_x >= min_disp_x && disp_match_x <= max_disp_x && abs(disp_match_y) < thres_dist_match_y)
    {
      validPts[++numValid] = id;
      siftData2.valid_disp_x.push_back(disp_match_x);
      siftData2.valid_disp_y.push_back(disp_match_y);
    }

    if(ambi > maxAmbiguity) break;
  }
  
  // cout << "vaild matching num: " << numValid << endl;
  validPts[0] = numValid;
}

// 注意，这里提供的seg_map是上一帧左图像的！！
// todo: use_masked_img
void Sift::select_flow_matching(float max_disp_y, const Mat &seg_map_prev, const Mat &seg_map_cur, Mat &mask_bg_prev, int num_flow_pt_in_bloc[][6], int num_temp_flow_pt_in_bloc[][6], 
                            int num_long_track_FAST_in_bloc[][6], vector<Point2f> &FAST_prev, const vector<int> &cnt_tracked, vector<int> &temp_flow_pt_id, const Mat &prev_l_img, bool use_masked_img)
{
  Mat mask(raw_img_H, raw_img_W, CV_8UC1);
  mask.setTo(255);

  int numPts = 0;
  float maxAmbiguity = 0.0;
  float thres_dist_match = 0.0;
  float thres_dist_match_x = 0.0;
  float thres_dist_match_y = 0.0;

  int numValid = 0;
  int *validPts;
  float minScore = sift_config_.minScore;
  float x_pos, y_pos, match_xpos, match_ypos, disp_match_x, disp_match_y, dist_flow;

  // 首先处理前后2帧的匹配
  int num_pt_flow = flow_sort_pt_id.size();
  int id;
  float ambi, score, max_score;

  validPts = img1.h_matching_pts_flow;
  maxAmbiguity = sift_config_.thres_Ambiguity_flow;
  thres_dist_match = sift_config_.thres_dist_match_flow;
  thres_dist_match_x = sift_config_.thres_dist_match_x_flow;
  thres_dist_match_y = sift_config_.thres_dist_match_y_flow;
  
  // 注意，需保证该二维数组至少有6行
  for(int i = 0; i < 6; ++i)
  {
    for(int j = 0; j < 6; ++j)
    {
      num_flow_pt_in_bloc[i][j] = 0;
      num_long_track_FAST_in_bloc[i][j] = 0;
      num_temp_flow_pt_in_bloc[i][j] = 0;
    }
  }
  
  if(!prev_flow_pt_with_stereo.empty()) prev_flow_pt_with_stereo.clear();
  if(!invalid_stereo_match.empty()) invalid_stereo_match.clear();
  // 添加上一帧剩下的立体匹配点，包括背景或物体上的点
  if(!temp_flow_pt_id.empty()) temp_flow_pt_id.clear();

  float min_disp_x_stereo = mbf/mThDepthBg;
  float min_disp_x_stereo_obj = mbf/mThDepthObj;
  float min_disp_x;
  // ORB-SLAM2中说，40倍基线以内的深度都属于是近点。KITTI的基线大约是0.54m，因此近点的阈值是21m。这里取18m以内的点属于近点
  float disp_x_close_pt = mbf/20.0;
  // 超近点是12倍基线以内
  float disp_x_very_close = mbf/6.0;

  float thres_stereo_match_disp_x = 256;

  float border = 5;

  float match_xpos_r, match_ypos_r, disp_match_x_stereo, disp_match_y_stereo;

  bool draw_tracked_fea = false;
  if(draw_tracked_fea && prev_l_img.data != nullptr)
  {
    tracked_fea_prev_img = prev_l_img.clone();
    stereo_fea_prev_img = prev_l_img.clone();
  }
  
  // 限制图像上半部分的 没有立体匹配 或者 较远 的点的数量上限
  int num_far_pt_up_half_img = 20;
  // 限制图像上半部分的 有立体匹配 且 较近 的点的数量上限
  int num_close_pt_up_half_img = 15;

  bool is_far_pt = false;
  bool is_bg_track = false;
  // 特征点需要尽可能均匀分布
  for(int i = 0; i < num_pt_flow; ++i)
  {
    is_far_pt = false;
    is_bg_track = false;
    // 注意，检测点数 可能比 允许保留的匹配数 要多
    if(numValid >= img1.max_valid) break;
    id = flow_sort_pt_id[i];
    
    // 是否应该先查看所有匹配的最小和最大score？是否会有负的score？理论上是有的，只有当两个点的描述子都一样时，其score理论上应该接近最大（不一定最大，但是比其大的情况应该很小，所以要搭配ambi值来抉择）
    if(siftData1.h_data[id].score <= minScore) continue;

    ambi = flow_sort_pt_ambi[i];
    // 因为点已经根据ambi进行排序了，如果某个点的ambi大于最大阈值，是否立马停止选取匹配点？从实验来看，路面上的点很难跟踪，即使ambi再大也很难出现大量路面点
    if (ambi > maxAmbiguity) break;
    
    x_pos = siftData1.h_data[id].xpos;
    y_pos = siftData1.h_data[id].ypos;
    match_xpos = siftData1.h_data[id].match_xpos; 
    match_ypos = siftData1.h_data[id].match_ypos;
    
    if(x_pos > raw_img_W-border || y_pos > raw_img_H-border || match_xpos > raw_img_W-border || match_ypos > raw_img_H-border) continue;
    if(x_pos < border || y_pos < border || match_xpos < border || match_ypos < border) continue;
    
    disp_match_x = x_pos - match_xpos;
    disp_match_y = y_pos - match_ypos;

    dist_flow = disp_match_x * disp_match_x + disp_match_y * disp_match_y;
    // cout << disp_match_x << " " << disp_match_y << endl;
    // float dist_match = sqrt(disp_match_x*disp_match_x + disp_match_y*disp_match_y);
    // if (pts_data[j].h_data[i].score>minScore && pts_data[j].h_data[i].ambiguity<maxAmbiguity && dist_match<thres_dist_match)
    // if(pts_data[j]->h_data[i].score>minScore && pts_data[j]->h_data[i].ambiguity<maxAmbiguity && disp_match_x < thres_dist_match_x && abs(disp_match_y - max_disp_y_) < thres_dist_match_y)
    
    if(abs(disp_match_x) < thres_dist_match_x && abs(disp_match_y) < thres_dist_match_y)
    {
      // 只使用当前帧的分割图来判断点的信息？
      Vec2b info_pt_prev = seg_map_prev.at<uchar>(y_pos, x_pos);

      uchar cls_prev = info_pt_prev(0);
      // 某些类别的物体不考虑
      if(cls_prev == 1 || cls_prev == 2 || cls_prev == 4 || cls_prev == 7) continue;

      Vec2b info_pt_cur = seg_map_cur.at<uchar>(match_ypos, match_xpos);

      uchar cls_cur = info_pt_cur(0);

      if (cls_cur == 1 || cls_cur == 2 || cls_cur == 4 || cls_cur == 7) continue;

      int id_row_block = y_pos/60;
      // 图像顶部的点需要仔细挑选，因为一般距离比较远，且容易是树叶之类的区域，会存在密集的特征点和匹配，需要减小ambi的值且增大点之间的距离
      // 可以越往下的图像区域，点之间的间隔越小
      int id_col_block = x_pos/200;

      // 底部多出的部分全都归到第6行的框内
      if(id_row_block == 6) id_row_block = 5;
      if(id_col_block == 6) id_col_block = 5;
      
      int radi = 20;

      // 其实上一帧seg_map中检测cls为0的点不一定就是背景点，其在上一帧可能是被确定为漏检的物体点，后续要如何更改？
      if(cls_prev == 0 && cls_cur == 0)
      {
        min_disp_x = min_disp_x_stereo;
        is_bg_track = true;
      }
      else
        min_disp_x = min_disp_x_stereo_obj;

      float *prev_stereo_ptr = prev_stereo_match_info + 5 * id;
      bool has_valid_dep = false;
      // 对于flow匹配点，有立体匹配但是不代表是有效的，这在最终添加flow点时要进行检查，如果是无效的，则重新为其寻找立体匹配点
      if(prev_stereo_ptr[0] > minScore) 
      {
        match_xpos_r = prev_stereo_ptr[3]; 
        match_ypos_r = prev_stereo_ptr[4];

        disp_match_x_stereo = x_pos - match_xpos_r;
        disp_match_y_stereo = y_pos - match_ypos_r;
        // 置信度不高的立体匹配
        if(abs(disp_match_y_stereo) > fmax(0.6, max_disp_y) || prev_stereo_ptr[1] > 0.88)
          invalid_stereo_match.insert(id);
        else if(disp_match_x_stereo <= min_disp_x || disp_match_x_stereo > thres_stereo_match_disp_x)
        {
          // 太远或太近的跟踪点直接放弃
          invalid_stereo_match.insert(id);
          continue;
        }
        else
        {
          has_valid_dep = true;
        }
      }

      // 背景跟踪点
      if(is_bg_track)
      {
        // 对图像顶部1/2区域的背景点提高要求
        if(id_row_block < 3)
        {
          // 提高匹配阈值
          if(ambi > 0.85) continue;
          // 图像上半部分中不允许出现距离太小的光流匹配。当然有可能当前相机静止，则光流小的匹配留到图像下半部分中提取
          if(dist_flow <= Min_dist_flow * Min_dist_flow) continue;
          // 最上面一行只要最左和最右两列的点
          if(id_row_block == 0 && (id_col_block == 0 || id_col_block == 5))
          {
            // 远处的背景跟踪点，如果没有有效深度，则不保留；对于远处的物体跟踪点，是否要保留？也不保留。仅保留近处的点
            if(!has_valid_dep)
            {
              if(num_far_pt_up_half_img <= 0)
                continue;
              else
              {
                is_far_pt = true;
              }
            }
            else if(disp_match_x_stereo < disp_x_close_pt)
            {
              if(num_far_pt_up_half_img <= 0)
              {
                invalid_stereo_match.insert(id);
                // 这些比较远的跟踪，是否要保留？可以选择保留那些匹配置信度较高的点，但是实际上证明远处的点跟踪大概率是错误的，尤其是植物上的点，或砖墙上的点
                continue;
              }
              else
              {
                is_far_pt = true;
              }
            }
            else
            {
              if(num_close_pt_up_half_img <= 0)
              {
                invalid_stereo_match.insert(id);
                continue;
              }
            }
          }
          // 第二和第三行 去除中间2列的点，大概率是地面点
          else if((id_row_block == 1 || id_row_block == 2) && (id_col_block != 2 && id_col_block != 3))
          {
            if(!has_valid_dep)
            {
              if(num_far_pt_up_half_img <= 0)
                continue;
              else
                is_far_pt = true;
            }
            else if(disp_match_x_stereo < disp_x_close_pt)
            {
              if(num_far_pt_up_half_img <= 0)
              {
                invalid_stereo_match.insert(id);
                // 这些比较远的跟踪，是否要保留？可以保留那些匹配置信度较高的点
                continue;
              }
              else
              {
                is_far_pt = true;
              } 
            }
            else
            {
              if(num_close_pt_up_half_img <= 0)
              {
                invalid_stereo_match.insert(id);
                continue;
              }
            }
          }
          else
          {
            continue;
          }

          if(has_valid_dep && disp_match_x_stereo > disp_x_very_close)
            radi = 8;
          if(has_valid_dep && disp_match_x_stereo >= disp_x_close_pt) 
            radi = 15;
          else
            radi = 30 - 2*id_row_block;
        }
        else
        {
          // 图像下半部分
          radi -= (id_row_block * 2);
          // 对于第4行的背景点也限制其光流跟踪的大小
          if(id_row_block == 3)
          {
            // if(dist_flow <= Min_dist_flow * Min_dist_flow) 
            //   continue;

            if(has_valid_dep && disp_match_x_stereo > disp_x_close_pt) radi = 15;
          }

          if(has_valid_dep && disp_match_x_stereo > disp_x_very_close) radi = 10;
        }
      }
      
      // 如果是物体的关联点
      // 注意，物体点在上一帧不一定有sift立体匹配，但是其深度可能在上一帧通过depth_map获取，因此这里先不论其是否有立体匹配，均保留，留到后面再排除
      if(cls_prev != 0 || cls_cur != 0) radi = 8;

      if(mask.at<uchar>(y_pos, x_pos) != 0)
      {
        // int *ptr_pt =  select_pt + 15 * (6*id_row_block + id_col_block) + num_pt;
        // ptr_pt[0] = id;

        // 记录背景点（不包括前后2帧属于漏检物体的匹配点）
        if(is_bg_track) 
        {
          int num_pt = num_flow_pt_in_bloc[id_row_block][id_col_block];

          int num_thres = NUM_FEA_IN_BLOC;
          if(id_row_block < 3)
          {
            num_thres = 5;
            if(has_valid_dep && disp_match_x_stereo > disp_x_very_close)
              num_thres = (num_far_pt_up_half_img + num_close_pt_up_half_img)/3;
            else if(has_valid_dep && disp_match_x_stereo > disp_x_close_pt)
              num_thres = (num_far_pt_up_half_img + num_close_pt_up_half_img)/4;
          }
          else if (id_row_block == 3)
            num_thres = 7;
          
          if(num_pt >= num_thres) continue;

          num_flow_pt_in_bloc[id_row_block][id_col_block] += 1;

          if(id_row_block < 3)
          {
            if(is_far_pt)
              --num_far_pt_up_half_img;
            else
              --num_close_pt_up_half_img;
          }
        }

        validPts[++numValid] = id;
        siftData1.valid_disp_x.push_back(disp_match_x);
        siftData1.valid_disp_y.push_back(disp_match_y);
        
        Point2f pt(x_pos, y_pos);
        circle(mask, pt, radi, 0, -1);
        // 专门标注背景点和背景区域的漏检物体点
        if(cls_prev == 0)
          circle(mask_bg_prev, pt, radi, 0, -1);
        
        if(has_valid_dep) 
        {
          prev_flow_pt_with_stereo.insert(id);
          // 记录有立体匹配的上一帧跟踪点。方便之后查询
          temp_flow_pt_id.push_back(id);
        }

        // 用于在彩色图像中显示各个被跟踪的点。BGR三色道，这里应该是红色
        if(draw_tracked_fea)
        {
          circle(tracked_fea_prev_img, pt, 4, Scalar(0,0,255), 1, 16);

          // 标记所有在上一帧中的立体匹配点.红色
          if(has_valid_dep)
            circle(stereo_fea_prev_img, pt, 4, Scalar(0,0,255), 1, 16);
        }
      }
    }
  }
  validPts[0] = numValid;

  // 在mask_bg_prev中标注上一帧就已经被跟踪的FAST点，这些点可以形成多帧跟踪。
  // FAST点只会在图像的下2/3区域，保证点不会太远
  if(!FAST_prev.empty())
  {
    for(int i = 0; i < FAST_prev.size(); ++i)
    {
      // 上一帧的新点
      if(cnt_tracked[i] == 1) continue;
      Point2f &pt = FAST_prev[i];
      x_pos = pt.x;
      y_pos = pt.y;
      // 只会标注背景点，和背景区域中未被sift跟踪点占据的区域
      if(mask_bg_prev.at<uchar>(y_pos, x_pos) == 0) continue;

      int id_row_block = y_pos/60;
      // 图像顶部的点需要仔细挑选，因为一般距离比较远，且容易是树叶之类的区域，会存在密集的特征点和匹配，需要减小ambi的值且增大点之间的距离
      // 可以越往下的图像区域，点之间的间隔越小
      int id_col_block = x_pos/200;
      int radi = 8;
      if(id_col_block == 6) id_col_block = 5;
      if(id_row_block == 6) id_row_block = 5;

      // Vec2b info_pt_prev = seg_map_prev.at<uchar>(y_pos, x_pos);

      // 对于背景FAST点而言，只在下2/3图像区域检测新的特征点
      // 但是上一帧部分跟踪的FAST点还可能是来自于上上帧的sift点（仅有stereo匹配的sift点被归为FAST点并尝试用金字塔光流寻找前后帧匹配）
      // if(info_pt_prev(0) == 0)
      //   assert(id_row_block >= 2);
      
      int num_flow = num_flow_pt_in_bloc[id_row_block][id_col_block];
      int num_pt = num_flow + num_long_track_FAST_in_bloc[id_row_block][id_col_block];

      int num_thres = (NUM_FEA_IN_BLOC-2);
      if(id_row_block < 3)
      {
        num_thres = (num_far_pt_up_half_img + num_close_pt_up_half_img)/5;
      }
      else if (id_row_block == 3)
        num_thres = 5;
      
      if(num_pt >= num_thres) continue;

      num_long_track_FAST_in_bloc[id_row_block][id_col_block] += 1;

      // 因为不知道这些点之后能否被跟踪到，因此将它们的半径画小一点
      circle(mask_bg_prev, pt, radi, 0, -1);

      // BGR. 绿色
      if(draw_tracked_fea)
      {
        circle(tracked_fea_prev_img, pt, 5, Scalar(0,255,0), 1, 16);

        // 上一帧保留的FAST背景点一定有立体匹配吗？不一定，但应该会有有效的深度值（如通过运动变换的投影）。如果没有有效深度，倾向于在上一帧就直接将其去除
        // circle(stereo_fea_prev_img, pt, 3, Scalar(0,0,255), 1, 16);
      }
    }
  }

  // 对于没有sift跟踪的上一帧背景点，限制其最大深度为20m
  min_disp_x_stereo = mbf/20.0;
  // 这个边界值要不小于下面画的点的最大半径
  border = 20;
  
  // 遍历上一帧的左右匹配点。如果是上面flow跟踪的点，则一律保留；剩下的则根据ambi从小到大保留，并且要均匀分布
  validPts = img1.h_matching_pts_stereo;
  maxAmbiguity = sift_config_.thres_Ambiguity_stereo;
  thres_dist_match = sift_config_.thres_dist_match_stereo;
  thres_dist_match_x = sift_config_.thres_dist_match_x_stereo;
  thres_dist_match_y = sift_config_.thres_dist_match_y_stereo;

  int num_stereo_prev = prev_stereo_sort_pt_id.size();
  assert(num_stereo_prev == num_pt_flow && "something wrong here!");

  for(int i = 0; i < num_stereo_prev; ++i)
  {
    is_far_pt = false;
    int id = prev_stereo_sort_pt_id[i];
    // 属于有效的跟踪点的立体匹配，不作为待跟踪点
    if(prev_flow_pt_with_stereo.find(id) != prev_flow_pt_with_stereo.end()) continue;
    // 属于无效的立体匹配（太远或太近），也不作为待跟踪点（因为认为光流金字塔的跟踪精度没有sift高）
    if(invalid_stereo_match.find(id) != invalid_stereo_match.end()) continue;

    x_pos = siftData1.h_data[id].xpos;
    y_pos = siftData1.h_data[id].ypos;

    if(x_pos > raw_img_W-border || y_pos > raw_img_H-border) continue;
    if(x_pos < border || y_pos < border) continue;

    // 只要背景区域，且没被占用的点。对于物体点，只要那些有前后2帧跟踪的点，有立体匹配的新物体点在上一帧时已经保存，这里不必再次保存
    if(mask_bg_prev.at<uchar>(y_pos, x_pos) == 0) continue;

    Vec2b info_pt_prev = seg_map_prev.at<uchar>(y_pos, x_pos);
    uchar cls_prev = info_pt_prev(0);
    if(cls_prev == 0)
      min_disp_x = min_disp_x_stereo;
    else
    {
      // 上一帧的物体点不要
      continue;
      // min_disp_x = min_disp_x_stereo_obj;
    }

    float *prev_info_ptr = prev_stereo_match_info + 5 * id;

    score = prev_info_ptr[0];
    if(score <= minScore) continue;

    match_xpos = prev_info_ptr[3]; 
    match_ypos = prev_info_ptr[4];

    disp_match_x = x_pos - match_xpos;
    disp_match_y = y_pos - match_ypos;

    // 只保留大于最小深度阈值，且小于18m的点作为待跟踪点
    if(disp_match_x <= min_disp_x || disp_match_x > thres_dist_match_x || abs(disp_match_y) > fmax(0.6, max_disp_y)) 
      continue;

    // dist_stereo = disp_match_x * disp_match_x + disp_match_y * disp_match_y;

    int id_row_block = y_pos/60;
    // 图像顶部的点需要仔细挑选，因为一般距离比较远，且容易是树叶之类的区域，会存在密集的特征点和匹配，需要减小ambi的值且增大点之间的距离
    // 可以越往下的图像区域，点之间的间隔越小
    int id_col_block = x_pos/200;
    int radi = 20;
    ambi = prev_stereo_sort_pt_ambi[i];

    if(id_col_block == 6) id_col_block = 5;

    // if(id_row_block == 0)
    // {
    //   // 虽然上面已经限制了点的深度值，但是为防止立体匹配错误（尤其是树叶处及其影子）的影响，还是将这部分最可能出现远点的区域去除
    //   // 仅针对背景点
    //   // if(id_col_block != 0 && id_col_block != 5)
    //   {
    //     // 大于近点阈值的则放弃
    //     if(disp_match_x < disp_x_close_pt)
    //       continue;
    //   }
    //   // 提高阈值.stereo一般可以匹配的比较准
    //   if(ambi > 0.82) continue;

    //   radi = 6;
    // }
    // else if((id_row_block == 1 && (id_col_block == 2 || id_col_block == 3)) || id_row_block == 2)
    // {
    //   // 大于近点阈值的则放弃
    //   if(disp_match_x < disp_x_close_pt)
    //     continue;

    //   // 提高阈值.stereo一般可以匹配的比较准
    //   if(ambi > 0.83) continue;

    //   radi = 6;
    // }

    if(id_row_block < 3)
    {
      if(ambi >= 0.85) continue;

      if(id_row_block == 0) 
      {
        if(id_col_block != 0 && id_col_block != 5)
          continue;
      }
      else if(id_col_block == 2 || id_col_block == 3) 
        continue;

      radi -= (2 * id_row_block);
      if(disp_match_x < disp_x_close_pt)
      {
        if(num_far_pt_up_half_img <= 0)
          continue;
        else
        {
          is_far_pt = true;
        }
      }
      else
      {
        if(num_close_pt_up_half_img <= 0) continue;
        radi = 13;
        if(disp_match_x > disp_x_very_close) radi = 8;
      }
    }
    else
    {
      // 立体匹配点最大的ambi不能超过0.88。所有立体匹配已经根据ambi排序过了
      if(ambi > maxAmbiguity) break;
      // 底部多出的部分全都归到第6行的框内
      if(id_row_block == 6) id_row_block = 5;

      radi = 25;
      radi -= (2 * id_row_block);

      if(disp_match_x > disp_x_very_close) radi = 10;
    }

    if(cls_prev != 0) 
      radi = 7;
    else
    {
      int num_flow = num_flow_pt_in_bloc[id_row_block][id_col_block];
      int num_long_track = num_long_track_FAST_in_bloc[id_row_block][id_col_block];
      int num_pt = num_flow + num_long_track + num_temp_flow_pt_in_bloc[id_row_block][id_col_block];
      
      int num_thres = NUM_FEA_IN_BLOC;
      if(id_row_block < 3)
      {
        num_thres = 5;
        if(disp_match_x > disp_x_very_close)
          num_thres = (num_far_pt_up_half_img + num_close_pt_up_half_img)/3;
        else if(disp_match_x > disp_x_close_pt)
          num_thres = (num_far_pt_up_half_img + num_close_pt_up_half_img)/4;
      }
      else if (id_row_block == 3)
        num_thres = 8;
      
      if(num_pt >= num_thres) continue;
      num_temp_flow_pt_in_bloc[id_row_block][id_col_block] += 1;

      if(id_row_block < 3)
      {
        if(is_far_pt)
          --num_far_pt_up_half_img;
        else
          --num_close_pt_up_half_img;
      }
    }
    
    // 有立体匹配但是还没有跟踪的上一帧背景点，需要在后续用光流金字塔寻找前后帧匹配
    temp_flow_pt_id.push_back(id);
    
    Point2f pt(x_pos, y_pos);
    circle(mask_bg_prev, pt, radi, 0, -1);

    if(draw_tracked_fea)
    {
      // BGR. 蓝色，代表的是待跟踪的sift点
      circle(tracked_fea_prev_img, pt, 4, Scalar(255,0,0), 1, 16);
      // 绿色代表仅有立体匹配的sift点
      circle(stereo_fea_prev_img, pt, 4, Scalar(0,255,0), 1, 16);
    }
  }

  if(draw_tracked_fea)
  {
    while(true)
    {
      cv::imshow("original tracked fea in prev left image", tracked_fea_prev_img);
      // 一直等待用户按下ESC键（ASCI码为27）
      if(waitKey(0) == 27)
      {
          break;
      }
    }
    
    while(true)
    {
      cv::imshow("original fea with stereo match in prev left image", stereo_fea_prev_img);
      // 一直等待用户按下ESC键（ASCI码为27）
      if(waitKey(0) == 27)
      {
          break;
      }
    }
  }
}

// 将当前帧的stereo match的信息复制到prev_stereo_match_info中
void Sift::Postprocess()
{
  // temp_numPts是当前帧的img2中的sift检测点数，数据已经复制到data1中了，但是前面这个点数还没赋值
  siftData1.numPts = siftData1.temp_numPts;

  // float *d2_ptr = &siftData2.d_data[0].xpos;

  // 将当前帧已经根据ambi排好序的stereo match部分信息复制出来
  float *h2_ptr = &siftData2.h_data[0].score;
  safeCall(cudaMemcpy2D(prev_stereo_match_info, sizeof(float)*5, h2_ptr, sizeof(SiftPoint), 5*sizeof(float), siftData2.numPts, cudaMemcpyHostToHost));
  if(!prev_stereo_sort_pt_id.empty()) prev_stereo_sort_pt_id.clear();
  prev_stereo_sort_pt_id = stereo_sort_pt_id;
  if(!prev_stereo_sort_pt_ambi.empty()) prev_stereo_sort_pt_ambi.clear();
  prev_stereo_sort_pt_ambi = stereo_sort_pt_ambi;
}

// 需要对图像进行填充，以便可以被64整除
void Sift::pad(const cv::Mat &raw_img, cv::Mat &padded_img, int* Padder, const string mode)
{
    int h_raw  = raw_img.rows;
    int w_raw  = raw_img.cols;

    int pad_ht = ((h_raw / 32 + 1) * 32 - h_raw) % 32;
    int pad_wd = ((w_raw / 128 + 1) * 128 - w_raw) % 128;

    // 只在下侧和右侧进行填充，这样最后特征点的坐标系只需要判断是否会超出原图的大小范围即可！
    Padder[0]  = 0;
    Padder[1]  = pad_ht;
    Padder[2]  = 0;
    Padder[3]  = pad_wd;
    
    if (mode == "constant"){
        int borderType = cv::BORDER_CONSTANT;
        // cv::Scalar color = Scalar(0, 0, 0);
        float color = 0.0;
        copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType, color);
    }
    else{
        int borderType = cv::BORDER_REPLICATE;
        // cv::Scalar color = Scalar(114, 114, 114);
        copyMakeBorder(raw_img, padded_img, Padder[0], Padder[1], Padder[2], Padder[3], borderType);
        // cout << "Succeeded copy make border!" << endl;
    }
}

void Sift::unpad(const cv::Mat &padded_img, cv::Mat &raw_img, const int raw_w, const int raw_h, const int* Padder){
  cv::Mat temp_Mat = padded_img(Range(Padder[0], raw_h+Padder[0]), Range(Padder[2], raw_w+Padder[2]));
  raw_img = temp_Mat.clone();
}

bool Sift::is_border_mask(float x, float y)
{
  if(x < 5 || x > img_W - 5 || y < 6 || y > img_H -6)
    return true;
  else
  {
    uchar s1 = raw_mask_map.at<uchar>(y-2, x-2);
    uchar s2 = raw_mask_map.at<uchar>(y-2, x+2);
    uchar s3 = raw_mask_map.at<uchar>(y+2, x-2);
    uchar s4 = raw_mask_map.at<uchar>(y+2, x+2);
    return (s1*s2*s3*s4 == 0);
  }
}

void Sift::MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
#if 1
  homography[0] = homography[4] = -1.0f;
  homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
  homography[2] = 1279.0f;
  homography[5] = 959.0f;
#endif
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f) // 100.0
	found = true;
      if (err<100.0f || j==sift1[i].match) { // 100.0
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match) 
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
  std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
  std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
  std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void Sift::PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img, int * ValidPts, int NumValid)
{
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  //for (int j=0;j<numPts;j++) 
  for (int j=0;j<NumValid;j++){ 
    //int k = sift1[j].match;
    int k = sift1[ValidPts[j]].match;
    if (sift1[ValidPts[j]].match_error<5) {
      float dx = sift2[k].xpos - sift1[ValidPts[j]].xpos;
      float dy = sift2[k].ypos - sift1[ValidPts[j]].ypos;
#if 0
      if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	std::cout << "scale=" << sift1[j].scale << "  ";
	std::cout << "error=" << (int)sift1[j].match_error << "  ";
	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
#if 1
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[ValidPts[j]].xpos + dx*l/len);
	int y = (int)(sift1[ValidPts[j]].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
#endif
    }
    int x = (int)(sift1[ValidPts[j]].xpos+0.5);
    int y = (int)(sift1[ValidPts[j]].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[ValidPts[j]].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}
