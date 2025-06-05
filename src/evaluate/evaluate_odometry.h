#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>

#include "mail.h"
#include "matrix.h"

using namespace std;

// static parameter
// float lengths[] = {5,10,50,100,150,200,250,300,350,400};
float lengths[] = {100,200,300,400,500,600,700,800};
int32_t num_lengths = 8;

struct errors {
    int32_t first_frame;
    float   r_err;
    float   t_err;
    float   len;
    float   speed;
    errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
      first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
  };


  vector<MatriX> loadPoses(string file_name);

  vector<float> trajectoryDistances (vector<MatriX> &poses);

  int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len);

  vector<errors> calcSequenceErrors (vector<MatriX> &poses_gt,vector<MatriX> &poses_result);

  void saveSequenceErrors (vector<errors> &err,string file_name);

  void savePathPlot (vector<MatriX> &poses_gt,vector<MatriX> &poses_result,vector<MatriX> &poses_result_imu,string file_name, bool plot_three_line);

  vector<int32_t> computeRoi (vector<MatriX> &poses_gt,vector<MatriX> &poses_result);

  void plotPathPlot (string dir,vector<int32_t> &roi,int32_t idx, bool plot_three_line);

  void saveErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix);

  void plotErrorPlots (string dir,char* prefix);

  void saveStats (vector<errors> err,string dir);

  bool eval (string result_sha);

  

