#include "evaluate_odometry.h"
#include "parameters.h"

vector<MatriX> loadPoses(string file_name) {
  vector<MatriX> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    MatriX P = MatriX::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

vector<float> trajectoryDistances (vector<MatriX> &poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int32_t i=1; i<poses.size(); i++) {
    MatriX P1 = poses[i-1];
    MatriX P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(dist[i-1]+sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}

int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len) {
  for (int32_t i=first_frame; i<dist.size(); i++)
    if (dist[i]>dist[first_frame]+len)
      return i;
  return -1;
}

inline float rotationError(MatriX &pose_error) {
  float a = pose_error.val[0][0];
  float b = pose_error.val[1][1];
  float c = pose_error.val[2][2];
  float d = 0.5*(a+b+c-1.0);
  return acos(max(min(d,1.0f),-1.0f));
}

inline float translationError(MatriX &pose_error) {
  float dx = pose_error.val[0][3];
  float dy = pose_error.val[1][3];
  float dz = pose_error.val[2][3];
  return sqrt(dx*dx+dy*dy+dz*dz);
}

vector<errors> calcSequenceErrors (vector<MatriX> &poses_gt,vector<MatriX> &poses_result) {

  // error vector
  vector<errors> err;

  // parameters
  int32_t step_size = 10; // every second
  
  // pre-compute distances (from ground truth as reference)
  vector<float> dist = trajectoryDistances(poses_gt);
 
  // for all start positions do
  for (int32_t first_frame=0; first_frame<poses_gt.size(); first_frame+=step_size) {
  
    // for all segment lengths do
    for (int32_t i=0; i<num_lengths; i++) {
    
      // current length
      float len = lengths[i];
      
      // compute last frame
      int32_t last_frame = lastFrameFromSegmentLength(dist,first_frame,len);
      
      // continue, if sequence not long enough
      if (last_frame==-1)
        continue;

      // compute rotational and translational errors
      MatriX pose_delta_gt     = MatriX::inv(poses_gt[first_frame])*poses_gt[last_frame];
      MatriX pose_delta_result = MatriX::inv(poses_result[first_frame])*poses_result[last_frame];
      MatriX pose_error        = MatriX::inv(pose_delta_result)*pose_delta_gt;
      float r_err = rotationError(pose_error);
      float t_err = translationError(pose_error);
      
      // compute speed
      float num_frames = (float)(last_frame-first_frame+1);
      float speed = len/(0.1*num_frames);
      
      // write to file
      err.push_back(errors(first_frame,r_err/len,t_err/len,len,speed));
    }
  }

  // return error vector
  return err;
}

void saveSequenceErrors (vector<errors> &err,string file_name) {

  // open file  
  FILE *fp;
  fp = fopen(file_name.c_str(),"w");
 
  // write to file
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++)
    fprintf(fp,"%d %f %f %f %f\n",it->first_frame,it->r_err,it->t_err,it->len,it->speed);
  
  // close file
  fclose(fp);
}

void savePathPlot (vector<MatriX> &poses_gt,vector<MatriX> &poses_result,vector<MatriX> &poses_result_imu,string file_name, bool plot_more_line, const vector<MatriX> &poses_result_VINS) {

  // parameters
  // 2 or 3
  int32_t step_size = 2;

  // open file  
  FILE *fp = fopen(file_name.c_str(),"w");

  if(plot_more_line)
  {
    if(poses_result_VINS.empty())
    {
      // save x/z coordinates of all frames to file
      for (int32_t i=0; i<poses_gt.size(); i+=step_size)
        fprintf(fp,"%f %f %f %f %f %f\n",poses_gt[i].val[0][3],poses_gt[i].val[2][3],
                                  poses_result[i].val[0][3],poses_result[i].val[2][3],
                                  poses_result_imu[i].val[0][3],poses_result_imu[i].val[2][3]);
    }
    else
    {
      // 认为最多只画4条线？
      for (int32_t i=0; i<poses_gt.size(); i+=step_size)
        fprintf(fp,"%f %f %f %f %f %f %f %f\n",poses_gt[i].val[0][3],poses_gt[i].val[2][3],
                                  poses_result[i].val[0][3],poses_result[i].val[2][3],
                                  poses_result_imu[i].val[0][3],poses_result_imu[i].val[2][3],
                                  poses_result_VINS[i].val[0][3],poses_result_VINS[i].val[2][3]);
    }
  }
  else
  {
    for (int32_t i=0; i<poses_gt.size(); i+=step_size)
      fprintf(fp,"%f %f %f %f\n",poses_gt[i].val[0][3],poses_gt[i].val[2][3],
                                poses_result[i].val[0][3],poses_result[i].val[2][3]);
  }

  // close file
  fclose(fp);
}

vector<int32_t> computeRoi (vector<MatriX> &poses_gt,vector<MatriX> &poses_result) {
  
  float x_min = numeric_limits<int32_t>::max();
  float x_max = numeric_limits<int32_t>::min();
  float z_min = numeric_limits<int32_t>::max();
  float z_max = numeric_limits<int32_t>::min();
  
  for (vector<MatriX>::iterator it=poses_gt.begin(); it!=poses_gt.end(); it++) {
    float x = it->val[0][3];
    float z = it->val[2][3];
    if (x<x_min) x_min = x; if (x>x_max) x_max = x;
    if (z<z_min) z_min = z; if (z>z_max) z_max = z;
  }
  
  for (vector<MatriX>::iterator it=poses_result.begin(); it!=poses_result.end(); it++) {
    float x = it->val[0][3];
    float z = it->val[2][3];
    if (x<x_min) x_min = x; if (x>x_max) x_max = x;
    if (z<z_min) z_min = z; if (z>z_max) z_max = z;
  }
  
  float dx = 1.1*(x_max-x_min);
  float dz = 1.1*(z_max-z_min);
  float mx = 0.5*(x_max+x_min);
  float mz = 0.5*(z_max+z_min);
  float r  = 0.5*max(dx,dz);
  
  vector<int32_t> roi;
  roi.push_back((int32_t)(mx-r));
  roi.push_back((int32_t)(mx+r));
  roi.push_back((int32_t)(mz-r));
  roi.push_back((int32_t)(mz+r));
  return roi;
}

void plotPathPlot (string dir,vector<int32_t> &roi,int32_t idx, bool plot_more_line, int num_line) {

  // gnuplot file name
  char command[1024];
  char file_name[256];
  sprintf(file_name,"%02d.gp",idx);
  string full_name = dir + "/" + file_name;
  
  // create png + eps
  for (int32_t i=0; i<2; i++) {

    // open file  
    FILE *fp = fopen(full_name.c_str(),"w");

    // save gnuplot instructions
    if (i==0) {
      fprintf(fp,"set term png size 900,900\n");
      fprintf(fp,"set output \"%02d.png\"\n",idx);
    } else {
      fprintf(fp,"set term postscript eps enhanced color\n");
      fprintf(fp,"set output \"%02d.eps\"\n",idx);
    }

    fprintf(fp,"set size ratio -1\n");
    fprintf(fp,"set xrange [%d:%d]\n",roi[0],roi[1]);
    fprintf(fp,"set yrange [%d:%d]\n",roi[2],roi[3]);
    fprintf(fp,"set xlabel \"x [m]\"\n");
    fprintf(fp,"set ylabel \"z [m]\"\n");
    fprintf(fp,"plot \"%02d.txt\" using 1:2 lc rgb \"#FF0000\" title 'Ground Truth' w lines,",idx);
    if(plot_more_line)
    {
      // fprintf(fp,"\"%02d.txt\" using 3:4 lc rgb \"#00FF00\" title 'RVIO(cam only)' w lines,",idx);
      // fprintf(fp,"\"%02d.txt\" using 5:6 lc rgb \"#0000FF\" title 'RVIO(cam+IMU)' w lines,",idx);
      fprintf(fp,"\"%02d.txt\" using 3:4 lc rgb \"#00FF00\" title 'RVIO(cam+IMU)' w lines,",idx);
      fprintf(fp,"\"%02d.txt\" using 5:6 lc rgb \"#0000FF\" title 'MVIO(cam+IMU)' w lines,",idx);
      // 默认最多只同时画4条线？如何自适用任意数量？
      if(num_line > 3)
        fprintf(fp,"\"%02d.txt\" using 7:8 lc rgb \"#FF00FF\" title 'VINS-FUSION(cam+IMU)' w lines,",idx);
    }
    else
    {
      fprintf(fp,"\"%02d.txt\" using 3:4 lc rgb \"#0000FF\" title 'Odometry' w lines,",idx);
    }
    fprintf(fp,"\"< head -1 %02d.txt\" using 1:2 lc rgb \"#000000\" pt 4 ps 1 lw 2 title 'Sequence Start' w points\n",idx);
    
    // close file
    fclose(fp);
    
    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir.c_str(),file_name);
    system(command);
  }
  
  // create pdf and crop
  // sprintf(command,"cd %s; ps2pdf %02d.eps %02d_large.pdf",dir.c_str(),idx,idx);
  // system(command);
  // sprintf(command,"cd %s; pdfcrop %02d_large.pdf %02d.pdf",dir.c_str(),idx,idx);
  // system(command);
  // sprintf(command,"cd %s; rm %02d_large.pdf",dir.c_str(),idx);
  // system(command);
}

void saveErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix) {

  // file names
  char file_name_tl[1024]; sprintf(file_name_tl,"%s/%s_tl.txt",plot_error_dir.c_str(),prefix);
  char file_name_rl[1024]; sprintf(file_name_rl,"%s/%s_rl.txt",plot_error_dir.c_str(),prefix);
  char file_name_ts[1024]; sprintf(file_name_ts,"%s/%s_ts.txt",plot_error_dir.c_str(),prefix);
  char file_name_rs[1024]; sprintf(file_name_rs,"%s/%s_rs.txt",plot_error_dir.c_str(),prefix);

  // open files
  FILE *fp_tl = fopen(file_name_tl,"w");
  FILE *fp_rl = fopen(file_name_rl,"w");
  FILE *fp_ts = fopen(file_name_ts,"w");
  FILE *fp_rs = fopen(file_name_rs,"w");
 
  // for each segment length do
  for (int32_t i=0; i<num_lengths; i++) {

    float t_err = 0;
    float r_err = 0;
    float num   = 0;

    // for all errors do
    for (vector<errors>::iterator it=seq_err.begin(); it!=seq_err.end(); it++) {
      if (fabs(it->len-lengths[i])<1.0) {
        t_err += it->t_err;
        r_err += it->r_err;
        num++;
      }
    }
    
    // we require at least 3 values
    if (num>2.5) {
      fprintf(fp_tl,"%f %f\n",lengths[i],t_err/num);
      fprintf(fp_rl,"%f %f\n",lengths[i],r_err/num);
    }
  }
  
  // for each driving speed do (in m/s)
  for (float speed=2; speed<25; speed+=2) {

    float t_err = 0;
    float r_err = 0;
    float num   = 0;

    // for all errors do
    for (vector<errors>::iterator it=seq_err.begin(); it!=seq_err.end(); it++) {
      if (fabs(it->speed-speed)<2.0) {
        t_err += it->t_err;
        r_err += it->r_err;
        num++;
      }
    }
    
    // we require at least 3 values
    if (num>2.5) {
      fprintf(fp_ts,"%f %f\n",speed,t_err/num);
      fprintf(fp_rs,"%f %f\n",speed,r_err/num);
    }
  }
  
  // close files
  fclose(fp_tl);
  fclose(fp_rl);
  fclose(fp_ts);
  fclose(fp_rs);
}

void plotErrorPlots (string dir,char* prefix) {

  char command[1024];

  // for all four error plots do
  for (int32_t i=0; i<4; i++) {
 
    // create suffix
    char suffix[16];
    switch (i) {
      case 0: sprintf(suffix,"tl"); break;
      case 1: sprintf(suffix,"rl"); break;
      case 2: sprintf(suffix,"ts"); break;
      case 3: sprintf(suffix,"rs"); break;
    }
       
    // gnuplot file name
    char file_name[1024]; char full_name[1024];
    sprintf(file_name,"%s_%s.gp",prefix,suffix);
    sprintf(full_name,"%s/%s",dir.c_str(),file_name);
    
    // create png + eps
    for (int32_t j=0; j<2; j++) {

      // open file  
      FILE *fp = fopen(full_name,"w");

      // save gnuplot instructions
      if (j==0) {
        fprintf(fp,"set term png size 500,250 font \"Helvetica\" 11\n");
        fprintf(fp,"set output \"%s_%s.png\"\n",prefix,suffix);
      } else {
        fprintf(fp,"set term postscript eps enhanced color\n");
        fprintf(fp,"set output \"%s_%s.eps\"\n",prefix,suffix);
      }
      
      // start plot at 0
      fprintf(fp,"set size ratio 0.5\n");
      fprintf(fp,"set yrange [0:*]\n");

      // x label
      if (i<=1) fprintf(fp,"set xlabel \"Path Length [m]\"\n");
      else      fprintf(fp,"set xlabel \"Speed [km/h]\"\n");
      
      // y label
      if (i==0 || i==2) fprintf(fp,"set ylabel \"Translation Error [%%]\"\n");
      else              fprintf(fp,"set ylabel \"Rotation Error [deg/m]\"\n");
      
      // plot error curve
      fprintf(fp,"plot \"%s_%s.txt\" using ",prefix,suffix);
      switch (i) {
        case 0: fprintf(fp,"1:($2*100) title 'Translation Error'"); break;
        case 1: fprintf(fp,"1:($2*57.3) title 'Rotation Error'"); break;
        case 2: fprintf(fp,"($1*3.6):($2*100) title 'Translation Error'"); break;
        case 3: fprintf(fp,"($1*3.6):($2*57.3) title 'Rotation Error'"); break;
      }
      fprintf(fp," lc rgb \"#0000FF\" pt 4 w linespoints\n");
      
      // close file
      fclose(fp);
      
      // run gnuplot => create png + eps
      sprintf(command,"cd %s; gnuplot %s",dir.c_str(),file_name);
      system(command);
    }
    
    // create pdf and crop
    sprintf(command,"cd %s; ps2pdf %s_%s.eps %s_%s_large.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; pdfcrop %s_%s_large.pdf %s_%s.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; rm %s_%s_large.pdf",dir.c_str(),prefix,suffix);
    system(command);
  }
}

void saveStats (vector<errors> err,string dir) {

  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  // open file  
  FILE *fp = fopen((dir + "/stats.txt").c_str(),"w");
 
  // save errors
  float num = err.size();
  fprintf(fp,"%f %f\n",t_err/num,r_err/num);
  
  // close file
  fclose(fp);
}

//bool eval (string result_sha,Mail* mail)
bool eval (string Dataset, string sequence) {

  // ground truth and result directories
  string result_sha = Dataset + "/" + sequence;
  string gt_dir         = "/home/crs/GEN_VIO/src/evaluate/data/" + Dataset + "/poses";  // "/home/crs/GEN_VIO/src/evaluate/data/KITTI_tracking/poses"
  string result_dir     = "/home/crs/GEN_VIO/src/evaluate/results/" + result_sha;
  string error_dir      = result_dir + "/errors";
  string plot_path_dir  = result_dir + "/plot_path";
  string plot_error_dir = result_dir + "/plot_error";
  
  // create output directories
  system(("mkdir " + error_dir).c_str());
  system(("mkdir " + plot_path_dir).c_str());
  system(("mkdir " + plot_error_dir).c_str());
  
  // total errors
  vector<errors> total_err;

  // for all sequences do
  // for (int32_t i=11; i<22; i++) {
   
    // file name
    // char file_name[256];
    // char file_name_est[256];
    // int i = 7;
    // sprintf(file_name,"%02d.txt",i);

    // 取sequence字符串前N个字符之后的子字符串
    string file_name = sequence.substr(sequence.length()-2) + ".txt";
    
    // read ground truth and result poses
    vector<MatriX> poses_gt     = loadPoses(gt_dir + "/" + file_name);
    // sprintf(file_name_est,"%04d.txt",i);
    vector<MatriX> poses_result = loadPoses(result_dir + "/data/vio_no_IMU.txt");

    // 读取之前保存的有IMU的估计结果，一起画图
    vector<MatriX> poses_result_imu, pose_result_VINS;

    vector<errors> seq_err;
    int num_line = 2;
    // 是仅画 gt / pure_stereo / stereo+IMU 三种设置下的结果图，还是 画gt和其中某种设置下的两种结果图并给出误差分析
    if(plot_line == 1)
    {
      poses_result_imu = loadPoses(result_dir + "/data/vio_with_IMU.txt");
      if(!poses_result_imu.empty()) 
        ++num_line;
      else
      {
        cout << "Please provide result of Vision-IMU-based system!" << endl;
        exit(-1);
      }
      
      pose_result_VINS = loadPoses(result_dir + "/data/vio_VINS_Fusion.txt");
      if(!pose_result_VINS.empty()) 
        ++num_line;
    }
    else
    {
      // plot status
      // mail->msg("Processing: %s, poses: %d/%d",file_name,poses_result.size(),poses_gt.size());
      
      // check for errors
      //if (poses_gt.size()==0 || poses_result.size()!=poses_gt.size()) {
        //mail->msg("ERROR: Couldn't read (all) poses of: %s", file_name);
        //return false;
      //}

      // compute sequence errors    
      seq_err = calcSequenceErrors(poses_gt,poses_result);
      saveSequenceErrors(seq_err,error_dir + "/" + file_name);
      
      // add to total errors
      total_err.insert(total_err.end(),seq_err.begin(),seq_err.end());
    }
    
    // for first half => plot trajectory and compute individual stats
    // if (i<=15) 
    {
      // save + plot bird's eye view trajectories
      savePathPlot(poses_gt,poses_result,poses_result_imu,plot_path_dir + "/" + file_name, (plot_line == 1), pose_result_VINS);
      vector<int32_t> roi = computeRoi(poses_gt,poses_result);
      int i = atoi(sequence.c_str());
      plotPathPlot(plot_path_dir,roi,i,plot_line,num_line);  
      
      if(!plot_line)
      {
        // save + plot individual errors
        char prefix[16];
        sprintf(prefix,"%02d",i);
        saveErrorPlots(seq_err,plot_error_dir,prefix);
        plotErrorPlots(plot_error_dir,prefix);
      }
    }

    if(!plot_line)
    {
      // save + plot total errors + summary statistics
      if (total_err.size()>0) {
        char prefix[16];
        sprintf(prefix,"avg");
        saveErrorPlots(total_err,plot_error_dir,prefix);
        plotErrorPlots(plot_error_dir,prefix);
        saveStats(total_err,result_dir);
      }
    }
  
  // }

  // success
	return true;
}

// int32_t main (int32_t argc,char *argv[]) {

//   // we need 2 or 4 arguments!
//   if (argc!=2 && argc!=4) {
//     cout << "Usage: ./eval_odometry result_sha [user_sha email]" << endl;
//     return 1;
//   }

//   // read arguments
//   string result_sha = argv[1];
  
//   // init notification mail
//   //Mail *mail;
//   //if (argc==4) mail = new Mail(argv[3]);
//   //else         mail = new Mail();
//   //mail->msg("Thank you for participating in our evaluation!");

//   // run evaluation
//   bool success = eval(result_sha);
//   //bool success = eval(result_sha,mail);
//   //if (argc==4) mail->finalize(success,"odometry",result_sha,argv[2]);
//   //else         mail->finalize(success,"odometry",result_sha);

//   // send mail and exit
//   //delete mail;
//   return 0;
// }

