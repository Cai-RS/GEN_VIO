#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
//#include <ros/ros.h>
//#include <sensor_msgs/Image.h>
//#include <cv_bridge/cv_bridge.h>
#include "estimator/estimator.h"
#include "parameters.h"

#include "evaluate/evaluate_odometry.h"

using namespace std;
using namespace Eigen;

void cal_error_pose_cam(const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, const Matrix3f &cur_cam_R_est, const Vector3f &cur_cam_P_est, 
						float &total_r_rpe, float &total_t_rpe, vector<float> &vec_err_cam_R, vector<float> &vec_err_cam_P);

set<int> cal_error_motion_objs(vector<float> &pt_dep_obj_no_gt, vector<string> &item_obj_gt, ifstream &file_est_obj, map<int,int> &last_frame_obj_id_est_gt, map<int,int> &cur_frame_obj_id_est_gt, 
							map<int,pair<Matrix3f,Vector3f>> &last_frame_obj_id_pose, map<int,pair<Matrix3f,Vector3f>> &cur_frame_obj_id_pose, const Matrix3f &last_cam_R_gt, const Vector3f &last_cam_P_gt, 
							const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, int &num_true_track, int &num_false_track, float &total_r_rpe, float &total_t_rpe);

bool match_gt_esti_cal_error(vector<string> &item_obj_gt, vector<uchar> &matched_gt_obj, int &obj_id, int &cls_label,  
							float* bbox, Vector3f &ave_pt, Matrix3f &motion_R_obj, Vector3f &motion_P_obj, const int &matched_obj_id_gt_prev, 
							map<int,int> &last_frame_obj_id_est_gt, map<int,pair<Matrix3f,Vector3f>> &cur_frame_obj_id_pose, 
							map<int,pair<Matrix3f,Vector3f>> &last_frame_obj_id_pose, const Matrix3f &last_cam_R_gt, const Vector3f &last_cam_P_gt, 
							const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, float &total_r_rpe, float &total_t_rpe, int &num_false_track);
float cal_iou_bbox(float bbox_est[], float bbox_gt[]);

void ObjPoseParsingKT(vector<string> &item_obj_gt, map<int,pair<Matrix3f, Vector3f>> &gt_RP_objs_prev);
void ObjPoseParsingKT(string item_obj_gt, int &obj_id_gt, int &obj_cls_gt, float bbox_gt[], Vector3f &dimen_3D_bbox_gt);
void cal_error_motion_cam(const Matrix3f &motion_R_gt, const Vector3f &motion_P_gt, const Matrix3f &motion_R_est, const Vector3f &motion_P_est, 
							float &total_r_rpe, float &total_t_rpe, vector<float> &vec_err_cam_motion_R, vector<float> &vec_err_cam_motion_P);

bool check_center_in_3D_bbox_gt(Vector3f dimen_3D_bbox_gt, Matrix3f R_obj_gt, Vector3f P_obj_gt, Vector3f ave_pt);

int main(int argc, char** argv)
{
    if(0 && argc != 4)
	{
		printf("please intput: KITTI_tracking_test [config file] [data base folder] [sequence] \n"
			   "for example: ./GEN_VIO "
			   "/home/crs/GEN_VIO/config/KITTI_tracking/kitti_config00-11.yaml "
			   "/home/crs/GEN_VIO/data/KITTI_tracking/sequences/ "
			   "0000\n");
		return 1;
	}

	// argv[0]是可执行函数的名称，如GEN—VIO，之后的才是此主函数的参数
    string config_file(argv[1]);
	printf("config_file: %s\n", argv[1]);
	string base_dir(argv[2]);
	string sequence(argv[3]);
	printf("read sequence: %s\n", argv[3]);
	string dataPath = base_dir + "/" + sequence + "/";

    readParameters(config_file);
	
	
	if(trans_result_format || evaluate_reslut)
	{
		string result_path(argv[2]);
		string name(argv[4]);
		cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
		if(!fsSettings.isOpened())
		{
			std::cerr << "ERROR: Wrong path to settings" << std::endl;
		}

		string Dataset;
		fsSettings["dataset"] >> Dataset;

		if(trans_result_format)
		{
			cv::Mat cv_T;
			fsSettings["body_T_cam0"] >> cv_T;
			Eigen::Matrix4f T;
			cv::cv2eigen(cv_T, T);
			Matrix3f R_imu_to_cam_gt = T.block<3, 3>(0, 0);
			Vector3f P_imu_to_cam_gt = T.block<3, 1>(0, 3);
			
			bool cam_gt = false;
			string DATA = "KITTI_odometry";
			if(Dataset == DATA)
				cam_gt = true;
			
			int use_imu = USE_IMU;

			fsSettings.release();

			ifstream file_est_cam;
			file_est_cam.open((result_path+"/vio(" + name + ").txt").c_str(), ios::in);
			if(!file_est_cam.is_open())
			{
				printf("cannot open file: %s\n", result_path.c_str());
				// ROS_BREAK();
				return 0;
			}
			
			string line_est_cam;
			Matrix3f cur_R_est, Corr_R_est, rot_diff;
			Vector3f cur_P_est, Corr_P_est, trans_diff;

			FILE* outFile;
			outFile = fopen(("/home/crs/GEN_VIO/src/evaluate/results/" + Dataset + "/" + sequence + "/data/vio.txt").c_str(),"w");
			if(outFile == NULL)
				printf("Output path dosen't exist: %s\n", ("/home/crs/GEN_VIO/src/evaluate/results/" + Dataset + "/" + sequence + "/data/vio.txt").c_str());
			
			int i = 0;
			while(getline(file_est_cam, line_est_cam))
			{
				istringstream istr_est(line_est_cam);
				istr_est >> cur_R_est(0,0) >> cur_R_est(0,1) >> cur_R_est(0,2) >> cur_P_est(0)
						>> cur_R_est(1,0) >> cur_R_est(1,1) >> cur_R_est(1,2) >> cur_P_est(1)
						>> cur_R_est(2,0) >> cur_R_est(2,1) >> cur_R_est(2,2) >> cur_P_est(2);
				
				if(!cam_gt)
				{
					if(use_imu)
					{
						// 由于有IMU模式下系统首帧的IMU位姿在优化中不是固定的，即会有roll和pitch角，为了和gt标签对齐，需要将首帧直接置为完全的全局参考系
						if (i == 0)
						{
							Matrix3f rot_diff_tmp = cur_R_est.transpose();
							Corr_R_est = rot_diff_tmp * cur_R_est;
							Corr_P_est = Vector3f(0.0,0.0,0.0);
							
							rot_diff = rot_diff_tmp;
							trans_diff = -rot_diff * cur_P_est;
						}
						else
						{
							Corr_R_est = rot_diff * cur_R_est;
							Corr_P_est = rot_diff * cur_P_est + trans_diff;
						}
					}
					else
					{
						// STEREO模式下输出的结果为相机的估计位姿
						// 转回为IMU的位姿，且初始IMU位姿就是全局位姿
						Corr_R_est = cur_R_est * R_imu_to_cam_gt;
						Corr_P_est = cur_R_est * P_imu_to_cam_gt + cur_P_est;
					}
				}
				else
				{
					if(use_imu)
					{
						// 估计的位姿结果从IMU转化为相机的
						Matrix3f R_est_cam = cur_R_est * R_imu_to_cam_gt.transpose();
						Vector3f P_est_cam = -R_est_cam * P_imu_to_cam_gt + cur_P_est;

						cur_R_est = R_est_cam;
						cur_P_est = P_est_cam;
					}

					// 如果是odometry数据集，则给定的gt是相机的位姿
					// 由于gt首帧是全局参考系，因此这里对估计的结果要进行校正
					if (i == 0)
					{
						Matrix3f rot_diff_tmp = cur_R_est.transpose();
						Corr_R_est = rot_diff_tmp * cur_R_est;
						Corr_P_est = Vector3f(0.0,0.0,0.0);

						rot_diff = rot_diff_tmp;
						trans_diff = -rot_diff * cur_P_est;
					}
					else
					{
						Corr_R_est = rot_diff * cur_R_est;
						Corr_P_est = rot_diff * cur_P_est + trans_diff;
					}
				}

				fprintf(outFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n", Corr_R_est(0,0), Corr_R_est(0,1), Corr_R_est(0,2), Corr_P_est(0),
																		Corr_R_est(1,0), Corr_R_est(1,1), Corr_R_est(1,2), Corr_P_est(1), 
																		Corr_R_est(2,0), Corr_R_est(2,1), Corr_R_est(2,2), Corr_P_est(2));

				++i;
			}
			
			fclose(outFile);
		}

		if(evaluate_reslut == 1)
			bool succ = eval(Dataset, sequence);
		
		return 1;
	}

	Estimator estimator;
    estimator.setParameter();

    FILE* file_time;
	file_time = std::fopen((dataPath + "image/timestamps.txt").c_str() , "r");
	if(file_time == NULL){
	    printf("cannot find file: %simage/timestamps.txt\n", dataPath.c_str());
	    //ROS_BREAK();
	    return 0;          
	}
	
	double imageTime;
	vector<double> imageTimeList;

    while (fscanf(file_time, "%lf", &imageTime) != EOF)
	{
	    imageTimeList.push_back(imageTime);
	}
	std::fclose(file_time);
	
	ifstream file_IMU;
	file_IMU.open((dataPath + "IMU/data.txt").c_str(), ios::in);
	if(!file_IMU.is_open())
	{
	    printf("cannot find file: %sIMU/data.txt\n", dataPath.c_str());
	    //ROS_BREAK();
	    return 0;          
	}

    string leftImagePath, rightImagePath, line_IMU;
	cv::Mat imLeft, imRight, imLeft_gray, imRight_gray;
	FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/" + sequence + "/result_cam/vio.txt").c_str(),"w");
	if(outFile == NULL)
		printf("Output path dosen't exist: %s\n", (OUTPUT_FOLDER + "/" + sequence + "/result_cam").c_str());
    
    // Eigen::Matrix<double, 4, 4> pose;
	Vector3d acc, ang_v;
	double dt = 0.0, elem = 0.0, t_frame = 0.0, ave_t_before_init = 0.0, ave_t_after_init = 0.0;
	int num_IMU = 0, num_frame_before_init = 0, num_frame_after_init = 0;
	bool init_succ = false;
	double t_cam;

	bool use_imu = USE_IMU;

	ifstream file_gt_pose;
	string line_gt_pose;
	Matrix3d R_gt_1, R_gt_2;
	Vector3d P_gt_1, P_gt_2;
	Matrix3d R_gt_motion = Matrix3d::Zero();
	Vector3d P_gt_motion = Vector3d::Zero();

	if(use_gt_to_show_match)
	{
		file_gt_pose.open((dataPath + "label/gt_pose_cam.txt").c_str(), ios::in);
		if(!file_gt_pose.is_open())
		{
			printf("cannot open file: %slabel/gt_pose_cam.txt\n", dataPath.c_str());
			return 0;          
		}
	}

	bool is_cam_gt = false;
	Matrix3d gt_R_imu_to_cam;
	Vector3d gt_P_imu_to_cam;
	if(use_gt_to_show_match)
	{
		cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
		if(!fsSettings.isOpened())
		{
			std::cerr << "ERROR: Wrong path to settings" << std::endl;
		}
		cv::Mat cv_T;
		fsSettings["body_T_cam0"] >> cv_T;
		Eigen::Matrix4d T;
		cv::cv2eigen(cv_T, T);
		gt_R_imu_to_cam = T.block<3, 3>(0, 0);
		gt_P_imu_to_cam = T.block<3, 1>(0, 3);
		
		string Dataset;
		fsSettings["dataset"] >> Dataset;
		
		string DATA = "KITTI_odometry";
		if(Dataset == DATA)
			is_cam_gt = true;
		
		fsSettings.release();
	}

	set<int> spec_frame;
	// spec_frame.insert(885);
	// spec_frame.insert(886);
	// spec_frame.insert(887);
	// spec_frame.insert(888);
	// spec_frame.insert(909);
	// spec_frame.insert(910);
	// spec_frame.insert(911);
	
    for (size_t i = 0; i < imageTimeList.size(); ++i)
	{
        printf("\nprocess image %d\n", (int)i);
        stringstream ss;
		// 6 or 10(for some KITTI_tracking sequence)
        ss << setfill('0') << setw(6) << i;
        leftImagePath = dataPath + "image/image_02/" + ss.str() + ".png";
        rightImagePath = dataPath + "image/image_03/" + ss.str() + ".png";	
		
		t_cam = imageTimeList[i];
		
		if(use_gt_to_show_match)
		{
			getline(file_gt_pose, line_gt_pose);
			istringstream istr_gt_pose(line_gt_pose);
			if(i == 0)
			{
				istr_gt_pose >> R_gt_1(0,0) >> R_gt_1(0,1) >> R_gt_1(0,2) >> P_gt_1(0)
							>> R_gt_1(1,0) >> R_gt_1(1,1) >> R_gt_1(1,2) >> P_gt_1(1)
							>> R_gt_1(2,0) >> R_gt_1(2,1) >> R_gt_1(2,2) >> P_gt_1(2);
				
				// 如果是KITTI-tracking数据集，则提供的真值文件是IMU的位姿,转换成相机的真实位姿
				if(!is_cam_gt)
				{
					Matrix3d cur_cam_R_gt = R_gt_1 * gt_R_imu_to_cam.transpose();
					Vector3d cur_cam_P_gt = -cur_cam_R_gt * gt_P_imu_to_cam + P_gt_1;

					R_gt_1 = cur_cam_R_gt;
					P_gt_1 = cur_cam_P_gt;
				}
				
			}
			else
			{
				istr_gt_pose >> R_gt_2(0,0) >> R_gt_2(0,1) >> R_gt_2(0,2) >> P_gt_2(0)
							>> R_gt_2(1,0) >> R_gt_2(1,1) >> R_gt_2(1,2) >> P_gt_2(1)
							>> R_gt_2(2,0) >> R_gt_2(2,1) >> R_gt_2(2,2) >> P_gt_2(2);
				
				if(!is_cam_gt)
				{
					Matrix3d cur_cam_R_gt = R_gt_2 * gt_R_imu_to_cam.transpose();
					Vector3d cur_cam_P_gt = -cur_cam_R_gt * gt_P_imu_to_cam + P_gt_2;

					R_gt_2 = cur_cam_R_gt;
					P_gt_2 = cur_cam_P_gt;
				}
				
				// 得到转换矩阵，用于将上一帧坐标系下的点坐标转换到当前帧坐标系下
				// gt文件中每一行表示的是首帧坐标系到某一帧坐标系的变换，其作用是将某一帧下的点坐标转换到首帧下坐标下的该点坐标
				// 要注意区分 坐标系间的变换 和 某个点在两个坐标系下的坐标值间的变换，表示“坐标系A变换到坐标系B“的 欧式变换 也是 用于 “将某点在坐标系B下的坐标值变换到坐标系A下的坐标值“
				R_gt_motion = R_gt_2.transpose() * R_gt_1;
				P_gt_motion = R_gt_2.transpose()*(P_gt_1 - P_gt_2);

            	Quaterniond delta_Q(R_gt_motion);
            	double delta_ang = fabs(acos(delta_Q.w()) * 2.0 / 3.1416 * 180.0);
				cout << "gt rotation angle: " << delta_ang << ", gt norm of translation: " << P_gt_motion.norm() << endl;
			}

		}

		if(use_imu)
		{
			while(dt < t_cam)
			{
				if(getline(file_IMU, line_IMU))
				{
					istringstream istr_IMU(line_IMU);
					// 注意，IMU的data数据比KITTI的原始IMU数据多了一列，即首列，为IMU的时间戳
					// 注意，用istringstream和>>来将字符串元素逐个赋值到变量时，如果遇到结束符";"，则该istr_IMU的内容就被清空，后续再使用该istr_IMU赋值给变量都是无效值（0）
					istr_IMU >> dt >> elem >> elem >> elem >> elem >> elem >> elem >> elem >> elem >> elem >> elem >> elem
							>> acc(0) >> acc(1) >> acc(2) >> elem >> elem >> elem >> ang_v(0) >> ang_v(1) >> ang_v(2);
					// cout << "IMU time: " << dt << endl;
					// for(int k = 0; k < 11; ++k)
					// {
					// 	istr_IMU >> elem;
					// 	cout << "elem: " << elem << endl;
					// }
					
					// 应该使用oxts文件中的哪个坐标系下的imu数据？这里使用ax,ay,az,wx,wy,wz，即跟汽车的前、左、上对齐的IMU数据
					// istr_IMU >> acc(0)   >> acc(1)   >> acc(2)
					// 		 >> elem     >> elem     >> elem
					// 		 >> ang_v(0) >> ang_v(1) >> ang_v(2);
					
					// 发布IMU数据和图像数据。给定imu到cam的外参矩阵！
					estimator.inputIMU(dt, acc, ang_v);
				}
				//cout << "measure g: " << acc(2) << endl;
			}
			// 最新的IMU的时刻需要大于或等于cam的时刻，这里假设比相机时刻已经多出了3帧IMU测量(注意在组织数据集时不要忘了这点！）。这对于初始化IMU的初始坐标系比较重要
			for(int k = 0; k < 3; ++k)
			{
				if(getline(file_IMU, line_IMU))
				{
					istringstream istr_IMU(line_IMU);
					istr_IMU >> dt;
					for(int k = 0; k < 11; ++k)
					{
						istr_IMU >> elem;
					}
					// 应该使用oxts文件中的哪个坐标系下的imu数据？这里使用ax,ay,az,wx,wy,wz，即跟汽车的前、左、上对齐的IMU数据
					istr_IMU >> acc(0)   >> acc(1)   >> acc(2)
								>> elem  >> elem     >> elem
								>> ang_v(0) >> ang_v(1) >> ang_v(2);
					
					// 发布IMU数据和图像数据。给定imu到cam的外参矩阵！
					estimator.inputIMU(dt, acc, ang_v);
				}
			}
		}

		imLeft = imread(leftImagePath);
		if(imLeft.empty()) 
		{
			cout << "Can not read left image" << ss.str() << ".png under path " << dataPath << "image/image_02" << endl;
			abort();
		}
		imRight = imread(rightImagePath);
		if(imLeft.empty()) 
		{
			cout << "Can not read right image" << ss.str() << ".png under path " << dataPath << "image/image_03" << endl;
			abort();
		}

		imLeft_gray  = imread(leftImagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
		if(imLeft_gray.empty()) 
		{
			cout << "Can not read left image " << ss.str() << ".png as grayscale under path " << dataPath << "image/image_02" << endl;
			abort();
		}

		imRight_gray = imread(rightImagePath, cv::ImreadModes::IMREAD_GRAYSCALE);
		if(imRight_gray.empty()) 
		{
			cout << "Can not read right image " << ss.str() << ".png as grayscale under path " << dataPath << "image/image_03" << endl;
			abort();
		}

		// cout << "psuh back image!" << endl;
        //estimator.inputImage(imageTimeList[i], imLeft, imRight);
		cout << "Start vio process!" << endl;
		if(spec_frame.find(i) != spec_frame.end())
			estimator.Fea_Obj_Extract_Track(t_frame, init_succ, t_cam, imLeft, imRight, imLeft_gray, imRight_gray, R_gt_motion, P_gt_motion, spec_frame);
		else
			estimator.Fea_Obj_Extract_Track(t_frame, init_succ, t_cam, imLeft, imRight, imLeft_gray, imRight_gray, R_gt_motion, P_gt_motion);
		
		if(use_gt_to_show_match)
		{
			if(i > 0)
			{
				R_gt_1 = R_gt_2;
				P_gt_1 = P_gt_2;
			}
		}

		if(i < 2)
		{
			printf("Process time for no.%d frame image: %fms\n", i, t_frame);
		}
		// 系统前2帧的用时不加入统计
		else
		{
			// 如果还没开始完成VI初始化
			if(!init_succ)
			{
				ave_t_before_init += t_frame;
				++num_frame_before_init;
			}
			else
			{
				ave_t_after_init += t_frame;
				++num_frame_after_init;
			}
		}
        //estimator.getPoseInWorldFrame(pose, true);

        // if(outFile != NULL)
        //     fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n", pose(0,0), pose(0,1), pose(0,2),pose(0,3),
        //                                                                 pose(1,0), pose(1,1), pose(1,2),pose(1,3),
        //                                                                 pose(2,0), pose(2,1), pose(2,2),pose(2,3));
		
        FILE* outFile_objs;
	    outFile_objs = fopen((OUTPUT_FOLDER_OBJS + "/" + sequence + "/result_objs/" + ss.str() + ".txt").c_str(),"w");
        if(outFile_objs == NULL)
			printf("Output path dosen't exist: %s\n", (OUTPUT_FOLDER_OBJS + "/" + sequence + "/result_objs").c_str());
		
        estimator.write_result_objs(outFile, outFile_objs);
        fclose(outFile_objs);
    }
	cout << "total num of found static objects: " << estimator.featureTracker.id_gl_sta_obj.size() << endl;
	// 注意，系统最后不一定使用IMU，中间可能切换配置
	use_imu = USE_IMU;
	estimator._shutdown = false;
	estimator.clearState();
	fclose(outFile);
	file_IMU.close();

	if(use_gt_to_show_match)
		file_gt_pose.close();
	
	if(USE_IMU) printf("Average process time before initialization: %fms\n", ave_t_before_init/num_frame_before_init);
	printf("Average process time after initialization: %fms\n", ave_t_after_init/num_frame_after_init);

	ifstream file_gt_pose_cam, file_gt_tracking, file_est_cam, file_est_obj;
	
	// 计算相机轨迹 和 运动物体的运动 的估计误差
	file_gt_pose_cam.open((dataPath + "label/gt_pose_cam.txt").c_str(), ios::in);
	if(!file_gt_pose_cam.is_open())
	{
	    printf("cannot open file: %slabel/gt_pose_cam.txt\n", dataPath.c_str());
	    //ROS_BREAK();
	    return 0;          
	}
	
	file_gt_tracking.open((dataPath + "label/gt_obj_detect.txt").c_str(), ios::in);
	if(!file_gt_tracking.is_open())
	{
	    printf("cannot open file: %slabel/gt_obj_detect.txt\n", dataPath.c_str());
	    //ROS_BREAK();
	    return 0;          
	}

	file_est_cam.open((OUTPUT_FOLDER + "/" + sequence + "/result_cam/vio.txt").c_str(), ios::in);
	if(!file_est_cam.is_open())
	{
		printf("cannot open file: %s/%s/result_cam/vio.txt\n", OUTPUT_FOLDER.c_str(),sequence.c_str());
	    // ROS_BREAK();
	    return 0;
	}
	
	string line_gt_cam, line_est_cam, line_gt_obj, line_est_obj;
	// 记录每一帧中少跟踪的动态物体（这是相对于gt的计数，也有可能部分物体是gt中没有3D位姿标注的，例如该物体太远了）
	vector<uchar> lost_dyn_objs_per_frame(imageTimeList.size(),0);   

	// int last_line = 0;
	vector<string> item_obj_gt;
	string first_str_frame;
	Matrix3f IMU_R_GT, last_cam_R_gt, cur_cam_R_gt, last_cam_R_est, cur_R_est, cur_cam_R_est, cam_motion_R_gt, cam_motion_R_est, rot_diff;
	Vector3f IMU_P_GT, last_cam_P_gt, cur_cam_P_gt, last_cam_P_est, cur_P_est, cur_cam_P_est, cam_motion_P_gt, cam_motion_P_est, trans_diff;

	Matrix3f Cam_init_R;
	Vector3f Cam_init_P;

	// 记录上一帧gt出现的gt物体id及其在相机坐标系下的位姿
	map<int,pair<Matrix3f,Vector3f>> last_frame_obj_id_pose, cur_frame_obj_id_pose;
	// 记录上一帧估计得到的动态物体的全局id 与 其匹配到的bg物体的id
	map<int,int> last_frame_obj_id_est_gt, cur_frame_obj_id_est_gt;
	vector<float> pt_dep_obj_no_gt;
	float total_err_cam_R = 0.0, total_err_cam_P = 0.0, total_err_cam_motion_R = 0.0, total_err_cam_motion_P = 0.0, total_r_rpe_obj = 0.0, total_t_rpe_obj = 0.0;
	int num_true_track = 0, num_false_track = 0, num_lost_track = 0;

	// 在全局文件中给定从IMU到左相机的投影变换矩阵
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
	cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
	Eigen::Matrix4f T;
    cv::cv2eigen(cv_T, T);
	Matrix3f R_imu_to_cam_gt = T.block<3, 3>(0, 0);
	Vector3f P_imu_to_cam_gt = T.block<3, 1>(0, 3);
	
	string Dataset;
	fsSettings["dataset"] >> Dataset;
	bool cam_gt = false;
	string DATA = "KITTI_odometry";
	if(Dataset == DATA)
		cam_gt = true;
	
	fsSettings.release();
	// 注意，parameters.h中的RIC是相机到IMU的旋转矩阵，跟标定文件中的相反
	// Matrix3f R_imu_to_cam_gt = RIC[0];
    // Vector3f P_imu_to_cam_gt = TIC[0];

	int num_img = imageTimeList.size();
	double cam_t, cam_t_prev;

	vector<float> vec_err_cam_R, vec_err_cam_P, vec_err_cam_motion_R, vec_err_cam_motion_P, vec_err_obj_motion_R, vec_err_obj_motion_P;

	for (size_t i = 0; i < num_img; ++i)
	{
		cam_t = imageTimeList[i];

		// 重复使用string变量来从getline中获取新行时，需要先clear该string变量吗?
		line_gt_cam.clear();
		getline(file_gt_pose_cam, line_gt_cam);
		
		istringstream istr_gt(line_gt_cam);
		istr_gt >> IMU_R_GT(0,0)  >> IMU_R_GT(0,1) >> IMU_R_GT(0,2) >> IMU_P_GT(0)
			    >> IMU_R_GT(1,0)  >> IMU_R_GT(1,1) >> IMU_R_GT(1,2) >> IMU_P_GT(1)
			    >> IMU_R_GT(2,0)  >> IMU_R_GT(2,1) >> IMU_R_GT(2,2) >> IMU_P_GT(2);
		
		// gt文件中的全局位姿来自于GPS数据的解算，然而其z轴正向是竖直向下；而KITTI坐标系标定中是设定IMU的z轴为数直向上（这与相机的坐标系规定通过相机位姿进行了强绑定）
		// IMU_P_GT(2) = -1 * IMU_P_GT(2);

		getline(file_est_cam, line_est_cam);
		
		istringstream istr_est(line_est_cam);
		istr_est >> cur_R_est(0,0)  >> cur_R_est(0,1) >> cur_R_est(0,2) >> cur_P_est(0)
			     >> cur_R_est(1,0)  >> cur_R_est(1,1) >> cur_R_est(1,2) >> cur_P_est(1)
			     >> cur_R_est(2,0)  >> cur_R_est(2,1) >> cur_R_est(2,2) >> cur_P_est(2);
		
		// 如果使用了IMU，则对VIO的估计结果进行对齐，使得首帧的IMU坐标系作为全局坐标系
		// 这是由于cam_gt文件中给出的实际上是IMU坐标系的位姿（且首帧的P为0，R为I）
		Matrix3f Corr_R_est;
		Vector3f Corr_P_est;

		// 其实VINS系统无论是STEREO还是STEREO-IMU，都是用首帧相机时刻的IMU的坐标系为全局参考系！！只不过如果有IMU，则还需要估计首帧IMU坐标系的roll和pitch角。
		if(!cam_gt)
		{
			// KITTI_tracking数据集中的位姿真值，都是IMU坐标系的，且其以首帧时刻IMU坐标系为全局参考系，即连roll和pitch角均为0.
			cur_cam_R_gt = IMU_R_GT * R_imu_to_cam_gt.transpose();
			cur_cam_P_gt = -cur_cam_R_gt * P_imu_to_cam_gt + IMU_P_GT;

			// STEREO+IMU模式下输出的结果为IMU的估计位姿
			if(use_imu)
			{
				// 由于有IMU模式下系统首帧的IMU位姿在优化中不是固定的，即会有roll和pitch角，为了和gt标签对齐，需要将首帧直接置为完全的全局参考系
				if (i == 0)
				{
					// Matrix3d tmp_R_d;
					// tmp_R_d(0,0) = (double)cur_cam_R_est(0,0);
					// tmp_R_d(0,1) = (double)cur_cam_R_est(0,1);
					// tmp_R_d(0,2) = (double)cur_cam_R_est(0,2);
					// tmp_R_d(1,0) = (double)cur_cam_R_est(1,0);
					// tmp_R_d(1,1) = (double)cur_cam_R_est(1,1);
					// tmp_R_d(1,2) = (double)cur_cam_R_est(1,2);
					// tmp_R_d(2,0) = (double)cur_cam_R_est(2,0);
					// tmp_R_d(2,1) = (double)cur_cam_R_est(2,1);
					// tmp_R_d(2,2) = (double)cur_cam_R_est(2,2);

					// Cam_init_R = Cam_R_GT;
					// Cam_init_P = Cam_P_GT;
					// float yaw = (float)Utility::R2ypr(tmp_R_d).x();
					// Matrix3f rot_diff_tmp = Utility::ypr2R(Eigen::Vector3f{-yaw, 0, 0});
					// Vector3f trans_diff_tmp = -rot_diff_tmp * Cam_P_GT;
					// cur_cam_R_gt = rot_diff_tmp * Cam_R_GT;
					// cur_cam_P_gt = Vector3f(0.0,0.0,0.0);

					// rot_diff = cur_cam_R_gt * Cam_init_R.transpose();
					// trans_diff = -rot_diff * Cam_init_P;

					Matrix3f rot_diff_tmp = cur_R_est.transpose();
					Corr_R_est = rot_diff_tmp * cur_R_est;
					Corr_P_est = Vector3f(0.0,0.0,0.0);
					
					rot_diff = rot_diff_tmp;
					trans_diff = -rot_diff * cur_P_est;
				}
				else
				{
					Corr_R_est = rot_diff * cur_R_est;
					Corr_P_est = rot_diff * cur_P_est + trans_diff;
				}
				
				cur_cam_R_est = Corr_R_est * R_imu_to_cam_gt.transpose();
				cur_cam_P_est = -cur_cam_R_est * P_imu_to_cam_gt + Corr_P_est;
			}
			else
			{
				// STEREO模式下输出的结果为相机的估计位姿
				cur_cam_R_est = cur_R_est;
				cur_cam_P_est = cur_P_est;

				// 如果是纯视觉，是否需要校正gt的首帧相机坐标系为全局参考坐标系。不需要，因为KITTI的位姿真值中一直都是以IMU首帧为全局参考系，因此MVIO系统也是
				// if(i == 0)
				// {
				// 	rot_diff = cur_cam_R_gt.transpose();
				// 	trans_diff = -rot_diff * cur_cam_P_gt;

				// 	cur_cam_R_gt = rot_diff * cur_cam_R_gt;
				// 	cur_cam_P_gt = Vector3f(0.0,0.0,0.0);
				// }
				// else
				// {
				// 	cur_cam_R_gt = rot_diff * cur_cam_R_gt;
				// 	cur_cam_P_gt = rot_diff * cur_cam_P_gt + trans_diff;
				// }
			}
		}
		else
		{
			cur_cam_R_gt = IMU_R_GT;
			cur_cam_P_gt = IMU_P_GT;
			
			if(use_imu)
			{
				// 估计的位姿结果从IMU转化为相机的
				Matrix3f R_est_cam = cur_R_est * R_imu_to_cam_gt.transpose();
				Vector3f P_est_cam = -R_est_cam * P_imu_to_cam_gt + cur_P_est;

				cur_R_est = R_est_cam;
				cur_P_est = P_est_cam;
			}

			// 如果是odometry数据集，则给定的gt是相机的位姿
			// 由于gt首帧是全局参考系，因此这里对估计的结果要进行校正
			if (i == 0)
			{
				Matrix3f rot_diff_tmp = cur_R_est.transpose();
				Corr_R_est = rot_diff_tmp * cur_R_est;
				Corr_P_est = Vector3f(0.0,0.0,0.0);

				rot_diff = rot_diff_tmp;
				trans_diff = -rot_diff * cur_P_est;
			}
			else
			{
				Corr_R_est = rot_diff * cur_R_est;
				Corr_P_est = rot_diff * cur_P_est + trans_diff;
			}

			cur_cam_R_est = Corr_R_est;
			cur_cam_P_est = Corr_P_est;
		}

		// 计算相机位姿的估计误差
		cal_error_pose_cam(cur_cam_R_gt, cur_cam_P_gt, cur_cam_R_est, cur_cam_P_est, total_err_cam_R, total_err_cam_P, vec_err_cam_R, vec_err_cam_P);
		
		int frame_id, obj_id;
		string elem;
		stringstream ss;
		// 6 or 10
		ss << setfill('0') << setw(6) << i;
		bool has_obj_cur_frame = true;
		// 物体的真值信息
		// 当前帧的第一个物体信息的行可能在上一帧已经读取了
		if (!first_str_frame.empty())
		{
			item_obj_gt.push_back(first_str_frame);
			istringstream istr(first_str_frame);
			string elem;
			istr >> elem;
			frame_id = atoi(elem.c_str());
			// 不一定每一帧都有跟踪物体的gt
			if (frame_id != i)
			{
				first_str_frame = item_obj_gt.back();
				item_obj_gt.pop_back();
				has_obj_cur_frame = false;
			}
			else
			{
				istr >> elem;
				obj_id = atoi(elem.c_str());
				if (obj_id == -1) 
				{
					item_obj_gt.pop_back();
				}
			}
		}

		if(has_obj_cur_frame)
		{
			// 继续获取当前帧中的有位姿估计的物体的真值
			while(getline(file_gt_tracking, line_est_obj))
			{
				item_obj_gt.push_back(line_est_obj);
				istringstream istr(line_est_obj);
				istr >> elem;
				frame_id = atoi(elem.c_str());
				
				if (frame_id != i)
				{
					first_str_frame = item_obj_gt.back();
					item_obj_gt.pop_back();
					line_est_obj.clear();
					break;
				}
				// 每一帧只保存那些有真值标注的物体（因为有些物体太远了，无法通过激光雷达来标注）	
				istr >> elem;
				obj_id = atoi(elem.c_str());
				if (obj_id == -1) 
				{
					item_obj_gt.pop_back();
					line_est_obj.clear();
					continue;
				}

				istr >> elem;
				// 忽略非刚体或铰接（太大，如火车或电车）的目标
				// 注意，C++中判断两个string对象是否完全相同时，可以用==，也可以使用string对象的成员函数compare()来判断（如果相等则返回0）
				// 如果需要比较的是C类型的字符串（char*)，则不能使用==（这是查看两个字符串g的地址是否相同），而是要用C语言库的strcmp()函数（且该函数只能用于比较字符串，不能比较数字，如果硬要比较，也是按照字符串来比较）！！
				if(elem == "Pedestrian" || elem == "Person_sitting" || elem == "Cyclist" || elem == "Tram" || elem == "Misc")
				{
					item_obj_gt.pop_back();
					line_est_obj.clear();
					continue;
				}
			}
		}

		// 保存当前帧所有gt物体的局部位姿
		ObjPoseParsingKT(item_obj_gt, cur_frame_obj_id_pose);

		if (i == 0) 
		{
			last_cam_R_gt = cur_cam_R_gt;
			last_cam_P_gt = cur_cam_P_gt;
			last_frame_obj_id_pose = cur_frame_obj_id_pose;
			last_cam_R_est = cur_cam_R_est;
			last_cam_P_est = cur_cam_P_est;
			cur_frame_obj_id_pose.clear();
			
			continue;
		}

		// 世界坐标系下 的两帧间的相机运动
		cam_motion_R_gt = cur_cam_R_gt * last_cam_R_gt.transpose();
		cam_motion_P_gt = -cam_motion_R_gt * last_cam_P_gt + cur_cam_P_gt;

		cam_motion_R_est = cur_cam_R_est * last_cam_R_est.transpose();
		cam_motion_P_est = -cam_motion_R_est * last_cam_P_est + cur_cam_P_est;

		// 各自后一帧相机坐标系下 的两帧间的相机运动
		// cam_motion_R_gt = cur_cam_R_gt.transpose() * last_cam_R_gt;
		// cam_motion_P_gt = cur_cam_R_gt.transpose() * (last_cam_P_gt - cur_cam_P_gt);
		
		// cam_motion_R_est = cur_cam_R_est.transpose() * last_cam_R_est;
		// cam_motion_P_est = cur_cam_R_est.transpose() * (last_cam_P_est - cur_cam_P_est);

		// 计算相对位姿（即前后两帧间的运动变换）的精度
		cal_error_motion_cam(cam_motion_R_gt, cam_motion_P_gt, cam_motion_R_est, cam_motion_P_est, total_err_cam_motion_R, total_err_cam_motion_P, vec_err_cam_motion_R, vec_err_cam_motion_P);
		
		// 物体的运动估计结果
		file_est_obj.open((OUTPUT_FOLDER_OBJS + "/"  + sequence + "/result_objs/" + ss.str() + ".txt").c_str(), ios::in);
		if(!file_est_obj.is_open())
		{
			printf("cannot open file: %s/%s/result_objs/%s.txt\n",OUTPUT_FOLDER_OBJS.c_str(),sequence.c_str(),ss.str());
			// ROS_BREAK();
			return 0;
		}
		// int line_num = 0, c;
		// do
		// {
		// 	c = fgetc(file_est_obj);
		//  注意，用每一行最后的换行符来统计行数，则如果最后一行数据之后没有多一行空行，则line_num的初始值应该是1？
		// 	if (c =='\n')
		// 		line_num++;
		// }while(c != EOF);
		
		set<int> matched_gt_obj = cal_error_motion_objs(pt_dep_obj_no_gt, item_obj_gt, file_est_obj, last_frame_obj_id_est_gt, cur_frame_obj_id_est_gt, last_frame_obj_id_pose,
														cur_frame_obj_id_pose, last_cam_R_gt, last_cam_P_gt, cur_cam_R_gt, cur_cam_P_gt, num_true_track, num_false_track, total_r_rpe_obj, total_t_rpe_obj);

		// 查找前后2帧的gt动态物体中是否还有被当前帧所漏检测和跟踪
		for(auto &iter: cur_frame_obj_id_pose)
		{
			int obj_id_gt = iter.first;
			if(matched_gt_obj.find(obj_id_gt) != matched_gt_obj.end()) continue;
			// 如果该gt物体在前后两帧均出现，则查看它是否为动态物体
			if(last_frame_obj_id_pose.find(obj_id_gt) != last_frame_obj_id_pose.end())
			{
				// 表示的是上一帧相机坐标系下的物体点坐标 转换到 当前帧相机坐标系下的物体坐标点，不论是否为动态
				// Matrix3f motion_R = cur_frame_obj_id_pose[obj_id_gt].first * last_frame_obj_id_pose[obj_id_gt].first.transpose();
				// Vector3f motion_P = -motion_R * last_frame_obj_id_pose[obj_id_gt].second + cur_frame_obj_id_pose[obj_id_gt].second;
				
				// 物体位姿的这个变换是表达相机坐标系下的，对应的，要用来做比较的相机运动变换也是要在相机坐标系下的
				Matrix3f motion_R = cur_frame_obj_id_pose[obj_id_gt].first.transpose() * last_frame_obj_id_pose[obj_id_gt].first;
				Vector3f motion_P = cur_frame_obj_id_pose[obj_id_gt].first.transpose() * (last_frame_obj_id_pose[obj_id_gt].second - cur_frame_obj_id_pose[obj_id_gt].second);

				Matrix3f cam_motion_R_gt_c = cur_cam_R_gt.transpose() * last_cam_R_gt;
				Vector3f cam_motion_P_gt_c = cur_cam_R_gt.transpose() * (last_cam_P_gt - cur_cam_P_gt);

				Matrix3f err_R = cam_motion_R_gt_c.transpose() * motion_R;
				Vector3f err_P = cam_motion_R_gt_c.transpose() * (motion_P - cam_motion_P_gt_c);

				float t_rpe = err_P.norm();
				float trace_rpe = 0;
				for (int i = 0; i < 3; ++i)
				{
					if (err_R(i,i)>1.0)
						trace_rpe = trace_rpe + 1.0-(err_R(i,i)-1.0);
					else
						trace_rpe = trace_rpe + err_R(i,i);
				}
				float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.14;
				// 衡量静态的标准？物体运动和相机运动之间的差距应该定为多少？
				if(r_rpe > 3.0 || t_rpe > 0.3) ++num_lost_track;
			}
		}

		last_cam_R_gt = cur_cam_R_gt;
		last_cam_P_gt = cur_cam_P_gt;
		last_frame_obj_id_pose = cur_frame_obj_id_pose;
		last_cam_R_est = cur_cam_R_est;
		last_cam_P_est = cur_cam_P_est;
		last_frame_obj_id_est_gt = cur_frame_obj_id_est_gt;
		item_obj_gt.clear();
		file_est_obj.close();
		cur_frame_obj_id_pose.clear();
		cur_frame_obj_id_est_gt.clear();
	}

	file_gt_pose_cam.close();
	file_gt_tracking.close(); 
	file_est_cam.close();

	float ave_err_cam_R = total_err_cam_R/num_img;
	float ave_err_cam_P = total_err_cam_P/num_img;
	float ave_err_cam_motion_R = total_err_cam_motion_R/(num_img-1);
	float ave_err_cam_motion_P = total_err_cam_motion_P/(num_img-1);
	float ave_r_rpe_obj, ave_t_rpe_obj;
	printf("Average error of absolute trajectory of camera: R_err:%6f, P_err:%6f\n", ave_err_cam_R, ave_err_cam_P);
	printf("Average error of relative motion of camera: R_err:%6f, P_err:%6f\n", ave_err_cam_motion_R, ave_err_cam_motion_P);
	if(num_true_track > 0)
	{
		ave_r_rpe_obj = total_r_rpe_obj/num_true_track;
		ave_t_rpe_obj = total_t_rpe_obj/num_true_track;
		printf("Average error of relative motion of all objects: R_err:%6f, P_err:%6f\n", ave_r_rpe_obj, ave_t_rpe_obj);
		printf("Number of true tracking is %d\n", num_true_track);
	}
	else
	{
		printf("Has no true obj tracking!\n");
	}

	printf("Number of false tracking is %d\n", num_false_track);
	printf("Number of lost tracking is %d\n", num_lost_track);


	// vec_err_cam_R, vec_err_cam_P, vec_err_cam_motion_R, vec_err_cam_motion_P;
	FILE* outFile_err_cam = fopen((OUTPUT_FOLDER + "/" + sequence + "/result_cam/err_est_cam.txt").c_str(),"w");
	if(outFile_err_cam == NULL) 
	{
		printf("Output path dosen't exist: %s\n", (OUTPUT_FOLDER + "/" + sequence + "/result_cam").c_str());
	}

	fprintf(outFile_err_cam, "%f %f %f %f\n", vec_err_cam_R[0], 0, 0.0, 0.0);
	for(int k = 0; k < vec_err_cam_motion_R.size(); ++k)
	{
		fprintf(outFile_err_cam, "%f %f %f %f\n", vec_err_cam_R[k+1], vec_err_cam_P[k+1], vec_err_cam_motion_R[k], vec_err_cam_motion_P[k]);																		
	}
	fclose(outFile_err_cam);
}

void cal_error_pose_cam(const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, const Matrix3f &cur_cam_R_est, const Vector3f &cur_cam_P_est, 
						float &total_r_rpe, float &total_t_rpe, vector<float> &vec_err_cam_R, vector<float> &vec_err_cam_P)
{
	// 相机坐标系下的位姿误差
	// Matrix3f err_R = cur_cam_R_gt.transpose() * cur_cam_R_est;
	// Vector3f err_P = cur_cam_R_gt.transpose() * (cur_cam_P_est - cur_cam_P_gt);

	// 世界坐标系下的位姿误差
	Matrix3f err_R = cur_cam_R_gt * cur_cam_R_est.transpose();
	Vector3f err_P = -err_R * cur_cam_P_est + cur_cam_P_gt;

	float t_rpe = err_P.norm();
	// float t_rpe = std::sqrt(err_P(0)*err_P(0) + err_P(1)*err_P(1) + err_P(2)*err_P(2));

	float trace_rpe = 0;
	for (int i = 0; i < 3; ++i)
	{
		if (err_R(i,i)>1.0)
			trace_rpe = trace_rpe + 1.0-(err_R(i,i)-1.0);
		else
			trace_rpe = trace_rpe + err_R(i,i);
	}
	float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.14;

	vec_err_cam_R.push_back(r_rpe);
	vec_err_cam_P.push_back(t_rpe);
	total_r_rpe += r_rpe;
	total_t_rpe += t_rpe;

}

void cal_error_motion_cam(const Matrix3f &motion_R_gt, const Vector3f &motion_P_gt, const Matrix3f &motion_R_est, const Vector3f &motion_P_est, 
						float &total_r_rpe, float &total_t_rpe, vector<float> &vec_err_cam_motion_R, vector<float> &vec_err_cam_motion_P)
{
	Matrix3f err_R = motion_R_gt.transpose() * motion_R_est;
	Vector3f err_P = motion_R_gt.transpose() * (motion_P_est - motion_P_gt);

	float t_rpe = err_P.norm();
	float trace_rpe = 0;
	for (int i = 0; i < 3; ++i)
	{
		if (err_R(i,i)>1.0)
			trace_rpe = trace_rpe + 1.0-(err_R(i,i)-1.0);
		else
			trace_rpe = trace_rpe + err_R(i,i);
	}
	float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.14;

	vec_err_cam_motion_R.push_back(r_rpe);
	vec_err_cam_motion_P.push_back(t_rpe);
	total_r_rpe += r_rpe;
	total_t_rpe += t_rpe;
}

// 计算当前帧所有估计的动态物体 的 运动误差
set<int> cal_error_motion_objs(vector<float> &pt_dep_obj_no_gt, vector<string> &item_obj_gt, ifstream &file_est_obj, map<int,int> &last_frame_obj_id_est_gt, map<int,int> &cur_frame_obj_id_est_gt, 
							map<int,pair<Matrix3f,Vector3f>> &last_frame_obj_id_pose, map<int,pair<Matrix3f,Vector3f>> &cur_frame_obj_id_pose, const Matrix3f &last_cam_R_gt, const Vector3f &last_cam_P_gt, 
							const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, int &num_true_track, int &num_false_track, float &total_r_rpe, float &total_t_rpe)					
{
	int obj_id;
	int cls_label;
	// left, right, top, bottom
	float bbox[4];
	Vector3f ave_pt;
	Matrix3f R_obj;
	Vector3f P_obj;
	int matched_obj_id_gt_prev;

	map<int,int> matched_obj_id = last_frame_obj_id_est_gt;

	last_frame_obj_id_est_gt.clear();

	vector<uchar> matched_gt_obj(item_obj_gt.size(),0);
	
	bool no_gt_last;
	string elem;
	string line_est_obj;

	while(getline(file_est_obj, line_est_obj))
	{
		matched_obj_id_gt_prev = -1;
		if(!line_est_obj.empty())
		{
			istringstream istr(line_est_obj);
			int id = 0;
			Matrix4f T;
			while(id < 21)
			{
				if(id == 0)
				{
					istr >> elem;
					obj_id = atoi(elem.c_str());
					++id;
				}
				else if(id == 1)
				{
					istr >> elem;
					cls_label = atoi(elem.c_str());
					++id; 
				}
				else if(id == 2)
				{
					for(int i = 0; i < 12; ++i)
					{
						istr >> elem;
						int row = i/4;
						int col = i%4;
						T(row,col) = atof(elem.c_str());
					}
					id += 12;
				}
				else if(id == 14)
				{
					for(int i = 0; i < 4; ++i)
					{
						istr >> elem;
						bbox[i] = atof(elem.c_str());
					}
					id += 4;
				}
				else if(id == 18)
				{
					for(int i = 0; i < 3; ++i)
					{
						istr >> elem;
						ave_pt(i) = atof(elem.c_str());
					}
					id += 3;
				}
				else
				{
					istr >> elem;
					++id;
				}	
			}

			R_obj(0,0) = T(0,0);
			R_obj(0,1) = T(0,1);
			R_obj(0,2) = T(0,2);
			P_obj(0)   = T(0,3);
			R_obj(1,0) = T(1,0);
			R_obj(1,1) = T(1,1);
			R_obj(1,2) = T(1,2);
			P_obj(1)   = T(1,3);
			R_obj(2,0) = T(2,0);
			R_obj(2,1) = T(2,1);
			R_obj(2,2) = T(2,2);
			P_obj(2)   = T(2,3);
			
			// 估计的动态物体在上一帧中是否有匹配到的gt物体
			if(matched_obj_id.find(obj_id) != matched_obj_id.end())
			{
				matched_obj_id_gt_prev = matched_obj_id[obj_id];
			}

			no_gt_last = match_gt_esti_cal_error(item_obj_gt, matched_gt_obj, obj_id, cls_label, bbox, ave_pt, R_obj, P_obj, 
												matched_obj_id_gt_prev, last_frame_obj_id_est_gt, cur_frame_obj_id_pose, last_frame_obj_id_pose, 
												last_cam_R_gt, last_cam_P_gt, cur_cam_R_gt, cur_cam_P_gt, total_r_rpe, total_t_rpe, num_false_track);

			// 完成了物体运动误差的计算
			if(!no_gt_last) ++num_true_track;
			// 剩下的情况就是如果某个动态物体是由于在上一帧所要匹配的gt没有标注，因此没有运动真值，则记录该物体的深度值，后续查看是否是因为该物体太近或者太远了
			if(no_gt_last && matched_obj_id_gt_prev == -1) pt_dep_obj_no_gt.push_back(ave_pt(2));

			line_est_obj.clear();
		}
		else
		{
			line_est_obj.clear();
			break;
		}
	}

	// 是否可以统计有多少动态物体在当前帧中没被跟踪到？可以统计，在主函数中进行

	// 当前帧的gt中有那些物体被匹配到了（必须该物体在上一帧中也有标注）
	set<int> matched_gt;
	if(!last_frame_obj_id_est_gt.empty())
	{
		for(auto &iter: last_frame_obj_id_est_gt)
		{
			matched_gt.insert(iter.second);
		}
	}
	
	return matched_gt;
}

// 将某一帧中的检测和运动估计的动态物体 与 该帧的gt信息 相匹配，并计算运动估计的误差
bool match_gt_esti_cal_error(vector<string> &item_obj_gt, vector<uchar> &matched_gt_obj, int &obj_id, int &cls_label,  
							float* bbox, Vector3f &ave_pt, Matrix3f &motion_R_obj, Vector3f &motion_P_obj, const int &matched_obj_id_gt_prev, 
							map<int,int> &last_frame_obj_id_est_gt, map<int,pair<Matrix3f,Vector3f>> &cur_frame_obj_id_pose, 
							map<int,pair<Matrix3f,Vector3f>> &last_frame_obj_id_pose, const Matrix3f &last_cam_R_gt, 
							const Vector3f &last_cam_P_gt, const Matrix3f &cur_cam_R_gt, const Vector3f &cur_cam_P_gt, float &total_r_rpe, float &total_t_rpe, int &num_false_track)
{
	bool no_gt_last = true;

	int obj_id_gt, obj_cls_gt;
	// left, top, right, bottom
	float bbox_gt[4];
	// 长宽高，底面中心点(x,y,z)，
	Vector3f dimen_3D_bbox_gt;
	Matrix3f cur_R_obj_gt, last_R_obj_gt, motion_R_obj_gt_c, motion_R_obj_gt_w;
	Vector3f cur_P_obj_gt, last_P_obj_gt, motion_P_obj_gt_c, motion_P_obj_gt_w;
	 
	bool false_track = false;

	for (int i = 0; i < matched_gt_obj.size(); ++i)
	{
		if(item_obj_gt[i].empty()) continue;
		// 如果该真值物体已经被匹配过，则跳过
		if (matched_gt_obj[i] == 1) continue;
		
		// 获取单个真值物体的标签信息
		ObjPoseParsingKT(item_obj_gt[i], obj_id_gt, obj_cls_gt, bbox_gt, dimen_3D_bbox_gt);
		
		{
			float score = cal_iou_bbox(bbox, bbox_gt);
			if (score >= 0.2)
			{
				// 这个位姿指的是将相机坐标系下的点投影到物体坐标系，但是要到规定的物体坐标系则还需要进行变换后的X轴和Z轴的交换
				cur_R_obj_gt = cur_frame_obj_id_pose[obj_id_gt].first;
				cur_P_obj_gt = cur_frame_obj_id_pose[obj_id_gt].second;
				// 物体特征点（或采样像素点）的3D坐标均值是否位于gt 3D bbox内部
				bool in_3D_bbox = check_center_in_3D_bbox_gt(dimen_3D_bbox_gt, cur_R_obj_gt, cur_P_obj_gt, ave_pt);
				if (in_3D_bbox) 
				{
					last_frame_obj_id_est_gt[obj_id] = obj_id_gt;
				}
				// 如果某个物体的平均3D采样点不在gt 3D边界框内，则其上点的深度估计很可能错误，则其运动估计的误差可能会很大
				// else if(score >= 0.35)
				// {
				// 	last_frame_obj_id_est_gt[obj_id] = obj_id_gt;
				// }
			}
		}
		
		// 如果该物体在当前帧找到匹配的真值物体
		if (last_frame_obj_id_est_gt.find(obj_id) != last_frame_obj_id_est_gt.end())
		{
			// 必须得该物体前后两帧均有真值位姿标注（及运动变换的真值），才认为完成运动误差计算
			// no_gt_last = false;

			// 如果前后两帧估计物体 所匹配的 gt物体 不相同，这可能吗？？如果真是如此，则说明估计物体 和 gt物体的 匹配方法有问题！！！或者是这两帧间跟踪错了物体?但是如果当前帧的匹配是对的，是上一帧错误，该怎么办？
			// 这种情况可能是因为当前帧物体的深度估计不准确，导致跟附近的物体相匹配了？可以尝试是否能与其他gt物体相匹配
			if(matched_obj_id_gt_prev != -1 && obj_id_gt != matched_obj_id_gt_prev) 
			{
				assert(false && "Something wrong with the method of matching gt object and estiamted object!!");
				last_frame_obj_id_est_gt.erase(obj_id);
				false_track = true;
				// 尝试是否能匹配到别的gt物体
				continue;
			}

			matched_gt_obj[i] = 1;

			// 如果该物体在上一帧中有位姿标注，则计算运动误差值
			if(last_frame_obj_id_pose.find(obj_id_gt) != last_frame_obj_id_pose.end())
			{
				false_track = false;
				no_gt_last = false;

				last_R_obj_gt = last_frame_obj_id_pose[obj_id_gt].first;
				last_P_obj_gt = last_frame_obj_id_pose[obj_id_gt].second;
				// 该用什么坐标系下的物体运动变换来跟真值做比较（该运动变换用于将不同坐标系下的两个刚体点直接相关联，由于同个时刻同一点在不同参考系下的坐标不同，因此这里的运动变换在不同参考系下肯定也不同）
				// gt文件下只有每一帧的物体在该帧相机坐标系下的位姿，以及各帧的相机的位姿（参考坐标系为初始帧时刻的IMU坐标系）。可以定义如下的三种坐标下的 两帧间物体运动变换
				// （1）物体在后一帧相机坐标系下的 物体运动变换 表示(从前一帧的该点在该帧相机坐标系下的坐标 直接 变换到 该点在当前帧相机坐标系下的坐标)，此时估计值motion_R_obj和motion_P_obj也是相应的坐标系下的估计值（那么物体的运动估计会受到相机运动估计的影响）
				// 用相机坐标系下的运动变换表达，方便BEV等模型直接将多帧的相机坐标系下的动态点转换到某一帧
				// motion_R_obj_gt_c = cur_R_obj_gt * last_R_obj_gt.transpose();
				// motion_P_obj_gt_c = -motion_R_obj_gt_c * last_P_obj_gt + cur_P_obj_gt;

				motion_R_obj_gt_c = cur_R_obj_gt.transpose() * last_R_obj_gt;
				motion_P_obj_gt_c = cur_R_obj_gt.transpose() * (last_P_obj_gt - cur_P_obj_gt);

				Matrix3f err_R = motion_R_obj.transpose() * motion_R_obj_gt_c;
				Vector3f err_P = motion_R_obj.transpose() * (motion_P_obj_gt_c - motion_P_obj);
				
				// （2）物体在全局参考坐标系下的运动变换，此时估计值motion_R_obj和motion_P_obj也要对应。
				// 但是由于我们使用了IMU，因此初始帧的位姿不再是I，后续计算误差时要校正初始帧的position和yaw角到全局值，因此下面的cam_P_gt和cur_cam_R_gt还要提前进行处理，使得其初始帧的位姿中的position和yaw角为0（则后续所有的gt的cam位姿都要校正）。
				// motion_R_obj_gt_w = cur_cam_R_gt * motion_R_obj_gt_c * last_cam_R_gt.transpose();
				// motion_P_obj_gt_w = - motion_R_obj_gt_w * last_cam_P_gt + cur_cam_R_gt * motion_P_obj_gt_c + cur_cam_P_gt;

				// Matrix3d err_R = motion_R_obj.transpose() * motion_R_obj_gt_w;
				// Vector3f err_P = motion_R_obj.transpose() * (motion_P_obj_gt_w - motion_P_obj);

				// （3）物体运动变换 在其 某一时刻的自身坐标系下的 表达。具体见VDO-SLAM的Tracking.cc的第978-986行代码（其中第（4）中感觉有点问题），这里暂时不用这个，因为这个同样涉及到参考坐标系的校正问题
				
				float t_rpe = std::sqrt(err_P(0)*err_P(0) + err_P(1)*err_P(1) + err_P(2)*err_P(2));
				float trace_rpe = 0;
				for (int i = 0; i < 3; ++i)
				{
					// 由旋转矩阵得到轴角中的旋转角度
					// 旋转矩阵行列式为1，且其对角线元素不能大于1。如果这里出现了大于1的情况，是不是说明这个旋转矩阵是无效的？需要怎么处理（一般这是由于计算误差导致的，大于1的情况应该是非常接近于1且略大于1，这里的处理时取其与1的对称值即可
					if (err_R(i,i)>1.0)
						trace_rpe = trace_rpe + 1.0-(err_R(i,i)-1.0);
					else
						trace_rpe = trace_rpe + err_R(i,i);
				}
				float r_rpe = acos( ( trace_rpe -1.0 )/2.0 )*180.0/3.14;
				
				if(!isfinite(r_rpe) || !isfinite(t_rpe) || abs(r_rpe) > 20.0f || abs(t_rpe) > 3.0f)
				{
					last_frame_obj_id_est_gt.erase(obj_id);
					false_track = true;
					// 尝试是否能匹配到别的gt物体
					continue;
				}
				else
				{
					total_r_rpe += abs(r_rpe);
					total_t_rpe += abs(t_rpe);
					false_track = false;
					no_gt_last = false;
				}
			}
			else
			{
				// 如果该估计物体当前帧中所匹配到的gt物体在上一帧中没有出现再gt。这有可能吗？
				// KITTI中物体的最大允许距离是非常远的（甚至超过50m),这比当前项目所考虑的物体距离(25m)要大很多，因此所跟踪的物体应该在前后两帧中都有位姿标注
				// assert(false && "Weired!");
				// ++num_false_track;
				last_frame_obj_id_est_gt.erase(obj_id);
				false_track = true;
				continue;
			}
		}
		
		// 找到了该物体的匹配，返回
		if(no_gt_last) break;
	}

	// 如果在当前帧中有物体，而在上一帧中没有，则认为是误检测或误跟踪
	if(false_track)
		++num_false_track;
	
	return no_gt_last;
}

// 获取某一帧的label文件中所有有效物体的obj id及对应的位姿
void ObjPoseParsingKT(vector<string> &item_obj_gt, map<int,pair<Matrix3f, Vector3f>> &gt_RP_objs_prev)
{
	if(item_obj_gt.size() == 0) return;
	int id = 0;
	int obj_id_gt;
	float ang_yaw;
	Vector3f P_obj_gt, P_obj_tmp;
	Matrix3f R_obj_gt;

	string obj_gt_info;
	string elem;

	for (int i = 0; i < item_obj_gt.size(); ++i)
	{
		obj_gt_info = item_obj_gt[i];
		istringstream istr(obj_gt_info);
		
		while(id < 17)
		{
			if (id == 1)
			{
				istr >> elem;
				obj_id_gt = atoi(elem.c_str());
				++id;
			}
			else if(id == 13)
			{
				for(int i = 0; i < 3; ++i)
				{
					istr >> elem;
					P_obj_gt(i) = atof(elem.c_str());
				}
				// 物体坐标系（前向为X轴，向右为Z轴，竖直向下为Y轴）的原点（物体3D bbox底面的中点）在相机坐标系（前向为Z轴，向右为X轴，竖直向下为Y轴）下的坐标
				id = id + 3;
			}
			else if(id == 16)
			{
				istr >> elem;
				ang_yaw = atof(elem.c_str());
				++id;
			}
			else
			{
				istr >> elem;
				++id;
			}
		}
		// 根据tracking数据集中的pdf中的图示，rotation_y的定义是物体坐标系的Z轴到相机坐标系的X轴的角度差值，要计算两个坐标系的旋转变换，需要计算两个坐标系的X轴的角度差值
		// 但是阅读deckit提供的相关数据处理代码后发现，实际的物体前进朝向应该是物体坐标系的X轴，Z轴应该是在侧向！
		// KITTI的第三方物体坐标系和相机坐标系的关系有点奇怪，给出的rotation_y是物体在相机坐标系中的前向朝向角度（相机的前向到物体朝向的角度，顺时针为正）再减去pi/2；
		// 使用此实际旋转角（减于pi/2后)得到物体的实际朝向之后，X轴和Z轴还要交换，这才是物体坐标系的规定坐标轴！
		// 不需要乘以-1，因为计算出的yaw角符合右手法则，即y轴向下，则顺时针的yaw角为正，逆时针为负，直接用这个角度值可以得到旋转矩阵
		float y = -1*(ang_yaw+3.1415926/2); 
		// 默认物体短时间内物体在相机坐标系中只有yaw角旋转；长期物体的roll和pitch是基于相机的roll和pitch来传递的
		float x = 0.0;
		float z = 0.0;
		// 用欧拉角的形式计算旋转矩阵
		// the angles are in radians.
		float cy = cos(y);
		float sy = sin(y);
		// float cx = cos(x);
		// float sx = sin(x);
		// float cz = cos(z);
		// float sz = sin(z);
		// float m00, m01, m02, m10, m11, m12, m20, m21, m22;
		// m00 = cy*cz+sy*sx*sz;
		// m01 = -cy*sz+sy*sx*cz;
		// m02 = sy*cx;
		// m10 = cx*sz;
		// m11 = cx*cz;
		// m12 = -sx;
		// m20 = -sy*cz+cy*sx*sz;
		// m21 = sy*sz+cy*sx*cz;
		// m22 = cy*cx;
		
		// 这里得到的R是从相机坐标系到物体坐标系的旋转矩阵
		// R_obj_gt(0,0) = m00;
		// R_obj_gt(0,1) = m01;
		// R_obj_gt(0,2) = m02;
		// R_obj_gt(1,0) = m10;
		// R_obj_gt(1,1) = m11;
		// R_obj_gt(1,2) = m12;
		// R_obj_gt(2,0) = m20;
		// R_obj_gt(2,1) = m21;
		// R_obj_gt(2,2) = m22;

		R_obj_gt(0,0) = cy;
		R_obj_gt(0,1) = 0.0;
		R_obj_gt(0,2) = -sy;
		R_obj_gt(1,0) = 0.0;
		R_obj_gt(1,1) = 1.0;
		R_obj_gt(1,2) = 0.0;
		R_obj_gt(2,0) = sy;
		R_obj_gt(2,1) = 0.0;
		R_obj_gt(2,2) = cy;

		// 注意，由于定义的旋转和位移是在相机坐标系下，因此需要先平移再旋转，否则无法按照KITTI所给定的rotation_y和物体中心坐标来正确得到相机到物体的位姿变换矩阵
		// 如果定义在局部坐标系下的旋转和平移，旋转和平移的先后顺序是有区别的！
		// https://blog.csdn.net/fanzy1234/article/details/131430384
		P_obj_tmp = -R_obj_gt*P_obj_gt;
		if(i == 0)
		{
			// R需要取逆吗？这里要得到的矩阵是将相机坐标系的点变换到物体坐标系下
			pair<Matrix3f, Vector3f> temp_pair(R_obj_gt, P_obj_tmp);
			// pair<Matrix3f, Vector3f> temp_pair(R_obj_gt.transpose(), -P_obj_gt);
			gt_RP_objs_prev.insert(make_pair(obj_id_gt,temp_pair));
		}
		else
			gt_RP_objs_prev[obj_id_gt] = std::pair<Matrix3f, Vector3f>(R_obj_gt, P_obj_tmp);
			// gt_RP_objs_prev[obj_id_gt] = std::pair<Matrix3f, Vector3f>(R_obj_gt.transpose(), -P_obj_gt);
	}
}

// 获取单个gt物体的一些信息
void ObjPoseParsingKT(string item_obj_gt, int &obj_id_gt, int &obj_cls_gt, float bbox_gt[], Vector3f &dimen_3D_bbox_gt)
{
	string elem;
	istringstream istr(item_obj_gt);
	int id = 0;
	while(id < 17)
	{
		if (id == 1)
		{
			istr >> elem;
			obj_id_gt = atoi(elem.c_str());
			++id;
		}
		else if(id == 2)
		{
			istr >> elem;
			obj_cls_gt = atoi(elem.c_str());
			++id;
		}
		else if(id == 6)
		{
			// gt中bbox四个坐标的顺序是left,top,right,bottom
			for(int i = 0; i < 4; ++i)
			{
				istr >> elem;
				bbox_gt[i] = atof(elem.c_str());
			}
			id = id+4;
		}
		else if(id == 10)
		{
			for(int i = 0; i < 3; ++i)
			{
				istr >> elem;
				// label中物体3D框维度的顺序是h、w、l，分别是物体坐标系的y、z和x轴
				dimen_3D_bbox_gt[i] = atof(elem.c_str());
			}
			id = id+3;
		}
		else
		{
			istr >> elem;
			++id;
		}
	}
	
}

float cal_iou_bbox(float bbox_est[], float bbox_gt[])
{
	float aleft   = bbox_est[0];
	float aright  = bbox_est[1];
	float atop    = bbox_est[2];
	float abottom = bbox_est[3];
	// 两者bbox的坐标存放顺序不同
	float bleft   = bbox_gt[0];
	float btop    = bbox_gt[1];
	float bright  = bbox_gt[2];
	float bbottom = bbox_gt[3];

	float cleft 	= max(aleft, bleft);
	float ctop 		= max(atop, btop);
	float cright 	= min(aright, bright);
	float cbottom 	= min(abottom, bbottom);
	
	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if(c_area == 0.0f)
		return 0.0f;
	
	float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
	float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
	return c_area / (a_area + b_area - c_area);
}

// 计算估计的运动物体的3D点的均值 是否 在该真值物体的3D bbox内部。这里由于物体坐标系和相机坐标系的相对位姿关系已知，且物体的bbox在物体坐标系下是与坐标轴平行的，因此可以将相机坐标系下的点变换到物体坐标系下，简单比较坐标极限值即可
// 如果长方体与估计点是在同一个坐标系下，且长方体的位姿为任意值，则判断该点是否在长方体内部最好的办法就是用向量叉积来判断 https://blog.csdn.net/hit1524468/article/details/79857665
bool check_center_in_3D_bbox_gt(Vector3f dimen_3D_bbox_gt, Matrix3f R_obj_gt, Vector3f P_obj_gt, Vector3f ave_pt)
{
	bool in_bbox = false;
	// 将相机坐标系下的3D点变换到物体坐标系下
	// Vector3f ave_pt_obj_cood = R_obj_gt.transpose() * ave_pt - R_obj_gt.transpose() * P_obj_gt;
	Vector3f ave_pt_obj_cood = R_obj_gt * ave_pt + P_obj_gt;
	cout << "ave_pt: " << ave_pt.transpose() << endl;
	// 注意，物体坐标系的X轴（前向）和Z轴（右向）相对于相机的坐标轴规定而言是互换的，因此这里变换后的第一个坐标其实是物体坐标系中的z坐标，而第三个坐标是物体坐标系中的x坐标！！
	float z = ave_pt_obj_cood(0);
	float y = ave_pt_obj_cood(1);
	float x = ave_pt_obj_cood(2);
	// dimen_3D_bbox_gt是物体的3D框尺寸真值，其顺序是h、w、l，即高、宽、长，对应的分别是车辆坐标系的y轴（向下）、z轴（向前）和x轴（向右）
	float h = dimen_3D_bbox_gt(0);
	float w = dimen_3D_bbox_gt(1);
	float l = dimen_3D_bbox_gt(2);
	if (x >= -l/2*1.25 && x <= l/2*1.25 && z >= -w/2*1.25 && z <= w/2*1.25 && y >= -h*1.15 && y <= 0) in_bbox = true;
	cout << "ets ave_x: " << x << " ets ave_y: " << y << " ets ave_z: " << z << " l/2: " << l/2 << " h: " << h << " w/2: " << w/2 << endl;
	return in_bbox;
}
