
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <regex>
#include <memory>
#include <vector>
#include "common.h"
#include "pnpsolve.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

bool downDirectory(const std::string& dir_src_path, std::vector<std::string>& paths, const std::regex& regex_str) {
    boost::filesystem::path dir_path(dir_src_path);
    if (!boost::filesystem::exists(dir_path))
        return false;

    if (!is_directory(dir_path)) {
        auto current_file_name = dir_path.filename().string();
        if (std::regex_match(current_file_name, regex_str))
            paths.push_back(dir_path.string());
    }
    else {
        for (auto iter : boost::filesystem::directory_iterator(dir_path)) {
            auto current_path = iter.path();
            downDirectory(current_path.string(), paths, regex_str);
        }
    }
    return true;
}

////单面cost 成立？
//int CaculateResidule(std::string name, std::vector<cv::Point2f> Ref_Pts, std::vector<cv::Point3f> Tar_Pts_normals,
//	cv::Mat intrisic, double * params, std::vector<float> &residule)
//{
//	//cv::Mat Rx = cv::Mat_<float>(3, 3) << ; 
//	std::ofstream Out_Residule(name, std::ios::out);
//	float cosTheta_x = ceres::cos(params[0]);
//	float sinTheta_x = ceres::sin(params[0]);
//	cv::Mat R_x = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, cosTheta_x, -sinTheta_x, 0, sinTheta_x, cosTheta_x);
//	float  cosTheta_y = ceres::cos(params[1]);
//	float  sinTheta_y = ceres::sin(params[1]);
//	cv::Mat R_y = (cv::Mat_<float>(3, 3) << cosTheta_y, 0, sinTheta_y, 0, 1, 0, -sinTheta_y, 0, cosTheta_y);
//	float cosTheta_z = ceres::cos(params[2]);
//	float sinTheta_z = ceres::sin(params[2]);
//	cv::Mat R_z = (cv::Mat_<float>(3, 3) << cosTheta_z, -sinTheta_z, 0, sinTheta_z, cosTheta_z, 0, 0, 0, 1);
//	cv::Mat RotationMatrix = R_x*R_y*R_z;
//	for (int i = 0; i < Tar_Pts_normals.size(); i++)
//	{
//		cv::Mat Pt_normal = (cv::Mat_<float>(3, 1) << Tar_Pts_normals[i].x, Tar_Pts_normals[i].y, Tar_Pts_normals[i].z);
//		cv::Mat AFT_Pt = RotationMatrix*Pt_normal;
//		AFT_Pt.at<float>(0, 0) = AFT_Pt.at<float>(0, 0) / AFT_Pt.at<float>(2, 0);
//		AFT_Pt.at<float>(1, 0) = AFT_Pt.at<float>(1, 0) / AFT_Pt.at<float>(2, 0);
//		AFT_Pt.at<float>(2, 0) = 1.0;
//		cv::Mat last_pt = intrisic*AFT_Pt;
//		float residle = Ref_Pts[i].y - last_pt.at<float>(1, 0);
//		residule.push_back(residle);
//		Out_Residule << last_pt.at<float>(0, 0) << "	" << last_pt.at<float>(1, 0) << "	" << residle << std::endl;
//	}
//	Out_Residule.clear();
//	Out_Residule.close();
//	return true;
//
//}

int main()
{
	std::string file_left = "E:/StereoCalibration/data/left";
	std::string file_right = "E:/StereoCalibration/data/right";
	std::regex regex_str = std::regex(".*\.jpg");
	std::vector<std::string> name_left, name_right;
	downDirectory(file_left, name_left, regex_str);
	downDirectory(file_right, name_right, regex_str);

	std::vector<std::vector<cv::Point2f> > imagePoints[2];
	std::vector<std::vector<cv::Point3f> > objectPoints;
	cv::Size boardSize(9, 6);
	if (name_left.size() != name_right.size()) {
		return -1;
	}
	int nimages = name_left.size();

	// corner detect can changged by CCT detect
	cv::Size imageSize;
	for (int i = 0; i < nimages; i++) {
		cv::Mat imgL = cv::imread(name_left[i], 0);
		cv::Mat imgR = cv::imread(name_right[i], 0);
		if (imgL.size() != imgR.size())
			continue;
		imageSize = imgL.size();
		std::vector<cv::Point2f> cornersL, cornersR;
		bool foundL = cv::findChessboardCorners(imgL, boardSize, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
		bool foundR = cv::findChessboardCorners(imgR, boardSize, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
		if (foundL && foundR && cornersL.size() == cornersR.size()) {
			cv::cornerSubPix(imgL, cornersL, cv::Size(7, 7), cv::Size(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.0001));
			cv::cornerSubPix(imgR, cornersR, cv::Size(7, 7), cv::Size(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.0001));
			imagePoints[0].push_back(cornersL);
			imagePoints[1].push_back(cornersR);
		}
		else
			continue;
	}
	objectPoints.resize(imagePoints[0].size());
	for (int i = 0; i < imagePoints[0].size(); i++) {
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(cv::Point3f(k*25, j*25, 0));
	}

	cv::Mat cameraMatrix[2], distCoeffs[2];
	
	//mono camera clibration can changged by camera calibration
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
	std::vector<cv::Mat> rvecsMat[2];
	std::vector<cv::Mat> tvecsMat[2];
	int flag = 0;
	// flag |= cv::CALIB_FIX_K1;
	//flag |= cv::CALIB_FIX_K2;
	//flag |= cv::CALIB_FIX_K3;
	 flag |= cv::CALIB_FIX_K4;
	 flag |= cv::CALIB_FIX_K5;
	// flag |= CALIB_FIX_PRINCIPAL_POINT;
	//flag |= cv::CALIB_FIX_ASPECT_RATIO;
	 //flag |= cv::CALIB_ZERO_TANGENT_DIST;
	//flag |= cv::CALIB_USE_INTRINSIC_GUESS;
	flag |= cv::CALIB_SAME_FOCAL_LENGTH;
	for (int i = 0; i < 2; i++) {
		double err = cv::calibrateCamera(objectPoints, imagePoints[i], imageSize, cameraMatrix[i], distCoeffs[i], rvecsMat[i], tvecsMat[i], flag);
		//std::cout << i << " cameraMatrix: " << cameraMatrix[i] << std::endl;
		//std::cout << i << " distCoeffs: " << distCoeffs[i] << std::endl;
		//std::cout << i << " err: " << err << std::endl << std::endl;
	}

	// point_R = R*point_L+T;

	//stereo camera calibration
	cv::Mat RR, TT, EE, FF;
	double rms = cv::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1],
		cv::Size(640, 480), RR, TT, EE, FF, cv::CALIB_FIX_INTRINSIC + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6));
	std::cout << "done with RMS error=" << rms << std::endl;

	//EssentialMat stereo camera calibration
	std::vector<std::vector<cv::Point2f>> ud_imagePoints[2];
	std::vector<cv::Point2f> ud_pts[2];
	for (int i = 0; i < 2; i++) {
		for (int k = 0; k < imagePoints[i].size(); k++) {
			std::vector<cv::Point2f> ud_pt;
			//undistortion point 2 camera1 internalMat
			cv::undistortPoints(imagePoints[i][k], ud_pt, cameraMatrix[i], distCoeffs[i], cv::noArray(), cameraMatrix[0]);
			ud_imagePoints[i].push_back(ud_pt);
			for (int t = 0; t < ud_pt.size(); t++) 
				ud_pts[i].push_back(ud_pt[t]);
		}
	}	
	// five points to find EssentialMat
	cv::Mat E_Mat = cv::findEssentialMat(ud_pts[0], ud_pts[1],
										cameraMatrix[0].at<double>(0, 0),
										cv::Point2d(cameraMatrix[0].at<double>(0, 2), 
										cameraMatrix[0].at<double>(1, 2)),
										/*cv::LMEDS*/cv::RANSAC, 
										0.99, 
										1.f);
	// recover RT , T is normal vector
	cv::Mat R, T;
	cv::recoverPose(E_Mat, ud_pts[0], ud_pts[1], R, T, cameraMatrix[0].at<double>(0, 0), cv::Point2d(cameraMatrix[0].at<double>(0, 2), cameraMatrix[0].at<double>(1, 2)));
	////T *= cv::norm(TT);
	cv::Mat rrrrrrr, tttttttt;
	{
		std::vector<Eigen::Vector4d> qvec;
		double tx = 0, ty = 0, tz = 0;
		for (int i = 0; i < rvecsMat[0].size(); i++) {
			cv::Mat RL, RR;
			cv::Rodrigues(rvecsMat[0][i], RL);
			cv::Rodrigues(rvecsMat[1][i], RR);
			cv::Mat RL2R = RR*RL.inv();
			cv::Mat TL2R = tvecsMat[1][i] - RL2R*tvecsMat[0][i];
			Eigen::Matrix3d RL2R_E;
			Eigen::Vector3d TL2R_E;
			cv::cv2eigen(RL2R, RL2R_E);
			cv::cv2eigen(TL2R, TL2R_E);
			tx += TL2R_E(0);
			ty += TL2R_E(1);
			tz += TL2R_E(2);
			Eigen::Quaterniond q(RL2R_E);
			Eigen::Vector4d q_(q.x(), q.y(), q.z(), q.w());
			qvec.push_back(q_);
		}
		Eigen::Quaterniond r = quatWAvgMarkley(qvec);
		Eigen::Matrix3d rrr = r.toRotationMatrix();
		Eigen::Vector3d ttt(tx / rvecsMat[0].size(), ty / rvecsMat[0].size(), tz / rvecsMat[0].size());
		//std::cout << rrr << std::endl;
		cv::eigen2cv(rrr, rrrrrrr);
		cv::eigen2cv(ttt, tttttttt);
	}
	//cv::Rodrigues(rvecsMat[1][0], R);
	//cv::Mat RL1;
	//cv::Rodrigues(rvecsMat[0][0], RL1);
	//R *= RL1.inv();
	//T = tvecsMat[1][0] - R*tvecsMat[0][0];
	//R = RR;
	//T = TT;
	//优化双目外参数 需要世界坐标点，初始值是平均四元素
	R = rrrrrrr;
	T = tttttttt;
	ceres_run(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], cameraMatrix[1], distCoeffs[0], distCoeffs[1], rvecsMat[0], tvecsMat[0], R, T);
	

	////stereo rectify
	//cv::Mat R1, R2, P1, P2, Q;
	//cv::Rect validRoi[2];
	////重投影矩阵Q = [[1, 0, 0, -cx]
	////	[0, 1, 0, -cy]
	////	[0, 0, 0, f]
	////	[1, 0, -1 / Tx, (cx - cx`) / Tx]]
	//cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
	//	cameraMatrix[1], distCoeffs[1],
	//	imageSize, R, T, R1, R2, P1, P2, Q,
	//	cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
	//cv::Mat rmap[2][2];
	//cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	//cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	//double f = double(Q.at<double>(2, 3)), b = -1 / double(Q.at<double>(3, 2));
	//double cx = -Q.at<double>(0, 3);
	//double cy = -Q.at<double>(1, 3);
	//double y_err = 0, num = 0;
	//for (int i = 0; i < nimages; i++) {
	//	cv::Mat imgL = cv::imread(name_left[i], 0);
	//	cv::Mat imgR = cv::imread(name_right[i], 0);
	//	remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
	//	remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
	//	std::vector<cv::Point2f> cornersL, cornersR;
	//	bool foundL = cv::findChessboardCorners(imgL, boardSize, cornersL, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
	//	bool foundR = cv::findChessboardCorners(imgR, boardSize, cornersR, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
	//	if (foundL && foundR && cornersL.size() == cornersR.size()) {
	//		auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
	//		for (int k = 0; k < cornersL.size(); k++) {
	//			y_err += std::fabs(cornersL[k].y - cornersR[k].y);
	//			num += 1;
	//			double z = f*b / (cornersR[k].x - cornersL[k].x);
	//			point_cloud->points_.push_back(Eigen::Vector3d((cornersL[k].x - cx)*z / f, (cornersL[k].y - cy)*z / f, z));
	//		}
	//		//open3d::visualization::DrawGeometries({ point_cloud }, "code plane", 640, 480);
	//	}
	//}

	//std::cout << "INTER_LINEAR y error : " << y_err / num << std::endl;

	// y_err pro_err plane_err
	double y_err = 0;
	double pro_err = 0.;

	std::vector<double> err_vec;
	std::vector<double> distance_err;
	for (int i = 0; i < nimages; i++) {
		std::vector<cv::Point3f> pt3ds;
		
		y_err += GetDyError(ud_imagePoints[0][i], cameraMatrix[0], ud_imagePoints[1][i], cameraMatrix[0], R, T);
		GetProjectError(ud_imagePoints[0][i], cameraMatrix[0], ud_imagePoints[1][i], cameraMatrix[0], R, T, pt3ds, err_vec);
		//std::vector<cv::Point3f> pt3ds1;
		//cv_triangulation(ud_imagePoints[0][i], cameraMatrix[0], ud_imagePoints[1][i], cameraMatrix[0], R, T, pt3ds1);
		
		//show 3d plane
		auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
		for (int k = 0; k < pt3ds.size(); k++){
			point_cloud->points_.push_back(Eigen::Vector3d(pt3ds[k].x, pt3ds[k].y, pt3ds[k].z));
		}
		GetPlaneFromPoints(point_cloud->points_, distance_err);
		//open3d::visualization::DrawGeometries({ point_cloud }, "code plane", 640, 480);
	}
	// plane error
	double sum = std::accumulate(std::begin(distance_err), std::end(distance_err), 0.0);
	double mean = sum / distance_err.size();
	double variance = 0.0;
	for (uint16_t i = 0; i < distance_err.size(); i++)
	{
		variance += std::pow(distance_err[i] - mean, 2);
	}
	variance = variance / distance_err.size();
	double standard_deviation = std::sqrt(variance);

	std::cout << "plane : " << std::endl;
	std::cout << "mean: " << mean << std::endl; // 均值
	std::vector<double>::iterator biggest = std::max_element(distance_err.begin(), distance_err.end()); //iterator
	std::cout << "max: " << *biggest << std::endl;
	//std::cout << <<variance << std::endl; // 方差
	std::cout << "3sigma: " << 3 * standard_deviation << std::endl << std::endl;// 标准差
	
	//y error
	double err_y = y_err / nimages;
	std::cout << "y error : " << err_y << std::endl << std::endl;

	//reproject error
	std::cout << "reproject: " << std::endl;
	sum = std::accumulate(std::begin(err_vec), std::end(err_vec), 0.0);
	mean = sum / err_vec.size();
	// 求方差与标准差
	variance = 0.0;
	for (uint16_t i = 0; i < err_vec.size(); i++)
	{
		variance += std::pow(err_vec[i] - mean, 2);
	}
	variance = variance / err_vec.size();
	standard_deviation = std::sqrt(variance);
	
	std::cout << "mean: " << mean << std::endl; // 均值
	//std::cout << variance << std::endl; // 方差
	std::cout << "3sigma: " << 3 * standard_deviation << std::endl; // 标准差
	
	return 0;
};