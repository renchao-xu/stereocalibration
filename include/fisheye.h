#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

struct Calib_param {
  Eigen::Matrix3d matrix;
  Eigen::Vector4d dist;
  Eigen::Vector3d rvec;
  Eigen::Vector3d tvec;
};

namespace fisheyemodel{
  class fisheye {
  public:
    fisheye() {}
    ~fisheye() {}
    
    void inline rodriguesRvec(
      const Eigen::Matrix3d &R,
      Eigen::Vector3d &rvec) {
        double cosa = (R.trace() - 1) / 2;
        const double theta = acos(cosa);
        if(theta < 1e-10) {
          rvec << 0.0, 0.0, 0.0;
          return;
        }
        const Eigen::Matrix3d K = (R - R.transpose()) / (2 * sin(theta));
        const Eigen::Vector3d r(K(2, 1), K(0, 2), K(1, 0));
        rvec = theta * r;

        return;
    }

    void inline rodriguesMatrix(
      const Eigen::Vector3d &rvec,
      Eigen::Matrix3d &R) {
        const double theta = rvec.norm();
        if (theta == 0) {
          R = Eigen::Matrix3d::Identity();
          return;
        }

        const Eigen::Vector3d r = rvec / theta;
        const Eigen::Matrix3d rrt = r * r.transpose();
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d N;
        N << 0, -1 * r(2), +1 * r(1),
            +1 * r(2), 0, -1 * r(0),
            -1 * r(1), +1 * r(0), 0;

        R = cos(theta) * I + (1 - cos(theta)) * rrt + sin(theta) * N;

        return;
    }

    void projectPoints(
      const Eigen::Matrix<double, Eigen::Dynamic, 3> &points,
      const Eigen::Vector3d &rvec,
      const Eigen::Vector3d &tvec,
      const Eigen::Matrix3d &matrix,
      const Eigen::Vector4d &dist,
      Eigen::Matrix<double, Eigen::Dynamic, 2> &imgps) {
        const double fx = matrix(0, 0);
        const double alpha = matrix(0, 1);
        const double u0 = matrix(0, 2);
        const double fy = matrix(1, 1);
        const double v0 = matrix(1, 2);

        const double k1 = dist(0);
        const double k2 = dist(1);
        const double k3 = dist(2);
        const double k4 = dist(3);

        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() << tvec;
        const double theta = rvec.norm();
        if (theta != 0) {
          const Eigen::Vector3d r = rvec / theta;
          transform.rotate(Eigen::AngleAxisd(theta, r));
        }

        const size_t N = points.rows();
        imgps.resize(N, 2);

        for (size_t i = 0; i < N; i++) {
          const Eigen::Vector3d point = points.row(i);
          const Eigen::Vector3d transformed_point = transform * point;

          const double X = transformed_point(0);
          const double Y = transformed_point(1);
          const double Z = transformed_point(2);

          const double x = X / Z;
          const double y = Y / Z;

          const double r = sqrt(x * x + y * y);
          double theta = atan(r);

          const double theta2 = theta * theta;
          const double theta4 = theta2 * theta2;
          const double theta6 = theta2 * theta4;
          const double theta8 = theta4 * theta4;

          const double F_theta = 1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
          const double theta_d = F_theta * theta;

          const double x1r = x / r;
          const double y1r = y / r;
          const double x_d = x1r * theta_d;
          const double y_d = y1r * theta_d;

          const double u = fx * x_d + alpha * y_d + u0;
          const double v = fy * y_d + v0;

          imgps(i, 0) = u;
          imgps(i, 1) = v;

        }

        return;
    }
    void undistortPoints(
      const Eigen::Matrix<double, Eigen::Dynamic, 2> &imgps,
      const Eigen::Matrix3d &matrix,
      const Eigen::Vector4d &dist,
      Eigen::Matrix<double, Eigen::Dynamic, 2> &undistortimgps) {
      const double fx = matrix(0, 0);
      const double alpha = matrix(0, 1);
      const double u0 = matrix(0, 2);
      const double fy = matrix(1, 1);
      const double v0 = matrix(1, 2);

      const double k1 = dist(0);
      const double k2 = dist(1);
      const double k3 = dist(2);
      const double k4 = dist(3);

      undistortimgps.resize(imgps.rows(), 2);
      for (size_t i = 0; i < imgps.rows(); i++) {
        const double u = imgps(i, 0);
        const double v = imgps(i, 1);

        const double y_d = (v - v0) / fy;
        const double x_d = (u - alpha * y_d - u0) / fx;

        double theta_d = sqrt(x_d * x_d + y_d * y_d);
        theta_d = std::min(std::max(-CV_PI/2.0, theta_d), CV_PI/2.0);

        bool converged = false;
        double theta = theta_d;

        double scale = 0.0;

        if (fabs(theta_d) > 1e-8) {
          for (int j = 0; j < cycle_max; j++) {
              double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
              double k1_theta2 = k1 * theta2, k2_theta4 = k2 * theta4, k3_theta6 = k3 * theta6, k4_theta8 = k4 * theta8;
              double theta_fix = (theta * (1 + k1_theta2 + k2_theta4 + k3_theta6 + k4_theta8) - theta_d) /
                                (1 + 3*k1_theta2 + 5*k2_theta4 + 7*k3_theta6 + 9*k4_theta8);
              theta = theta - theta_fix;
              if (fabs(theta_fix) < EPS) {
                  converged = true;
                  break;
              }
          }

          scale = std::tan(theta) / theta_d;
        }
        else {
            converged = true;
        }

        double u_d_ = -10000.0, v_d_ = -10000.0;
        if (converged) {
          double ys = x_d * scale;
          double y_d_ = y_d * scale;
          u_d_ = ys * fx + alpha * y_d_ + u0;
          v_d_ = ys * fy + v0;
        }
        
        undistortimgps(i, 0) = u_d_;
        undistortimgps(i, 1) = v_d_;
      }

      return;
  }

  void initUndistortRectifyMap(
    const size_t &width,
    const size_t &height,
    const Eigen::Matrix3d &matrix,
    const Eigen::Vector4d &dist,
    const double &ratio1,
    const double &ratio2,
    cv::Mat &mapx,
    cv::Mat &mapy) {
    Eigen::Matrix3d matrix_new;
    matrix_new << matrix;
    matrix_new(0, 0) *= ratio2;
    matrix_new(1, 1) *= ratio2;
    matrix_new(0, 2) *= ratio1 * ratio2;
    matrix_new(1, 2) *= ratio1 * ratio2;

    const size_t width_new = ratio1 * ratio2 * width;
    const size_t height_new = ratio1 * ratio2 * height;
    mapx.create(height_new, width_new, CV_32FC1);
    mapy.create(height_new, width_new, CV_32FC1);

    const double fx = matrix(0, 0);
    const double alpha = matrix(0, 1);
    const double u0 = matrix(0, 2);
    const double fy = matrix(1, 1);
    const double v0 = matrix(1, 2);

    const double k1 = dist(0);
    const double k2 = dist(1);
    const double k3 = dist(2);
    const double k4 = dist(3);

    const double fx_new = matrix_new(0, 0);
    const double alpha_new = matrix_new(0, 1);
    const double u0_new = matrix_new(0, 2);
    const double fy_new = matrix_new(1, 1);
    const double v0_new = matrix_new(1, 2);

    for (size_t u = 0; u < width_new; u++) {
      for (size_t v = 0; v < height_new; v++) {
        const double y = (v - v0_new) / fy_new;
        const double x = (u - alpha_new * y - u0_new) / fx_new;
        const double r = sqrt(x * x + y * y);
        const double theta = atan(r);

        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta2 * theta4;
        const double theta8 = theta4 * theta4;

        const double F_theta = 1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        const double theta_d = F_theta * theta;

        const double x1r = x / r;
        const double y1r = y / r;
        const double x_d = x1r * theta_d;
        const double y_d = y1r * theta_d;

        const double u_d = fx * x_d + alpha * y_d + u0;
        const double v_d = fy * y_d + v0;

        mapx.ptr<float>(v)[u] = u_d;
        mapy.ptr<float>(v)[u] = v_d;
      }
    }

      return;
  }

  void undistortImage(
    cv::Mat &distort_img,
    cv::Mat &undistort_img,
    const Eigen::Matrix3d &matrix,
    const Eigen::Vector4d &dist,
    const double &ratio1,
    const double &ratio2) {
    cv::Mat mapx, mapy;
    size_t img_wight = distort_img.cols;
    size_t img_height = distort_img.rows;
    initUndistortRectifyMap(img_wight, img_height, matrix, dist, ratio1, ratio2, mapx, mapy);
    cv::remap(distort_img, undistort_img, mapx, mapy, cv::INTER_LINEAR);
    
    return;
  }

  private:
    const double EPS = 1e-8;
    const size_t cycle_max = 5e2;
  };
}