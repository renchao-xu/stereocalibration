#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include "fisheye.h"

#include <omp.h>



namespace BROWNPNP {
  class pnp {
    public:
      static pnp& getInstance() {
        static pnp instance;
        return instance;
      }
      void setpcdPattern(const std::vector<Eigen::Vector3d> &pcdPattern) {
        std::vector<Eigen::Vector3d> ().swap(_pcdPattern);
        _pcdPattern = pcdPattern;
      }
      void setimgPattern(const std::vector<Eigen::Vector2d> &imgPattern) {
        std::vector<Eigen::Vector2d> ().swap(_imgPattern);
        _imgPattern = imgPattern;
      }
      void setcamMatrix(const Eigen::Matrix3d &camMatrix) {
        _camMatrix << camMatrix;
      }
      void setcamDis(const Eigen::Vector3d &k_dis, const Eigen::Vector2d &p_dis) {
		  _k_dis << k_dis;
		  _p_dis << p_dis;
      }
      inline void setInitGuess(const Eigen::Matrix4d& init_guess) {
        init_guess_ << init_guess;
        final_tf_ = init_guess_;
      }
      bool solve() {
          return estimateTransformation();
      }

      Eigen::Matrix4d getFinalTransformation() {
        return final_tf_;
      }

    protected:
      std::vector<Eigen::Vector3d> _pcdPattern;
      std::vector<Eigen::Vector2d> _imgPattern;
      Eigen::Matrix4d init_guess_, final_tf_;
      Eigen::Matrix3d _camMatrix;
	  Eigen::Vector3d _k_dis;
	  Eigen::Vector2d _p_dis;

      struct PNPCost {
      public:
		  PNPCost(Eigen::Vector3d &p_pcd, Eigen::Vector2d &p_img, Eigen::Matrix3d &cam_matrix, Eigen::Vector3d &k_dist, Eigen::Vector2d &p_dist)
			  : _p_pcd(p_pcd), _p_img(p_img), _cam_matrix(cam_matrix), _k_dist(k_dist), _p_dist(p_dist)
        {}

        ~PNPCost() {}

        template <typename T>
        bool operator() (const T* const q_, const T* const tvec_, T* residual) const {
          Eigen::Quaternion<T> q(q_[0], q_[1], q_[2], q_[3]);
          Eigen::Matrix<T, 3, 1> _tvec(tvec_[0], tvec_[1], tvec_[2]);

          const T fx = T(_cam_matrix(0, 0));
          const T alpha = T(_cam_matrix(0, 1));
          const T u0 = T(_cam_matrix(0, 2));
          const T fy = T(_cam_matrix(1, 1));
          const T v0 = T(_cam_matrix(1, 2));

          const T k1 = T(_k_dist(0));
          const T k2 = T(_k_dist(1));
          const T p1 = T(_p_dist(0));
          const T p2 = T(_p_dist(1));
		  const T k3 = T(_k_dist(2));

          // ceres::QuaternionRotatePoint(q, p_21, p_21);
          Eigen::Matrix<T, 3, 1> p_dev = q.toRotationMatrix() * _p_pcd + _tvec ;
          const T X = T(p_dev(0, 0));
          const T Y = T(p_dev(1, 0));
          const T Z = T(p_dev(2, 0));

          const T xp = X / Z;
          const T yp = Y / Z;

          T r_2 = xp*xp + yp*yp;
    
          // 径向畸变
          T xdis = xp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2);
          T ydis = yp*(T(1.) + k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2);

          // 切向畸变
          xdis = xdis + T(2.)*p1*xp*yp + p2*(r_2 + T(2.)*xp*xp);
          ydis = ydis + p1*(r_2 + T(2.)*yp*yp) + T(2.)*p2*xp*yp;

          const T u = fx * xdis + alpha * ydis + u0;
          const T v = fy * ydis + v0;

          residual[0] = u - T(_p_img(0));
          residual[1] = v - T(_p_img(1));

          return true;
        }

        static ceres::CostFunction* Create(Eigen::Vector3d p_pcd, Eigen::Vector2d p_img, Eigen::Matrix3d cam_matrix, Eigen::Vector3d k_dist, Eigen::Vector2d p_dist) {
          return (new ceres::AutoDiffCostFunction<PNPCost, 2, 4, 3>(new PNPCost(p_pcd, p_img, cam_matrix, k_dist, p_dist)));
        }

      private:
        const Eigen::Vector3d _p_pcd;
        const Eigen::Vector2d _p_img;
        const Eigen::Matrix3d _cam_matrix;
        const Eigen::Vector3d _k_dist;
		const Eigen::Vector2d _p_dist;
      };

    private:
      bool estimateTransformation() {
        Eigen::Matrix4d tf = init_guess_;
        Eigen::Quaterniond q(tf.block<3, 3>(0, 0));
        double q_[4] = {q.w(), q.x(), q.y(), q.z()};
        Eigen::Vector3d tvec = tf.block<3,1>(0, 3);
        double tvec_[3] = {tvec[0], tvec[1], tvec[2]};

        ceres::Problem problem;
        ceres::LocalParameterization* local_param = new ceres::QuaternionParameterization();
        for (int i = 1; i < _pcdPattern.size(); i++) {
          Eigen::Vector3d p_src = _pcdPattern[i];
          Eigen::Vector2d p_dst = _imgPattern[i];
          problem.AddParameterBlock(q_, 4, local_param);
          problem.AddParameterBlock(tvec_, 3);

          ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<PNPCost, 2, 4, 3>(new PNPCost(p_src, p_dst, _camMatrix, _k_dis, _p_dis));
          problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.8), q_, tvec_);
        }

        ceres::Solver::Options options;
        // options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-10;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        //std::cout << summary.BriefReport() << std::endl;

        q.coeffs() << q_[1], q_[2], q_[3], q_[0];
        tvec << tvec_[0], tvec_[1], tvec_[2];
        final_tf_.block<3, 3>(0, 0) = q.toRotationMatrix();
        final_tf_.block<3, 1>(0, 3) = tvec;
        return true;
      }

      pnp() {}
      ~pnp() {}
      pnp(const pnp&) = delete;
      pnp& operator = (const pnp&) = delete;
	};
}


namespace FISHEYEPNP {

  double repro_err(const Eigen::Matrix<double, Eigen::Dynamic, 2> &imgps,
                    const Eigen::Matrix<double, Eigen::Dynamic, 3> &points,
                    const struct Calib_param &calib_param) {
    const size_t N = imgps.rows();

    Eigen::Matrix<double, Eigen::Dynamic, 2> uv_points;
    fisheyemodel::fisheye Fisheye;
    Fisheye.projectPoints(points, calib_param.rvec, calib_param.tvec, calib_param.matrix, calib_param.dist, uv_points);
    const Eigen::MatrixXd uv_err = (uv_points - imgps).transpose();
    const Eigen::VectorXd errs = Eigen::VectorXd::Map(uv_err.data(), 2 * N, 1);

    const double err = errs.dot(errs);
    
    return sqrt(err/imgps.rows());
  }

  class pnp {
   public:
    static pnp& getInstance() {
      static pnp instance;
      return instance;
    }

    void setpcdPattern(const std::vector<Eigen::Vector3d> &pcdPattern) {
      std::vector<Eigen::Vector3d> ().swap(_pcdPattern);
      _pcdPattern = pcdPattern;
    }
    void setimgPattern(const std::vector<Eigen::Vector2d> &imgPattern) {
      std::vector<Eigen::Vector2d> ().swap(_imgPattern);
      _imgPattern = imgPattern;
    }
    void setcamMatrix(const Eigen::Matrix3d &camMatrix) {
      _camMatrix << camMatrix;
    }
    void setcamDis(const Eigen::Vector4d& dist) {
      _dist << dist;
    }
    inline void setInitGuess(const Eigen::Matrix4d& init_guess) {
      init_guess_ << init_guess;
      final_tf_ = init_guess_;
    }

    bool solve() {
        return estimateTransformation();
    }

    Eigen::Matrix4d getFinalTransformation() {
      return final_tf_;
    }

   protected:
    std::vector<Eigen::Vector3d> _pcdPattern;
    std::vector<Eigen::Vector2d> _imgPattern;
    Eigen::Matrix4d init_guess_, final_tf_;
    Eigen::Matrix3d _camMatrix;
    Eigen::Vector4d _dist;

    struct PNPCost {
     public:
      PNPCost (Eigen::Vector3d p_pcd, Eigen::Vector2d p_img, Eigen::Matrix3d cam_matrix, Eigen::Vector4d dist)
      :  _p_pcd(p_pcd), _p_img(p_img), _cam_matrix(cam_matrix), _dist(dist)
      {}

      ~PNPCost() {}

      template <typename T>
      bool operator() (const T* const q_, const T* const tvec_, T* residual) const {
        Eigen::Quaternion<T> q(q_[0], q_[1], q_[2], q_[3]);
        Eigen::Matrix<T, 3, 1> _tvec(tvec_[0], tvec_[1], tvec_[2]);

        const T fx = T(_cam_matrix(0, 0));
        const T alpha = T(_cam_matrix(0, 1));
        const T u0 = T(_cam_matrix(0, 2));
        const T fy = T(_cam_matrix(1, 1));
        const T v0 = T(_cam_matrix(1, 2));

        const T k1 = T(_dist(0));
        const T k2 = T(_dist(1));
        const T k3 = T(_dist(2));
        const T k4 = T(_dist(3));

        // ceres::QuaternionRotatePoint(q, p_21, p_21);
        Eigen::Matrix<T, 3, 1> p_dev = q.toRotationMatrix() * _p_pcd + _tvec ;
        const T X = T(p_dev(0, 0));
        const T Y = T(p_dev(1, 0));
        const T Z = T(p_dev(2, 0));

        const T x = X / Z;
        const T y = Y / Z;


        const T r = T(ceres::sqrt(x * x + y * y));
        T theta = T(ceres::atan(r));

        const T theta2 = theta * theta;
        const T theta4 = theta2 * theta2;
        const T theta6 = theta2 * theta4;
        const T theta8 = theta4 * theta4;

        const T F_theta = T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        const T theta_d = F_theta * theta;

        const T x1r = x / r;
        const T y1r = y / r;
        const T x_d = x1r * theta_d;
        const T y_d = y1r * theta_d;

        const T u = fx * x_d + alpha * y_d + u0;
        const T v = fy * y_d + v0;

        residual[0] = u - T(_p_img(0));
        residual[1] = v - T(_p_img(1));

        return true;
      }

      static ceres::CostFunction* Create(Eigen::Vector3d p_pcd, Eigen::Vector2d p_img, Eigen::Matrix3d cam_matrix, Eigen::Vector4d dist) {
        return (new ceres::AutoDiffCostFunction<PNPCost, 2, 4, 3>(new PNPCost(p_pcd, p_img, cam_matrix, dist)));
      }

     private:
      const Eigen::Vector3d _p_pcd;
      const Eigen::Vector2d _p_img;
      const Eigen::Matrix3d _cam_matrix;
      const Eigen::Vector4d _dist;
    };

   private:
    bool estimateTransformation() {
      Eigen::Matrix4d tf = init_guess_;
      Eigen::Quaterniond q(tf.block<3, 3>(0, 0));
      double q_[4] = {q.w(), q.x(), q.y(), q.z()};
      Eigen::Vector3d tvec = tf.block<3,1>(0, 3);
      double tvec_[3] = {tvec[0], tvec[1], tvec[2]};

      ceres::Problem problem;
      ceres::LocalParameterization* local_param = new ceres::QuaternionParameterization();
      for (int i = 1; i < _pcdPattern.size(); i++) {
        Eigen::Vector3d p_src = _pcdPattern[i];
        Eigen::Vector2d p_dst = _imgPattern[i];
        problem.AddParameterBlock(q_, 4, local_param);
        problem.AddParameterBlock(tvec_, 3);

        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<PNPCost, 2, 4, 3>(new PNPCost(p_src, p_dst, _camMatrix, _dist));
        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.), q_, tvec_);
        // problem.AddResidualBlock(cost_function, NULL, q_, tvec_);
      }
      // mm
      problem.SetParameterLowerBound(tvec_, 2, 200);
      problem.SetParameterUpperBound(tvec_, 2, 4000);

      ceres::Solver::Options options;
      // options.num_threads = thread_num;
      // options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
      options.linear_solver_type = ceres::DENSE_QR;
      options.trust_region_strategy_type = ceres::DOGLEG;
      options.minimizer_progress_to_stdout = false;
      options.max_num_iterations = 500;
      // options.function_tolerance = 1e-15;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      // std::cout << summary.BriefReport() << std::endl;

      q.coeffs() << q_[1], q_[2], q_[3], q_[0];
      tvec << tvec_[0], tvec_[1], tvec_[2];
      final_tf_.block<3, 3>(0, 0) = q.toRotationMatrix();
      final_tf_.block<3, 1>(0, 3) = tvec;

      fisheyemodel::fisheye Fisheye;

      Eigen::Matrix3d RM = q.toRotationMatrix();

      Eigen::Vector3d rvec;
      Fisheye.rodriguesRvec(RM, rvec);
      struct Calib_param calib_param;
      calib_param.matrix = _camMatrix;
      calib_param.dist = _dist;
      calib_param.rvec = rvec;
      calib_param.tvec = tvec;
      Eigen::Matrix<double, Eigen::Dynamic, 2> imgps;
      Eigen::Matrix<double, Eigen::Dynamic, 3> points;
      imgps.resize(_pcdPattern.size(), 2);
      points.resize(_pcdPattern.size(), 3);
      for(int i=0; i<_pcdPattern.size(); i++) {
        imgps(i, 0) = _imgPattern[i](0);
        imgps(i, 1) = _imgPattern[i](1);
        points(i, 0) = _pcdPattern[i][0];
        points(i, 1) = _pcdPattern[i][1];
        points(i, 2) = _pcdPattern[i][2];
      }

      double err = repro_err(imgps, points, calib_param);
      std::cout<<"err: "<< err<<std::endl;
      if (err > 1.0)
        return false;
      return true;
    }

    pnp() {}
    ~pnp() {}
    pnp(const pnp&) = delete;
    pnp& operator = (const pnp&) = delete;

  };

} // namespace  pnp
