#ifndef COMMON_H
#define COMMON_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "pnpsolve.h"


using Eigen::Map;

/// Trait used for double type
using EigenDoubleTraits = Eigen::NumTraits<double>;

/// 3d vector using double internal format
using Vec3 = Eigen::Vector3d;

/// 2d vector using int internal format
using Vec2i = Eigen::Vector2i;

/// 2d vector using float internal format
using Vec2f = Eigen::Vector2f;

/// 3d vector using float internal format
using Vec3f = Eigen::Vector3f;

/// 9d vector using double internal format
using Vec9 = Eigen::Matrix<double, 9, 1>;

/// Quaternion type
using Quaternion = Eigen::Quaternion<double>;

/// 3x3 matrix using double internal format
using Mat3 = Eigen::Matrix<double, 3, 3>;

/// 3x4 matrix using double internal format
using Mat34 = Eigen::Matrix<double, 3, 4>;

/// 2d vector using double internal format
using Vec2 = Eigen::Vector2d;

/// 4d vector using double internal format
using Vec4 = Eigen::Vector4d;

/// 6d vector using double internal format
using Vec6 = Eigen::Matrix<double, 6, 1>;

/// 4x4 matrix using double internal format
using Mat4 = Eigen::Matrix<double, 4, 4>;

/// generic matrix using unsigned int internal format
using Matu = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>;

/// 3x3 matrix using double internal format with RowMajor storage
using RMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

//-- General purpose Matrix and Vector
/// Unconstrained matrix using double internal format
using Mat = Eigen::MatrixXd;

/// Unconstrained vector using double internal format
using Vec = Eigen::VectorXd;

/// Unconstrained vector using unsigned int internal format
using Vecu = Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>;

/// Unconstrained matrix using float internal format
using Matf = Eigen::MatrixXf;

/// Unconstrained vector using float internal format
using Vecf = Eigen::VectorXf;

/// 2xN matrix using double internal format
using Mat2X = Eigen::Matrix<double, 2, Eigen::Dynamic>;

/// 3xN matrix using double internal format
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;

/// 4xN matrix using double internal format
using Mat4X = Eigen::Matrix<double, 4, Eigen::Dynamic>;

/// 9xN matrix using double internal format
using MatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;

//-- Sparse Matrix (Column major, and row major)
/// Sparse unconstrained matrix using double internal format
using sMat = Eigen::SparseMatrix<double>;

/// Sparse unconstrained matrix using double internal format and Row Major storage
using sRMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

void hStack(double *mat33, double *vec3, double *mat34)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mat34[i * 4 + j] = mat33[i * 3 + j];
		}
		mat34[i * 4 + 3] = vec3[i];
	}
	/*for(int i = 0; i < 3*3; i++)
	{
	mat34[i] = mat33[i];
	}
	for(int i = 0; i < 3; i++)
	{
	mat34[9+i] = vec3[i];
	}*/
}

int sgn(double pa)
{
	if (pa > 0.0)
	{
		return 1;
	}
	else if (pa < 0.0)
	{
		return -1;
	}
	return 0;
}

void svdcmp(double a[], int m, int n, double w[], double v[])
{
	int i, j, k, l, its, nm, /*aaaaa,*/ pp;
	double rv1[100];
	double x, y, z, c, f, h, s, g = 0.0;
	double scale1 = 0.0;
	double anorm = 0.0;

	for (i = 1; i <= n; i++)
	{
		l = i + 1;
		rv1[i] = scale1 * g;
		g = 0.0;
		s = 0.0;
		scale1 = 0.0;
		if (i <= m)
		{
			for (k = i; k <= m; k++)
			{
				double temp = fabs(a[(k - 1)*n + i]);
				scale1 = scale1 + fabs(a[(k - 1)*n + i]);
			}
			if (scale1 != 0.0)
			{
				for (k = i; k <= m; k++)
				{
					a[(k - 1)*n + i] = a[(k - 1)*n + i] / scale1;
					s = s + a[(k - 1)*n + i] * a[(k - 1)*n + i];
				}
				f = a[(i - 1)*n + i];
				g = -sqrt(s) * sgn(f);
				h = f * g - s;
				a[(i - 1)*n + i] = f - g;
				if (i != n)
				{
					for (j = l; j <= n; j++)
					{
						s = 0.0;
						for (k = i; k <= m; k++)
						{
							s = s + a[(k - 1)*n + i] * a[(k - 1)*n + j];
						}
						f = s / h;
						for (k = i; k <= m; k++)
						{
							a[(k - 1)*n + j] = a[(k - 1)*n + j] + f * a[(k - 1)*n + i];
						}
					}
				}
				for (k = i; k <= m; k++)
				{
					a[(k - 1)*n + i] = scale1 * a[(k - 1)*n + i];
				}
			}
		}
		w[i] = scale1 * g;
		g = 0.0;
		s = 0.0;
		scale1 = 0.0;
		if ((i <= m) && (i != n))
		{
			for (k = l; k <= n; k++)
			{
				scale1 = scale1 + fabs(a[(i - 1)*n + k]);
			}
			if (scale1 != 0.0)
			{
				for (k = l; k <= n; k++)
				{
					a[(i - 1)*n + k] = a[(i - 1)*n + k] / scale1;
					s = s + a[(i - 1)*n + k] * a[(i - 1)*n + k];
				}
				f = a[(i - 1)*n + l];
				g = -sqrt(s) * sgn(f);
				h = f * g - s;
				a[(i - 1)*n + l] = f - g;
				for (k = l; k <= n; k++)
				{
					rv1[k] = a[(i - 1)*n + k] / h;
				}
				if (i != m)
				{
					for (j = l; j <= m; j++)
					{
						s = 0.0;
						for (k = l; k <= n; k++)
						{
							s = s + a[(j - 1)*n + k] * a[(i - 1)*n + k];
						}
						for (k = l; k <= n; k++)
						{
							a[(j - 1)*n + k] = a[(j - 1)*n + k] + s * rv1[k];
						}
					}
				}
				for (k = l; k <= n; k++)
				{
					a[(i - 1)*n + k] = scale1 * a[(i - 1)*n + k];
				}
			}
		}
		if (anorm > (fabs(w[i]) + fabs(rv1[i])))
		{
			anorm = anorm;
		}
		else
		{
			anorm = fabs(w[i]) + fabs(rv1[i]);
		}
	}

	for (i = n; i >= 1; i--)
	{
		if (i < n)
		{
			if (g != 0.0)
			{
				for (j = l; j <= n; j++)
				{
					v[(j - 1)*n + i] = (a[(i - 1)*n + j] / a[(i - 1)*n + l]) / g;
				}
				for (j = l; j <= n; j++)
				{
					s = 0.0;
					for (k = l; k <= n; k++)
					{
						s = s + a[(i - 1)*n + k] * v[(k - 1)*n + j];
					}
					for (k = l; k <= n; k++)
					{
						v[(k - 1)*n + j] = v[(k - 1)*n + j] + s * v[(k - 1)*n + i];
					}
				}
			}
			for (j = l; j <= n; j++)
			{
				v[(i - 1)*n + j] = 0.0;
				v[(j - 1)*n + i] = 0.0;
			}
		}
		v[(i - 1)*n + i] = 1.0;
		g = rv1[i];
		l = i;
	}

	for (i = n; i >= 1; i--)
	{
		l = i + 1;
		g = w[i];
		if (i < n)
		{
			for (j = l; j <= n; j++)
			{
				a[(i - 1)*n + j] = 0.0;
			}
		}
		if (g != 0.0)
		{
			g = 1.0 / g;
			if (i != n)
			{
				for (j = l; j <= n; j++)
				{
					s = 0.0;
					for (k = l; k <= m; k++)
					{
						s = s + a[(k - 1)*n + i] * a[(k - 1)*n + j];
					}
					f = (s / a[(i - 1)*n + i]) * g;
					for (k = i; k <= m; k++)
					{
						a[(k - 1)*n + j] = a[(k - 1)*n + j] + f * a[(k - 1)*n + i];
					}
				}
			}
			for (j = i; j <= m; j++)
			{
				a[(j - 1)*n + i] = a[(j - 1)*n + i] * g;
			}
		}
		else
		{
			for (j = i; j <= m; j++)
			{
				a[(j - 1)*n + i] = 0.0;
			}
		}
		a[(i - 1)*n + i] = a[(i - 1)*n + i] + 1.0;
	}

	for (k = n; k >= 1; k--)
	{
		for (its = 1; its <= 30; its++)
		{
			for (l = k; l >= 1; l--)
			{
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					goto r2;
				}
				if (fabs(w[nm]) + anorm == anorm)
				{
					goto r1;
				}
			}
		r1: 		c = 0.0;
			s = 1.0;
			for (i = l; i <= k; i++)
			{
				f = s * rv1[i];
				if (fabs(f) + anorm != anorm)
				{
					g = w[i];
					h = sqrt(f * f + g * g);
					w[i] = h;
					h = 1.0 / h;
					c = (g * h);
					s = -(f * h);
					for (j = 1; j <= m; j++)
					{
						y = a[(j - 1)*n + nm];
						z = a[(j - 1)*n + i];
						a[(j - 1)*n + nm] = (y * c) + (z * s);
						a[(j - 1)*n + i] = -(y * s) + (z * c);
					}
				}
			}
		r2:         z = w[k];
			if (l == k)
			{
				if (z < 0.0)
				{
					w[k] = -z;
					for (j = 1; j <= n; j++)
					{
						v[(j - 1)*n + k] = -v[(j - 1)*n + k];
					}
				}
				//                goto r3;
				// goto r3 等同于 break
				break;
			}
			//if (its == 30)
			//{
			//	cout << "no convergence in 30 iterations" << endl;
			//}
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = sqrt(f * f + 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + fabs(g) * sgn(f))) - h)) / x;
			c = 1.0;
			s = 1.0;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = g * c;
				z = sqrt(f * f + h * h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = (x * c) + (g * s);
				g = -(x * s) + (g * c);
				h = y * s;
				y = y * c;
				for (pp = 1; pp <= n; pp++)
				{
					x = v[(pp - 1)*n + j];
					z = v[(pp - 1)*n + i];
					v[(pp - 1)*n + j] = (x * c) + (z * s);
					v[(pp - 1)*n + i] = -(x * s) + (z * c);
				}
				z = sqrt(f * f + h * h);
				w[j] = z;
				if (z != 0.0)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = -(s * g) + (c * y);
				for (pp = 1; pp <= m; pp++)
				{
					y = a[(pp - 1)*n + j];
					z = a[(pp - 1)*n + i];
					a[(pp - 1)*n + j] = (y * c) + (z * s);
					a[(pp - 1)*n + i] = -(y * s) + (z * c);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
		//r3:		aaaaa = 1;
	}

	return;
}

int svdDec(double *A, int rowM, int colN, double *U, double *W, double *V)
{
	double* Aa = (double*)malloc((1 + colN*rowM)*sizeof(double));
	double* Vd = (double*)malloc((1 + colN*colN)*sizeof(double));
	double* S = (double*)malloc((1 + rowM)*sizeof(double));
	//U.SetSize(A.NbRow, A.NbCol);
	//V.SetSize(A.NbCol, A.NbCol);
	//W.SetSize(A.NbRow, 1);
	int i, j;
	for (i = 1; i < rowM + 1; i++)
	{
		for (j = 1; j < colN + 1; j++)
		{
			Aa[(i - 1)*colN + j] = A[(i - 1) * colN + j - 1];
		}
	}

	svdcmp(Aa, rowM, colN, S, Vd);

	for (i = 1; i < rowM + 1; i++)
	{
		for (j = 1; j < colN + 1; j++)
		{
			U[(i - 1) * colN + j - 1] = Aa[(i - 1)*colN + j];
			//V(i - 1, j - 1) = Vd[(i - 1)*A.NbRow + j];
		}
		W[i - 1] = S[i];
	}
	for (i = 1; i < colN + 1; i++)
	{
		for (j = 1; j < colN + 1; j++)
			V[(i - 1) * colN + j - 1] = Vd[(i - 1)*colN + j];
	}

	free(Aa);
	free(Vd);
	free(S);

	return 0;
}

void getMinEigenValIdx(double *W, int dim, int *index)
{
	int idx = 0;
	double min_eigen_val = W[0];
	for (int i = 1; i < dim; i++)
	{
		if (W[i] < min_eigen_val)
		{
			min_eigen_val = W[i];
			idx = i;
		}
	}

	*index = idx;
}

void triangulateDLT(double *P1, double *x1, double *P2, double *x2, double *x_euclidean)
{
	int rowM = 4;
	int colN = 4;
	double *A = (double*)malloc(rowM * colN * sizeof(double));

	for (int i = 0; i < 4; i++)
	{
		A[i] = x1[0] * P1[8 + i] - x1[2] * P1[i];
		A[4 + i] = x1[1] * P1[8 + i] - x1[2] * P1[4 + i];
		A[8 + i] = x2[0] * P2[8 + i] - x2[2] * P2[i];
		A[12 + i] = x2[1] * P2[8 + i] - x2[2] * P2[4 + i];
	}

	double *U = (double*)malloc(rowM * colN * sizeof(double));
	double *V = (double*)malloc(colN * colN * sizeof(double));
	double *W = (double*)malloc(rowM * sizeof(double));

	svdDec(A, rowM, colN, U, W, V);

	int idx = 0;
	getMinEigenValIdx(W, 4, &idx);

	x_euclidean[0] = (V[idx] / V[12 + idx]);
	x_euclidean[1] = (V[4 + idx] / V[12 + idx]);
	x_euclidean[2] = (V[8 + idx] / V[12 + idx]);

	free(A); A = NULL;
	free(U); U = NULL;
	free(V); V = NULL;
	free(W); W = NULL;
}

void TriangulateDLT(const Mat34 &P1, const Vec3 &x1, const Mat34 &P2, const Vec3 &x2, Vec4 *X_homogeneous) {
	// Solve:
	// [cross(x0,P0) X = 0]
	// [cross(x1,P1) X = 0]
	Mat4 design;
	design.row(0) = x1[0] * P1.row(2) - x1[2] * P1.row(0);
	design.row(1) = x1[1] * P1.row(2) - x1[2] * P1.row(1);
	design.row(2) = x2[0] * P2.row(2) - x2[2] * P2.row(0);
	design.row(3) = x2[1] * P2.row(2) - x2[2] * P2.row(1);

	Eigen::JacobiSVD<Mat4> svd(design, Eigen::ComputeFullV);
	(*X_homogeneous) = svd.matrixV().col(3);
}

void TriangulateDLT(const Mat34 &P1, const Vec3 &x1, const Mat34 &P2, const Vec3 &x2, Vec3 *X_euclidean) {
	Vec4 X_homogeneous;
	TriangulateDLT(P1, x1, P2, x2, &X_homogeneous);
	(*X_euclidean) = X_homogeneous.hnormalized();
}

static inline int vector_cross_3x1(const double vec_left[3], const double vec_right[3], double out[3])
{
	out[0] = vec_left[1] * vec_right[2] - vec_left[2] * vec_right[1];
	out[1] = vec_left[2] * vec_right[0] - vec_left[0] * vec_right[2];
	out[2] = vec_left[0] * vec_right[1] - vec_left[1] * vec_right[0];

	return 0;
}

int Trans2Rot(const double trans[3], double rot[9], int is_disp_along_x_direction) {

	double norm_trans = sqrt(trans[0] * trans[0] + trans[1] * trans[1] + trans[2] * trans[2]);
	if (is_disp_along_x_direction == 1) {
		//x direction
		rot[0] = trans[0] / norm_trans;
		rot[1] = trans[1] / norm_trans;
		rot[2] = trans[2] / norm_trans;

		//y direction
		double vect_y[3] = { 0,1,0 };
		double vect_temp[3] = { 0,0,0 };
		vector_cross_3x1(vect_y, &rot[0], vect_temp);
		vector_cross_3x1(&rot[0], vect_temp, &rot[3]);

		norm_trans = sqrt(rot[3] * rot[3] + rot[4] * rot[4] + rot[5] * rot[5]);
		rot[3] = rot[3] / norm_trans;
		rot[4] = rot[4] / norm_trans;
		rot[5] = rot[5] / norm_trans;

		//z direction z=y*x
		vector_cross_3x1(&rot[0], &rot[3], &rot[6]);
		norm_trans = sqrt(rot[6] * rot[6] + rot[7] * rot[7] + rot[8] * rot[8]);
		rot[6] = rot[6] / norm_trans;
		rot[7] = rot[7] / norm_trans;
		rot[8] = rot[8] / norm_trans;

	}
	else {
		rot[3] = trans[0] / norm_trans;
		rot[4] = trans[1] / norm_trans;
		rot[5] = trans[2] / norm_trans;

		//x direction
		double vect_x[3] = { 1,0,0 };
		double vect_temp[3] = { 0,0,0 };
		vector_cross_3x1(vect_x, &rot[3], vect_temp);
		vector_cross_3x1(&rot[3], vect_temp, &rot[0]);

		norm_trans = sqrt(rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]);
		rot[0] = rot[0] / norm_trans;
		rot[1] = rot[1] / norm_trans;
		rot[2] = rot[2] / norm_trans;

		//z direction z=y*x
		vector_cross_3x1(&rot[0], &rot[3], &rot[6]);
		norm_trans = sqrt(rot[6] * rot[6] + rot[7] * rot[7] + rot[8] * rot[8]);
		rot[6] = rot[6] / norm_trans;
		rot[7] = rot[7] / norm_trans;
		rot[8] = rot[8] / norm_trans;
	}

	return 0;
}

void dgemm(double *A, double *B, double *C, int M)
{
	if (NULL == A || NULL == B || NULL == C)
		return;

	memset(C, 0, M*M*sizeof(double));

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			for (int k = 0; k < M; k++)
			{
				C[M * i + j] += A[M * i + k] * B[M * k + j];
			}
		}
	}
}

double dnormVec(double *x, int len, double *res)
{
	if (NULL == x || len == 0)
		return 0.0;

	double squareSum = 0.0;
	for (int i = 0; i < len; i++)
	{
		squareSum += x[i] * x[i];
	}

	*res = sqrt(squareSum);

	return 0.0;
}

double GetDyError(std::vector<cv::Point2f> &tie_pts1, cv::Mat &cam_intrin1, 
					std::vector<cv::Point2f> &tie_pts2, cv::Mat &cam_intrin2,
					cv::Mat &R, cv::Mat &T)
{
	double y_err = 0;
	double P1[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
	double P2[12];
	double rotation[9] = { R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
						R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
						R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2) };
	double trans[3] = { T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0) };
	hStack(rotation, trans, P2);

	//验证dy的变化
	double rot_right[9], rot_left[9],verify_trans[3];
	Trans2Rot(trans, rot_right, 1);
	dgemm(rot_right, rotation, rot_left, 3);

	double xp_left, yp_left, zp_left, xp_right, yp_right, zp_right;
	double xp_left_rot, yp_left_rot, zp_left_rot, xp_right_rot, yp_right_rot, zp_right_rot;

	double x1[3], x2[3];
	double pt3d[3] = { 0, 0, 0 };

	double focal_x1 = cam_intrin1.at<double>(0, 0);
	double focal_y1 = cam_intrin1.at<double>(1, 1);
	double cx1 = cam_intrin1.at<double>(0, 2);
	double cy1 = cam_intrin1.at<double>(1, 2);
	double focal_x2 = cam_intrin2.at<double>(0, 0);
	double focal_y2 = cam_intrin2.at<double>(1, 1);
	double cx2 = cam_intrin2.at<double>(0, 2);
	double cy2 = cam_intrin2.at<double>(1, 2);

	for (int i = 0; i < tie_pts1.size(); i++)
	{
		
		x1[0] = (tie_pts1[i].x - cx1) / focal_x1;
		x1[1] = (tie_pts1[i].y- cy1) / focal_y1;
		x1[2] = 1;
		x2[0] = (tie_pts2[i].x - cx2) / focal_x2;
		x2[1] = (tie_pts2[i].y - cy2) / focal_y2;
		x2[2] = 1;

		 // 计算dy的值
		 xp_left_rot = rot_left[0] * x1[0] + rot_left[1] * x1[1] + rot_left[2];
		 yp_left_rot = rot_left[3] * x1[0] + rot_left[4] * x1[1] + rot_left[5];
		 zp_left_rot = rot_left[6] * x1[0] + rot_left[7] * x1[1] + rot_left[8];

		 xp_left = xp_left_rot*focal_x1 + zp_left_rot *cx1;
		 yp_left = yp_left_rot*focal_x1 + zp_left_rot * cy1;
		 zp_left = zp_left_rot;


		 xp_right_rot = rot_right[0] * x2[0] + rot_right[1] * x2[1] + rot_right[2];
		 yp_right_rot = rot_right[3] * x2[0] + rot_right[4] * x2[1] + rot_right[5];
		 zp_right_rot = rot_right[6] * x2[0] + rot_right[7] * x2[1] + rot_right[8];

		 xp_right = xp_right_rot*focal_x2 + zp_right_rot * cx2;
		 yp_right = yp_right_rot*focal_x2 + zp_right_rot * cy2;
		 zp_right = zp_right_rot;

		 double vdiff = fabs(yp_left/zp_left - yp_right/zp_right);
		 y_err += vdiff;
		 /*printf("vdiff= %.3f\n", vdiff);*/
	}

	return y_err / tie_pts1.size();
}

void GetProjectError(std::vector<cv::Point2f> &tie_pts1, cv::Mat &cam_intrin1, 
						std::vector<cv::Point2f> &tie_pts2, cv::Mat &cam_intrin2,
						cv::Mat &R, cv::Mat &T, std::vector<cv::Point3f> &pt3ds,
						std::vector<double> &err_vec) {
	double P1[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
	double P2[12];
	double rotation[9] = { R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
		R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
		R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2) };
	double trans[3] = { T.at<double>(0,0), T.at<double>(1,0), T.at<double>(2,0) };
	hStack(rotation, trans, P2);
	double x1[3], x2[3];
	double pt3d[3] = { 0, 0, 0 };

	double focal_x1 = cam_intrin1.at<double>(0, 0);
	double focal_y1 = cam_intrin1.at<double>(1, 1);
	double cx1 = cam_intrin1.at<double>(0, 2);
	double cy1 = cam_intrin1.at<double>(1, 2);
	double focal_x2 = cam_intrin2.at<double>(0, 0);
	double focal_y2 = cam_intrin2.at<double>(1, 1);
	double cx2 = cam_intrin2.at<double>(0, 2);
	double cy2 = cam_intrin2.at<double>(1, 2);

	double resNorm = 0.0;
	for (int i = 0; i < tie_pts1.size(); i++)
	{
		x1[0] = (tie_pts1[i].x - cx1) / focal_x1;
		x1[1] = (tie_pts1[i].y - cy1) / focal_y1;
		x1[2] = 1;
		x2[0] = (tie_pts2[i].x - cx2) / focal_x2;
		x2[1] = (tie_pts2[i].y - cy2) / focal_y2;
		x2[2] = 1;
	
		triangulateDLT(P1, x1, P2, x2, pt3d);
		cv::Point3f pt(pt3d[0], pt3d[1], pt3d[2]);
		pt3ds.push_back(pt);
	
		double pt3dHnorm[2] = { pt3d[0] / pt3d[2], pt3d[1] / pt3d[2] };
		double x1p[2];
		x1p[0] = focal_x1 * pt3dHnorm[0] + cx1;
		x1p[1] = focal_y1 * pt3dHnorm[1] + cy1;
	
		x1p[0] -= tie_pts1[i].x;
		x1p[1] -= tie_pts1[i].y;
		int res = dnormVec(x1p, 2, &resNorm);
		err_vec.push_back(resNorm);
		//if(resNorm > 1)
		//    continue;
	
		//vec_err[vec_err_num++] = resNorm;
		//printf("vec_err %d : %f\n", i, resNorm);
	
	}
	return;
}

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
	return cv::Point2d
		(
			(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
			(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
			);
}

void cv_triangulation(std::vector<cv::Point2f> &tie_pts1, cv::Mat &cam_intrin1, std::vector<cv::Point2f> &tie_pts2, cv::Mat &cam_intrin2,
	cv::Mat &R, cv::Mat &t, std::vector<cv::Point3f> &pt3ds) {
	cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);
	std::vector<cv::Point2f> pts_1, pts_2;
	for (int i = 0; i < tie_pts1.size(); i++) {
		pts_1.push_back(pixel2cam(tie_pts1[i], cam_intrin1));
		pts_2.push_back(pixel2cam(tie_pts2[i], cam_intrin2));
	}
	cv::Mat pts_4d;
	cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
	for (int i = 0; i<pts_4d.cols; i++)
	{
		cv::Mat x = pts_4d.col(i);
		x /= x.at<float>(3, 0); // 从齐次坐标变换到非齐次坐标
		cv::Point3f p(
			x.at<float>(0, 0),
			x.at<float>(1, 0),
			x.at<float>(2, 0)
			);
		pt3ds.push_back(p);
	}
}

void ceres_run(const std::vector<std::vector<cv::Point3f>> &objectPoints, 
				const std::vector<std::vector<cv::Point2f>> imgpoints1,
				const std::vector<std::vector<cv::Point2f>> imgpoints2,
				const cv::Mat &cameraMatrix1, const cv::Mat &cameraMatrix2,
				const cv::Mat &distCoeffs1, const cv::Mat &distCoeffs2,
				std::vector<cv::Mat> &rvecsMat, std::vector<cv::Mat> &tvecsMat,
				cv::Mat &R, cv::Mat &T) {
	//cam1
	Eigen::Matrix3d matrix;
	cv::cv2eigen(cameraMatrix1, matrix);

	Eigen::Vector3d k_dist;
	Eigen::Vector2d p_dist;
	for (int i = 0; i < 2; i++) {
		k_dist(i) = distCoeffs1.at<double>(0, i);
	}
	k_dist(2) = distCoeffs1.at<double>(0, 4);
	for (int i = 2; i < 4; i++)
		p_dist(i) = distCoeffs1.at<double>(0, i);

	BROWNPNP::pnp& Pnp = BROWNPNP::pnp::getInstance();
	Eigen::Matrix4d init = Eigen::Matrix4d::Identity();
	Pnp.setcamMatrix(matrix);
	Pnp.setcamDis(k_dist, p_dist);
	std::vector<Eigen::Matrix4d> tfs;
	std::vector<std::vector<Eigen::Vector3d>> wp;
	std::vector<Eigen::Vector2d> img_pts_R;
	std::cout << "start pnp ceres" << std::endl;
	for (int i = 0; i < objectPoints.size(); i++) {
		std::vector<Eigen::Vector3d> world_pts;
		std::vector<Eigen::Vector2d> img_pts;

		Eigen::Matrix3d r;
		cv::Mat rmat;
		cv::Rodrigues(rvecsMat[i], rmat);
		cv::cv2eigen(rmat, r);
		init.block<3, 3>(0, 0) = r;
		init(0, 3) = tvecsMat[i].at<double>(0, 0);
		init(1, 3) = tvecsMat[i].at<double>(1, 0);
		init(2, 3) = tvecsMat[i].at<double>(2, 0);
		init(3, 3) = 1;
		//std::cout << "before : " << std::endl << init << std::endl;
		Pnp.setInitGuess(init);
		for (int k = 0; k < objectPoints[i].size(); k++) {
			world_pts.push_back(Eigen::Vector3d(objectPoints[i][k].x, objectPoints[i][k].y, objectPoints[i][k].z));
			img_pts.push_back(Eigen::Vector2d(imgpoints1[i][k].x, imgpoints1[i][k].y));
			img_pts_R.push_back(Eigen::Vector2d(imgpoints2[i][k].x, imgpoints2[i][k].y));
		}
		wp.push_back(world_pts);
		Pnp.setpcdPattern(world_pts);
		Pnp.setimgPattern(img_pts);
		Pnp.solve();
		Eigen::Matrix4d final_tf = Pnp.getFinalTransformation();

		//todo 筛选重投影误差小的点，作为外参数优化点对

		tfs.push_back(final_tf);
	}
	std::vector<Eigen::Vector3d> world_pts;
	for (int i = 0; i < tfs.size(); i++) {
		Eigen::Matrix3d r = tfs[i].block<3, 3>(0, 0);
		Eigen::Vector3d t = tfs[i].block<3, 1>(0, 3);
		for (int k = 0; k < wp[i].size(); k++) {
			Eigen::Vector3d pt = r*wp[i][k] + t;
			world_pts.push_back(pt);
		}
	}
	
	Eigen::Matrix3d r;
	cv::cv2eigen(R, r);
	init.block<3, 3>(0, 0) = r;
	init(0, 3) = T.at<double>(0, 0);
	init(1, 3) = T.at<double>(1, 0);
	init(2, 3) = T.at<double>(2, 0);
	init(3, 3) = 1;
	for (int i = 0; i < 2; i++) {
		k_dist(i) = distCoeffs2.at<double>(0, i);
	}
	k_dist(2) = distCoeffs2.at<double>(0, 4);
	for (int i = 2; i < 4; i++)
		p_dist(i) = distCoeffs2.at<double>(0, i);
	cv::cv2eigen(cameraMatrix2, matrix);
	Pnp.setcamMatrix(matrix);
	Pnp.setcamDis(k_dist, p_dist);
	Pnp.setInitGuess(init);
	std::cout << "before : " << std::endl << init << std::endl;
	Pnp.setpcdPattern(world_pts);
	Pnp.setimgPattern(img_pts_R);
	Pnp.solve();
	Eigen::Matrix4d final_tf = Pnp.getFinalTransformation();
	std::cout << "after : " << std::endl << final_tf << std::endl << std::endl;
	r = final_tf.block<3, 3>(0, 0);
	Eigen::Vector3d t = final_tf.block<3, 1>(0, 3);
	cv::eigen2cv(r, R);
	for (int i = 0; i < 3; i++) {
		T.at<double>(i, 0) = t(i);
	}
	return;
}

//% Q is an Nx4 matrix of quaternions.weights is an Nx1 vector, a weight for each quaternion.
//% Qavg is the weightedaverage quaternion
//% Markley, F.Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
//% "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
//% no. 4 (2007) : 1193 - 1197.
//function[Qavg] = quatWAvgMarkley(Q, weights)
//% Form the symmetric accumulator matrix
//M = zeros(4, 4);
//n = size(Q, 1); % amount of quaternions
//wSum = 0;
//for i = 1:n
//q = Q(i, :)';
//w_i = weights(i);
//M = M + w_i.*(q*q');
//	wSum = wSum + w_i;
//end
//% scale
//M = (1.0 / wSum)*M;
//% The average quaternion is the eigenvector of M corresponding to the maximum eigenvalue.
//% Get the eigenvector corresponding to largest eigen value
//[Qavg, ~] = eigs(M, 1);
//end

Eigen::Quaterniond quatWAvgMarkley(std::vector<Eigen::Vector4d> &q) {
	Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
	int N = q.size();
	for (int i = 0; i < N; i++) {
		M = M + q[i]*q[i].transpose();
		//std::cout << q[i] * q[i].transpose() << std::endl << std::endl;
	}
	Eigen::EigenSolver<Eigen::MatrixXd> es(M);
	Eigen::Vector4d q_(es.eigenvectors().col(0)[0].real(), es.eigenvectors().col(0)[1].real(), es.eigenvectors().col(0)[2].real(), es.eigenvectors().col(0)[3].real());
	Eigen::Quaterniond r;
	r.coeffs() << q_[0], q_[1], q_[2], q_[3];
	return r;
}


////光流法，连带世界坐标一起优化，优化过于非线性，并且没有绑定世界坐标与RT的关系，所以不建议优化
//struct SnavelyReprojectionError {
//	SnavelyReprojectionError(double observed_x, double observed_y)
//		: observed_x(observed_x), observed_y(observed_y) {}
//
//	template <typename T>
//	bool operator()(const T* const camera,
//		const T* const point,
//		T* residuals) const {
//
//		// camera[0,1,2] 是三个方向上的旋转
//		T p[3];
//
//		//根据旋转，预测空间特征点在 相机的坐标
//		ceres::AngleAxisRotatePoint(camera, point, p);
//
//		// camera[3,4,5] 是XYZ三个方向上的平移
//		p[0] += camera[3];
//		p[1] += camera[4];
//		p[2] += camera[5];
//
//		//归一化坐标
//		T xp = -p[0] / p[2];
//		T yp = -p[1] / p[2];
//
//		// 下面的程序是为了得到 经过修正后的 坐标 
//		const T& l1 = camera[7];
//		const T& l2 = camera[8];
//		T r2 = xp*xp + yp*yp;
//		T distortion = 1.0 + r2  * (l1 + l2  * r2);
//
//		const T& focal = camera[6];
//		T predicted_x = focal * distortion * xp;
//		T predicted_y = focal * distortion * yp;
//
//
//		// 计算两个归一化 坐标的差：预测值 和 实际的测量值
//		residuals[0] = predicted_x - observed_x;
//		residuals[1] = predicted_y - observed_y;
//
//		return true;
//	}
//
//
//	static ceres::CostFunction* Create(const double observed_x,
//		const double observed_y) {
//		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
//	}
//
//	double observed_x;
//	double observed_y;
//};
//

//svd fit plane
Eigen::Vector4d GetPlaneFromPoints(const std::vector<Eigen::Vector3d> &points, std::vector<double> &distance_err) {
	Eigen::Vector3d centroid(0, 0, 0);
	for (auto point : points) {
		centroid += point;
	}
	centroid /= double(points.size());

	double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

	for (auto point : points) {
		Eigen::Vector3d r = point - centroid;
		xx += r(0) * r(0);
		xy += r(0) * r(1);
		xz += r(0) * r(2);
		yy += r(1) * r(1);
		yz += r(1) * r(2);
		zz += r(2) * r(2);
	}

	double det_x = yy * zz - yz * yz;
	double det_y = xx * zz - xz * xz;
	double det_z = xx * yy - xy * xy;

	Eigen::Vector3d abc;
	if (det_x > det_y && det_x > det_z) {
		abc = Eigen::Vector3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
	}
	else if (det_y > det_z) {
		abc = Eigen::Vector3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
	}
	else {
		abc = Eigen::Vector3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
	}

	double norm = abc.norm();
	// Return invalid plane if the points don't span a plane.
	if (norm == 0) {
		return Eigen::Vector4d(0, 0, 0, 0);
	}
	abc /= abc.norm();
	double d = -abc.dot(centroid);
	Eigen::Vector4d plane_model(abc(0), abc(1), abc(2), d);

	//std::cout << "d: " << d << std::endl;

	for (auto point : points) {
		Eigen::Vector4d point1(point[0], point[1], point[2], 1.0);
		double distance = std::fabs(plane_model.dot(point1));
		distance_err.push_back(distance);
	}
	return plane_model;
};

#endif