#pragma once
#ifndef utility_h_
#define utility_h_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <string>

class Utility
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            typedef Eigen::Matrix<double, 6, 1> Vector6d;
        static Eigen::Matrix<double, 3, 3> skewSymmetric(const Eigen::Vector3d &p)
        {
            Eigen::Matrix<double, 3, 3> res;
            res << 0, -p(2), p(1),
                p(2), 0, -p(0),
                -p(1), p(0), 0;
            return res;
        }

        //wk: 位姿更新, 旋转上采用left Lie perturbation, 且error state采用hybrid(混合式)而非se3李代数
        static Eigen::Matrix<double, 4, 4> leftLieHybridPoseUpdate(const Vector6d &delta_chi, const Eigen::Matrix<double, 4, 4> &pose)
        {
            Eigen::Matrix<double, 4, 4> res = pose;
            Eigen::Vector3d dp = delta_chi.segment<3>(0);
            Eigen::Vector3d dtheta = delta_chi.segment<3>(3);
            //wk:旋转我懒得写exp映射了，先用四元数
            Eigen::Matrix<double, 3, 1> trans = res.block<3, 1>(0, 3);
            Eigen::Quaterniond cur_q(res.block<3, 3>(0, 0));
            Eigen::Quaterniond dq;
            dtheta /= 2.0;
            double sinc;
            if(dtheta.norm()>1e-6)
            {
                sinc = sin(dtheta.norm()) / dtheta.norm();
            }
            else
            {
                const double x = dtheta.norm();
                static const double c_2 = 1.0 / 6.0;
                static const double c_4 = 1.0 / 120.0;
                static const double c_6 = 1.0 / 5040.0;
                const double x_2 = x * x;
                const double x_4 = x_2 * x_2;
                const double x_6 = x_2 * x_2 * x_2;
                sinc = 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
            }
            dq.w() = cos(dtheta.norm());
            dq.x() = dtheta(0) * sinc;
            dq.y() = dtheta(1) * sinc;
            dq.z() = dtheta(2) * sinc;

            cur_q = dq * cur_q;
            cur_q.normalized();

            res.block<3, 3>(0, 0) = cur_q.toRotationMatrix();
            res.block<3, 1>(0, 3) = trans + dp; 
            return res;
        }

        static std::string i_to_s(int i)
        {
            std::string res = std::to_string(i);
            while(res.size()<4)
                res = "0" + res;
            return res;
        } 
};

#endif
