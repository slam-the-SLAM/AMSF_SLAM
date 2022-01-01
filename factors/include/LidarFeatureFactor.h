#ifndef LidarFeatureFactor_h_
#define LidarFeatureFactor_h_
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../../utility/utility.h"

typedef std::vector<Eigen::Vector3d> LiDARfeatures;
//wk: use left Lie hybrid mode
class LidarFeatureFactor
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LidarFeatureFactor() {}
        ~LidarFeatureFactor() {}
        void get_jacobian_N_residual(Eigen::Matrix<double, 2, 6> &jacobian,
                Eigen::Vector2d &residual,
                Eigen::Matrix2d &information,
                const Eigen::Matrix4d &pose,
                const LiDARfeatures &surfFeature,
                const LiDARfeatures &cornerFeature) const
        {
            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            //surf feature points
            //curr点在curr坐标系下
            Eigen::Vector3d curr_sp_curr = surfFeature.at(0);
            Eigen::Vector3d ref_sp1 = surfFeature.at(1);
            Eigen::Vector3d ref_sp2 = surfFeature.at(2);
            Eigen::Vector3d ref_sp3 = surfFeature.at(3);
            //curr点在ref坐标系下
            Eigen::Vector3d curr_sp = rot.transpose() * (curr_sp_curr - trans);
            //corner feature points
            //curr点在curr坐标系下
            Eigen::Vector3d curr_cp_curr = cornerFeature.at(0);
            Eigen::Vector3d ref_cp1 = cornerFeature.at(1);
            Eigen::Vector3d ref_cp2 = cornerFeature.at(2);
            //curr点在ref坐标系下
            Eigen::Vector3d curr_cp = rot.transpose() * (curr_cp_curr - trans);
            //residual of surf feature
            Eigen::Vector3d surfRhomboid =  Utility::skewSymmetric(ref_sp1 - ref_sp2) * (ref_sp1 - ref_sp3);//wk:在平面特征中由平面三个点构成的平行四边形的法向量并乘上该平行四边形的面积
            residual(0) = (curr_sp - ref_sp1).transpose() * (surfRhomboid.normalized());
            //residual of corner feature
            Eigen::Vector3d cornerRhomboid = Utility::skewSymmetric(curr_cp - ref_cp1) * (curr_cp - ref_cp2);//wk:在边缘特征中由当前特征点与上一时刻边缘上的两个点构成的平行四边形的法向量并乘上该平行四边形的面积
            residual(1) = cornerRhomboid.norm() / (ref_cp1 - ref_cp2).norm();
            //jacobian of surf feature
            Eigen::Matrix3d dcurr_sp_dp = -rot.transpose();
            Eigen::Matrix3d dcurr_sp_dtheta = rot.transpose() * Utility::skewSymmetric(curr_sp_curr - trans);
            jacobian.block<1, 3>(0, 0) = surfRhomboid.normalized().transpose() * dcurr_sp_dp;
            jacobian.block<1, 3>(0, 3) = surfRhomboid.normalized().transpose() * dcurr_sp_dtheta;
            //jacobian of corner feature
            Eigen::Matrix3d dcurr_cp_dp = -rot.transpose();
            Eigen::Matrix3d dcurr_cp_dtheta = rot.transpose() * Utility::skewSymmetric(curr_cp_curr - trans);
            Eigen::Vector3d normhight = ref_cp2 - ref_cp1;
            normhight.normalize();
            jacobian.block<1, 3>(1, 0) = cornerRhomboid.normalized().transpose() * Utility::skewSymmetric(normhight) * dcurr_cp_dp;
            jacobian.block<1, 3>(1, 3) = cornerRhomboid.normalized().transpose() * Utility::skewSymmetric(normhight) * dcurr_cp_dtheta;
            //information
            information = Eigen::Matrix<double, 2, 2>::Identity();
        }
        Eigen::Vector2d get_residual(const Eigen::Matrix4d &pose,
                const LiDARfeatures &surfFeature,
                const LiDARfeatures &cornerFeature) const
        {
            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            Eigen::Vector2d residual;
            Eigen::Vector3d surfRhomboid = Utility::skewSymmetric(surfFeature.at(1) - surfFeature.at(2)) * (surfFeature.at(1) - surfFeature.at(3));
            residual(0) = ((rot.transpose() * (surfFeature.at(0) - trans)) - surfFeature.at(1)).transpose() * (surfRhomboid.normalized());

            Eigen::Vector3d curr_cp = rot.transpose() * (cornerFeature.at(0) - trans);
            Eigen::Vector3d cornerRhomboid = Utility::skewSymmetric(curr_cp - cornerFeature.at(1)) * (curr_cp - cornerFeature.at(2));//wk:在边缘特征中由当前特征点与上一时刻边缘上的两个点构成的平行四边形的法向量并乘上该平行四边形的面积
            residual(1) = cornerRhomboid.norm() / (cornerFeature.at(1) - cornerFeature.at(2)).norm();
            return residual;
        }

        void get_surf_jacobian_N_residual(Eigen::Matrix<double, 1, 3> &jacobian,
                double &residual,
                double &information,
                const Eigen::Matrix4d &pose,
                const LiDARfeatures &surfFeature) const
        {
            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            //surf feature points
            //curr点在curr坐标系下
            Eigen::Vector3d curr_sp_curr = surfFeature.at(0);
            Eigen::Vector3d ref_sp1 = surfFeature.at(1);
            Eigen::Vector3d ref_sp2 = surfFeature.at(2);
            Eigen::Vector3d ref_sp3 = surfFeature.at(3);
            //curr点在ref坐标系下
            Eigen::Vector3d curr_sp = rot.transpose() * (curr_sp_curr - trans);
            //residual of surf feature
            Eigen::Vector3d surfRhomboid =  Utility::skewSymmetric(ref_sp1 - ref_sp2) * (ref_sp1 - ref_sp3);//wk:在平面特征中由平面三个点构成的平行四边形的法向量并乘上该平行四边形的面积
            residual = (curr_sp - ref_sp1).transpose() * (surfRhomboid.normalized());
            //jacobian of surf feature
            Eigen::Matrix3d dcurr_sp_dp = -rot.transpose();
            Eigen::Matrix3d dcurr_sp_dtheta = rot.transpose() * Utility::skewSymmetric(curr_sp_curr - trans);
            //surf jacobian only contains tz, roll, pitch
            jacobian(0, 0) = surfRhomboid.normalized().transpose() * (dcurr_sp_dp.col(2));
            jacobian(0, 1) = surfRhomboid.normalized().transpose() * (dcurr_sp_dtheta.col(0));
            jacobian(0, 2) = surfRhomboid.normalized().transpose() * (dcurr_sp_dtheta.col(1));
            //information
            information = 1.0;
        }
        double get_surf_residual(const Eigen::Matrix4d &pose,
                const LiDARfeatures &surfFeature) const
        {
            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            Eigen::Vector3d surfRhomboid = Utility::skewSymmetric(surfFeature.at(1) - surfFeature.at(2)) * (surfFeature.at(1) - surfFeature.at(3));
            return ((rot.transpose() * (surfFeature.at(0) - trans)) - surfFeature.at(1)).transpose() * (surfRhomboid.normalized());
        }
        void get_corner_jacobian_N_residual(Eigen::Matrix<double, 1, 3> &jacobian,
                double &residual,
                double &information,
                const Eigen::Matrix4d &pose,
                const LiDARfeatures &cornerFeature) const
        {

            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            //corner feature points
            //curr点在curr坐标系下
            Eigen::Vector3d curr_cp_curr = cornerFeature.at(0);
            Eigen::Vector3d ref_cp1 = cornerFeature.at(1);
            Eigen::Vector3d ref_cp2 = cornerFeature.at(2);
            //curr点在ref坐标系下
            Eigen::Vector3d curr_cp = rot.transpose() * (curr_cp_curr - trans);
            //residual of corner feature
            Eigen::Vector3d cornerRhomboid = Utility::skewSymmetric(curr_cp - ref_cp1) * (curr_cp - ref_cp2);//wk:在边缘特征中由当前特征点与上一时刻边缘上的两个点构成的平行四边形的法向量并乘上该平行四边形的面积
            residual = cornerRhomboid.norm() / (ref_cp1 - ref_cp2).norm();
            //jacobian of corner feature
            Eigen::Matrix3d dcurr_cp_dp = -rot.transpose();
            Eigen::Matrix3d dcurr_cp_dtheta = rot.transpose() * Utility::skewSymmetric(curr_cp_curr - trans);
            Eigen::Vector3d normhight = ref_cp2 - ref_cp1;
            normhight.normalize();
            //corner jacobian only contains tx, ty, yaw
            jacobian(0, 0) = cornerRhomboid.normalized().transpose() * Utility::skewSymmetric(normhight) * (dcurr_cp_dp.col(0));
            jacobian(0, 1) = cornerRhomboid.normalized().transpose() * Utility::skewSymmetric(normhight) * (dcurr_cp_dp.col(1));
            jacobian(0, 2) = cornerRhomboid.normalized().transpose() * Utility::skewSymmetric(normhight) * (dcurr_cp_dtheta.col(2));
            //information
            information = 1.0;
        }

        double get_corner_residual(const Eigen::Matrix4d &pose,
                const LiDARfeatures &cornerFeature) const
        {
            //trans and rot
            Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
            Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
            Eigen::Vector3d curr_cp = rot.transpose() * (cornerFeature.at(0) - trans);
            Eigen::Vector3d cornerRhomboid = Utility::skewSymmetric(curr_cp - cornerFeature.at(1)) * (curr_cp - cornerFeature.at(2));//wk:在边缘特征中由当前特征点与上一时刻边缘上的两个点构成的平行四边形的法向量并乘上该平行四边形的面积
            return cornerRhomboid.norm() / (cornerFeature.at(1) - cornerFeature.at(2)).norm();
        }
};

#endif //LidarFeatureFactor_h_
