//wk: This class is for Reprojection Error, 
//for the jacobian calculation, here use error-state, left Lie, hybrid form pertubation on pose
#ifndef REPROJECTIONFACTOR_H
#define REPROJECTIONFACTOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>

typedef std::shared_ptr<std::vector<Eigen::Vector3d>> PointPtr;
typedef std::shared_ptr<std::vector<Eigen::Vector2d>> PixelPtr;
class ReprojectionFactor
{
    public:
        ReprojectionFactor(const PointPtr &p3dp, const PixelPtr &p2dp, const Eigen::Matrix<double, 3, 3> &intrinsics): 
            p3dp_(p3dp), p2dp_(p2dp), ResidualNum(p3dp->size()) 
    {
        fx_ = intrinsics(0, 0);
        fy_ = intrinsics(1, 1);
        cx_ = intrinsics(0, 2);
        cy_ = intrinsics(1, 2);
    }
        ~ReprojectionFactor() {}
        //wk: calculate Jacobian and Residual
        void getJacobian_N_Residual(Eigen::Matrix<double, Eigen::Dynamic, 6> &curJacobian, Eigen::Matrix<double, Eigen::Dynamic, 1> &curResidual, const Eigen::Matrix<double, 4, 4> &cur_pose) const 
        {
            //wk: the size is not known at compile-time, we use dynamic-size matrix and relative API, Resize and init at every iteration
            curJacobian.resize(ResidualNum * 2, 6);
            curJacobian.Constant(ResidualNum * 2, 6, 0);
            Eigen::Vector3d trans = cur_pose.block<3, 1>(0, 3);
            Eigen::Matrix<double, 3, 3> rot = cur_pose.block<3, 3>(0, 0);
            //wk: the size is not known at compile-time, we use dynamic-size matrix and relative API, Resize and init at every iteration
            curResidual.resize(ResidualNum * 2, 1);
            curResidual.Constant(ResidualNum * 2, 1, 0);
            for(int i=0; i<ResidualNum; ++i)
            {
                //wk: 3d points in curr camera
                Eigen::Vector3d reproj_3dp = rot * p3dp_->at(i) + trans;
                //wk: check this point
                //std::cout << "reproj_3dp: " << reproj_3dp << std::endl;
                //wk: jacobian part of du/dg
                Eigen::Matrix<double, 2, 3> du_dg = Eigen::Matrix<double, 2, 3>::Zero();
                du_dg(0, 0) = fx_ / reproj_3dp(2);
                du_dg(0, 2) = -fx_ * reproj_3dp(0) / (reproj_3dp(2) * reproj_3dp(2));
                du_dg(1, 1) = fy_ / reproj_3dp(2);
                du_dg(1, 2) = -fy_ * reproj_3dp(1) / (reproj_3dp(2) * reproj_3dp(2));
                //check this matrix
                //std::cout << "du_dg: " << du_dg << std::endl;
                //wk: jacobian part of dg/ddelta_theta
                Eigen::Matrix<double, 3, 3> dg_dtheta = Eigen::Matrix<double, 3, 3>::Zero();
                //wk: 小心这里不是reproj_3dp!
                Eigen::Vector3d temp_3dp = reproj_3dp - trans;
                dg_dtheta << 0, temp_3dp(2), -temp_3dp(1),
                          -temp_3dp(2), 0, temp_3dp(0),
                          temp_3dp(1), -temp_3dp(0), 0;
                //wk: check this dg_dtheta
                //std::cout << "dg_dtheta: " << dg_dtheta << std::endl;
                //wk: jacobian part of pertubation of translation(the pertubation is in hybrid form)
                curJacobian.block<2, 3>(i, 0) = du_dg;
                //wk: jacobian part of pertubation of rotation(the pertubation is in hybrid form)
                curJacobian.block<2, 3>(i, 3)  = du_dg * dg_dtheta;
                //wk: residuals
                curResidual(i, 0) = fx_ * reproj_3dp(0) / reproj_3dp(2) + cx_ - p2dp_->at(i)(0);
                curResidual(i+1, 0) = fy_ * reproj_3dp(1) / reproj_3dp(2) + cy_ - p2dp_->at(i)(1);
            }
        }
    private:
        const int ResidualNum;
        PointPtr p3dp_;
        PixelPtr p2dp_;
        double fx_, fy_, cx_, cy_;
};

#endif//wk: ReprojectioinFactor
