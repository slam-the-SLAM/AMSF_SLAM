#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "ReprojectionFactor.h"

typedef std::shared_ptr<std::vector<Eigen::Vector3d>> PointPtr;
typedef std::shared_ptr<std::vector<Eigen::Vector2d>> PixelPtr;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

Eigen::Matrix<double, 4, 4> leftLieHybridPoseUpdate(const Vector6d &delta_chi, const Eigen::Matrix<double, 4, 4> &pose)
{
    Eigen::Matrix<double, 4, 4> res = pose;
    Eigen::Vector3d dp = delta_chi.segment<3>(0);
    Eigen::Vector3d dtheta = delta_chi.segment<3>(3);
    //wk:旋转我懒得写exp映射了，先用四元数
    Eigen::Matrix<double, 3, 1> trans = res.block<3, 1>(0, 3);
    Eigen::Quaterniond cur_q(res.block<3, 3>(0, 0));
    Eigen::Quaterniond dq;
    dtheta /= 2.0;
    dq.w() = 1.0;
    dq.x() = dtheta(0);
    dq.y() = dtheta(1);
    dq.z() = dtheta(2);
    cur_q = dq * cur_q;
    res.block<3, 3>(0, 0) = cur_q.toRotationMatrix();
    res.block<3, 1>(0, 3) = trans + dp; 
    return res;
}

Eigen::Vector2d pertub_reprojectError(const Eigen::Matrix<double, 3, 3> &intrinsics, 
        const Vector6d &delta_chi, 
        const Eigen::Matrix<double, 4, 4> &pose, 
        const Eigen::Vector3d &p3d, 
        const Eigen::Vector2d &p2d)
{
    Eigen::Matrix<double, 4, 4> curPose = leftLieHybridPoseUpdate(delta_chi, pose);
    Eigen::Vector2d res;
    Eigen::Vector3d repr_p = curPose.block<3, 3>(0, 0) * p3d + curPose.block<3, 1>(0, 3);
    //wk: 不要忘记除以深度!
    Eigen::Vector3d repr_u = intrinsics * repr_p / repr_p(2);
    res = repr_u.segment<2>(0) - p2d;
    return res;
}
int main()
{
    //intrinsics
    double fx = 200;
    double fy = 200;
    double cx = 150;
    double cy = 120;
    Eigen::Matrix<double, 3, 3> intrinsics;
    intrinsics << fx, 0, cx,
               0, fy, cy,
               0, 0, 1;

    //pose
    Eigen::Matrix<double, 4, 4> pose = Eigen::Matrix<double, 4, 4>::Identity();
    Eigen::AngleAxisd rotVec(M_PI/4, Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d trans(0, 0, 0.3);
    pose.block<3, 3>(0, 0) = rotVec.matrix();
    pose.block<3, 1>(0, 3) = trans;

    //points
    PointPtr dp3(new std::vector<Eigen::Vector3d>());
    PointPtr dp3_repr(new std::vector<Eigen::Vector3d>());
    PixelPtr dp2(new std::vector<Eigen::Vector2d>());
    PixelPtr dp2_repr(new std::vector<Eigen::Vector2d>());
    //can change this part to use multi points here
    //residual_number
    int residualNum = 1;
    for(int i=0; i<residualNum; ++i)
    {
        //double x = 0;
        //double y = 0;
        //double depth = 3.5;
        //Eigen::Vector4d p_ori(x, y, depth, 1);
        Eigen::Vector3d rdm_p = Eigen::Vector3d::Random();
        Eigen::Vector4d p_ori(2 * rdm_p(0), 2 * rdm_p(1), 5 * rdm_p(2), 1);
        Eigen::Vector4d p_repr = pose * p_ori;
        Eigen::Vector2d u_ori(fx * p_repr(0) / p_repr(2) + cx, 
                fy * p_repr(1) / p_repr(2) + cy);
        //pixel pertubation
        Eigen::Vector2d u_pertubation = Eigen::Vector2d::Random();
        Eigen::Vector2d u_repr = u_ori + u_pertubation;
        dp3->push_back(p_ori.segment<3>(0));
        dp3_repr->push_back(p_repr.segment<3>(0));
        dp2->push_back(u_ori);
        dp2_repr->push_back(u_repr);
    }
    //Reprojection Factor
    ReprojectionFactor ReprFactor(dp3, dp2_repr, intrinsics);
    Eigen::Matrix<double, Eigen::Dynamic, 6> jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residual;
    ReprFactor.getJacobian_N_Residual(jacobian, residual, pose);

    //check residual
    double r_thres = 1e-10;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ref_residual;
    ref_residual.resize(residual.rows(), residual.cols());
    ref_residual.Constant(ref_residual.rows(), ref_residual.cols(), 0);
    for(int i=0; i<dp3->size(); ++i)
    {
        ref_residual.block<2, 1>(i, 0) = dp2->at(i) - dp2_repr->at(i);
    }
    Eigen::Matrix<double, Eigen::Dynamic, 1> diff_residual = residual - ref_residual;
    if(diff_residual.norm() > r_thres)
        std::cout << "residual calculation WRONG!" << std::endl;
    else
        std::cout << "residual calculation right!!!" << std::endl;
    std::cout << "the diff of residual is: " << diff_residual.norm() << std::endl;

    //check jacobian
    double j_thres = 1e-6;
    double delta = 1e-6;
    Eigen::Matrix<double, Eigen::Dynamic, 6> ref_jacobian;
    ref_jacobian.resize(jacobian.rows(), jacobian.cols());
    ref_jacobian.Constant(ref_jacobian.rows(), ref_jacobian.cols(), 0);
    Eigen::Vector2d positive_result, negative_result;
    for(int i=0; i<residualNum; ++i)
    {
        //check points
        //std::cout << "point3d: " << dp3->at(i) << std::endl;
        //std::cout << "point2d: " << dp2->at(i) << std::endl;
        for(int j=0; j<6; ++j)
        {
            //translational
            //rotational
            //init delta_chi every col of jacobian
            Vector6d delta_chi = Vector6d::Zero();
            delta_chi(j) = delta;
            std::cout.precision(9);
            positive_result = pertub_reprojectError(intrinsics, delta_chi, pose, dp3->at(i), dp2->at(i));
            //std::cout << "positive_result " << i << " " << j << ":\n" << positive_result << std::endl;
            delta_chi(j) = -delta;
            negative_result = pertub_reprojectError(intrinsics, delta_chi, pose, dp3->at(i), dp2->at(i));
            //std::cout << "negative_result " << i << " " << j << ":\n" << negative_result << std::endl;
            ref_jacobian.block<2, 1>(i, j) = 0.5 * (positive_result - negative_result) / delta;
        }
    }
    Eigen::Matrix<double, Eigen::Dynamic, 6> diff_jacobian = jacobian - ref_jacobian;
    if(diff_jacobian.norm() > j_thres)
        std::cout << "jacobian calculation WRONG" << std::endl;
    else
        std::cout << "jacobian calculation right!!!" << std::endl;
    std::cout << "the diff of jacobian is: " << diff_jacobian.norm() << std::endl;
    //std::cout.precision(9);
    //std::cout << "jacobian:\n" << jacobian << std::endl;
    //std::cout << "ref_jacobian:\n" << ref_jacobian << std::endl;

    //check optimization
    return 0;
}
