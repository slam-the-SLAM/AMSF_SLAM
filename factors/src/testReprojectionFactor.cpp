#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "ReprojectionFactor.h"

typedef std::shared_ptr<std::vector<Eigen::Vector3d>> PointPtr;
typedef std::shared_ptr<std::vector<Eigen::Vector2d>> PixelPtr;
typedef std::shared_ptr<ReprojectionFactor> ReprojectionFactorPtr;
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

void Optimization(Eigen::Matrix<double, 4, 4> &pose, const ReprojectionFactorPtr &factor, int it_num, const Eigen::Quaterniond &q_gt, const Eigen::Vector3d &p_gt)
{
    Eigen::Matrix<double, Eigen::Dynamic, 6> jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residual;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> information;
    Vector6d delta_chi;
    for(int i=0; i<it_num; ++i)
    {
        factor->getJacobian_N_Residual(jacobian, residual, information, pose);
        Eigen::Matrix<double, 6, 6> Hessian = jacobian.transpose() * information * jacobian;
        Eigen::Matrix<double, 6, 1> bres = -1.0 * jacobian.transpose() * information * residual;
        delta_chi = Hessian.ldlt().solve(bres);
        Eigen::Matrix<double, 4, 4> curPose = leftLieHybridPoseUpdate(delta_chi, pose);
        pose = curPose;
        //std::cout << "delta_chi norm: " << delta_chi.norm() << std::endl;
        //std::cout << "delta_chi: " << delta_chi << std::endl;
        //std::cout << "residual norm: " << residual.norm() << std::endl;
        Eigen::Quaterniond q_res(pose.block<3, 3>(0, 0));
        Eigen::Vector3d p_res(pose.block<3, 1>(0, 3));
        std::cout << "diff_q is: " << 2 * (q_gt.inverse() * q_res).vec().norm() << std::endl;
        std::cout << "diff_p is: " << (p_gt - p_res).norm() << std::endl;
    }
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
    Eigen::Vector3d axis(1, 1, 1);
    axis = axis / axis.norm();
    Eigen::AngleAxisd rotVec(M_PI/4, axis);
    Eigen::Vector3d trans(0, 0, 0.3);
    pose.block<3, 3>(0, 0) = rotVec.matrix();
    pose.block<3, 1>(0, 3) = trans;

    Eigen::Matrix<double, 4, 4> pose_disturb = Eigen::Matrix<double, 4, 4>::Identity();
    Eigen::Vector3d rdmvec = Eigen::Vector3d::Random();
    rdmvec = rdmvec / rdmvec.norm();
    Eigen::AngleAxisd rotVec_disturb(0.1, rdmvec);
    rdmvec = Eigen::Vector3d::Random();
    rdmvec = rdmvec / rdmvec.norm();
    std::cout << "rdmvec after normalize: " << rdmvec << std::endl;
    Eigen::Vector3d trans_disturb = rdmvec * 0.1;
    pose_disturb.block<3, 3>(0, 0) = rotVec_disturb.matrix() * pose.block<3, 3>(0, 0);
    pose_disturb.block<3, 1>(0, 3) = trans_disturb + pose.block<3, 1>(0, 3);

    //points
    PointPtr dp3(new std::vector<Eigen::Vector3d>());
    PointPtr dp3_repr(new std::vector<Eigen::Vector3d>());
    PixelPtr dp2(new std::vector<Eigen::Vector2d>());
    PixelPtr dp2_repr(new std::vector<Eigen::Vector2d>());
    //can change this part to use multi points here
    //residual_number
    int residualNum = 10;
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
        //Eigen::Vector2d u_pertubation = Eigen::Vector2d::Random();
        //Eigen::Vector2d u_repr = u_ori + u_pertubation;
        //pose disturb
        Eigen::Vector4d p_repr_disturb = pose_disturb * p_ori;
        Eigen::Vector2d u_repr(fx * p_repr_disturb(0) / p_repr_disturb(2) + cx, 
                fy * p_repr_disturb(1) / p_repr_disturb(2) + cy);
        dp3->push_back(p_ori.segment<3>(0));
        dp3_repr->push_back(p_repr.segment<3>(0));
        dp2->push_back(u_ori);
        dp2_repr->push_back(u_repr);
    }
    //Reprojection Factor
    ReprojectionFactorPtr ReprFactor(new ReprojectionFactor(dp3, dp2, intrinsics, 1.0));
    Eigen::Matrix<double, Eigen::Dynamic, 6> jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residual;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> information;
    ReprFactor->getJacobian_N_Residual(jacobian, residual, information, pose_disturb);

    //check residual
    double r_thres = 1e-10;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ref_residual;
    ref_residual.resize(residual.rows(), residual.cols());
    ref_residual.Constant(ref_residual.rows(), ref_residual.cols(), 0);
    for(int i=0; i<dp3->size(); ++i)
    {
        ref_residual.block<2, 1>(i, 0) = dp2_repr->at(i) - dp2->at(i);
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
            positive_result = pertub_reprojectError(intrinsics, delta_chi, pose_disturb, dp3->at(i), dp2->at(i));
            //std::cout << "positive_result " << i << " " << j << ":\n" << positive_result << std::endl;
            delta_chi(j) = -delta;
            negative_result = pertub_reprojectError(intrinsics, delta_chi, pose_disturb, dp3->at(i), dp2->at(i));
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
    double p_thres = 1e-1;
    double q_thres = 1e-2;
    Eigen::Quaterniond q_gt(pose.block<3, 3>(0, 0));
    Eigen::Vector3d p_gt(pose.block<3, 1>(0, 3));
    Eigen::Quaterniond q_res(pose_disturb.block<3, 3>(0, 0));
    Eigen::Vector3d p_res(pose_disturb.block<3, 1>(0, 3));
    std::cout << "start diff_q is: " << 2 * (q_gt.inverse() * q_res).vec().norm() << std::endl;
    std::cout << "start diff_p is: " << (p_gt - p_res).norm() << std::endl;
    Optimization(pose_disturb, ReprFactor, 20, q_gt, p_gt);

    return 0;
}
