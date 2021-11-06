#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "ReprojectionFactor.h"

typedef std::shared_ptr<std::vector<Eigen::Vector3d>> PointPtr;
typedef std::shared_ptr<std::vector<Eigen::Vector2d>> PixelPtr;

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
    for(int i=0; i<10; ++i)
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
    Eigen::Matrix<double, Eigen::Dynamic, 2> jacobian;
    Eigen::Matrix<double, Eigen::Dynamic, 1> residual;
    ReprFactor.getJacobian_N_Residual(jacobian, residual, pose);

    //check residual
    double thres = 1e-10;
    Eigen::Matrix<double, Eigen::Dynamic, 1> ref_residual;
    ref_residual.resize(residual.rows(), residual.cols());
    ref_residual.Constant(ref_residual.rows(), ref_residual.cols(), 0);
    for(int i=0; i<dp3->size(); ++i)
    {
        ref_residual.block<2, 1>(i, 0) = dp2->at(i) - dp2_repr->at(i);
    }
    Eigen::Matrix<double, Eigen::Dynamic, 1> diff_residual = residual - ref_residual;
    if(diff_residual.norm() > thres)
        std::cout << "residual calculation wrong! the diff of residual is: " << diff_residual.norm() << std::endl;
    else
        std::cout << "residual calculation right!!!" << std::endl;

    return 0;
}
