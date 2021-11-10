#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "ReprojectionFactor.h"
//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

typedef std::shared_ptr<std::vector<Eigen::Vector3d>> PointPtr;
typedef std::shared_ptr<std::vector<Eigen::Vector2d>> PixelPtr;
typedef std::shared_ptr<ReprojectionFactor> ReprojectionFactorPtr;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

//wk: 位姿更新, 旋转上采用left Lie perturbation, 且error state采用hybrid(混合式)而非se3李代数
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

Eigen::Vector2d perturb_reprojectError(const Eigen::Matrix<double, 3, 3> &intrinsics,
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

void Optimization_GN(Eigen::Matrix<double, 4, 4> &pose,
        const PointPtr &p3dp,
        const PixelPtr &p2dp,
        const PointPtr &p3dp_repr, //wk: just for checking residual and jacobian
        const PixelPtr &p2dp_repr, //wk: just for checking residual and jacobian
        const Eigen::Matrix3d &intrinsics, //wk: just for checking residual and jacobian
        const ReprojectionFactorPtr &factor,
        int it_num,
        double opt_thres,
        /*const Eigen::Quaterniond &q_gt,*/
        /*const Eigen::Vector3d &p_gt,*/
        bool check=false)
{
    Eigen::Matrix<double, 2, 6> jacobian;
    Eigen::Vector2d residual;
    Eigen::Matrix<double, 2, 2> information;
    Eigen::Matrix<double, 6, 6> Hessian;
    Vector6d bres;
    Vector6d delta_chi = Vector6d::Zero();
    double cost=0, curcost=0;
    bool residualOK = true;
    bool jacobianOK = true;
    for(int i=0; i<it_num; ++i)
    {
        Hessian.setZero();
        bres.setZero();
        curcost = 0;
        for(int j=0; j<p3dp->size(); ++j)
        {
            factor->get_jacobian_N_residual(jacobian, residual, information, pose, p3dp->at(j), p2dp->at(j));
            Hessian += jacobian.transpose() * information * jacobian;
            bres += -1.0 * jacobian.transpose() * information * residual;
            curcost += residual.transpose() * information * residual;
            if(check && 0==i)
            {
                //check residual
                double r_thres = 1e-10;
                Eigen::Vector2d ref_residual = p2dp_repr->at(j) - p2dp->at(j);
                Eigen::Vector2d diff_residual = residual - ref_residual;
                if(diff_residual.norm() > r_thres)
                {
                    std::cout << "residual calculation WRONG!" << std::endl;
                    std::cout << "the diff of residual is: " << diff_residual.norm() << std::endl;
                    residualOK = false;
                    break;
                }
                /*else*/
                    /*std::cout << "residual calculation right!!!" << std::endl;*/
                //check jacobian
                double j_thres = 1e-5;
                double delta = 1e-6;
                Eigen::Matrix<double, 2, 6> ref_jacobian;
                Eigen::Vector2d positive_result, negative_result;
                for(int k=0; k<6; ++k)
                {
                    //init delta_chi_chk every col of jacobian
                    Vector6d delta_chi_chk = Vector6d::Zero();
                    delta_chi_chk(k) = delta;
                    positive_result = perturb_reprojectError(intrinsics, delta_chi_chk, pose, p3dp->at(j), p2dp->at(j));
                    //std::cout << "positive_result " << i << " " << j << ":\n" << positive_result << std::endl;
                    delta_chi_chk(k) = -delta;
                    negative_result = perturb_reprojectError(intrinsics, delta_chi_chk, pose, p3dp->at(j), p2dp->at(j));
                    //std::cout << "negative_result " << i << " " << j << ":\n" << negative_result << std::endl;
                    ref_jacobian.block<2, 1>(0, k) = 0.5 * (positive_result - negative_result) / delta;
                }
                Eigen::Matrix<double, 2, 6> diff_jacobian = jacobian - ref_jacobian;
                if(diff_jacobian.norm() > j_thres)
                {
                    std::cout << "jacobian calculation WRONG" << std::endl;
                    std::cout << "the diff of jacobian is: " << diff_jacobian.norm() << std::endl;
                    jacobianOK = false;
                    break;
                }
                /*else*/
                    /*std::cout << "jacobian calculation right!!!" << std::endl;*/
            }
        }
        if(check && 0==i && (!residualOK || !jacobianOK))
            break;
        //wk: solve delta_chi
        delta_chi = Hessian.ldlt().solve(bres);
        //wk: in case that delta_chi result is nan
        if(std::isnan(delta_chi[0]))
        {
            std::cout << "stop as delta_chi is nan!" << std::endl;
            break;
        }
        std::cout << "cost: " << cost << std::endl;
        //wk: in case that cost increase, or update current cost
  /*      if(i>0 && curcost>=cost)*/
        /*{*/
            /*std::cout << "stop as cost increase at iteration" << i << std::endl;*/
            /*break;*/
        /*}*/
        /*else*/
            cost = curcost;
        //wk: in case that converge before reach the last iteration
        if(i>0 && delta_chi.norm() < opt_thres)
        {
            std::cout << "stop as converge at iteration" << i << std::endl;
            break;
        }
        Eigen::Matrix<double, 4, 4> curPose = leftLieHybridPoseUpdate(delta_chi, pose);
        pose = curPose;
        //std::cout << "delta_chi norm: " << delta_chi.norm() << std::endl;
        //std::cout << "delta_chi: " << delta_chi << std::endl;
        //std::cout << "residual norm: " << residual.norm() << std::endl;
        //Eigen::Quaterniond q_res(pose.block<3, 3>(0, 0));
        //Eigen::Vector3d p_res(pose.block<3, 1>(0, 3));
        //std::cout << "diff_q is: " << 2 * (q_gt.inverse() * q_res).vec().norm() << std::endl;
        //std::cout << "diff_p is: " << (p_gt - p_res).norm() << std::endl;
    }
}

//wk: see https://github.com/jiajunhua/gaoxiang12-slambook/blob/master/ch7/pose_estimation_3d2d.cpp
void matchORBFeatures(const Mat &last_img, const Mat &curr_img,
        std::vector<KeyPoint> &last_kp, std::vector<KeyPoint> &curr_kp,
        std::vector<DMatch> &matches)
{
    Ptr<FeatureDetector> fdetector = ORB::create();
    Ptr<DescriptorExtractor> descrpt = ORB::create();
    Ptr<DescriptorMatcher> dmatcher = DescriptorMatcher::create("BruteForce-Hamming");

    fdetector->detect(last_img, last_kp);
    fdetector->detect(curr_img, curr_kp);

    Mat last_desc, curr_desc;
    descrpt->compute(last_img, last_kp, last_desc);
    descrpt->compute(curr_img, curr_kp, curr_desc);

    std::vector<DMatch> temp_matches;
    dmatcher->match(last_desc, curr_desc, temp_matches);

    std::cout << "筛选前有" << temp_matches.size() << "组匹配点" << std::endl;

    double minDist = 10000;
    for(auto m : temp_matches)
        if(m.distance < minDist) minDist = m.distance;
    double implDist = std::max(2 * minDist, 30.0);
    std::cout << "--implDist : " << implDist << std::endl;
    for(auto m : temp_matches)
        if(m.distance <= implDist)
            matches.push_back(m);

    std::cout << "筛选后有" << matches.size() << "组匹配点" << std::endl;
}

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cout << "parameter error!" << std::endl;
        return 1;
    }
    //intrinsics
    double fx = 520.9;
    double fy = 521.0;
    double cx = 325.1;
    double cy = 249.7;
    Eigen::Matrix<double, 3, 3> intrinsics;
    intrinsics << fx, 0, cx,
               0, fy, cy,
               0, 0, 1;

    //pose
    Eigen::Matrix<double, 4, 4> pose = Eigen::Matrix<double, 4, 4>::Identity();

    Eigen::Matrix<double, 4, 4> pose_disturb = Eigen::Matrix<double, 4, 4>::Identity();
    /*Eigen::Vector3d rdmvec = Eigen::Vector3d::Random();*/
    /*rdmvec = rdmvec / rdmvec.norm();*/
    /*Eigen::AngleAxisd rotVec_disturb(0.3, rdmvec);*/
    /*rdmvec = Eigen::Vector3d::Random();*/
    /*rdmvec = rdmvec / rdmvec.norm();*/
    //std::cout << "rdmvec after normalize: " << rdmvec << std::endl;
/*    Eigen::Vector3d trans_disturb = rdmvec * 0.1;*/
    /*pose_disturb.block<3, 3>(0, 0) = rotVec_disturb.matrix() * pose.block<3, 3>(0, 0);*/
    /*pose_disturb.block<3, 1>(0, 3) = trans_disturb + pose.block<3, 1>(0, 3);*/

    //points
    PointPtr dp3(new std::vector<Eigen::Vector3d>());
    PointPtr dp3_repr(new std::vector<Eigen::Vector3d>());
    PixelPtr dp2(new std::vector<Eigen::Vector2d>());
    PixelPtr dp2_repr(new std::vector<Eigen::Vector2d>());
    //wk: generate points with opencv orb
    std::cout << "generate points with CV ORB." << std::endl;
    Mat last_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat curr_img = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    std::vector<KeyPoint> last_kp;
    std::vector<KeyPoint> curr_kp;
    std::vector<DMatch> matches;

    matchORBFeatures(last_img, curr_img, last_kp, curr_kp, matches);
    //check matches
    /*std::cout << "matches[0]: " << matches[0].queryIdx << std::endl;*/
    /*std::cout << "matches_currkeypoint[0]: " << curr_kp[matches[0].queryIdx].pt << std::endl;*/
    Mat depth_img = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    for(auto m : matches)
    {
        Point2d last_2dp = last_kp[m.queryIdx].pt;
        int u = last_2dp.x;
        int v = last_2dp.y;
        unsigned short sdepth = depth_img.ptr<unsigned short>(v)[u];
        if(0 == sdepth)
            continue;
        double z = sdepth/5000.0;

        //last
        Eigen::Vector3d Elast_3dp((u - cx) * z / fx,
                (v - cy) * z / fy,
                z);
        Eigen::Vector2d Elast_2dp(u, v);

        //current
        Eigen::Vector3d Ecurr_3dp = pose_disturb.block<3, 3>(0, 0) * Elast_3dp + pose_disturb.block<3, 1>(0, 3);
        //wk: 注意这里是trainIdx,不是queryIdx
        Point2d curr_2dp = curr_kp[m.trainIdx].pt;
        int cur_u = curr_2dp.x;
        int cur_v = curr_2dp.y;
        Eigen::Vector2d Ecurr_2dp(cur_u, cur_v);
        //push into vectors of Eigen Matrix
        dp3->push_back(Elast_3dp);
        dp2->push_back(Ecurr_2dp);
        dp3_repr->push_back(Ecurr_3dp);
        //if no perturbation on pixel
        dp2_repr->push_back(Ecurr_2dp);
    }
    std::cout << "3d-2d pairs: " << dp3->size() << std::endl;
    //wk: check some points in dp3 and dp2
    /*std::cout << "dp3[0]: " << dp3->at(0) << std::endl;*/
    /*std::cout << "dp2[0]: " << dp2->at(0) << std::endl;*/
    /*for(auto p : (*dp2))*/
        /*std::cout <<"p: " << p << "\n";*/
    /*std::cout << std::endl;*/
    //Reprojection Factor
    ReprojectionFactorPtr ReprFactor(new ReprojectionFactor(intrinsics, 1.0));

    //check optimization
    double p_thres = 1e-1;
    double q_thres = 1e-2;
    double op_thres = 1e-6;
    /*Eigen::Quaterniond q_gt(pose.block<3, 3>(0, 0));*/
    /*Eigen::Vector3d p_gt(pose.block<3, 1>(0, 3));*/
    /*Eigen::Quaterniond q_res(pose_disturb.block<3, 3>(0, 0));*/
    /*Eigen::Vector3d p_res(pose_disturb.block<3, 1>(0, 3));*/
    /*std::cout << "start diff_q is: " << 2 * (q_gt.inverse() * q_res).vec().norm() << std::endl;*/
    /*std::cout << "start diff_p is: " << (p_gt - p_res).norm() << std::endl;*/
    Optimization_GN(pose_disturb,
            dp3,
            dp2,
            dp3_repr,
            dp2_repr,
            intrinsics,
            ReprFactor,
            30,
            op_thres,
            /*q_gt, */
            /*p_gt, */
            false);
    /*q_res = pose_disturb.block<3, 3>(0, 0);*/
    /*p_res = pose_disturb.block<3, 1>(0, 3);*/
    /*std::cout << "end diff_q is: " << 2 * (q_gt.inverse() * q_res).vec().norm() << std::endl;*/
    /*std::cout << "end diff_p is: " << (p_gt - p_res).norm() << std::endl;*/
    std::cout << "optimized pose is:\n" << pose_disturb << std::endl;

    return 0;
}
