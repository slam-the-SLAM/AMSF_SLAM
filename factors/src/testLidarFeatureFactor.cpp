#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <eigen3/Eigen/Core>
#include "../../utility/utility.h"
#include "../include/LidarFeatureFactor.h"
using namespace std;
typedef vector<Eigen::Vector3d> LiDARfeatures;

int main()
{
    ifstream SurfInClouds, CornerInClouds, SurfOdomIn, SurfOdomOut, CornerOdomIn, CornerOdomOut;
    string Path = "/home/wk/Output/";
    string SurfPath = "Corner_";
    string CornerPath = "Surf_";
    string SurfOdomInName = Path + "SurfOdomInput.txt";
    string SurfOdomOutName = Path + "SurfOdomOutput.txt";
    string CornerOdomInName = Path + "CornerOdomInput.txt";
    string CornerOdomOutName = Path + "CornerOdomOutput.txt";

    //read surf OdomOutput
    SurfOdomOut.open(SurfOdomOutName);
    double stx_o, sty_o, stz_o, sqx_o, sqy_o, sqz_o, sqw_o;
    int seq_so;
    SurfOdomOut >> seq_so >> stx_o >> sty_o >> stz_o >> sqx_o >> sqy_o >> sqz_o >> sqw_o;
    cout << "seq surf out: " << seq_so << endl;
    SurfOdomOut.close();

    //read relative OdomInput
    SurfOdomIn.open(SurfOdomInName);
    double stx_i, sty_i, stz_i, sqx_i, sqy_i, sqz_i, sqw_i;
    int seq_si;
    while(seq_si != seq_so)
        SurfOdomIn >> seq_si >> stx_i >> sty_i >> stz_i >> sqx_i >> sqy_i >> sqz_i >> sqw_i;
    cout << "seq surf in: " << seq_si << endl;
    SurfOdomIn.close();

    //read corner OdomOutput
    CornerOdomOut.open(CornerOdomOutName);
    double ctx_o, cty_o, ctz_o, cqx_o, cqy_o, cqz_o, cqw_o;
    int seq_co;
    bool check_jacobian = true;
    if(check_jacobian)
    {
        while(seq_co != seq_so)
        {
            CornerOdomOut >> seq_co >> ctx_o >> cty_o >> ctz_o >> cqx_o >> cqy_o >> cqz_o >> cqw_o;
        }
    }
    else
        CornerOdomOut >> seq_co >> ctx_o >> cty_o >> ctz_o >> cqx_o >> cqy_o >> cqz_o >> cqw_o;
    cout << "seq corner out: " << seq_co << endl;
    CornerOdomOut.close();

    //read relative OdomInput
    CornerOdomIn.open(CornerOdomInName);
    double ctx_i, cty_i, ctz_i, cqx_i, cqy_i, cqz_i, cqw_i;
    int seq_ci;
    while(seq_ci != seq_co)
        CornerOdomIn >> seq_ci >> ctx_i >> cty_i >> ctz_i >> cqx_i >> cqy_i >> cqz_i >> cqw_i;
    cout << "seq corner out: " << seq_ci << endl;
    CornerOdomIn.close();

    //read surface feature points
    string SurfName = Path + SurfPath + Utility::i_to_s(seq_so); 
    string CornerName = Path + CornerPath + Utility::i_to_s(seq_co);
    cout << "SurfName: " << SurfName << endl;
    cout << "CornerName: " << CornerName << endl;

    //queue<LiDARfeatures> surfFeatures, cornerFeatures;
    vector<LiDARfeatures> surfFeatures, cornerFeatures;
    SurfInClouds.open(SurfName);
    while(SurfInClouds.good())
        //while(!SurfInClouds.eof())
    {
        Eigen::Vector3d curr_sp;
        Eigen::Vector3d ref_sp1;
        Eigen::Vector3d ref_sp2;
        Eigen::Vector3d ref_sp3;
        SurfInClouds >> curr_sp(0)
            >> curr_sp(1)
            >> curr_sp(2)
            >> ref_sp1(0)
            >> ref_sp1(1)
            >> ref_sp1(2)
            >> ref_sp2(0)
            >> ref_sp2(1)
            >> ref_sp2(2)
            >> ref_sp3(0)
            >> ref_sp3(1)
            >> ref_sp3(2);
        LiDARfeatures surff;
        surff.push_back(curr_sp);
        surff.push_back(ref_sp1);
        surff.push_back(ref_sp2);
        surff.push_back(ref_sp3);
        //surfFeatures.push(surff);
        surfFeatures.push_back(surff);
    }
    SurfInClouds.close();
    cout << "Surf features read in " << surfFeatures.size() << " features" << endl;
    //read corner feature points
    CornerInClouds.open(CornerName);
    while(CornerInClouds.good())
    {
        Eigen::Vector3d curr_cp;
        Eigen::Vector3d ref_cp1;
        Eigen::Vector3d ref_cp2;
        CornerInClouds >> curr_cp(0)
            >> curr_cp(1)
            >> curr_cp(2)
            >> ref_cp1(0)
            >> ref_cp1(1)
            >> ref_cp1(2)
            >> ref_cp2(0)
            >> ref_cp2(1)
            >> ref_cp2(2);
        LiDARfeatures cornerf;
        cornerf.push_back(curr_cp);
        cornerf.push_back(ref_cp1);
        cornerf.push_back(ref_cp2);
        //cornerFeatures.push(cornerf);
        cornerFeatures.push_back(cornerf);
    }
    CornerInClouds.close();
    cout << "Corner features read in " << cornerFeatures.size() << " features" << endl;
    //optimize
    //pose_s
    Eigen::Quaterniond qs(sqw_i, sqx_i, sqy_i, sqz_i);
    Eigen::Vector3d ts(stx_i, sty_i, stz_i);
    Eigen::Matrix4d pose_s;
    pose_s.block<3, 3>(0, 0) = qs.toRotationMatrix();
    pose_s.block<3, 1>(0, 3) = ts;

    Eigen::Quaterniond qs_o(sqw_o, sqx_o, sqy_o, sqz_o);
    Eigen::Vector3d ts_o(stx_o, sty_o, stz_o);
    Eigen::Matrix4d pose_so;
    pose_so.block<3, 3>(0, 0) = qs_o.toRotationMatrix();
    pose_so.block<3, 1>(0, 3) = ts_o;

    LidarFeatureFactor FactorSurf;
    int print_step_s = 0;
    bool check_surf_jacobian = false;
    Eigen::Matrix<double, 3, 3> Hessian_s = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> b_s = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> delta_s = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix4d pose_s_ite = Eigen::Matrix4d::Zero();
    double cost_surf = 0;
    for(int ite=0; ite<1; ++ite)
    {
        cost_surf = 0;
        for(int j=0; j<surfFeatures.size(); ++j)
        {
            Eigen::Matrix<double, 1, 3> jacobian_s;
            double residual_s, information_s;
            FactorSurf.get_surf_jacobian_N_residual(jacobian_s,
                    residual_s,
                    information_s,
                    pose_s,
                    surfFeatures.at(j));
            //whole hessian and b calculate
            Hessian_s += jacobian_s.transpose() * information_s * jacobian_s;
            //b_s += -1.0 * jacobian_s.transpose() * information_s * residual_s;
            b_s += -0.05 * jacobian_s.transpose() * information_s * residual_s;
            cost_surf += residual_s * residual_s;
            if(0==j)
                cout << "residual_s: " << residual_s << endl;

            if(check_surf_jacobian)
            {
                Eigen::Matrix<double, 1, 3> jacobian_auto_s;
                double residual_auto_plus_s, residual_auto_minus_s;
                Eigen::Matrix4d pose_auto_s;
                Eigen::Matrix<double, 6, 1> delta_chi_s = Eigen::Matrix<double, 6, 1>::Zero();
                double s_interval = 1e-8;
                double s_thres = 1e-8;
                vector<int> vec_s = {2, 3, 4};
                for(int i=0; i<3; ++i)
                {
                    delta_chi_s(vec_s.at(i)) = s_interval;
                    pose_auto_s = Utility::leftLieHybridPoseUpdate(delta_chi_s, pose_s);
                    residual_auto_plus_s = FactorSurf.get_surf_residual(pose_auto_s, surfFeatures.at(j));
                    delta_chi_s(vec_s.at(i)) = -s_interval;
                    pose_auto_s = Utility::leftLieHybridPoseUpdate(delta_chi_s, pose_s);
                    residual_auto_minus_s = FactorSurf.get_surf_residual(pose_auto_s, surfFeatures.at(j));
                    jacobian_auto_s(0, i) = 0.5 * (residual_auto_plus_s - residual_auto_minus_s) / s_interval;
                    delta_chi_s(vec_s.at(i)) = 0;
                }
                if((jacobian_s - jacobian_auto_s).norm() > s_thres)
                {
                    cout << "surf jacobian - jacobian_auto 's norm: " << (jacobian_s - jacobian_auto_s).norm() << endl;
                    if(print_step_s < 5 )
                        cout << "surf jacobian - jacobian_auto:\n" << jacobian_s - jacobian_auto_s << endl;
                }
            }
            ++print_step_s;
            //cout << surfFeatures.size() << endl;
        }
        delta_s = Hessian_s.ldlt().solve(b_s);
        //cout << "surf delta: " << delta_s << endl;
        //print cost
        cout << "surf cost: " << cost_surf << endl;
        Eigen::Matrix<double, 6, 1> delta_full_s = Eigen::Matrix<double, 6, 1>::Zero();
        delta_full_s(2) = delta_s(0);
        delta_full_s(3) = delta_s(1);
        delta_full_s(4) = delta_s(2);
        pose_s_ite = Utility::leftLieHybridPoseUpdate(delta_full_s, pose_s);
        pose_s = pose_s_ite;
    }
    Eigen::Quaterniond qs_ite(pose_s_ite.block<3, 3>(0, 0));
    Eigen::Vector3d ts_ite = pose_s_ite.block<3, 1>(0, 3);
    cout << "surf q diff: " << 2.0 * ((qs_ite.inverse() * qs_o).vec()).norm() << endl;
    cout << "surf t diff: " << (ts_ite - ts).norm() << endl;

    //pose_c
    Eigen::Quaterniond qc(cqw_i, cqx_i, cqy_i, cqz_i);
    Eigen::Vector3d tc(ctx_i, cty_i, ctz_i);
    Eigen::Matrix4d pose_c;
    pose_c.block<3, 3>(0, 0) = qc.toRotationMatrix();
    pose_c.block<3, 1>(0, 3) = tc;

    Eigen::Quaterniond qc_o(cqw_o, cqx_o, cqy_o, cqz_o);
    Eigen::Vector3d tc_o(ctx_o, cty_o, ctz_o);
    Eigen::Matrix4d pose_co;
    pose_co.block<3, 3>(0, 0) = qc_o.toRotationMatrix();
    pose_co.block<3, 1>(0, 3) = tc_o;

    LidarFeatureFactor FactorCorner;
    int print_step_c = 0;
    bool check_corner_jacobian = false;
    Eigen::Matrix<double, 3, 3> Hessian_c = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> b_c = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> delta_c = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix4d pose_c_ite = Eigen::Matrix4d::Zero();
    double cost_corner = 0;
    for(int ite=0; ite<1; ++ite)
    {
        cost_corner = 0;
        for(int j=0; j<cornerFeatures.size(); ++j)
        {
            Eigen::Matrix<double, 1, 3> jacobian_c;
            double residual_c, information_c;
            FactorCorner.get_corner_jacobian_N_residual(jacobian_c,
                    residual_c,
                    information_c,
                    pose_c,
                    cornerFeatures.at(j));
            //whole hessian and b calculate
            Hessian_c += jacobian_c.transpose() * information_c * jacobian_c;
            //b_c += -1.0 * jacobian_c.transpose() * information_c * residual_c;
            b_c += -0.05 * jacobian_c.transpose() * information_c * residual_c;
            cost_corner += residual_c * residual_c;
            if(0==j)
                cout << "residual_c: " << residual_c << endl;
            if(check_corner_jacobian)
            {
                Eigen::Matrix<double, 1, 3> jacobian_auto_c;
                double residual_auto_plus_c, residual_auto_minus_c;
                Eigen::Matrix4d pose_auto_c;
                Eigen::Matrix<double, 6, 1> delta_chi_c = Eigen::Matrix<double, 6, 1>::Zero();
                double c_interval = 1e-8;
                double c_thres = 1e-8;
                vector<int> vec_c = {0, 1, 5};
                for(int i=0; i<3; ++i)
                {
                    delta_chi_c(vec_c.at(i)) = c_interval;
                    pose_auto_c = Utility::leftLieHybridPoseUpdate(delta_chi_c, pose_c);
                    residual_auto_plus_c = FactorCorner.get_corner_residual(pose_auto_c, cornerFeatures.at(j));
                    delta_chi_c(vec_c.at(i)) = -c_interval;
                    pose_auto_c = Utility::leftLieHybridPoseUpdate(delta_chi_c, pose_c);
                    residual_auto_minus_c = FactorCorner.get_corner_residual(pose_auto_c, cornerFeatures.at(j));
                    jacobian_auto_c(0, i) = 0.5 * (residual_auto_plus_c - residual_auto_minus_c) / c_interval;
                    delta_chi_c(vec_c.at(i)) = 0;
                }
                if((jacobian_c - jacobian_auto_c).norm() > c_thres)
                {
                    cout << "corner jacobian - jacobian_auto 's norm: " << (jacobian_c - jacobian_auto_c).norm() << endl;
                    if(print_step_c < 5 )
                        cout << "corner jacobian - jacobian_auto:\n" << jacobian_c - jacobian_auto_c << endl;
                }
            }
            ++print_step_c;
            //cout << cornerFeatures.size() << endl;
        }
        delta_c = Hessian_c.ldlt().solve(b_c);
        //cout << "corner delta: " << delta_c << endl;
        //print cost
        cout << "corner cost: " << cost_corner << endl;
        Eigen::Matrix<double, 6, 1> delta_full_c = Eigen::Matrix<double, 6, 1>::Zero();
        delta_full_c(0) = delta_c(0);
        delta_full_c(1) = delta_c(1);
        delta_full_c(5) = delta_c(2);
        pose_c_ite = Utility::leftLieHybridPoseUpdate(delta_full_c, pose_c);
        pose_c = pose_c_ite;
    }
    Eigen::Quaterniond qc_ite(pose_c_ite.block<3, 3>(0, 0));
    Eigen::Vector3d tc_ite = pose_c_ite.block<3, 1>(0, 3);
    cout << "corner q diff: " << 2.0 * ((qc_ite.inverse() * qc_o).vec()).norm() << endl;
    cout << "corner t diff: " << (tc_ite - tc).norm() << endl;

    //LidarFeatureFactor Factor;
    //int print_step = 0;
    //while(!surfFeatures.empty() && !cornerFeatures.empty())
    //{
    ////cout << __FILE__ << __LINE__ <<endl;
    //Eigen::Matrix<double, 2, 6> jacobian;
    //Eigen::Vector2d residual;
    //Eigen::Matrix2d information;
    //Factor.get_jacobian_N_residual(jacobian,
    //residual,
    //information,
    //pose_s,
    //surfFeatures.front(),
    //cornerFeatures.front());
    //if(check_jacobian)
    //{
    //Eigen::Matrix<double, 2, 6> jacobian_auto;
    //Eigen::Vector2d residual_auto_plus, residual_auto_minus;
    //Eigen::Matrix4d pose_auto;
    //Eigen::Matrix<double, 6, 1> delta_chi = Eigen::Matrix<double, 6, 1>::Zero();
    //double interval = 1e-8;
    //double thres = 1e-8;
    //for(int i=0; i<6; ++i)
    //{
    //delta_chi(i) = interval;
    //pose_auto = Utility::leftLieHybridPoseUpdate(delta_chi, pose_s);
    //residual_auto_plus = Factor.get_residual(pose_auto, surfFeatures.front(), cornerFeatures.front());
    //delta_chi(i) = -interval;
    //pose_auto = Utility::leftLieHybridPoseUpdate(delta_chi, pose_s);
    //residual_auto_minus = Factor.get_residual(pose_auto, surfFeatures.front(), cornerFeatures.front());
    //jacobian_auto.col(i) = 0.5 * (residual_auto_plus - residual_auto_minus) / interval;
    //delta_chi(i) = 0;
    //}
    //if((jacobian - jacobian_auto).norm() > thres)
    //{
    //cout << "whole jacobian - jacobian_auto 's norm: " << (jacobian - jacobian_auto).norm() << endl;
    //if(print_step < 5 )
    //cout << "whole jacobian - jacobian_auto:\n" << jacobian - jacobian_auto << endl;
    //}
    //}
    //surfFeatures.pop();
    //cornerFeatures.pop();
    //++print_step;
    ////cout << cornerFeatures.size() << endl;
    //}
    return 0;
}
