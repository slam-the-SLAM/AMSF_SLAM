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
    string SurfPath = "/Corner_";
    string CornerPath = "/Surf_";
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

    queue<LiDARfeatures> surfFeatures, cornerFeatures;
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
        surfFeatures.push(surff);
    }
    SurfInClouds.close();
    cout << "Surf features read in!" << endl;
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
        cornerFeatures.push(cornerf);
    }
    CornerInClouds.close();
    cout << "Corner features read in!" << endl;
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
    while(!surfFeatures.empty())
    {
        Eigen::Matrix<double, 1, 3> jacobian_s;
        double residual_s, information_s;
        FactorSurf.get_surf_jacobian_N_residual(jacobian_s,
                residual_s,
                information_s,
                pose_s,
                surfFeatures.front());
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
                residual_auto_plus_s = FactorSurf.get_surf_residual(pose_auto_s, surfFeatures.front());
                delta_chi_s(vec_s.at(i)) = -s_interval;
                pose_auto_s = Utility::leftLieHybridPoseUpdate(delta_chi_s, pose_s);
                residual_auto_minus_s = FactorSurf.get_surf_residual(pose_auto_s, surfFeatures.front());
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
        surfFeatures.pop();
        ++print_step_s;
        //cout << surfFeatures.size() << endl;
    }

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
    while(!cornerFeatures.empty())
    {
        Eigen::Matrix<double, 1, 3> jacobian_c;
        double residual_c, information_c;
        FactorCorner.get_corner_jacobian_N_residual(jacobian_c,
                residual_c,
                information_c,
                pose_c,
                cornerFeatures.front());
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
                residual_auto_plus_c = FactorCorner.get_corner_residual(pose_auto_c, cornerFeatures.front());
                delta_chi_c(vec_c.at(i)) = -c_interval;
                pose_auto_c = Utility::leftLieHybridPoseUpdate(delta_chi_c, pose_c);
                residual_auto_minus_c = FactorCorner.get_corner_residual(pose_auto_c, cornerFeatures.front());
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
        cornerFeatures.pop();
        ++print_step_c;
        //cout << cornerFeatures.size() << endl;
    }
    return 0;
}
