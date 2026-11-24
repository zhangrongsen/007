#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <optional>
#include <Eigen/Dense>
#include "filter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// ===================== 题目固定参数 =====================
const double W_ARMOR = 0.135;
const double H_ARMOR = 0.056;
const double R_OUTPOST = 0.275;
const double R_BUFF = 0.15;

const Mat K = (Mat_<double>(3, 3) <<
    1781.053454, 0, 510.5751561,
    0, 1781.686873, 472.9735763,
    0, 0, 1);
const Mat D = (Mat_<double>(1, 5) <<
    -0.04420740074, -0.1081537096,
    -0.0006506219039, 0.0002357380215, 0);
const double FPS_REF = 119.997728;

const double MIN_TZ = 0.01;
const double MAX_REPROJ_ERR = 2.5;
const double MAX_PNP_POS_RESIDUAL = 0.15;
const int MIN_OBS_PER_LAYER = 10;
const double MIN_PHASE_SPAN = M_PI / 2.0;
const double MIN_TIME_SPAN = 0.5;
const double RMSE_RATIO_THRESH = 0.9;
const int MASK_START = 1200;
const int MASK_END = 1260;

const vector<Point3d> armor_world_pts = {
    Point3d(-W_ARMOR / 2, -H_ARMOR / 2, 0),
    Point3d( W_ARMOR / 2, -H_ARMOR / 2, 0),
    Point3d( W_ARMOR / 2,  H_ARMOR / 2, 0),
    Point3d(-W_ARMOR / 2,  H_ARMOR / 2, 0)
};

// 灯条结构体
struct LightBar {
    RotatedRect rect;
    Point2f center;
    float angle;
    float length;
};

// 装甲结构体
struct Armor {
    vector<Point2f> pts;
    Point2f center;
    Mat rvec, tvec;
    bool valid_pnp = false;
    int track_id = -1;
};

// 跟踪器：每个装甲ID绑定一个KF
struct TrackArmor
{
    int id;
    Point2f last_center;
    DistanceKF kf;
    double last_t;
};

// 观测点，用于前哨站LM拟合
struct Obs
{
    double t;
    Eigen::Vector3d Pc; // 装甲板中心在相机坐标系下3D点
};

vector<TrackArmor> tracks;
int next_id = 0;

bool isLightPair(const LightBar& l1, const LightBar& l2)
{
    float angle_diff = fabs(l1.angle - l2.angle);
    if (angle_diff > 15) return false;
    float dist = norm(l1.center - l2.center);
    float avg_len = (l1.length + l2.length) / 2.0f;
    if (dist < avg_len * 0.8 || dist > avg_len * 4.0) return false;
    return true;
}

vector<Point2f> getArmorPoints(const LightBar& left, const LightBar& right)
{
    vector<Point2f> res(4);
    Point2f l_top, l_bot, r_top, r_bot;
    auto getEnds = [](RotatedRect r, Point2f& p1, Point2f& p2)
    {
        Point2f pts[4];
        r.points(pts);
        sort(pts, pts + 4, [](const Point2f& a, const Point2f& b) { return a.y < b.y; });
        p1 = pts[0]; p2 = pts[3];
    };
    getEnds(left.rect, l_top, l_bot);
    getEnds(right.rect, r_top, r_bot);
    res[0] = l_top;
    res[1] = r_top;
    res[2] = r_bot;
    res[3] = l_bot;
    return res;
}

vector<LightBar> findLightBars(const Mat& src)
{
    Mat gray, bin;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold(gray, bin, 180, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(bin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<LightBar> lights;
    for (auto& cnt : contours)
    {
        if (contourArea(cnt) < 10) continue;
        RotatedRect r = minAreaRect(cnt);
        float w = r.size.width;
        float h = r.size.height;
        if (max(w, h) / (min(w, h) + 1e-6) < 2.5) continue;
        LightBar lb;
        lb.rect = r;
        lb.center = r.center;
        lb.angle = r.angle;
        lb.length = max(w, h);
        lights.push_back(lb);
    }
    return lights;
}

vector<Armor> detectArmor(const Mat& src)
{
    vector<LightBar> lights = findLightBars(src);
    vector<Armor> armors;
    if (lights.size() < 2) return armors;
    for (int i = 0; i < lights.size(); i++)
    {
        for (int j = i + 1; j < lights.size(); j++)
        {
            LightBar l1 = lights[i];
            LightBar l2 = lights[j];
            if (l1.center.x > l2.center.x) swap(l1, l2);
            if (!isLightPair(l1, l2)) continue;
            Armor a;
            a.pts = getArmorPoints(l1, l2);
            a.center = (l1.center + l2.center) / 2.0f;
            bool ok = solvePnP(armor_world_pts, a.pts, K, D, a.rvec, a.tvec, false, SOLVEPNP_ITERATIVE);
            if (ok)
            {
                a.valid_pnp = true;
                armors.push_back(a);
            }
        }
    }
    return armors;
}

// 跨帧装甲匹配（简单最近邻）
void associateArmor(vector<Armor>& armors, double t_now)
{
    const double max_match_dist = 60.0;
    for (auto& ar : armors)
    {
        double min_d = 1e9;
        int best_idx = -1;
        for (int i = 0; i < tracks.size(); i++)
        {
            double d = norm(ar.center - tracks[i].last_center);
            if (d < max_match_dist && d < min_d)
            {
                min_d = d;
                best_idx = i;
            }
        }
        if (best_idx >= 0)
        {
            ar.track_id = tracks[best_idx].id;
            tracks[best_idx].last_center = ar.center;
            tracks[best_idx].last_t = t_now;
        }
        else
        {
            ar.track_id = next_id++;
            TrackArmor tr;
            tr.id = ar.track_id;
            tr.last_center = ar.center;
            tr.last_t = t_now;
            tr.kf.reset();
            tracks.push_back(tr);
        }
    }
}

void drawArmor(Mat& img, const Armor& ar)
{
    if (!ar.valid_pnp) return;
    for (int i = 0; i < 4; i++)
    {
        line(img, ar.pts[i], ar.pts[(i+1)%4], Scalar(0,255,0), 2);
    }
    stringstream ss;
    ss << "ID:" << ar.track_id << " Z:" << fixed << setprecision(2) << ar.tvec.at<double>(2,0);
    putText(img, ss.str(), ar.center, FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0,255,255),1);
}

// LM拟合前哨站圆周运动模型：Pc = C + R * [cos(theta0 + w*t), sin(theta0 + w*t), 0]
// 待估：Cx,Cy,Cz, w, theta0
Eigen::VectorXd lmFitOutpost(const vector<Obs>& obs)
{
    Eigen::VectorXd x0(5);
    x0 << 0,0,0, 2.0, 0.0; // 初值 Cx,Cy,Cz, w, theta0
    const int max_iter = 30;
    const double eps = 1e-6;
    for(int iter=0;iter<max_iter;iter++)
    {
        Eigen::MatrixXd J(obs.size(),5);
        Eigen::VectorXd f(obs.size());
        for(int i=0;i<obs.size();i++)
        {
            double t = obs[i].t;
            double Cx = x0(0), Cy=x0(1), Cz=x0(2), w=x0(3), th0=x0(4);
            double th = th0 + w * t;
            double cx = cos(th);
            double sx = sin(th);
            Eigen::Vector3d pred;
            pred << Cx + R_OUTPOST * cx, Cy + R_OUTPOST * sx, Cz;
            Eigen::Vector3d res = obs[i].Pc - pred;
            f(i) = res.squaredNorm();

            J(i,0) = -2 * res(0);
            J(i,1) = -2 * res(1);
            J(i,2) = -2 * res(2);
            J(i,3) = -2 * ( -res(0)*R_OUTPOST*sx*t + res(1)*R_OUTPOST*cx*t );
            J(i,4) = -2 * ( -res(0)*R_OUTPOST*sx + res(1)*R_OUTPOST*cx );
        }
        Eigen::VectorXd delta = (J.transpose() * J + 1e-6*Eigen::MatrixXd::Identity(5,5)).inverse() * J.transpose() * f;
        x0 = x0 - delta;
        if(delta.norm() < eps) break;
    }
    return x0;
}

int main()
{
    // ========== 修改这里为你的视频路径 ==========
    VideoCapture cap("outpost.mp4");
    if (!cap.isOpened())
    {
        cout << "视频打开失败" << endl;
        return -1;
    }
    ofstream csv("result.csv");
    csv << "frame,t,id,z_raw,z_filtered,Cx,Cy,Cz,omega,theta0\n";

    Mat frame;
    int frame_cnt = 0;
    vector<Obs> all_obs;
    Eigen::VectorXd est_param(5);
    bool has_fit = false;

    while (cap.read(frame))
    {
        double t = frame_cnt / FPS_REF;
        vector<Armor> armors = detectArmor(frame);
        associateArmor(armors, t);

        // 距离KF更新
        for(auto& ar : armors)
        {
            double z_raw = ar.tvec.at<double>(2,0);
            if(z_raw < MIN_TZ) continue;
            // 找到对应track
            TrackArmor* tr = nullptr;
            for(auto& track : tracks)
            {
                if(track.id == ar.track_id) { tr = &track; break; }
            }
            if(!tr) continue;
            double dt = t - tr->last_t;
            if(dt > 0) tr->kf.predict(dt);
            tr->kf.update(z_raw, 0.005);
            double z_filt = tr->kf.getDist();

            // 构造3D观测（装甲板中心相机坐标系）
            Eigen::Vector3d Pc(ar.tvec.at<double>(0,0), ar.tvec.at<double>(1,0), ar.tvec.at<double>(2,0));
            all_obs.push_back({t, Pc});

            // 写csv
            csv << frame_cnt << "," << t << "," << ar.track_id << "," << z_raw << "," << z_filt << ",";
            if(has_fit)
            {
                csv << est_param(0) << "," << est_param(1) << "," << est_param(2) << "," << est_param(3) << "," << est_param(4);
            }else{
                csv << ",,,,,";
            }
            csv << "\n";
        }

        // 收集足够观测后LM拟合（你可以自己调触发条件）
        if(all_obs.size() > MIN_OBS_PER_LAYER && !has_fit)
        {
            est_param = lmFitOutpost(all_obs);
            has_fit = true;
            cout << "==== 拟合结果 ====" << endl;
            cout << "旋转中心 C = (" << est_param(0) << "," << est_param(1) << "," << est_param(2) << ")" << endl;
            cout << "角速度 w = " << est_param(3) << " rad/s" << endl;
            cout << "初始相位 theta0 = " << est_param(4) << endl;
        }

        for (auto& ar : armors) drawArmor(frame, ar);
        imshow("Armor Detector", frame);
        if (waitKey(1) == 27) break;
        frame_cnt++;
    }
    cap.release();
    csv.close();
    destroyAllWindows();
    cout << "完成，结果输出到 result.csv" << endl;
    return 0;
}
