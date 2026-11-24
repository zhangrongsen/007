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

using namespace cv;
using namespace std;

// ===================== 题目固定参数 =====================
const double W_ARMOR = 0.135;
const double H_ARMOR = 0.056;
const double R_OUTPOST = 0.275;
const double R_BUFF = 0.15;

// 相机内参、畸变（和你之前一致）
const Mat K = (Mat_<double>(3, 3) <<
    1781.053454, 0, 510.5751561,
    0, 1781.686873, 472.9735763,
    0, 0, 1);
const Mat D = (Mat_<double>(1, 5) <<
    -0.04420740074, -0.1081537096,
    -0.0006506219039, 0.0002357380215, 0);
const double FPS_REF = 119.997728;

// ===================== 质量筛选阈值 =====================
const double MIN_TZ = 0.01;
const double MAX_REPROJ_ERR = 2.5;
const double MAX_PNP_POS_RESIDUAL = 0.15;
const int MIN_OBS_PER_LAYER = 10;
const double MIN_PHASE_SPAN = CV_PI / 2.0;
const double MIN_TIME_SPAN = 0.5;
const double RMSE_RATIO_THRESH = 0.9;

// 屏蔽帧范围
const int MASK_START = 1200;
const int MASK_END = 1260;

// 装甲板世界坐标（标准大小装甲）
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
};

// 灯条匹配：角度差、距离、长宽比筛选
bool isLightPair(const LightBar& l1, const LightBar& l2)
{
    float angle_diff = fabs(l1.angle - l2.angle);
    if (angle_diff > 15) return false;

    float dist = norm(l1.center - l2.center);
    float avg_len = (l1.length + l2.length) / 2.0f;
    if (dist < avg_len * 0.8 || dist > avg_len * 4.0) return false;

    return true;
}

// 从一对灯条，按顺序生成装甲四点
vector<Point2f> getArmorPoints(const LightBar& left, const LightBar& right)
{
    vector<Point2f> res(4);
    // 取灯条的上下端点
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

// 图像预处理：二值化、找灯条
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

// 装甲匹配 + PnP求解
vector<Armor> detectArmor(const Mat& src)
{
    vector<LightBar> lights = findLightBars(src);
    vector<Armor> armors;
    if (lights.size() < 2) return armors;

    // 两两配对
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

            // PnP
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

// 绘制装甲
void drawArmor(Mat& img, const Armor& ar)
{
    if (!ar.valid_pnp) return;
    for (int i = 0; i < 4; i++)
    {
        line(img, ar.pts[i], ar.pts[(i+1)%4], Scalar(0,255,0), 2);
    }
    stringstream ss;
    ss << "Z:" << fixed << setprecision(2) << ar.tvec.at<double>(2,0);
    putText(img, ss.str(), ar.center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255),1);
}

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "无法打开摄像头" << endl;
        return -1;
    }
    Mat frame;
    while (cap.read(frame))
    {
        vector<Armor> armors = detectArmor(frame);
        for (auto& ar : armors)
        {
            drawArmor(frame, ar);
        }
        imshow("Armor Detector", frame);
        if (waitKey(1) == 27) break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}