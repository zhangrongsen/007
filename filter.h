#ifndef FILTER_H
#define FILTER_H
#include <Eigen/Dense>

// 一维距离卡尔曼滤波：状态 [距离d, 距离导数dd]
class DistanceKF
{
public:
    DistanceKF();
    void reset();
    void predict(double dt);
    void update(double z, double R);
    double getDist() const { return x(0); }
    double getVel() const { return x(1); }

private:
    Eigen::Vector2d x;
    Eigen::Matrix2d P;
    Eigen::Matrix2d Q;
    double R_meas;
    bool initialized;
};

#endif
