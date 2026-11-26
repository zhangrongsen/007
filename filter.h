#include "filter.h"
#include <cmath>

DistanceKF::DistanceKF()
{
    reset();
    Q << 1e-3, 0,
         0, 1e-2;   // 适度放松过程噪声，避免过度信任模型
}

void DistanceKF::reset()
{
    x.setZero();
    P = Eigen::Matrix2d::Identity() * 1.0;
    initialized = false;
}

void DistanceKF::predict(double dt)
{
    if (dt <= 0.0) return;
    Eigen::Matrix2d F;
    F << 1, dt,
         0, 1;
    x = F * x;
    P = F * P * F.transpose() + Q * dt;
}

void DistanceKF::update(double z, double R)
{
    if (!initialized)
    {
        x(0) = z;
        x(1) = 0.0;
        P = Eigen::Matrix2d::Identity() * R;
        initialized = true;
        return;
    }

    // H = [1, 0]，只观测距离
    double y = z - x(0);                    // 标量残差
    double S = P(0, 0) + R;                 // 标量新息协方差
    Eigen::Vector2d K = P.col(0) / S;       // 2×1 卡尔曼增益

    x = x + K * y;
    P = (Eigen::Matrix2d::Identity() - K * Eigen::RowVector2d(1.0, 0.0)) * P;
    P = 0.5 * (P + P.transpose());          // 数值对称化
}
