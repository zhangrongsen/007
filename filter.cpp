#include "filter.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DistanceKF::DistanceKF()
{
    reset();
    R_meas = 0.005;
    Q << 1e-4, 0,
         0, 1e-3;
}

void DistanceKF::reset()
{
    x.setZero();
    P = Eigen::Matrix2d::Identity() * 10.0;
    initialized = false;
}

void DistanceKF::predict(double dt)
{
    Eigen::Matrix2d F;
    F << 1, dt,
         0, 1;
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void DistanceKF::update(double z, double R)
{
    if (!initialized)
    {
        x(0) = z;
        x(1) = 0.0;
        initialized = true;
        return;
    }
    Eigen::Matrix<double,1,2> H;
    H << 1, 0;
    Eigen::Matrix<double,1,1> z_hat = H * x;
    Eigen::Matrix<double,1,1> y = z - z_hat;
    Eigen::Matrix<double,1,1> S = H * P * H.transpose() + Eigen::Matrix<double,1,1>(R);
    Eigen::Matrix2d Kgain = P * H.transpose() * S.inverse();
    x = x + Kgain * y;
    P = (Eigen::Matrix2d::Identity() - Kgain * H) * P;
}
