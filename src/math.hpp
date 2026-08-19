#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>

namespace kami {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

template<typename S>
Eigen::Matrix<S, 3, 1> cayley_rotation_vector(const Eigen::Matrix<S, 3, 3> &q)
{
    Eigen::Matrix<S, 3, 1> v;
    v << (q(2, 1) - q(1, 2)) * S(0.5),
         (q(0, 2) - q(2, 0)) * S(0.5),
         (q(1, 0) - q(0, 1)) * S(0.5);
    const S c = (q.trace() - S(1)) * S(0.5);
    return (S(2) / (S(1) + c + S(1.0e-12))) * v;
}

inline Mat3 frame_from_tangent(const Vec3 &tangent, const Vec3 &hint)
{
    Vec3 d3 = tangent.normalized();
    Vec3 d1 = hint - d3 * hint.dot(d3);
    if (d1.squaredNorm() < 1.0e-16) {
        const Vec3 axis = std::abs(d3.x()) < 0.8 ? Vec3::UnitX() : Vec3::UnitY();
        d1 = axis - d3 * axis.dot(d3);
    }
    d1.normalize();
    Vec3 d2 = d3.cross(d1).normalized();
    d1 = d2.cross(d3).normalized();
    Mat3 r;
    r.col(0) = d1;
    r.col(1) = d2;
    r.col(2) = d3;
    return r;
}

inline Vec3 parallel_transport(const Vec3 &director, const Vec3 &from, const Vec3 &to)
{
    const Vec3 a = from.normalized();
    const Vec3 b = to.normalized();
    const Vec3 axis = a.cross(b);
    const double s = axis.norm();
    const double c = std::clamp(a.dot(b), -1.0, 1.0);
    if (s < 1.0e-12) {
        if (c > 0.0) return director;
        Vec3 n = a.unitOrthogonal();
        return 2.0 * n * n.dot(director) - director;
    }
    const Vec3 n = axis / s;
    return director * c + n.cross(director) * s + n * n.dot(director) * (1.0 - c);
}

inline bool finite(const Vec3 &v)
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

} // namespace kami
