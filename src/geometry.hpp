#pragma once

#include "math.hpp"

#include <array>

namespace kami {

struct ClosestPair {
    double distance = std::numeric_limits<double>::infinity();
    double segment_t = 0.0;
    Eigen::Vector3d triangle_bary = Eigen::Vector3d(1.0, 0.0, 0.0);
};

inline std::pair<Vec3, Eigen::Vector3d> closest_point_triangle(
    const Vec3 &p, const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return {a, {1.0, 0.0, 0.0}};
    const Vec3 bp = p - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return {b, {0.0, 1.0, 0.0}};
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return {a + v * ab, {1.0 - v, v, 0.0}};
    }
    const Vec3 cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return {c, {0.0, 0.0, 1.0}};
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return {a + w * ac, {1.0 - w, 0.0, w}};
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {b + w * (c - b), {0.0, 1.0 - w, w}};
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    return {a + ab * v + ac * w, {1.0 - v - w, v, w}};
}

struct SegmentPair {
    Vec3 first;
    Vec3 second;
    double s = 0.0;
    double t = 0.0;
};

inline SegmentPair closest_segment_segment(
    const Vec3 &p1, const Vec3 &q1, const Vec3 &p2, const Vec3 &q2)
{
    const double eps = 1.0e-14;
    const Vec3 d1 = q1 - p1;
    const Vec3 d2 = q2 - p2;
    const Vec3 r = p1 - p2;
    const double a = d1.dot(d1);
    const double e = d2.dot(d2);
    const double f = d2.dot(r);
    double s = 0.0;
    double t = 0.0;
    if (a <= eps && e <= eps) return {p1, p2, 0.0, 0.0};
    if (a <= eps) t = std::clamp(f / e, 0.0, 1.0);
    else {
        const double c = d1.dot(r);
        if (e <= eps) s = std::clamp(-c / a, 0.0, 1.0);
        else {
            const double b = d1.dot(d2);
            const double denom = a * e - b * b;
            if (denom != 0.0) s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, 0.0, 1.0); }
            else if (t > 1.0) { t = 1.0; s = std::clamp((b - c) / a, 0.0, 1.0); }
        }
    }
    return {p1 + s * d1, p2 + t * d2, s, t};
}

inline bool segment_triangle_intersection(const Vec3 &p0, const Vec3 &p1,
                                          const Vec3 &a, const Vec3 &b, const Vec3 &c,
                                          double &t, Eigen::Vector3d &bary)
{
    const Vec3 d = p1 - p0;
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;
    const Vec3 h = d.cross(e2);
    const double det = e1.dot(h);
    if (std::abs(det) < 1.0e-14) return false;
    const double inv = 1.0 / det;
    const Vec3 s = p0 - a;
    const double u = inv * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;
    const Vec3 q = s.cross(e1);
    const double v = inv * d.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;
    t = inv * e2.dot(q);
    if (t < 0.0 || t > 1.0) return false;
    bary = {1.0 - u - v, u, v};
    return true;
}

inline ClosestPair closest_segment_triangle(const Vec3 &p0, const Vec3 &p1,
                                            const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    ClosestPair best;
    double hit_t = 0.0;
    Eigen::Vector3d hit_bary;
    if (segment_triangle_intersection(p0, p1, a, b, c, hit_t, hit_bary)) {
        best.distance = 0.0;
        best.segment_t = hit_t;
        best.triangle_bary = hit_bary;
        return best;
    }
    auto consider_point = [&](const Vec3 &p, double st) {
        const auto cp = closest_point_triangle(p, a, b, c);
        const double d = (p - cp.first).norm();
        if (d < best.distance) {
            best.distance = d;
            best.segment_t = st;
            best.triangle_bary = cp.second;
        }
    };
    consider_point(p0, 0.0);
    consider_point(p1, 1.0);
    const std::array<Vec3, 3> tv{a, b, c};
    for (int e = 0; e < 3; ++e) {
        const int n = (e + 1) % 3;
        const SegmentPair pair = closest_segment_segment(p0, p1, tv[e], tv[n]);
        const double d = (pair.first - pair.second).norm();
        if (d < best.distance) {
            best.distance = d;
            best.segment_t = pair.s;
            best.triangle_bary.setZero();
            best.triangle_bary[e] = 1.0 - pair.t;
            best.triangle_bary[n] = pair.t;
        }
    }
    return best;
}

inline bool aabb_overlap(const Vec3 &amin, const Vec3 &amax,
                         const Vec3 &bmin, const Vec3 &bmax, double padding)
{
    return (amin.array() <= bmax.array() + padding).all() &&
           (bmin.array() <= amax.array() + padding).all();
}

} // namespace kami
