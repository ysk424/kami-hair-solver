#pragma once

#include "kami_hair_solver.h"
#include "cuda_backend.hpp"
#include "geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kami {

struct Triangle {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t i2 = 0;
};

struct RootBinding {
    uint32_t left = 0;
    uint32_t right = 0;
    double t = 0.0;
};

struct Node {
    Vec3 x = Vec3::Zero();
    Vec3 rotation = Vec3::Zero();
    Mat3 reference_frame = Mat3::Identity();
    double mass = 0.0;
    double rotational_mass = 0.0;
    bool fixed = false;
    bool virtual_extension = false;
    uint32_t strand = 0;
    RootBinding binding;
};

struct Element {
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t strand = 0;
    bool collider_contact = true;
    double rest_length = 0.0;
    Vec3 rest_shear = Vec3::Zero();
    Vec3 rest_curvature = Vec3::Zero();
};

struct BvhNode {
    Vec3 lower = Vec3::Zero();
    Vec3 upper = Vec3::Zero();
    uint32_t begin = 0;
    uint32_t count = 0;
    int left = -1;
    int right = -1;
};

class ColliderBvh {
public:
    void build(const std::vector<Vec3> &vertices, const std::vector<Triangle> &triangles);
    void query(const Vec3 &lower, const Vec3 &upper, double padding,
               std::vector<uint32_t> &out) const;

private:
    int build_node(uint32_t begin, uint32_t end, const std::vector<Vec3> &vertices,
                   const std::vector<Triangle> &triangles);
    std::vector<uint32_t> order_;
    std::vector<BvhNode> nodes_;
};

class Solver {
public:
    explicit Solver(const KhsSolverDesc &desc);

    KhsResult set_hair(const KhsVec3 *points, uint32_t point_count,
                       const uint32_t *offsets, uint32_t strand_count,
                       const KhsHairMaterial &material);
    KhsResult set_collider(const KhsVec3 *vertices, uint32_t vertex_count,
                           const KhsTriangle *triangles, uint32_t triangle_count);
    KhsResult build();
    KhsResult update_runtime_parameters(const KhsSolverDesc &desc,
                                        const KhsHairMaterial &material);
    KhsResult update_collider(const KhsVec3 *vertices, uint32_t vertex_count);
    KhsResult update_roots(const KhsVec3 *points, uint32_t point_count);
    KhsResult step(double frame_dt);
    KhsResult allocate_animation(uint32_t frame_count);
    KhsResult set_root_animation_frame(uint32_t frame, const KhsVec3 *points, uint32_t point_count);
    KhsResult set_collider_animation_frame(uint32_t frame, const KhsVec3 *vertices, uint32_t vertex_count);
    KhsResult finalize_animation();
    KhsResult step_animation_frame(uint32_t frame, double frame_dt);

    uint32_t original_point_count() const { return static_cast<uint32_t>(input_points_.size()); }
    uint32_t internal_node_count() const { return static_cast<uint32_t>(nodes_.size()); }
    KhsResult copy_original_positions(KhsVec3 *positions, uint32_t capacity) const;
    KhsResult copy_internal_positions(KhsVec3 *positions, uint32_t capacity) const;
    KhsResult copy_mapping(uint32_t *indices, uint32_t capacity) const;
    KhsResult build_stats(KhsBuildStats *stats) const;
    KhsResult step_stats(KhsStepStats *stats) const;
    KhsResult gpu_stats(KhsGpuStats *stats) const;
    KhsResult progress(KhsProgress *progress) const;
    KhsResult request_cancel();
    const std::string &last_error() const { return error_; }
    void set_error(std::string message) { error_ = std::move(message); }

private:
    void clear_built_data();
    bool make_internal_mesh();
    void initialize_frames_and_elements();
    void initialize_masses();
    void count_free_dofs();
    bool compute_root_targets(const std::vector<Vec3> &points,
                              std::vector<Vec3> &positions,
                              std::vector<Vec3> &rotations);
    void validate_collider_topology();
    bool initial_feasibility_check();

    bool point_inside_closed_collider(const Vec3 &point) const;
    double current_minimum_gap(uint64_t *candidate_count = nullptr) const;

    KhsSolverDesc desc_{};
    KhsHairMaterial material_{};
    bool has_hair_ = false;
    bool has_collider_ = false;
    bool built_ = false;
    std::string error_;

    std::vector<Vec3> input_points_;
    std::vector<uint32_t> input_offsets_;
    std::vector<Vec3> pending_root_input_;
    std::vector<uint32_t> original_to_internal_;
    std::vector<Node> nodes_;
    std::vector<Element> elements_;
    std::vector<std::vector<uint32_t>> strand_nodes_;
    uint32_t free_dof_count_ = 0;

    std::vector<Vec3> root_x_current_;
    std::vector<Vec3> root_x_pending_;
    std::vector<Vec3> root_rotation_current_;
    std::vector<Vec3> root_rotation_pending_;

    std::vector<Vec3> collider_pending_;
    std::vector<Vec3> collider_current_;
    std::vector<Triangle> collider_triangles_;
    ColliderBvh collider_bvh_;
    uint32_t excluded_triangles_ = 0;
    uint32_t boundary_edges_ = 0;
    uint32_t nonmanifold_edges_ = 0;
    uint32_t inconsistent_edges_ = 0;
    uint32_t inverted_closed_components_ = 0;
    bool collider_closed_manifold_ = false;
    uint32_t merged_segments_ = 0;
    uint32_t virtual_extension_strands_ = 0;
    uint32_t virtual_extension_nodes_ = 0;
    double virtual_extension_rest_length_ = 0.0;

    KhsBuildStats build_stats_{};
    KhsStepStats step_stats_{};
    std::unique_ptr<CudaBackend> cuda_;
};

} // namespace kami
