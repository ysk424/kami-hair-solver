#include "solver.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace kami {
namespace {

Vec3 from_c(const KhsVec3 &v) { return {v.x, v.y, v.z}; }
KhsVec3 to_c(const Vec3 &v) { return {v.x(), v.y(), v.z()}; }

Vec3 rotation_vector(const Mat3 &r)
{
    Eigen::AngleAxisd aa(r);
    if (!std::isfinite(aa.angle()) || !finite(aa.axis())) return Vec3::Zero();
    double angle = aa.angle();
    if (angle > 3.14159265358979323846) angle -= 2.0 * 3.14159265358979323846;
    return aa.axis() * angle;
}

uint64_t edge_key(uint32_t a, uint32_t b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32u) | static_cast<uint64_t>(b);
}

} // namespace

void ColliderBvh::build(const std::vector<Vec3> &vertices,
                        const std::vector<Triangle> &triangles)
{
    order_.resize(triangles.size());
    std::iota(order_.begin(), order_.end(), 0u);
    nodes_.clear();
    if (!triangles.empty()) build_node(0, static_cast<uint32_t>(triangles.size()), vertices, triangles);
}

int ColliderBvh::build_node(uint32_t begin, uint32_t end,
                            const std::vector<Vec3> &vertices,
                            const std::vector<Triangle> &triangles)
{
    const int index = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    Vec3 lower = Vec3::Constant(std::numeric_limits<double>::infinity());
    Vec3 upper = Vec3::Constant(-std::numeric_limits<double>::infinity());
    Vec3 centroid_lower = lower;
    Vec3 centroid_upper = upper;
    for (uint32_t k = begin; k < end; ++k) {
        const Triangle &t = triangles[order_[k]];
        const Vec3 a = vertices[t.i0];
        const Vec3 b = vertices[t.i1];
        const Vec3 c = vertices[t.i2];
        lower = lower.cwiseMin(a).cwiseMin(b).cwiseMin(c);
        upper = upper.cwiseMax(a).cwiseMax(b).cwiseMax(c);
        const Vec3 centroid = (a + b + c) / 3.0;
        centroid_lower = centroid_lower.cwiseMin(centroid);
        centroid_upper = centroid_upper.cwiseMax(centroid);
    }
    nodes_[index].lower = lower;
    nodes_[index].upper = upper;
    nodes_[index].begin = begin;
    nodes_[index].count = end - begin;
    if (end - begin <= 8) return index;
    Eigen::Index axis = 0;
    (centroid_upper - centroid_lower).maxCoeff(&axis);
    const uint32_t middle = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
        [&](uint32_t lhs, uint32_t rhs) {
            const Triangle &l = triangles[lhs];
            const Triangle &r = triangles[rhs];
            const double lc = (vertices[l.i0][axis] + vertices[l.i1][axis] + vertices[l.i2][axis]) / 3.0;
            const double rc = (vertices[r.i0][axis] + vertices[r.i1][axis] + vertices[r.i2][axis]) / 3.0;
            return lc < rc;
        });
    const int left = build_node(begin, middle, vertices, triangles);
    const int right = build_node(middle, end, vertices, triangles);
    nodes_[index].left = left;
    nodes_[index].right = right;
    nodes_[index].count = 0;
    return index;
}

void ColliderBvh::query(const Vec3 &lower, const Vec3 &upper, double padding,
                        std::vector<uint32_t> &out) const
{
    out.clear();
    if (nodes_.empty()) return;
    std::vector<int> stack{0};
    while (!stack.empty()) {
        const int index = stack.back();
        stack.pop_back();
        const BvhNode &node = nodes_[index];
        if (!aabb_overlap(lower, upper, node.lower, node.upper, padding)) continue;
        if (node.left >= 0) {
            stack.push_back(node.left);
            stack.push_back(node.right);
        }
        else {
            for (uint32_t k = node.begin; k < node.begin + node.count; ++k) out.push_back(order_[k]);
        }
    }
}

Solver::Solver(const KhsSolverDesc &desc) : desc_(desc)
{
    build_stats_.struct_size = sizeof(KhsBuildStats);
    step_stats_.struct_size = sizeof(KhsStepStats);
    step_stats_.phase = KHS_PHASE_IDLE;
}

void Solver::clear_built_data()
{
    built_ = false;
    cuda_.reset();
    nodes_.clear();
    elements_.clear();
    strand_nodes_.clear();
    original_to_internal_.clear();
    root_x_current_.clear();
    root_x_pending_.clear();
    root_rotation_current_.clear();
    root_rotation_pending_.clear();
    merged_segments_ = 0;
    virtual_extension_strands_ = 0;
    virtual_extension_nodes_ = 0;
    virtual_extension_rest_length_ = 0.0;
}

KhsResult Solver::set_hair(const KhsVec3 *points, uint32_t point_count,
                           const uint32_t *offsets, uint32_t strand_count,
                           const KhsHairMaterial &material)
{
    if (!points || !offsets || point_count < 2 || strand_count == 0 ||
        offsets[0] != 0 || offsets[strand_count] != point_count) {
        error_ = "髪入力の点配列またはストランドオフセットが不正です。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    for (uint32_t s = 0; s < strand_count; ++s) {
        if (offsets[s + 1] <= offsets[s] || offsets[s + 1] > point_count) {
            error_ = "髪ストランドのオフセットが昇順ではありません。";
            return KHS_ERROR_INVALID_HAIR;
        }
    }
    if (!(material.density > 0.0 && material.radius > 0.0 &&
          material.young_modulus > 0.0 && material.poisson_ratio > -1.0 &&
          material.poisson_ratio < 0.5 && material.shear_correction > 0.0 &&
          material.contact_stiffness > 0.0 && material.barrier_distance > 0.0 &&
          material.friction >= 0.0 && material.friction_smoothing > 0.0)) {
        error_ = "髪材料値が許容範囲外です。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    std::vector<Vec3> converted(point_count);
    for (uint32_t i = 0; i < point_count; ++i) {
        converted[i] = from_c(points[i]);
        if (!finite(converted[i])) {
            std::ostringstream ss;
            ss << "髪の点 " << i << " に非有限座標があります。";
            error_ = ss.str();
            return KHS_ERROR_INVALID_HAIR;
        }
    }
    clear_built_data();
    input_points_ = converted;
    pending_root_input_ = converted;
    input_offsets_.assign(offsets, offsets + strand_count + 1);
    material_ = material;
    has_hair_ = true;
    error_.clear();
    return KHS_OK;
}

KhsResult Solver::set_collider(const KhsVec3 *vertices, uint32_t vertex_count,
                               const KhsTriangle *triangles, uint32_t triangle_count)
{
    if (triangle_count == 0) {
        has_collider_ = false;
        collider_pending_.clear();
        collider_current_.clear();
        collider_triangles_.clear();
        excluded_triangles_ = 0;
        built_ = false;
        return KHS_OK;
    }
    if (!vertices || !triangles || vertex_count < 3) {
        error_ = "コライダーの頂点または三角形配列が不正です。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    collider_pending_.resize(vertex_count);
    for (uint32_t i = 0; i < vertex_count; ++i) collider_pending_[i] = from_c(vertices[i]);
    collider_triangles_.clear();
    excluded_triangles_ = 0;
    for (uint32_t k = 0; k < triangle_count; ++k) {
        const KhsTriangle &source = triangles[k];
        if (source.i0 >= vertex_count || source.i1 >= vertex_count || source.i2 >= vertex_count ||
            source.i0 == source.i1 || source.i1 == source.i2 || source.i2 == source.i0 ||
            !finite(collider_pending_[source.i0]) || !finite(collider_pending_[source.i1]) ||
            !finite(collider_pending_[source.i2])) {
            ++excluded_triangles_;
            continue;
        }
        const Vec3 cross = (collider_pending_[source.i1] - collider_pending_[source.i0]).cross(
            collider_pending_[source.i2] - collider_pending_[source.i0]);
        if (cross.squaredNorm() <= 1.0e-24) {
            ++excluded_triangles_;
            continue;
        }
        collider_triangles_.push_back({source.i0, source.i1, source.i2});
    }
    if (collider_triangles_.empty()) {
        error_ = "有効なコライダー三角形がありません。";
        return KHS_ERROR_INVALID_COLLIDER;
    }
    collider_current_ = collider_pending_;
    has_collider_ = true;
    built_ = false;
    error_.clear();
    return KHS_OK;
}

bool Solver::make_internal_mesh()
{
    nodes_.clear();
    elements_.clear();
    strand_nodes_.assign(input_offsets_.size() - 1, {});
    original_to_internal_.assign(input_points_.size(), 0);
    const double hmax = desc_.maximum_element_length;
    const double zero_tolerance = 1.0e-12;
    virtual_extension_strands_ = 0;
    virtual_extension_nodes_ = 0;
    virtual_extension_rest_length_ = 0.0;
    for (uint32_t s = 0; s + 1 < input_offsets_.size(); ++s) {
        const uint32_t begin = input_offsets_[s];
        const uint32_t end = input_offsets_[s + 1];
        Node first;
        first.x = input_points_[begin];
        first.strand = s;
        first.binding = {begin, begin, 0.0};
        nodes_.push_back(first);
        strand_nodes_[s].push_back(static_cast<uint32_t>(nodes_.size() - 1));
        original_to_internal_[begin] = static_cast<uint32_t>(nodes_.size() - 1);
        Vec3 previous = input_points_[begin];
        uint32_t unique_segments = 0;
        double strand_rest_length = 0.0;
        uint32_t terminal_left = begin;
        uint32_t terminal_right = begin;
        double terminal_segment_length = 0.0;
        Vec3 terminal_direction = Vec3::Zero();
        for (uint32_t p = begin + 1; p < end; ++p) {
            const Vec3 target = input_points_[p];
            const double length = (target - previous).norm();
            if (length <= zero_tolerance) {
                original_to_internal_[p] = strand_nodes_[s].back();
                ++merged_segments_;
                previous = target;
                continue;
            }
            const uint32_t divisions = std::max(1u, static_cast<uint32_t>(std::ceil(length / hmax)));
            const uint32_t left_original = p - 1;
            for (uint32_t d = 1; d <= divisions; ++d) {
                const double t = static_cast<double>(d) / divisions;
                Node node;
                node.x = (1.0 - t) * previous + t * target;
                node.strand = s;
                node.binding = {left_original, p, t};
                nodes_.push_back(node);
                strand_nodes_[s].push_back(static_cast<uint32_t>(nodes_.size() - 1));
            }
            original_to_internal_[p] = strand_nodes_[s].back();
            previous = target;
            strand_rest_length += length;
            terminal_left = p - 1;
            terminal_right = p;
            terminal_segment_length = length;
            terminal_direction = (target - input_points_[p - 1]) / length;
            ++unique_segments;
        }
        if (unique_segments == 0 || strand_nodes_[s].size() < 2) {
            std::ostringstream ss;
            ss << "ストランド " << s << " は全点が同じ位置です（点 "
               << begin << "〜" << (end - 1) << "）。";
            error_ = ss.str();
            return false;
        }
        if (desc_.minimum_dynamic_length > strand_rest_length + zero_tolerance) {
            const double extension_length = desc_.minimum_dynamic_length - strand_rest_length;
            const uint32_t divisions = std::max(
                1u, static_cast<uint32_t>(std::ceil(extension_length / hmax)));
            const Vec3 tip = input_points_[end - 1];
            for (uint32_t d = 1; d <= divisions; ++d) {
                const double distance = extension_length * static_cast<double>(d) / divisions;
                Node node;
                node.x = tip + distance * terminal_direction;
                node.strand = s;
                node.virtual_extension = true;
                node.binding = {
                    terminal_left,
                    terminal_right,
                    1.0 + distance / terminal_segment_length};
                nodes_.push_back(node);
                strand_nodes_[s].push_back(static_cast<uint32_t>(nodes_.size() - 1));
            }
            ++virtual_extension_strands_;
            virtual_extension_nodes_ += divisions;
            virtual_extension_rest_length_ += extension_length;
        }
        const uint32_t fixed_count = std::min<uint32_t>(
            desc_.fixed_root_nodes, static_cast<uint32_t>(strand_nodes_[s].size()));
        for (uint32_t k = 0; k < fixed_count; ++k) nodes_[strand_nodes_[s][k]].fixed = true;
    }
    return true;
}

void Solver::initialize_frames_and_elements()
{
    elements_.clear();
    for (uint32_t s = 0; s < strand_nodes_.size(); ++s) {
        const auto &indices = strand_nodes_[s];
        std::vector<Vec3> tangents(indices.size());
        for (size_t k = 0; k < indices.size(); ++k) {
            Vec3 t;
            if (k == 0) t = nodes_[indices[1]].x - nodes_[indices[0]].x;
            else if (k + 1 == indices.size()) t = nodes_[indices[k]].x - nodes_[indices[k - 1]].x;
            else t = nodes_[indices[k + 1]].x - nodes_[indices[k - 1]].x;
            tangents[k] = t.normalized();
        }
        Mat3 frame = frame_from_tangent(tangents[0], Vec3::UnitX());
        nodes_[indices[0]].reference_frame = frame;
        for (size_t k = 1; k < indices.size(); ++k) {
            const Vec3 d1 = parallel_transport(frame.col(0), tangents[k - 1], tangents[k]);
            frame = frame_from_tangent(tangents[k], d1);
            nodes_[indices[k]].reference_frame = frame;
        }
        for (size_t k = 0; k + 1 < indices.size(); ++k) {
            Element e;
            e.i = indices[k];
            e.j = indices[k + 1];
            e.strand = s;
            e.collider_contact = !nodes_[e.j].virtual_extension;
            e.rest_length = (nodes_[e.j].x - nodes_[e.i].x).norm();
            const Mat3 &ri = nodes_[e.i].reference_frame;
            const Mat3 &rj = nodes_[e.j].reference_frame;
            const Vec3 tangent = (nodes_[e.j].x - nodes_[e.i].x) / e.rest_length;
            e.rest_shear = 0.5 * (ri.transpose() + rj.transpose()) * tangent;
            const Mat3 relative_rotation = ri.transpose() * rj;
            e.rest_curvature = cayley_rotation_vector(relative_rotation) / e.rest_length;
            elements_.push_back(e);
        }
    }
}

void Solver::initialize_masses()
{
    constexpr double pi = 3.14159265358979323846;
    const double area = pi * material_.radius * material_.radius;
    const double inertia = pi * std::pow(material_.radius, 4.0) / 4.0;
    const double polar = 2.0 * inertia;
    for (Node &node : nodes_) { node.mass = 0.0; node.rotational_mass = 0.0; }
    for (const Element &e : elements_) {
        const double mass = material_.density * area * e.rest_length;
        const double rotational_mass = material_.density * (2.0 * inertia + polar) / 3.0 * e.rest_length;
        nodes_[e.i].mass += 0.5 * mass;
        nodes_[e.j].mass += 0.5 * mass;
        nodes_[e.i].rotational_mass += 0.5 * rotational_mass;
        nodes_[e.j].rotational_mass += 0.5 * rotational_mass;
    }
}

void Solver::count_free_dofs()
{
    free_dof_count_ = 0;
    for (const Node &node : nodes_) if (!node.fixed) free_dof_count_ += 6;
}

void Solver::validate_collider_topology()
{
    boundary_edges_ = 0;
    nonmanifold_edges_ = 0;
    inconsistent_edges_ = 0;
    inverted_closed_components_ = 0;
    struct EdgeUse { uint32_t count = 0; int direction_sum = 0; };
    std::unordered_map<uint64_t, EdgeUse> uses;
    uses.reserve(collider_triangles_.size() * 3);
    for (const Triangle &t : collider_triangles_) {
        const std::array<std::pair<uint32_t, uint32_t>, 3> edges{{
            {t.i0, t.i1}, {t.i1, t.i2}, {t.i2, t.i0}}};
        for (const auto &edge : edges) {
            EdgeUse &use = uses[edge_key(edge.first, edge.second)];
            ++use.count;
            use.direction_sum += edge.first < edge.second ? 1 : -1;
        }
    }
    for (const auto &entry : uses) {
        const EdgeUse &use = entry.second;
        if (use.count == 1) ++boundary_edges_;
        else if (use.count > 2) ++nonmanifold_edges_;
        else if (use.direction_sum != 0) ++inconsistent_edges_;
    }
    collider_closed_manifold_ = boundary_edges_ == 0 && nonmanifold_edges_ == 0 && inconsistent_edges_ == 0;
    if (collider_closed_manifold_) {
        double signed_six_volume = 0.0;
        for (const Triangle &t : collider_triangles_) {
            signed_six_volume += collider_current_[t.i0].dot(
                collider_current_[t.i1].cross(collider_current_[t.i2]));
        }
        if (signed_six_volume < 0.0) inverted_closed_components_ = 1;
    }
}

bool Solver::compute_root_targets(const std::vector<Vec3> &points,
                                  std::vector<Vec3> &positions,
                                  std::vector<Vec3> &rotations)
{
    if (points.size() != input_points_.size()) return false;
    positions.resize(nodes_.size());
    rotations.resize(nodes_.size());
    for (uint32_t n = 0; n < nodes_.size(); ++n) {
        const RootBinding &b = nodes_[n].binding;
        positions[n] = (1.0 - b.t) * points[b.left] + b.t * points[b.right];
        rotations[n] = nodes_[n].rotation;
    }
    for (const auto &indices : strand_nodes_) {
        std::vector<Vec3> tangents(indices.size());
        for (size_t k = 0; k < indices.size(); ++k) {
            Vec3 t;
            if (k == 0) t = positions[indices[1]] - positions[indices[0]];
            else if (k + 1 == indices.size()) t = positions[indices[k]] - positions[indices[k - 1]];
            else t = positions[indices[k + 1]] - positions[indices[k - 1]];
            if (t.squaredNorm() < 1.0e-24) return false;
            tangents[k] = t.normalized();
        }
        Mat3 frame = frame_from_tangent(tangents[0], nodes_[indices[0]].reference_frame.col(0));
        rotations[indices[0]] = rotation_vector(frame * nodes_[indices[0]].reference_frame.transpose());
        for (size_t k = 1; k < indices.size(); ++k) {
            const Vec3 d1 = parallel_transport(frame.col(0), tangents[k - 1], tangents[k]);
            frame = frame_from_tangent(tangents[k], d1);
            rotations[indices[k]] = rotation_vector(frame * nodes_[indices[k]].reference_frame.transpose());
        }
    }
    return true;
}

KhsResult Solver::build()
{
    if (!has_hair_) {
        error_ = "先に Hair Curves を設定してください。";
        return KHS_ERROR_INVALID_STATE;
    }
    step_stats_.phase = KHS_PHASE_PREPARING;
    if (!make_internal_mesh()) {
        step_stats_.phase = KHS_PHASE_FAILED;
        return KHS_ERROR_INVALID_HAIR;
    }
    initialize_frames_and_elements();
    initialize_masses();
    count_free_dofs();
    if (!compute_root_targets(input_points_, root_x_current_, root_rotation_current_)) {
        error_ = "毛根の初期方向を構築できません。";
        step_stats_.phase = KHS_PHASE_FAILED;
        return KHS_ERROR_INVALID_HAIR;
    }
    root_x_pending_ = root_x_current_;
    root_rotation_pending_ = root_rotation_current_;
    if (has_collider_) {
        validate_collider_topology();
        collider_current_ = collider_pending_;
        collider_bvh_.build(collider_current_, collider_triangles_);
        if (!initial_feasibility_check()) {
            step_stats_.phase = KHS_PHASE_FAILED;
            return KHS_ERROR_INITIAL_INTERSECTION;
        }
    }
    cuda_ = make_cuda_backend();
    if (!cuda_) {
        error_ = "CUDAバックエンドを作成できません。CUDA 12.9対応GPUを確認してください。";
        step_stats_.phase = KHS_PHASE_FAILED;
        return KHS_ERROR_INVALID_STATE;
    }
    std::vector<CudaNodeInit> cuda_nodes(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node &source = nodes_[i];
        CudaNodeInit &target = cuda_nodes[i];
        target.position = to_c(source.x);
        target.rotation = to_c(source.rotation);
        for (int row = 0; row < 3; ++row) {
            target.reference[row] = {
                source.reference_frame(row, 0), source.reference_frame(row, 1),
                source.reference_frame(row, 2)};
        }
        target.mass = source.mass;
        target.rotational_mass = source.rotational_mass;
        target.strand = source.strand;
        target.binding_left = source.binding.left;
        target.binding_right = source.binding.right;
        target.binding_t = source.binding.t;
        target.fixed = source.fixed;
    }
    std::vector<CudaElementInit> cuda_elements(elements_.size());
    for (size_t i = 0; i < elements_.size(); ++i) {
        const Element &source = elements_[i];
        cuda_elements[i] = {source.i, source.j, source.strand,
                            source.collider_contact ? 1u : 0u, source.rest_length,
                            to_c(source.rest_shear), to_c(source.rest_curvature)};
    }
    std::vector<KhsVec3> cuda_collider(collider_current_.size());
    for (size_t i = 0; i < collider_current_.size(); ++i) cuda_collider[i] = to_c(collider_current_[i]);
    std::vector<CudaTriangleInit> cuda_triangles(collider_triangles_.size());
    for (size_t i = 0; i < collider_triangles_.size(); ++i) {
        const Triangle &source = collider_triangles_[i];
        cuda_triangles[i] = {source.i0, source.i1, source.i2};
    }
    const KhsResult cuda_result = cuda_->initialize(
        desc_, material_, cuda_nodes, cuda_elements, strand_nodes_, original_to_internal_,
        cuda_collider, cuda_triangles);
    if (cuda_result != KHS_OK) {
        error_ = cuda_->last_error();
        cuda_.reset();
        step_stats_.phase = KHS_PHASE_FAILED;
        return cuda_result;
    }
    built_ = true;
    build_stats_ = {};
    build_stats_.struct_size = sizeof(KhsBuildStats);
    build_stats_.strand_count = static_cast<uint32_t>(strand_nodes_.size());
    build_stats_.original_point_count = static_cast<uint32_t>(input_points_.size());
    build_stats_.internal_node_count = static_cast<uint32_t>(nodes_.size());
    build_stats_.element_count = static_cast<uint32_t>(elements_.size());
    build_stats_.fixed_node_count = static_cast<uint32_t>(std::count_if(
        nodes_.begin(), nodes_.end(), [](const Node &n) { return n.fixed; }));
    build_stats_.collider_vertex_count = static_cast<uint32_t>(collider_current_.size());
    build_stats_.collider_triangle_count = static_cast<uint32_t>(collider_triangles_.size());
    build_stats_.excluded_collider_triangle_count = excluded_triangles_;
    build_stats_.collider_boundary_edge_count = boundary_edges_;
    build_stats_.collider_nonmanifold_edge_count = nonmanifold_edges_;
    build_stats_.collider_inconsistent_edge_count = inconsistent_edges_;
    build_stats_.collider_inverted_closed_component_count = inverted_closed_components_;
    build_stats_.merged_zero_length_segment_count = merged_segments_;
    build_stats_.virtual_extension_strand_count = virtual_extension_strands_;
    build_stats_.virtual_extension_node_count = virtual_extension_nodes_;
    build_stats_.virtual_extension_rest_length = virtual_extension_rest_length_;
    build_stats_.degree_of_freedom_count = free_dof_count_;
    build_stats_.estimated_bytes =
        nodes_.size() * sizeof(Node) + elements_.size() * sizeof(Element) +
        collider_current_.size() * sizeof(Vec3) * 4 + collider_triangles_.size() * sizeof(Triangle);
    build_stats_.initial_minimum_gap = has_collider_ ? current_minimum_gap() :
        std::numeric_limits<double>::infinity();
    step_stats_.phase = KHS_PHASE_IDLE;
    error_.clear();
    return KHS_OK;
}

KhsResult Solver::update_runtime_parameters(const KhsSolverDesc &desc,
                                            const KhsHairMaterial &material)
{
    if (!built_ || !cuda_) {
        error_ = "実行時パラメーターを更新する前にソルバーを構築してください。";
        return KHS_ERROR_INVALID_STATE;
    }
    if (desc.maximum_element_length != desc_.maximum_element_length ||
        desc.minimum_dynamic_length != desc_.minimum_dynamic_length ||
        desc.fixed_root_nodes != desc_.fixed_root_nodes) {
        error_ =
            "再開中は最大要素長、最小動力学長、固定する毛根節点数を変更できません。"
            "これらを変更する場合は最初から計算してください。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    if (!(material.density > 0.0 && material.radius > 0.0 &&
          material.young_modulus > 0.0 && material.poisson_ratio > -1.0 &&
          material.poisson_ratio < 0.5 && material.shear_correction > 0.0 &&
          material.mass_damping >= 0.0 && material.contact_stiffness > 0.0 &&
          material.barrier_distance > 0.0 && material.friction >= 0.0 &&
          material.friction_smoothing > 0.0 && material.collider_offset >= 0.0)) {
        error_ = "再開用の髪材料値が許容範囲外です。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }

    const KhsSolverDesc old_desc = desc_;
    const KhsHairMaterial old_material = material_;
    desc_ = desc;
    material_ = material;
    initialize_masses();

    std::vector<double> masses(nodes_.size());
    std::vector<double> rotational_masses(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        masses[i] = nodes_[i].mass;
        rotational_masses[i] = nodes_[i].rotational_mass;
    }
    const KhsResult result = cuda_->update_runtime_parameters(
        desc_, material_, masses, rotational_masses);
    if (result != KHS_OK) {
        desc_ = old_desc;
        material_ = old_material;
        initialize_masses();
        error_ = cuda_->last_error();
        return result;
    }
    error_.clear();
    return KHS_OK;
}

bool Solver::point_inside_closed_collider(const Vec3 &point) const
{
    uint32_t hits = 0;
    const Vec3 direction(1.0, 0.0000012345, 0.0000023456);
    for (const Triangle &t : collider_triangles_) {
        const Vec3 &a = collider_current_[t.i0];
        const Vec3 &b = collider_current_[t.i1];
        const Vec3 &c = collider_current_[t.i2];
        const Vec3 e1 = b - a;
        const Vec3 e2 = c - a;
        const Vec3 h = direction.cross(e2);
        const double det = e1.dot(h);
        if (std::abs(det) < 1.0e-14) continue;
        const double inv = 1.0 / det;
        const Vec3 s = point - a;
        const double u = inv * s.dot(h);
        if (u < 0.0 || u > 1.0) continue;
        const Vec3 q = s.cross(e1);
        const double v = inv * direction.dot(q);
        if (v < 0.0 || u + v > 1.0) continue;
        const double distance = inv * e2.dot(q);
        if (distance > 1.0e-12) ++hits;
    }
    return (hits & 1u) != 0;
}

bool Solver::initial_feasibility_check()
{
    if (collider_closed_manifold_) {
        for (uint32_t n = 0; n < nodes_.size(); ++n) {
            if (nodes_[n].fixed || nodes_[n].virtual_extension) continue;
            if (point_inside_closed_collider(nodes_[n].x)) {
                std::ostringstream ss;
                ss << "初期交差: 内部節点 " << n << " が閉じたコライダー内部にあります。";
                error_ = ss.str();
                return false;
            }
        }
    }
    uint64_t candidates = 0;
    const double gap = current_minimum_gap(&candidates);
    if (gap <= desc_.minimum_gap) {
        std::ostringstream ss;
        ss << "初期交差または負のギャップがあります。最小ギャップ=" << gap
           << " m、候補数=" << candidates << "。";
        error_ = ss.str();
        return false;
    }
    return true;
}

double Solver::current_minimum_gap(uint64_t *candidate_count) const
{
    double minimum = std::numeric_limits<double>::infinity();
    uint64_t count = 0;
    std::vector<uint32_t> candidates;
    const double padding = material_.radius + material_.collider_offset + material_.barrier_distance;
    for (const Element &e : elements_) {
        if (nodes_[e.i].fixed || !e.collider_contact) continue;
        const Vec3 lower = nodes_[e.i].x.cwiseMin(nodes_[e.j].x);
        const Vec3 upper = nodes_[e.i].x.cwiseMax(nodes_[e.j].x);
        collider_bvh_.query(lower, upper, padding, candidates);
        for (uint32_t triangle_index : candidates) {
            const Triangle &t = collider_triangles_[triangle_index];
            const ClosestPair pair = closest_segment_triangle(
                nodes_[e.i].x, nodes_[e.j].x,
                collider_current_[t.i0], collider_current_[t.i1], collider_current_[t.i2]);
            minimum = std::min(minimum, pair.distance - material_.radius - material_.collider_offset);
            ++count;
        }
    }
    if (candidate_count) *candidate_count = count;
    return minimum;
}

KhsResult Solver::update_collider(const KhsVec3 *vertices, uint32_t vertex_count)
{
    if (!built_ || !has_collider_) {
        error_ = "構築済みコライダーがありません。";
        return KHS_ERROR_INVALID_STATE;
    }
    if (!vertices || vertex_count != collider_pending_.size()) {
        error_ = "アニメーションコライダーの頂点数が変化しました。";
        return KHS_ERROR_INVALID_COLLIDER;
    }
    for (uint32_t i = 0; i < vertex_count; ++i) {
        collider_pending_[i] = from_c(vertices[i]);
        if (!finite(collider_pending_[i])) {
            error_ = "アニメーションコライダーに非有限座標があります。";
            return KHS_ERROR_INVALID_COLLIDER;
        }
    }
    if (cuda_) {
        const KhsResult result = cuda_->update_collider(vertices, vertex_count);
        if (result != KHS_OK) {
            error_ = cuda_->last_error();
            return result;
        }
    }
    return KHS_OK;
}

KhsResult Solver::update_roots(const KhsVec3 *points, uint32_t point_count)
{
    if (!built_ || !points || point_count != input_points_.size()) {
        error_ = "毛根目標の点数が元 Hair Curves と一致しません。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    pending_root_input_.resize(point_count);
    for (uint32_t i = 0; i < point_count; ++i) {
        pending_root_input_[i] = from_c(points[i]);
        if (!finite(pending_root_input_[i])) {
            error_ = "毛根目標に非有限座標があります。";
            return KHS_ERROR_INVALID_HAIR;
        }
    }
    if (!compute_root_targets(pending_root_input_, root_x_pending_, root_rotation_pending_)) {
        error_ = "毛根目標から初期生え方向を計算できません。";
        return KHS_ERROR_INVALID_HAIR;
    }
    if (cuda_) {
        std::vector<KhsVec3> positions(root_x_pending_.size());
        std::vector<KhsVec3> rotations(root_rotation_pending_.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = to_c(root_x_pending_[i]);
            rotations[i] = to_c(root_rotation_pending_[i]);
        }
        const KhsResult result = cuda_->update_roots(positions, rotations);
        if (result != KHS_OK) {
            error_ = cuda_->last_error();
            return result;
        }
    }
    return KHS_OK;
}

KhsResult Solver::step(double frame_dt)
{
    if (!built_ || !cuda_) {
        error_ = "CUDA髪ソルバーを先に準備してください。";
        return KHS_ERROR_INVALID_STATE;
    }
    if (!(frame_dt > 0.0) || !std::isfinite(frame_dt)) {
        error_ = "時間刻みは正の有限値でなければなりません。";
        return KHS_ERROR_INVALID_ARGUMENT;
    }
    const KhsResult result = cuda_->step(frame_dt);
    step_stats_ = cuda_->stats();
    if (result != KHS_OK) error_ = cuda_->last_error();
    else error_.clear();
    return result;
}

KhsResult Solver::copy_original_positions(KhsVec3 *positions, uint32_t capacity) const
{
    if (!built_ || !cuda_ || !positions || capacity < original_to_internal_.size()) return KHS_ERROR_INVALID_ARGUMENT;
    return cuda_->copy_original_positions(positions, capacity);
}

KhsResult Solver::copy_internal_positions(KhsVec3 *positions, uint32_t capacity) const
{
    if (!built_ || !cuda_ || !positions || capacity < nodes_.size()) return KHS_ERROR_INVALID_ARGUMENT;
    return cuda_->copy_internal_positions(positions, capacity);
}

KhsResult Solver::copy_mapping(uint32_t *indices, uint32_t capacity) const
{
    if (!built_ || !indices || capacity < original_to_internal_.size()) return KHS_ERROR_INVALID_ARGUMENT;
    std::copy(original_to_internal_.begin(), original_to_internal_.end(), indices);
    return KHS_OK;
}

KhsResult Solver::build_stats(KhsBuildStats *stats) const
{
    if (!built_ || !stats || stats->struct_size < sizeof(KhsBuildStats)) return KHS_ERROR_INVALID_ARGUMENT;
    *stats = build_stats_;
    return KHS_OK;
}

KhsResult Solver::step_stats(KhsStepStats *stats) const
{
    if (!stats || stats->struct_size < sizeof(KhsStepStats)) return KHS_ERROR_INVALID_ARGUMENT;
    *stats = step_stats_;
    return KHS_OK;
}

KhsResult Solver::allocate_animation(uint32_t frame_count)
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    const KhsResult result = cuda_->allocate_animation(frame_count);
    if (result != KHS_OK) error_ = cuda_->last_error();
    return result;
}

KhsResult Solver::set_root_animation_frame(uint32_t frame, const KhsVec3 *points,
                                           uint32_t point_count)
{
    if (!built_ || !cuda_ || !points || point_count != input_points_.size())
        return KHS_ERROR_INVALID_ARGUMENT;
    std::vector<Vec3> converted(point_count);
    for (uint32_t i = 0; i < point_count; ++i) {
        converted[i] = from_c(points[i]);
        if (!finite(converted[i])) return KHS_ERROR_INVALID_HAIR;
    }
    std::vector<Vec3> positions;
    std::vector<Vec3> rotations;
    if (!compute_root_targets(converted, positions, rotations)) return KHS_ERROR_INVALID_HAIR;
    std::vector<KhsVec3> c_positions(positions.size());
    std::vector<KhsVec3> c_rotations(rotations.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        c_positions[i] = to_c(positions[i]);
        c_rotations[i] = to_c(rotations[i]);
    }
    const KhsResult result = cuda_->set_root_animation_frame(frame, c_positions, c_rotations);
    if (result != KHS_OK) error_ = cuda_->last_error();
    return result;
}

KhsResult Solver::set_collider_animation_frame(uint32_t frame, const KhsVec3 *vertices,
                                               uint32_t vertex_count)
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    const KhsResult result = cuda_->set_collider_animation_frame(frame, vertices, vertex_count);
    if (result != KHS_OK) error_ = cuda_->last_error();
    return result;
}

KhsResult Solver::finalize_animation()
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    const KhsResult result = cuda_->finalize_animation();
    if (result != KHS_OK) error_ = cuda_->last_error();
    return result;
}

KhsResult Solver::step_animation_frame(uint32_t frame, double frame_dt)
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    const KhsResult result = cuda_->step_animation_frame(frame, frame_dt);
    step_stats_ = cuda_->stats();
    if (result != KHS_OK) error_ = cuda_->last_error();
    else error_.clear();
    return result;
}

uint64_t Solver::animation_checkpoint_size() const
{
    return built_ && cuda_ ? cuda_->animation_checkpoint_size() : 0;
}

KhsResult Solver::save_animation_checkpoint(void *data, uint64_t capacity)
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    return cuda_->save_animation_checkpoint(data, capacity);
}

KhsResult Solver::restore_animation_checkpoint(const void *data, uint64_t size)
{
    if (!built_ || !cuda_) return KHS_ERROR_INVALID_STATE;
    const KhsResult result = cuda_->restore_animation_checkpoint(data, size);
    if (result != KHS_OK) error_ = cuda_->last_error();
    else error_.clear();
    return result;
}

KhsResult Solver::gpu_stats(KhsGpuStats *stats) const
{
    if (!cuda_ || !stats || stats->struct_size < sizeof(KhsGpuStats)) return KHS_ERROR_INVALID_ARGUMENT;
    *stats = cuda_->gpu_stats();
    return KHS_OK;
}

KhsResult Solver::progress(KhsProgress *value) const
{
    if (!cuda_ || !value || value->struct_size < sizeof(KhsProgress)) return KHS_ERROR_INVALID_ARGUMENT;
    *value = cuda_->progress();
    return KHS_OK;
}

KhsResult Solver::failure_diagnostics(KhsFailureDiagnostics *diagnostics) const
{
    if (!cuda_ || !diagnostics || diagnostics->struct_size < sizeof(KhsFailureDiagnostics))
        return KHS_ERROR_INVALID_ARGUMENT;
    *diagnostics = cuda_->failure_diagnostics();
    return KHS_OK;
}

KhsResult Solver::request_cancel()
{
    if (!cuda_) return KHS_ERROR_INVALID_STATE;
    cuda_->request_cancel();
    return KHS_OK;
}

} // namespace kami
