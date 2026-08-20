#pragma once

#include "kami_hair_solver.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kami {

struct CudaNodeInit {
    KhsVec3 position{};
    KhsVec3 rotation{};
    KhsVec3 reference[3]{};
    double mass = 0.0;
    double rotational_mass = 0.0;
    uint32_t strand = 0;
    uint32_t binding_left = 0;
    uint32_t binding_right = 0;
    double binding_t = 0.0;
    bool fixed = false;
};

struct CudaElementInit {
    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t strand = 0;
    uint32_t collider_contact = 1;
    double rest_length = 0.0;
    KhsVec3 rest_shear{};
    KhsVec3 rest_curvature{};
};

struct CudaTriangleInit {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t i2 = 0;
};

class CudaBackend {
public:
    virtual ~CudaBackend() = default;

    virtual KhsResult initialize(
        const KhsSolverDesc &desc,
        const KhsHairMaterial &material,
        const std::vector<CudaNodeInit> &nodes,
        const std::vector<CudaElementInit> &elements,
        const std::vector<std::vector<uint32_t>> &strand_nodes,
        const std::vector<uint32_t> &original_to_internal,
        const std::vector<KhsVec3> &collider_vertices,
        const std::vector<CudaTriangleInit> &collider_triangles) = 0;

    virtual KhsResult update_runtime_parameters(
        const KhsSolverDesc &desc,
        const KhsHairMaterial &material,
        const std::vector<double> &node_masses,
        const std::vector<double> &node_rotational_masses) = 0;

    virtual KhsResult update_roots(const std::vector<KhsVec3> &positions,
                                   const std::vector<KhsVec3> &rotations) = 0;
    virtual KhsResult update_collider(const KhsVec3 *vertices, uint32_t count) = 0;
    virtual KhsResult allocate_animation(uint32_t frame_count) = 0;
    virtual KhsResult set_root_animation_frame(uint32_t frame,
                                               const std::vector<KhsVec3> &positions,
                                               const std::vector<KhsVec3> &rotations) = 0;
    virtual KhsResult set_collider_animation_frame(uint32_t frame,
                                                   const KhsVec3 *vertices,
                                                   uint32_t count) = 0;
    virtual KhsResult finalize_animation() = 0;
    virtual KhsResult step(double frame_dt) = 0;
    virtual KhsResult step_animation_frame(uint32_t frame, double frame_dt) = 0;
    virtual uint64_t animation_checkpoint_size() const = 0;
    virtual KhsResult save_animation_checkpoint(void *data, uint64_t capacity) = 0;
    virtual KhsResult restore_animation_checkpoint(const void *data, uint64_t size) = 0;
    virtual KhsResult copy_original_positions(KhsVec3 *positions, uint32_t capacity) const = 0;
    virtual KhsResult copy_internal_positions(KhsVec3 *positions, uint32_t capacity) const = 0;
    virtual KhsStepStats stats() const = 0;
    virtual KhsGpuStats gpu_stats() const = 0;
    virtual KhsProgress progress() const = 0;
    virtual KhsFailureDiagnostics failure_diagnostics() const = 0;
    virtual void request_cancel() = 0;
    virtual std::string last_error() const = 0;
};

std::unique_ptr<CudaBackend> make_cuda_backend();
bool query_cuda_device(KhsGpuInfo &info, std::string &error);

} // namespace kami
