#include "kami_hair_solver.h"
#include "solver.hpp"

#include <cmath>
#include <cstring>
#include <exception>
#include <new>

struct KhsSolver {
    explicit KhsSolver(const KhsSolverDesc &desc) : implementation(desc) {}
    kami::Solver implementation;
};

namespace {

template<typename Function>
KhsResult guarded(KhsSolver *solver, Function &&function)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    try {
        return function();
    }
    catch (const std::bad_alloc &) {
        solver->implementation.set_error("髪ソルバーのメモリを確保できません。 ");
        return KHS_ERROR_OUT_OF_MEMORY;
    }
    catch (const std::exception &exception) {
        solver->implementation.set_error(std::string("髪ソルバー内部例外: ") + exception.what());
        return KHS_ERROR_INTERNAL;
    }
    catch (...) {
        solver->implementation.set_error("髪ソルバーで不明な内部例外が発生しました。");
        return KHS_ERROR_INTERNAL;
    }
}

bool valid_desc(const KhsSolverDesc *desc)
{
    return desc && desc->struct_size >= sizeof(KhsSolverDesc) && desc->substeps > 0 &&
        desc->newton_iterations > 0 && desc->line_search_iterations > 0 &&
        desc->absolute_tolerance > 0.0 && desc->relative_tolerance > 0.0 &&
        desc->increment_tolerance > 0.0 && desc->minimum_line_search_step > 0.0 &&
        desc->minimum_gap > 0.0 && desc->maximum_element_length > 0.0 &&
        desc->minimum_dynamic_length >= 0.0 && std::isfinite(desc->minimum_dynamic_length) &&
        desc->fixed_root_nodes > 0;
}

} // namespace

extern "C" {

uint32_t khsGetAbiVersion(void) { return KHS_ABI_VERSION; }

KhsResult khsGetGpuInfo(KhsGpuInfo *info)
{
    if (!info || info->struct_size < sizeof(KhsGpuInfo)) return KHS_ERROR_INVALID_ARGUMENT;
    std::string error;
    KhsGpuInfo result{};
    if (!kami::query_cuda_device(result, error)) {
        *info = result;
        return KHS_ERROR_INVALID_STATE;
    }
    *info = result;
    return KHS_OK;
}

void khsDefaultSolverDesc(KhsSolverDesc *desc)
{
    if (!desc) return;
    std::memset(desc, 0, sizeof(*desc));
    desc->struct_size = sizeof(*desc);
    desc->gravity = {0.0, 0.0, -9.81};
    desc->substeps = 8;
    desc->newton_iterations = 24;
    desc->line_search_iterations = 20;
    desc->absolute_tolerance = 1.0e-8;
    desc->relative_tolerance = 1.0e-5;
    desc->increment_tolerance = 1.0e-8;
    desc->minimum_line_search_step = 1.0e-9;
    desc->minimum_gap = 1.0e-7;
    desc->maximum_element_length = 0.01;
    desc->minimum_dynamic_length = 0.0;
    desc->fixed_root_nodes = 2;
    desc->thread_count = 0;
}

void khsDefaultHairMaterial(KhsHairMaterial *material)
{
    if (!material) return;
    std::memset(material, 0, sizeof(*material));
    material->struct_size = sizeof(*material);
    material->density = 1300.0;
    material->radius = 4.0e-5;
    material->young_modulus = 4.0e9;
    material->poisson_ratio = 0.38;
    material->shear_correction = 0.9;
    material->mass_damping = 8.0;
    material->contact_stiffness = 1.0e4;
    material->barrier_distance = 2.0e-4;
    material->friction = 0.35;
    material->friction_smoothing = 1.0e-6;
    material->collider_offset = 0.0;
}

KhsSolver *khsCreate(const KhsSolverDesc *desc)
{
    if (!valid_desc(desc)) return nullptr;
    try { return new KhsSolver(*desc); }
    catch (...) { return nullptr; }
}

void khsDestroy(KhsSolver *solver) { delete solver; }

KhsResult khsSetHairCurves(KhsSolver *solver, const KhsVec3 *points,
                           uint32_t point_count, const uint32_t *offsets,
                           uint32_t strand_count, const KhsHairMaterial *material)
{
    if (!material || material->struct_size < sizeof(KhsHairMaterial)) return KHS_ERROR_INVALID_ARGUMENT;
    return guarded(solver, [&] {
        return solver->implementation.set_hair(points, point_count, offsets, strand_count, *material);
    });
}

KhsResult khsSetColliderMesh(KhsSolver *solver, const KhsVec3 *vertices,
                             uint32_t vertex_count, const KhsTriangle *triangles,
                             uint32_t triangle_count)
{
    return guarded(solver, [&] {
        return solver->implementation.set_collider(vertices, vertex_count, triangles, triangle_count);
    });
}

KhsResult khsBuild(KhsSolver *solver)
{
    return guarded(solver, [&] { return solver->implementation.build(); });
}

KhsResult khsUpdateColliderVertices(KhsSolver *solver, const KhsVec3 *vertices,
                                    uint32_t vertex_count)
{
    return guarded(solver, [&] {
        return solver->implementation.update_collider(vertices, vertex_count);
    });
}

KhsResult khsUpdateRootTargets(KhsSolver *solver, const KhsVec3 *points,
                               uint32_t point_count)
{
    return guarded(solver, [&] {
        return solver->implementation.update_roots(points, point_count);
    });
}

KhsResult khsStep(KhsSolver *solver, double frame_dt)
{
    return guarded(solver, [&] { return solver->implementation.step(frame_dt); });
}

KhsResult khsAllocateAnimation(KhsSolver *solver, uint32_t frame_count)
{
    return guarded(solver, [&] { return solver->implementation.allocate_animation(frame_count); });
}

KhsResult khsSetRootAnimationFrame(KhsSolver *solver, uint32_t frame_index,
                                   const KhsVec3 *points, uint32_t point_count)
{
    return guarded(solver, [&] {
        return solver->implementation.set_root_animation_frame(frame_index, points, point_count);
    });
}

KhsResult khsSetColliderAnimationFrame(KhsSolver *solver, uint32_t frame_index,
                                       const KhsVec3 *vertices, uint32_t vertex_count)
{
    return guarded(solver, [&] {
        return solver->implementation.set_collider_animation_frame(frame_index, vertices, vertex_count);
    });
}

KhsResult khsFinalizeAnimation(KhsSolver *solver)
{
    return guarded(solver, [&] { return solver->implementation.finalize_animation(); });
}

KhsResult khsStepAnimationFrame(KhsSolver *solver, uint32_t frame_index, double frame_dt)
{
    return guarded(solver, [&] {
        return solver->implementation.step_animation_frame(frame_index, frame_dt);
    });
}

uint32_t khsGetOriginalPointCount(const KhsSolver *solver)
{
    return solver ? solver->implementation.original_point_count() : 0;
}

uint32_t khsGetInternalNodeCount(const KhsSolver *solver)
{
    return solver ? solver->implementation.internal_node_count() : 0;
}

KhsResult khsCopyOriginalPositions(const KhsSolver *solver, KhsVec3 *positions,
                                   uint32_t capacity)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.copy_original_positions(positions, capacity);
}

KhsResult khsCopyInternalPositions(const KhsSolver *solver, KhsVec3 *positions,
                                   uint32_t capacity)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.copy_internal_positions(positions, capacity);
}

KhsResult khsCopyOriginalToInternalMap(const KhsSolver *solver, uint32_t *indices,
                                       uint32_t capacity)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.copy_mapping(indices, capacity);
}

KhsResult khsGetBuildStats(const KhsSolver *solver, KhsBuildStats *stats)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.build_stats(stats);
}

KhsResult khsGetLastStepStats(const KhsSolver *solver, KhsStepStats *stats)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.step_stats(stats);
}

KhsResult khsGetGpuStats(const KhsSolver *solver, KhsGpuStats *stats)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.gpu_stats(stats);
}

KhsResult khsGetProgress(const KhsSolver *solver, KhsProgress *progress)
{
    if (!solver) return KHS_ERROR_INVALID_ARGUMENT;
    return solver->implementation.progress(progress);
}

KhsResult khsRequestCancel(KhsSolver *solver)
{
    return guarded(solver, [&] { return solver->implementation.request_cancel(); });
}

const char *khsGetLastError(const KhsSolver *solver)
{
    static const char *invalid = "髪ソルバーがありません。";
    return solver ? solver->implementation.last_error().c_str() : invalid;
}

} // extern "C"
