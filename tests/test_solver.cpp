#include "kami_hair_solver.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

void require(KhsSolver *solver, KhsResult result)
{
    if (result != KHS_OK) {
        std::fprintf(stderr, "solver error: %s\n", khsGetLastError(solver));
        std::abort();
    }
}

void test_rest_state_and_subdivision()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 0.006;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.mass_damping = 0.0;
    const KhsVec3 points[]{{0.0, 0.0, 0.0}, {0.0, 0.0, -0.01}, {0.0, 0.0, -0.02}};
    const uint32_t offsets[]{0, 3};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsBuild(solver));
    assert(khsGetInternalNodeCount(solver) == 5);
    require(solver, khsStep(solver, 1.0 / 24.0));
    KhsVec3 output[3];
    require(solver, khsCopyOriginalPositions(solver, output, 3));
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(output[i].x - points[i].x) < 1.0e-10);
        assert(std::abs(output[i].y - points[i].y) < 1.0e-10);
        assert(std::abs(output[i].z - points[i].z) < 1.0e-10);
    }
    KhsStepStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetLastStepStats(solver, &stats));
    assert(stats.phase == KHS_PHASE_FINISHED);
    assert(stats.final_residual_norm < 1.0e-5);
    khsDestroy(solver);
}

void test_duplicate_mapping()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.maximum_element_length = 1.0;
    desc.fixed_root_nodes = 1;
    KhsSolver *solver = khsCreate(&desc);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    const KhsVec3 points[]{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.1}};
    const uint32_t offsets[]{0, 3};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsBuild(solver));
    uint32_t mapping[3];
    require(solver, khsCopyOriginalToInternalMap(solver, mapping, 3));
    assert(mapping[0] == mapping[1]);
    assert(mapping[2] != mapping[1]);
    KhsBuildStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetBuildStats(solver, &stats));
    assert(stats.merged_zero_length_segment_count == 1);
    khsDestroy(solver);
}

void test_initial_intersection_rejected()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.maximum_element_length = 1.0;
    desc.fixed_root_nodes = 1;
    KhsSolver *solver = khsCreate(&desc);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.radius = 1.0e-3;
    material.barrier_distance = 2.0e-3;
    const KhsVec3 points[]{{-0.2, 0.0, 0.2}, {-0.1, 0.0, 0.1}, {0.1, 0.0, -0.1}};
    const uint32_t offsets[]{0, 3};
    const KhsVec3 collider[]{{-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {0.0, 1.0, 0.0}};
    const KhsTriangle triangle[]{0, 1, 2};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, collider, 3, triangle, 1));
    assert(khsBuild(solver) == KHS_ERROR_INITIAL_INTERSECTION);
    khsDestroy(solver);
}

void test_barrier_contact_remains_feasible()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.substeps = 4;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 0.02;
    desc.newton_iterations = 40;
    KhsSolver *solver = khsCreate(&desc);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.radius = 2.0e-4;
    material.young_modulus = 1.0e6;
    material.contact_stiffness = 100.0;
    material.barrier_distance = 2.0e-3;
    material.mass_damping = 2.0;
    const KhsVec3 points[]{{0.0, 0.0, 0.03}, {0.0, 0.0, 0.02},
                           {0.0, 0.0, 0.01}, {0.0, 0.0, 0.001}};
    const uint32_t offsets[]{0, 4};
    const KhsVec3 collider[]{{-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
                             {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}};
    const KhsTriangle triangles[]{{0, 1, 2}, {0, 2, 3}};
    require(solver, khsSetHairCurves(solver, points, 4, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, collider, 4, triangles, 2));
    require(solver, khsBuild(solver));
    for (int frame = 0; frame < 3; ++frame) require(solver, khsStep(solver, 1.0 / 60.0));
    KhsStepStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetLastStepStats(solver, &stats));
    assert(stats.active_contact_count > 0);
    assert(stats.minimum_gap > desc.minimum_gap);
    assert(stats.phase == KHS_PHASE_FINISHED);
    khsDestroy(solver);
}

void test_moving_collider_tunnelling_rejected()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.substeps = 1;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.radius = 1.0e-3;
    const KhsVec3 points[]{{-0.2, 0.0, 0.0}, {-0.1, 0.0, 0.0}, {0.1, 0.0, 0.0}};
    const uint32_t offsets[]{0, 3};
    const KhsVec3 below[]{{-1.0, -1.0, -0.1}, {1.0, -1.0, -0.1}, {0.0, 1.0, -0.1}};
    const KhsVec3 above[]{{-1.0, -1.0, 0.1}, {1.0, -1.0, 0.1}, {0.0, 1.0, 0.1}};
    const KhsTriangle triangle[]{0, 1, 2};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, below, 3, triangle, 1));
    require(solver, khsBuild(solver));
    require(solver, khsUpdateColliderVertices(solver, above, 3));
    assert(khsStep(solver, 1.0 / 24.0) == KHS_ERROR_NOT_CONVERGED);
    khsDestroy(solver);
}

void test_gpu_resident_animation()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.newton_iterations = 100;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    const KhsVec3 first[]{{0.0, 0.0, 0.0}, {0.0, 0.0, -0.1}, {0.0, 0.0, -0.2}};
    const KhsVec3 second[]{{0.001, 0.0, 0.0}, {0.001, 0.0, -0.1}, {0.001, 0.0, -0.2}};
    const uint32_t offsets[]{0, 3};
    require(solver, khsSetHairCurves(solver, first, 3, offsets, 1, &material));
    require(solver, khsBuild(solver));
    require(solver, khsAllocateAnimation(solver, 2));
    require(solver, khsSetRootAnimationFrame(solver, 0, first, 3));
    require(solver, khsSetRootAnimationFrame(solver, 1, second, 3));
    require(solver, khsSetColliderAnimationFrame(solver, 0, nullptr, 0));
    require(solver, khsSetColliderAnimationFrame(solver, 1, nullptr, 0));
    require(solver, khsFinalizeAnimation(solver));
    require(solver, khsStepAnimationFrame(solver, 1, 1.0 / 24.0));
    KhsGpuStats gpu{};
    gpu.struct_size = sizeof(gpu);
    require(solver, khsGetGpuStats(solver, &gpu));
    assert(gpu.animation_frame_count == 2);
    assert(gpu.resident_bytes > 0);
    assert(gpu.last_frame_milliseconds > 0.0);
    KhsProgress progress{};
    progress.struct_size = sizeof(progress);
    require(solver, khsGetProgress(solver, &progress));
    assert(progress.phase == KHS_PHASE_FINISHED);
    khsDestroy(solver);
}

} // namespace

int main()
{
    assert(khsGetAbiVersion() == KHS_ABI_VERSION);
    test_rest_state_and_subdivision();
    test_duplicate_mapping();
    test_initial_intersection_rejected();
    test_barrier_contact_remains_feasible();
    test_moving_collider_tunnelling_rejected();
    test_gpu_resident_animation();
    std::puts("Kami Hair Solver tests passed");
    return 0;
}
