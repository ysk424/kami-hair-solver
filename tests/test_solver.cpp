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
    desc.substeps = 2;
    desc.newton_iterations = 32;
    material.density = 1400.0;
    material.radius = 5.0e-5;
    material.young_modulus = 2.0e9;
    require(solver, khsUpdateRuntimeParameters(solver, &desc, &material));
    KhsSolverDesc structural_change = desc;
    structural_change.maximum_element_length = 0.007;
    assert(khsUpdateRuntimeParameters(solver, &structural_change, &material) ==
           KHS_ERROR_INVALID_ARGUMENT);
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

void test_minimum_dynamic_length_preserves_visible_prefix()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 0.06;
    desc.minimum_dynamic_length = 0.30;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.mass_damping = 0.0;
    const KhsVec3 points[]{{0.0, 0.0, 0.0}, {0.05, 0.0, 0.0}, {0.10, 0.0, 0.0}};
    const uint32_t offsets[]{0, 3};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsBuild(solver));

    KhsBuildStats build{};
    build.struct_size = sizeof(build);
    require(solver, khsGetBuildStats(solver, &build));
    assert(build.original_point_count == 3);
    assert(build.internal_node_count == 7);
    assert(build.element_count == 6);
    assert(build.virtual_extension_strand_count == 1);
    assert(build.virtual_extension_node_count == 4);
    assert(std::abs(build.virtual_extension_rest_length - 0.20) < 1.0e-12);

    std::vector<KhsVec3> internal(build.internal_node_count);
    require(solver, khsCopyInternalPositions(
        solver, internal.data(), static_cast<uint32_t>(internal.size())));
    assert(std::abs(internal.back().x - 0.30) < 1.0e-12);
    assert(std::abs(internal.back().y) < 1.0e-12);
    assert(std::abs(internal.back().z) < 1.0e-12);

    require(solver, khsStep(solver, 1.0 / 24.0));
    KhsVec3 output[3];
    require(solver, khsCopyOriginalPositions(solver, output, 3));
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(output[i].x - points[i].x) < 1.0e-10);
        assert(std::abs(output[i].y - points[i].y) < 1.0e-10);
        assert(std::abs(output[i].z - points[i].z) < 1.0e-10);
    }
    khsDestroy(solver);
}

void test_virtual_extension_ignores_collider_contact()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 0.05;
    desc.minimum_dynamic_length = 0.30;
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.mass_damping = 0.0;
    const KhsVec3 points[]{{0.0, 0.0, 0.0}, {0.05, 0.0, 0.0}};
    const uint32_t offsets[]{0, 2};
    const KhsTriangle triangle[]{0, 1, 2};

    // The static triangle intersects only the invisible extension. Building
    // must still succeed, while visible-hair intersection tests remain active.
    {
        KhsSolver *solver = khsCreate(&desc);
        assert(solver);
        const KhsVec3 collider[]{{0.10, -1.0, 0.0}, {0.40, -1.0, 0.0},
                                 {0.25, 1.0, 0.0}};
        require(solver, khsSetHairCurves(solver, points, 2, offsets, 1, &material));
        require(solver, khsSetColliderMesh(solver, collider, 3, triangle, 1));
        require(solver, khsBuild(solver));
        KhsBuildStats build{};
        build.struct_size = sizeof(build);
        require(solver, khsGetBuildStats(solver, &build));
        assert(build.virtual_extension_strand_count == 1);
        assert(std::isinf(build.initial_minimum_gap));
        require(solver, khsStep(solver, 1.0 / 24.0));
        KhsStepStats step{};
        step.struct_size = sizeof(step);
        require(solver, khsGetLastStepStats(solver, &step));
        assert(step.contact_candidate_count == 0);
        assert(step.active_contact_count == 0);
        khsDestroy(solver);
    }

    // A moving triangle sweeps through only the invisible extension. It must
    // not trigger animated-collider crossing or generate contact candidates.
    {
        KhsSolver *solver = khsCreate(&desc);
        assert(solver);
        const KhsVec3 below[]{{0.10, -1.0, -0.1}, {0.40, -1.0, -0.1},
                              {0.25, 1.0, -0.1}};
        const KhsVec3 above[]{{0.10, -1.0, 0.1}, {0.40, -1.0, 0.1},
                              {0.25, 1.0, 0.1}};
        require(solver, khsSetHairCurves(solver, points, 2, offsets, 1, &material));
        require(solver, khsSetColliderMesh(solver, below, 3, triangle, 1));
        require(solver, khsBuild(solver));
        require(solver, khsUpdateColliderVertices(solver, above, 3));
        require(solver, khsStep(solver, 1.0 / 24.0));
        KhsStepStats step{};
        step.struct_size = sizeof(step);
        require(solver, khsGetLastStepStats(solver, &step));
        assert(step.contact_candidate_count == 0);
        assert(step.active_contact_count == 0);
        khsDestroy(solver);
    }
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
    material.collider_offset = 0.0;
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
    material.collider_offset = 0.0;
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
    desc.maximum_substeps = 4;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.collider_offset = 0.0;
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
    KhsFailureDiagnostics diagnostics{};
    diagnostics.struct_size = sizeof(diagnostics);
    require(solver, khsGetFailureDiagnostics(solver, &diagnostics));
    assert(diagnostics.kind == KHS_FAILURE_MOVING_COLLIDER_SWEEP);
    assert(diagnostics.requested_substeps == 1);
    assert(diagnostics.attempted_substeps == 4);
    assert(diagnostics.maximum_substeps == 4);
    assert(diagnostics.adaptive_attempt_count == 3);
    assert(diagnostics.strand_index == 0);
    assert(diagnostics.collider_triangle_index == 0);
    assert(diagnostics.distance <= diagnostics.required_distance + 1.0e-12);
    assert(std::abs(diagnostics.collider_frame_displacement - 0.2) < 1.0e-12);
    khsDestroy(solver);
}

void test_moving_collider_uses_variable_toi_steps()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.maximum_substeps = 512;
    desc.newton_iterations = 64;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.collider_offset = 0.0;
    material.radius = 1.0e-3;
    material.young_modulus = 1.0e6;
    material.contact_stiffness = 100.0;
    material.barrier_distance = 2.0e-3;
    material.mass_damping = 2.0;
    const KhsVec3 points[]{{-0.2, 0.0, 0.02}, {-0.1, 0.0, 0.0}, {0.1, 0.0, 0.02}};
    const uint32_t offsets[]{0, 3};
    const KhsVec3 below[]{{-1.0, -1.0, -0.1}, {1.0, -1.0, -0.1}, {0.0, 1.0, -0.1}};
    const KhsVec3 above[]{{-1.0, -1.0, 0.1}, {1.0, -1.0, 0.1}, {0.0, 1.0, 0.1}};
    const KhsTriangle triangle[]{0, 1, 2};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, below, 3, triangle, 1));
    require(solver, khsBuild(solver));
    require(solver, khsUpdateColliderVertices(solver, above, 3));
    const KhsResult step_result = khsStep(solver, 1.0 / 24.0);
    if (step_result != KHS_OK) {
        KhsFailureDiagnostics diagnostics{};
        diagnostics.struct_size = sizeof(diagnostics);
        require(solver, khsGetFailureDiagnostics(solver, &diagnostics));
        KhsStepStats failed_stats{};
        failed_stats.struct_size = sizeof(failed_stats);
        require(solver, khsGetLastStepStats(solver, &failed_stats));
        std::fprintf(stderr,
                     "variable TOI failure: %s kind=%u sub=%u attempted=%u limited=%u "
                     "distance=%.12g required=%.12g clearance=%.12g movement=%.12g "
                     "accepted=%u min_gap=%.12g residual=%.12g increment=%.12g alpha=%.12g\n",
                     khsGetLastError(solver), diagnostics.kind, diagnostics.substep,
                     diagnostics.attempted_substeps, diagnostics.adaptive_attempt_count,
                     diagnostics.distance, diagnostics.required_distance,
                     diagnostics.clearance, diagnostics.collider_substep_displacement,
                     failed_stats.substeps, failed_stats.minimum_gap,
                     failed_stats.final_residual_norm, failed_stats.increment_norm,
                     failed_stats.accepted_step_length);
    }
    require(solver, step_result);
    KhsStepStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetLastStepStats(solver, &stats));
    assert(stats.phase == KHS_PHASE_FINISHED);
    assert(stats.substeps > desc.substeps);
    assert(stats.substeps <= desc.maximum_substeps);
    assert(stats.converged_substeps == stats.substeps);
    assert(stats.ccd_step_limit < 1.0);
    KhsVec3 output[3];
    require(solver, khsCopyOriginalPositions(solver, output, 3));
    assert(output[2].z > material.radius);
    khsDestroy(solver);
}

void test_moving_collider_preserves_small_existing_clearance()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 1;
    desc.maximum_substeps = 64;
    desc.newton_iterations = 64;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.collider_offset = 0.0;
    material.radius = 1.0e-3;
    material.young_modulus = 1.0e6;
    material.contact_stiffness = 100.0;
    material.barrier_distance = 2.0e-3;
    material.mass_damping = 2.0;
    const KhsVec3 points[]{{-0.2, 0.0, 0.02}, {-0.1, 0.0, 0.0}, {0.1, 0.0, 0.02}};
    const uint32_t offsets[]{0, 3};
    const KhsVec3 start[]{{-1.0, -1.0, -0.0012}, {1.0, -1.0, -0.0012}, {0.0, 1.0, -0.0012}};
    const KhsVec3 end[]{{-1.0, -1.0, -0.0008}, {1.0, -1.0, -0.0008}, {0.0, 1.0, -0.0008}};
    const KhsTriangle triangle[]{0, 1, 2};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, start, 3, triangle, 1));
    require(solver, khsBuild(solver));
    require(solver, khsUpdateColliderVertices(solver, end, 3));
    require(solver, khsStep(solver, 1.0 / 24.0));
    KhsStepStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetLastStepStats(solver, &stats));
    assert(stats.phase == KHS_PHASE_FINISHED);
    assert(stats.substeps > desc.substeps);
    assert(stats.substeps <= desc.maximum_substeps);
    assert(stats.minimum_gap > desc.minimum_gap);
    khsDestroy(solver);
}

void test_moving_collider_transfers_normal_velocity_to_hair()
{
    KhsSolverDesc desc;
    khsDefaultSolverDesc(&desc);
    desc.gravity = {0.0, 0.0, 0.0};
    desc.substeps = 8;
    desc.maximum_substeps = 32;
    desc.fixed_root_nodes = 1;
    desc.maximum_element_length = 1.0;
    KhsSolver *solver = khsCreate(&desc);
    assert(solver);
    KhsHairMaterial material;
    khsDefaultHairMaterial(&material);
    material.collider_offset = 0.0;
    material.radius = 4.0e-5;
    material.young_modulus = 4.0e9;
    material.contact_stiffness = 1.0e4;
    material.barrier_distance = 2.0e-4;
    const KhsVec3 points[]{{-0.02, 0.0, 0.02}, {-0.01, 0.0, 0.0}, {0.01, 0.0, 0.02}};
    const uint32_t offsets[]{0, 3};
    KhsVec3 collider[]{{-1.0, -1.0, -9.0e-5}, {1.0, -1.0, -9.0e-5}, {0.0, 1.0, -9.0e-5}};
    const KhsTriangle triangle[]{0, 1, 2};
    require(solver, khsSetHairCurves(solver, points, 3, offsets, 1, &material));
    require(solver, khsSetColliderMesh(solver, collider, 3, triangle, 1));
    require(solver, khsBuild(solver));
    for (int frame = 0; frame < 3; ++frame) {
        for (auto &vertex : collider) vertex.z += 4.4e-4;
        require(solver, khsUpdateColliderVertices(solver, collider, 3));
        const KhsResult result = khsStep(solver, 1.0 / 24.0);
        if (result != KHS_OK) {
            KhsFailureDiagnostics diagnostics{};
            diagnostics.struct_size = sizeof(diagnostics);
            require(solver, khsGetFailureDiagnostics(solver, &diagnostics));
            KhsStepStats failed_stats{};
            failed_stats.struct_size = sizeof(failed_stats);
            require(solver, khsGetLastStepStats(solver, &failed_stats));
            std::fprintf(stderr,
                         "normal transport frame=%d: %s kind=%u attempted=%u limited=%u "
                         "accepted=%u gap=%.12g residual=%.12g alpha=%.12g\n",
                         frame, khsGetLastError(solver), diagnostics.kind,
                         diagnostics.attempted_substeps,
                         diagnostics.adaptive_attempt_count, failed_stats.substeps,
                         failed_stats.minimum_gap, failed_stats.final_residual_norm,
                         failed_stats.accepted_step_length);
        }
        require(solver, result);
    }
    KhsStepStats stats{};
    stats.struct_size = sizeof(stats);
    require(solver, khsGetLastStepStats(solver, &stats));
    assert(stats.phase == KHS_PHASE_FINISHED);
    assert(stats.minimum_gap > desc.minimum_gap);
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
    const uint64_t checkpoint_size = khsGetAnimationCheckpointSize(solver);
    assert(checkpoint_size > 0);
    std::vector<unsigned char> checkpoint(checkpoint_size);
    require(solver, khsSaveAnimationCheckpoint(
        solver, checkpoint.data(), checkpoint.size()));
    require(solver, khsStepAnimationFrame(solver, 1, 1.0 / 24.0));
    std::vector<KhsVec3> first_result(3);
    require(solver, khsCopyOriginalPositions(solver, first_result.data(), 3));
    require(solver, khsRestoreAnimationCheckpoint(
        solver, checkpoint.data(), checkpoint.size()));
    require(solver, khsStepAnimationFrame(solver, 1, 1.0 / 24.0));
    std::vector<KhsVec3> repeated_result(3);
    require(solver, khsCopyOriginalPositions(solver, repeated_result.data(), 3));
    for (size_t i = 0; i < first_result.size(); ++i) {
        assert(std::abs(first_result[i].x - repeated_result[i].x) < 1.0e-14);
        assert(std::abs(first_result[i].y - repeated_result[i].y) < 1.0e-14);
        assert(std::abs(first_result[i].z - repeated_result[i].z) < 1.0e-14);
    }
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
    test_minimum_dynamic_length_preserves_visible_prefix();
    test_virtual_extension_ignores_collider_contact();
    test_duplicate_mapping();
    test_initial_intersection_rejected();
    test_barrier_contact_remains_feasible();
    test_moving_collider_tunnelling_rejected();
    test_moving_collider_uses_variable_toi_steps();
    test_moving_collider_preserves_small_existing_clearance();
    test_moving_collider_transfers_normal_velocity_to_hair();
    test_gpu_resident_animation();
    std::puts("Kami Hair Solver tests passed");
    return 0;
}
