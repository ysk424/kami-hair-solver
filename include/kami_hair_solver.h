#ifndef KAMI_HAIR_SOLVER_H
#define KAMI_HAIR_SOLVER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(KHS_BUILD_DLL)
#    define KHS_API __declspec(dllexport)
#  else
#    define KHS_API __declspec(dllimport)
#  endif
#else
#  define KHS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KHS_ABI_VERSION 5u

typedef struct KhsSolver KhsSolver;

typedef struct KhsVec3 {
    double x;
    double y;
    double z;
} KhsVec3;

typedef struct KhsTriangle {
    uint32_t i0;
    uint32_t i1;
    uint32_t i2;
} KhsTriangle;

typedef enum KhsResult {
    KHS_OK = 0,
    KHS_ERROR_INVALID_ARGUMENT = 1,
    KHS_ERROR_INVALID_STATE = 2,
    KHS_ERROR_INVALID_HAIR = 3,
    KHS_ERROR_INVALID_COLLIDER = 4,
    KHS_ERROR_INITIAL_INTERSECTION = 5,
    KHS_ERROR_NOT_CONVERGED = 6,
    KHS_ERROR_NUMERICAL_FAILURE = 7,
    KHS_ERROR_OUT_OF_MEMORY = 8,
    KHS_ERROR_INTERNAL = 9
} KhsResult;

typedef enum KhsProgressPhase {
    KHS_PHASE_IDLE = 0,
    KHS_PHASE_PREPARING = 1,
    KHS_PHASE_ASSEMBLING = 2,
    KHS_PHASE_LINEAR_SOLVE = 3,
    KHS_PHASE_CCD = 4,
    KHS_PHASE_LINE_SEARCH = 5,
    KHS_PHASE_FINISHED = 6,
    KHS_PHASE_FAILED = 7
} KhsProgressPhase;

typedef struct KhsSolverDesc {
    uint32_t struct_size;
    KhsVec3 gravity;
    /* 1フレームを分ける基本区間数。 */
    uint32_t substeps;
    /* TOI制限と局所再試行を含む、1フレームの可変時間区間上限。 */
    uint32_t maximum_substeps;
    uint32_t newton_iterations;
    uint32_t line_search_iterations;
    double absolute_tolerance;
    double relative_tolerance;
    double increment_tolerance;
    double minimum_line_search_step;
    double minimum_gap;
    double maximum_element_length;
    /* 0 で無効。短いストランドは毛先接線方向へこの自然長まで物理的に延長する。 */
    double minimum_dynamic_length;
    uint32_t fixed_root_nodes;
    uint32_t thread_count;
} KhsSolverDesc;

typedef struct KhsHairMaterial {
    uint32_t struct_size;
    double density;
    double radius;
    double young_modulus;
    double poisson_ratio;
    double shear_correction;
    double mass_damping;
    double contact_stiffness;
    double barrier_distance;
    double friction;
    double friction_smoothing;
    double collider_offset;
} KhsHairMaterial;

typedef struct KhsBuildStats {
    uint32_t struct_size;
    uint32_t strand_count;
    uint32_t original_point_count;
    uint32_t internal_node_count;
    uint32_t element_count;
    uint32_t fixed_node_count;
    uint32_t collider_vertex_count;
    uint32_t collider_triangle_count;
    uint32_t excluded_collider_triangle_count;
    uint32_t collider_boundary_edge_count;
    uint32_t collider_nonmanifold_edge_count;
    uint32_t collider_inconsistent_edge_count;
    uint32_t collider_inverted_closed_component_count;
    uint32_t merged_zero_length_segment_count;
    uint32_t virtual_extension_strand_count;
    uint32_t virtual_extension_node_count;
    double virtual_extension_rest_length;
    uint64_t degree_of_freedom_count;
    uint64_t estimated_bytes;
    double initial_minimum_gap;
} KhsBuildStats;

typedef struct KhsStepStats {
    uint32_t struct_size;
    uint32_t substeps;
    uint32_t converged_substeps;
    uint32_t newton_iterations;
    uint32_t linear_solves;
    uint32_t line_search_evaluations;
    uint64_t contact_candidate_count;
    uint64_t active_contact_count;
    double initial_residual_norm;
    double final_residual_norm;
    double relative_residual_norm;
    double increment_norm;
    double objective_change;
    double minimum_gap;
    double accepted_step_length;
    double ccd_step_limit;
    double kinetic_energy;
    double elastic_energy;
    double contact_energy;
    double friction_energy;
    KhsProgressPhase phase;
} KhsStepStats;

typedef struct KhsGpuInfo {
    uint32_t struct_size;
    uint32_t available;
    int32_t device_ordinal;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
    uint64_t total_vram_bytes;
    char device_name[128];
} KhsGpuInfo;

typedef struct KhsGpuStats {
    uint32_t struct_size;
    uint32_t animation_frame_count;
    uint64_t resident_bytes;
    uint64_t peak_temporary_bytes;
    double last_frame_milliseconds;
    double last_assembly_milliseconds;
    double last_collision_milliseconds;
    double last_optimization_milliseconds;
} KhsGpuStats;

typedef struct KhsProgress {
    uint32_t struct_size;
    uint32_t phase;
    uint32_t frame_index;
    uint32_t frame_count;
    uint32_t substep;
    uint32_t substep_count;
    uint32_t nonlinear_iteration;
    uint32_t nonlinear_iteration_limit;
    uint32_t cancelled;
    double frame_elapsed_seconds;
} KhsProgress;

typedef enum KhsFailureKind {
    KHS_FAILURE_NONE = 0,
    KHS_FAILURE_MOVING_COLLIDER_SWEEP = 1,
    KHS_FAILURE_BARRIER_INFEASIBLE = 2,
    KHS_FAILURE_NONLINEAR_SOLVE = 3
} KhsFailureKind;

typedef struct KhsFailureDiagnostics {
    uint32_t struct_size;
    uint32_t kind;
    uint32_t frame_index;
    uint32_t substep;
    uint32_t requested_substeps;
    /* 失敗までに試した可変時間区間の総数。 */
    uint32_t attempted_substeps;
    uint32_t maximum_substeps;
    /* TOIによって提案区間が制限された回数。ABI互換のため名称を維持する。 */
    uint32_t adaptive_attempt_count;
    uint32_t strand_index;
    uint32_t element_index;
    uint32_t collider_triangle_index;
    uint32_t reserved;
    double distance;
    double required_distance;
    double clearance;
    double collider_substep_displacement;
    double collider_frame_displacement;
    KhsVec3 hair_start;
    KhsVec3 hair_end;
    KhsVec3 collider_point;
} KhsFailureDiagnostics;

KHS_API uint32_t khsGetAbiVersion(void);
KHS_API KhsResult khsGetGpuInfo(KhsGpuInfo *info);
KHS_API void khsDefaultSolverDesc(KhsSolverDesc *desc);
KHS_API void khsDefaultHairMaterial(KhsHairMaterial *material);
KHS_API KhsSolver *khsCreate(const KhsSolverDesc *desc);
KHS_API void khsDestroy(KhsSolver *solver);

/* offsets は strand_count + 1 個。offsets[0] == 0、末尾は point_count。 */
KHS_API KhsResult khsSetHairCurves(KhsSolver *solver,
                                   const KhsVec3 *points,
                                   uint32_t point_count,
                                   const uint32_t *offsets,
                                   uint32_t strand_count,
                                   const KhsHairMaterial *material);

/* 三角形 0 個でコライダーを解除する。構築時に縮退面を除外する。 */
KHS_API KhsResult khsSetColliderMesh(KhsSolver *solver,
                                     const KhsVec3 *vertices,
                                     uint32_t vertex_count,
                                     const KhsTriangle *triangles,
                                     uint32_t triangle_count);

KHS_API KhsResult khsBuild(KhsSolver *solver);

/*
 * 構築済みソルバーの現在状態を保持したまま実行時パラメーターを更新する。
 * 内部メッシュ構造を決める maximum_element_length、minimum_dynamic_length、
 * fixed_root_nodes は構築時の値から変更できない。
 */
KHS_API KhsResult khsUpdateRuntimeParameters(KhsSolver *solver,
                                              const KhsSolverDesc *desc,
                                              const KhsHairMaterial *material);

/* 評価後の同じトポロジーを次フレームの目標として渡す。 */
KHS_API KhsResult khsUpdateColliderVertices(KhsSolver *solver,
                                            const KhsVec3 *vertices,
                                            uint32_t vertex_count);

/* 元 Hair Curves の評価後ワールド座標。固定根元の移動目標にだけ使う。 */
KHS_API KhsResult khsUpdateRootTargets(KhsSolver *solver,
                                       const KhsVec3 *points,
                                       uint32_t point_count);

KHS_API KhsResult khsStep(KhsSolver *solver, double frame_dt);

/* 全フレームを事前確保し、各フレームを直接GPU常駐領域へ転送する。 */
KHS_API KhsResult khsAllocateAnimation(KhsSolver *solver, uint32_t frame_count);
KHS_API KhsResult khsSetRootAnimationFrame(KhsSolver *solver,
                                           uint32_t frame_index,
                                           const KhsVec3 *points,
                                           uint32_t point_count);
KHS_API KhsResult khsSetColliderAnimationFrame(KhsSolver *solver,
                                               uint32_t frame_index,
                                               const KhsVec3 *vertices,
                                               uint32_t vertex_count);
KHS_API KhsResult khsFinalizeAnimation(KhsSolver *solver);
KHS_API KhsResult khsStepAnimationFrame(KhsSolver *solver,
                                        uint32_t frame_index,
                                        double frame_dt);

/*
 * 完成フレーム境界のCUDA状態（変位・回転・速度・アニメーション位置）を
 * 不透明なメモリへ保存・復元する。同じ構築済みソルバー内だけで使用できる。
 */
KHS_API uint64_t khsGetAnimationCheckpointSize(const KhsSolver *solver);
KHS_API KhsResult khsSaveAnimationCheckpoint(KhsSolver *solver,
                                              void *data,
                                              uint64_t capacity);
KHS_API KhsResult khsRestoreAnimationCheckpoint(KhsSolver *solver,
                                                 const void *data,
                                                 uint64_t size);

KHS_API uint32_t khsGetOriginalPointCount(const KhsSolver *solver);
KHS_API uint32_t khsGetInternalNodeCount(const KhsSolver *solver);
KHS_API KhsResult khsCopyOriginalPositions(const KhsSolver *solver,
                                           KhsVec3 *positions,
                                           uint32_t capacity);
KHS_API KhsResult khsCopyInternalPositions(const KhsSolver *solver,
                                           KhsVec3 *positions,
                                           uint32_t capacity);
KHS_API KhsResult khsCopyOriginalToInternalMap(const KhsSolver *solver,
                                               uint32_t *indices,
                                               uint32_t capacity);
KHS_API KhsResult khsGetBuildStats(const KhsSolver *solver,
                                   KhsBuildStats *stats);
KHS_API KhsResult khsGetLastStepStats(const KhsSolver *solver,
                                      KhsStepStats *stats);
KHS_API KhsResult khsGetGpuStats(const KhsSolver *solver, KhsGpuStats *stats);
KHS_API KhsResult khsGetProgress(const KhsSolver *solver, KhsProgress *progress);
KHS_API KhsResult khsGetFailureDiagnostics(const KhsSolver *solver,
                                            KhsFailureDiagnostics *diagnostics);
KHS_API KhsResult khsRequestCancel(KhsSolver *solver);
KHS_API const char *khsGetLastError(const KhsSolver *solver);

#ifdef __cplusplus
}
#endif

#endif
