from __future__ import annotations

import ctypes
import os
from pathlib import Path
import numpy as np


ABI_VERSION = 5
OK = 0


class Vec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("z", ctypes.c_double)]


class Triangle(ctypes.Structure):
    _fields_ = [("i0", ctypes.c_uint32), ("i1", ctypes.c_uint32), ("i2", ctypes.c_uint32)]


class SolverDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("gravity", Vec3),
        ("substeps", ctypes.c_uint32),
        ("maximum_substeps", ctypes.c_uint32),
        ("newton_iterations", ctypes.c_uint32),
        ("line_search_iterations", ctypes.c_uint32),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("increment_tolerance", ctypes.c_double),
        ("minimum_line_search_step", ctypes.c_double),
        ("minimum_gap", ctypes.c_double),
        ("maximum_element_length", ctypes.c_double),
        ("minimum_dynamic_length", ctypes.c_double),
        ("fixed_root_nodes", ctypes.c_uint32),
        ("thread_count", ctypes.c_uint32),
    ]


class HairMaterial(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("density", ctypes.c_double),
        ("radius", ctypes.c_double),
        ("young_modulus", ctypes.c_double),
        ("poisson_ratio", ctypes.c_double),
        ("shear_correction", ctypes.c_double),
        ("mass_damping", ctypes.c_double),
        ("contact_stiffness", ctypes.c_double),
        ("barrier_distance", ctypes.c_double),
        ("friction", ctypes.c_double),
        ("friction_smoothing", ctypes.c_double),
        ("collider_offset", ctypes.c_double),
    ]


class BuildStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("strand_count", ctypes.c_uint32),
        ("original_point_count", ctypes.c_uint32),
        ("internal_node_count", ctypes.c_uint32),
        ("element_count", ctypes.c_uint32),
        ("fixed_node_count", ctypes.c_uint32),
        ("collider_vertex_count", ctypes.c_uint32),
        ("collider_triangle_count", ctypes.c_uint32),
        ("excluded_collider_triangle_count", ctypes.c_uint32),
        ("collider_boundary_edge_count", ctypes.c_uint32),
        ("collider_nonmanifold_edge_count", ctypes.c_uint32),
        ("collider_inconsistent_edge_count", ctypes.c_uint32),
        ("collider_inverted_closed_component_count", ctypes.c_uint32),
        ("merged_zero_length_segment_count", ctypes.c_uint32),
        ("virtual_extension_strand_count", ctypes.c_uint32),
        ("virtual_extension_node_count", ctypes.c_uint32),
        ("virtual_extension_rest_length", ctypes.c_double),
        ("degree_of_freedom_count", ctypes.c_uint64),
        ("estimated_bytes", ctypes.c_uint64),
        ("initial_minimum_gap", ctypes.c_double),
    ]


class StepStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("substeps", ctypes.c_uint32),
        ("converged_substeps", ctypes.c_uint32),
        ("newton_iterations", ctypes.c_uint32),
        ("linear_solves", ctypes.c_uint32),
        ("line_search_evaluations", ctypes.c_uint32),
        ("contact_candidate_count", ctypes.c_uint64),
        ("active_contact_count", ctypes.c_uint64),
        ("initial_residual_norm", ctypes.c_double),
        ("final_residual_norm", ctypes.c_double),
        ("relative_residual_norm", ctypes.c_double),
        ("increment_norm", ctypes.c_double),
        ("objective_change", ctypes.c_double),
        ("minimum_gap", ctypes.c_double),
        ("accepted_step_length", ctypes.c_double),
        ("ccd_step_limit", ctypes.c_double),
        ("kinetic_energy", ctypes.c_double),
        ("elastic_energy", ctypes.c_double),
        ("contact_energy", ctypes.c_double),
        ("friction_energy", ctypes.c_double),
        ("phase", ctypes.c_int),
    ]


class GpuInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("available", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("compute_capability_major", ctypes.c_uint32),
        ("compute_capability_minor", ctypes.c_uint32),
        ("total_vram_bytes", ctypes.c_uint64),
        ("device_name", ctypes.c_char * 128),
    ]


class GpuStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("animation_frame_count", ctypes.c_uint32),
        ("resident_bytes", ctypes.c_uint64),
        ("peak_temporary_bytes", ctypes.c_uint64),
        ("last_frame_milliseconds", ctypes.c_double),
        ("last_assembly_milliseconds", ctypes.c_double),
        ("last_collision_milliseconds", ctypes.c_double),
        ("last_optimization_milliseconds", ctypes.c_double),
    ]


class Progress(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("frame_index", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("substep", ctypes.c_uint32),
        ("substep_count", ctypes.c_uint32),
        ("nonlinear_iteration", ctypes.c_uint32),
        ("nonlinear_iteration_limit", ctypes.c_uint32),
        ("cancelled", ctypes.c_uint32),
        ("frame_elapsed_seconds", ctypes.c_double),
    ]


class FailureDiagnostics(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("frame_index", ctypes.c_uint32),
        ("substep", ctypes.c_uint32),
        ("requested_substeps", ctypes.c_uint32),
        ("attempted_substeps", ctypes.c_uint32),
        ("maximum_substeps", ctypes.c_uint32),
        ("adaptive_attempt_count", ctypes.c_uint32),
        ("strand_index", ctypes.c_uint32),
        ("element_index", ctypes.c_uint32),
        ("collider_triangle_index", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("distance", ctypes.c_double),
        ("required_distance", ctypes.c_double),
        ("clearance", ctypes.c_double),
        ("collider_substep_displacement", ctypes.c_double),
        ("collider_frame_displacement", ctypes.c_double),
        ("hair_start", Vec3),
        ("hair_end", Vec3),
        ("collider_point", Vec3),
    ]


def _vec_array(values):
    storage = np.ascontiguousarray(values, dtype=np.float64).reshape((-1, 3))
    pointer = storage.ctypes.data_as(ctypes.POINTER(Vec3))
    return storage, pointer


class HairSolver:
    def __init__(self, library_path: str | Path | None = None):
        path = Path(library_path) if library_path else Path(__file__).parent / "bin" / "kami_hair_solver.dll"
        self._handle = None
        self._dll_directories = []
        cuda_candidates = []
        if os.environ.get("CUDA_PATH"):
            cuda_candidates.append(Path(os.environ["CUDA_PATH"]) / "bin")
        cuda_candidates.append(Path(os.environ.get("ProgramFiles", "C:/Program Files")) /
                               "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.9" / "bin")
        if hasattr(os, "add_dll_directory"):
            for directory in cuda_candidates:
                if directory.is_dir():
                    self._dll_directories.append(os.add_dll_directory(str(directory)))
        self._library = ctypes.CDLL(str(path))
        self._bind()
        if self._library.khsGetAbiVersion() != ABI_VERSION:
            raise RuntimeError("髪ソルバーDLLのABIバージョンが一致しません。")
        self.desc = SolverDesc()
        self.material = HairMaterial()
        self._library.khsDefaultSolverDesc(ctypes.byref(self.desc))
        self._library.khsDefaultHairMaterial(ctypes.byref(self.material))

    def _bind(self):
        lib = self._library
        lib.khsGetAbiVersion.restype = ctypes.c_uint32
        lib.khsGetGpuInfo.argtypes = [ctypes.POINTER(GpuInfo)]
        lib.khsGetGpuInfo.restype = ctypes.c_int
        lib.khsDefaultSolverDesc.argtypes = [ctypes.POINTER(SolverDesc)]
        lib.khsDefaultHairMaterial.argtypes = [ctypes.POINTER(HairMaterial)]
        lib.khsCreate.argtypes = [ctypes.POINTER(SolverDesc)]
        lib.khsCreate.restype = ctypes.c_void_p
        lib.khsDestroy.argtypes = [ctypes.c_void_p]
        lib.khsSetHairCurves.argtypes = [ctypes.c_void_p, ctypes.POINTER(Vec3), ctypes.c_uint32,
                                         ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                         ctypes.POINTER(HairMaterial)]
        lib.khsSetHairCurves.restype = ctypes.c_int
        lib.khsSetColliderMesh.argtypes = [ctypes.c_void_p, ctypes.POINTER(Vec3), ctypes.c_uint32,
                                           ctypes.POINTER(Triangle), ctypes.c_uint32]
        lib.khsSetColliderMesh.restype = ctypes.c_int
        lib.khsBuild.argtypes = [ctypes.c_void_p]
        lib.khsBuild.restype = ctypes.c_int
        lib.khsUpdateRuntimeParameters.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(SolverDesc), ctypes.POINTER(HairMaterial)]
        lib.khsUpdateRuntimeParameters.restype = ctypes.c_int
        lib.khsUpdateColliderVertices.argtypes = [ctypes.c_void_p, ctypes.POINTER(Vec3), ctypes.c_uint32]
        lib.khsUpdateColliderVertices.restype = ctypes.c_int
        lib.khsUpdateRootTargets.argtypes = [ctypes.c_void_p, ctypes.POINTER(Vec3), ctypes.c_uint32]
        lib.khsUpdateRootTargets.restype = ctypes.c_int
        lib.khsStep.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.khsStep.restype = ctypes.c_int
        lib.khsAllocateAnimation.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.khsAllocateAnimation.restype = ctypes.c_int
        lib.khsSetRootAnimationFrame.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.POINTER(Vec3), ctypes.c_uint32]
        lib.khsSetRootAnimationFrame.restype = ctypes.c_int
        lib.khsSetColliderAnimationFrame.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                      ctypes.POINTER(Vec3), ctypes.c_uint32]
        lib.khsSetColliderAnimationFrame.restype = ctypes.c_int
        lib.khsFinalizeAnimation.argtypes = [ctypes.c_void_p]
        lib.khsFinalizeAnimation.restype = ctypes.c_int
        lib.khsStepAnimationFrame.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double]
        lib.khsStepAnimationFrame.restype = ctypes.c_int
        lib.khsGetAnimationCheckpointSize.argtypes = [ctypes.c_void_p]
        lib.khsGetAnimationCheckpointSize.restype = ctypes.c_uint64
        lib.khsSaveAnimationCheckpoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.khsSaveAnimationCheckpoint.restype = ctypes.c_int
        lib.khsRestoreAnimationCheckpoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.khsRestoreAnimationCheckpoint.restype = ctypes.c_int
        lib.khsGetOriginalPointCount.argtypes = [ctypes.c_void_p]
        lib.khsGetOriginalPointCount.restype = ctypes.c_uint32
        lib.khsCopyOriginalPositions.argtypes = [ctypes.c_void_p, ctypes.POINTER(Vec3), ctypes.c_uint32]
        lib.khsCopyOriginalPositions.restype = ctypes.c_int
        lib.khsGetBuildStats.argtypes = [ctypes.c_void_p, ctypes.POINTER(BuildStats)]
        lib.khsGetBuildStats.restype = ctypes.c_int
        lib.khsGetLastStepStats.argtypes = [ctypes.c_void_p, ctypes.POINTER(StepStats)]
        lib.khsGetLastStepStats.restype = ctypes.c_int
        lib.khsGetGpuStats.argtypes = [ctypes.c_void_p, ctypes.POINTER(GpuStats)]
        lib.khsGetGpuStats.restype = ctypes.c_int
        lib.khsGetProgress.argtypes = [ctypes.c_void_p, ctypes.POINTER(Progress)]
        lib.khsGetProgress.restype = ctypes.c_int
        lib.khsGetFailureDiagnostics.argtypes = [ctypes.c_void_p, ctypes.POINTER(FailureDiagnostics)]
        lib.khsGetFailureDiagnostics.restype = ctypes.c_int
        lib.khsRequestCancel.argtypes = [ctypes.c_void_p]
        lib.khsRequestCancel.restype = ctypes.c_int
        lib.khsGetLastError.argtypes = [ctypes.c_void_p]
        lib.khsGetLastError.restype = ctypes.c_char_p

    def create(self):
        if self._handle:
            self.close()
        self._handle = self._library.khsCreate(ctypes.byref(self.desc))
        if not self._handle:
            raise RuntimeError("髪ソルバーを作成できません。設定値を確認してください。")

    def gpu_info(self):
        info = GpuInfo()
        info.struct_size = ctypes.sizeof(info)
        result = self._library.khsGetGpuInfo(ctypes.byref(info))
        if result != OK:
            raise RuntimeError("CUDA GPUを取得できません。")
        return info

    def _check(self, result):
        if result != OK:
            message = self._library.khsGetLastError(self._handle)
            raise RuntimeError(message.decode("utf-8", errors="replace") if message else "髪ソルバーでエラーが発生しました。")

    def set_hair(self, points, offsets):
        _storage, values = _vec_array(points)
        curve_offsets = (ctypes.c_uint32 * len(offsets))(*offsets)
        self._check(self._library.khsSetHairCurves(
            self._handle, values, len(points), curve_offsets, len(offsets) - 1,
            ctypes.byref(self.material)))

    def set_collider(self, vertices, triangles):
        if not triangles:
            self._check(self._library.khsSetColliderMesh(self._handle, None, 0, None, 0))
            return
        _storage, values = _vec_array(vertices)
        faces = (Triangle * len(triangles))()
        for i, triangle in enumerate(triangles):
            faces[i] = Triangle(int(triangle[0]), int(triangle[1]), int(triangle[2]))
        self._check(self._library.khsSetColliderMesh(
            self._handle, values, len(vertices), faces, len(triangles)))

    def build(self):
        self._check(self._library.khsBuild(self._handle))
        stats = BuildStats()
        stats.struct_size = ctypes.sizeof(stats)
        self._check(self._library.khsGetBuildStats(self._handle, ctypes.byref(stats)))
        return stats

    def update_runtime_parameters(self):
        self._check(self._library.khsUpdateRuntimeParameters(
            self._handle, ctypes.byref(self.desc), ctypes.byref(self.material)))

    def update_roots(self, points):
        _storage, values = _vec_array(points)
        self._check(self._library.khsUpdateRootTargets(self._handle, values, len(points)))

    def update_collider(self, vertices):
        _storage, values = _vec_array(vertices)
        self._check(self._library.khsUpdateColliderVertices(self._handle, values, len(vertices)))

    def step(self, dt):
        self._check(self._library.khsStep(self._handle, float(dt)))
        count = self._library.khsGetOriginalPointCount(self._handle)
        output = (Vec3 * count)()
        self._check(self._library.khsCopyOriginalPositions(self._handle, output, count))
        positions = [(v.x, v.y, v.z) for v in output]
        return positions, self.stats()

    def allocate_animation(self, frame_count):
        self._check(self._library.khsAllocateAnimation(self._handle, int(frame_count)))

    def set_root_animation_frame(self, frame, points):
        _storage, values = _vec_array(points)
        self._check(self._library.khsSetRootAnimationFrame(
            self._handle, int(frame), values, len(points)))

    def set_collider_animation_frame(self, frame, vertices):
        if not vertices:
            self._check(self._library.khsSetColliderAnimationFrame(
                self._handle, int(frame), None, 0))
            return
        _storage, values = _vec_array(vertices)
        self._check(self._library.khsSetColliderAnimationFrame(
            self._handle, int(frame), values, len(vertices)))

    def finalize_animation(self):
        self._check(self._library.khsFinalizeAnimation(self._handle))

    def step_animation_frame(self, frame, dt):
        self._check(self._library.khsStepAnimationFrame(self._handle, int(frame), float(dt)))
        return self.positions(), self.stats()

    def save_checkpoint(self):
        size = self.checkpoint_size()
        if not size:
            raise RuntimeError("CUDAアニメーション状態をまだ保存できません。")
        storage = (ctypes.c_ubyte * size)()
        self._check(self._library.khsSaveAnimationCheckpoint(self._handle, storage, size))
        return bytes(storage)

    def restore_checkpoint(self, checkpoint):
        storage = (ctypes.c_ubyte * len(checkpoint)).from_buffer_copy(checkpoint)
        self._check(self._library.khsRestoreAnimationCheckpoint(
            self._handle, storage, len(checkpoint)))

    def checkpoint_size(self):
        return int(self._library.khsGetAnimationCheckpointSize(self._handle))

    def stats(self):
        stats = StepStats()
        stats.struct_size = ctypes.sizeof(stats)
        self._check(self._library.khsGetLastStepStats(self._handle, ctypes.byref(stats)))
        return stats

    def gpu_stats(self):
        stats = GpuStats()
        stats.struct_size = ctypes.sizeof(stats)
        self._check(self._library.khsGetGpuStats(self._handle, ctypes.byref(stats)))
        return stats

    def progress(self):
        progress = Progress()
        progress.struct_size = ctypes.sizeof(progress)
        self._check(self._library.khsGetProgress(self._handle, ctypes.byref(progress)))
        return progress

    def failure_diagnostics(self):
        diagnostics = FailureDiagnostics()
        diagnostics.struct_size = ctypes.sizeof(diagnostics)
        self._check(self._library.khsGetFailureDiagnostics(
            self._handle, ctypes.byref(diagnostics)))
        return diagnostics

    def cancel(self):
        self._check(self._library.khsRequestCancel(self._handle))

    def positions(self):
        count = self._library.khsGetOriginalPointCount(self._handle)
        output = (Vec3 * count)()
        self._check(self._library.khsCopyOriginalPositions(self._handle, output, count))
        return [(v.x, v.y, v.z) for v in output]

    def close(self):
        if self._handle:
            self._library.khsDestroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
