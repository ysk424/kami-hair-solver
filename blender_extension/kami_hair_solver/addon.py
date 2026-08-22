from __future__ import annotations

from array import array
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import socket
import struct
import textwrap
import threading
import time

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix

from .native import HairSolver


_CACHE_MAGIC = b"KAMIHC1\0"
_COLLIDER_CACHE_MAGIC = b"KAMISC1\0"
_CACHE_HEADER = struct.Struct("<8sIII")
_SESSIONS = {}
_ACTIVE_BAKES = set()
_RESUMABLE_BAKES = {}
_PROGRESS_PHASE_NAMES = {
    0: "待機",
    1: "準備",
    2: "方程式組立",
    3: "線形求解",
    4: "衝突判定",
    5: "ラインサーチ",
    6: "完了",
    7: "失敗",
    8: "sweep事前制限",
    9: "相対運動sweep",
    10: "ラインサーチCCD",
}
_UINT32_MAX = (1 << 32) - 1
_FAILURE_KIND_NAMES = {
    1: "髪・コライダー相対移動TOI制限",
    2: "バリア実行可能領域外",
    3: "非線形求解失敗",
}
_PARAMETER_INFO = {
    "substeps": ("基本サブステップ", "same_frame"),
    "maximum_substeps": ("可変時間ステップ上限", "same_frame"),
    "newton_iterations": ("Newton反復上限", "same_frame"),
    "density": ("密度", "rewind"),
    "radius": ("物理半径", "rewind"),
    "young_modulus": ("Young率", "rewind"),
    "poisson_ratio": ("Poisson比", "rewind"),
    "mass_damping": ("質量比例減衰", "rewind"),
    "contact_stiffness": ("接触バリア剛性", "rewind"),
    "barrier_distance": ("バリア活性距離", "rewind"),
    "friction": ("摩擦係数", "rewind"),
    "collider_offset": ("コライダー間隔", "rewind"),
    "soft_collider": ("softコライダー", "restart"),
    "collider_anchor_stiffness": ("コライダーアンカー剛性", "rewind"),
    "maximum_element_length": ("最大要素長", "restart"),
    "minimum_dynamic_length": ("最小動力学長", "restart"),
    "fixed_root_nodes": ("固定する毛根節点数", "restart"),
}
_DISTANCE_PARAMETERS = {
    "radius", "barrier_distance", "collider_offset",
    "maximum_element_length", "minimum_dynamic_length",
}


def _hair_poll(_self, obj):
    return obj is not None and obj.type == "CURVES"


def _mesh_poll(_self, obj):
    return obj is not None and obj.type == "MESH"


def _get_maximum_substeps(settings):
    stored = settings.get("_maximum_substeps")
    if stored is None:
        stored = settings.get("maximum_substeps")
    if stored is not None:
        return int(stored)
    return min(4096, max(128, int(settings.substeps)))


def _set_maximum_substeps(settings, value):
    settings["_maximum_substeps"] = int(value)


class HairSettings3(PropertyGroup):
    hair: PointerProperty(name="入力する髪", type=bpy.types.Object, poll=_hair_poll)
    collider: PointerProperty(name="衝突メッシュ", type=bpy.types.Object, poll=_mesh_poll)
    result: PointerProperty(name="計算結果の髪", type=bpy.types.Object)
    collider_result: PointerProperty(name="softコライダー結果", type=bpy.types.Object)
    frame_start: IntProperty(name="開始フレーム", default=1, min=-1048574, max=1048574)
    frame_end: IntProperty(name="終了フレーム", default=100, min=-1048574, max=1048574)
    substeps: IntProperty(name="基本サブステップ", default=8, min=1, max=4096)
    maximum_substeps: IntProperty(
        name="可変時間ステップ上限",
        description="TOI制限や求解失敗の局所再試行を含め、1フレームで使用できる時間区間の上限",
        min=1, max=4096, get=_get_maximum_substeps, set=_set_maximum_substeps)
    newton_iterations: IntProperty(name="Newton反復上限", default=32, min=2, max=100)
    checkpoint_frames: IntProperty(
        name="巻き戻し保持フレーム数",
        description="デバッグ再開用に完全なCUDA状態をメモリへ保持する過去フレーム数",
        default=10, min=1, max=100)
    resume_frame: IntProperty(
        name="再開フレーム", default=1, min=-1048574, max=1048574)
    maximum_element_length: FloatProperty(
        name="最大要素長", default=0.01, min=1.0e-5, soft_max=0.05, subtype="DISTANCE", unit="LENGTH")
    minimum_dynamic_length: FloatProperty(
        name="最小動力学長",
        description="短い髪を毛先接線方向へ非表示延長し、この自然長で物理計算します。延長部分はコライダーと接触しません（0で無効）",
        default=0.0, min=0.0, soft_max=0.5, subtype="DISTANCE", unit="LENGTH")
    fixed_root_nodes: IntProperty(name="固定する毛根節点数", default=2, min=1, max=32)
    density: FloatProperty(name="密度", default=1300.0, min=1.0, soft_max=2000.0, unit="MASS")
    radius: FloatProperty(
        name="物理半径", default=4.0e-5, min=1.0e-7, soft_max=0.002, subtype="DISTANCE", unit="LENGTH")
    young_modulus: FloatProperty(name="Young率", default=4.0e9, min=1.0e3, soft_max=1.0e10)
    poisson_ratio: FloatProperty(name="Poisson比", default=0.38, min=-0.99, max=0.499)
    mass_damping: FloatProperty(name="質量比例減衰", default=8.0, min=0.0, soft_max=30.0)
    contact_stiffness: FloatProperty(name="接触バリア剛性", default=1.0e5, min=1.0e-3, soft_max=1.0e6)
    barrier_distance: FloatProperty(
        name="バリア活性距離", default=7.0e-4, min=1.0e-7, soft_max=0.01,
        subtype="DISTANCE", unit="LENGTH")
    friction: FloatProperty(name="摩擦係数", default=0.35, min=0.0, soft_max=1.5)
    collider_offset: FloatProperty(
        name="コライダー間隔", default=5.0e-4, min=0.0, soft_max=0.01,
        subtype="DISTANCE", unit="LENGTH")
    soft_collider: BoolProperty(
        name="softコライダー（実験）",
        description="コライダー頂点を目標位置へ戻す面積集中アンカーばねで変形可能にします",
        default=False)
    collider_anchor_stiffness: FloatProperty(
        name="アンカー剛性密度",
        description="softコライダーをアニメーション目標へ戻す面積当たり剛性 [N/m³]",
        default=1.0e8, min=1.0, soft_min=1.0e5, soft_max=1.0e11)
    cache_path: StringProperty(name="髪キャッシュ", default="//髪キャッシュ.khc", subtype="FILE_PATH")
    show_advanced: BoolProperty(name="詳細設定", default=False)
    status: StringProperty(name="状態", default="未準備")
    summary: StringProperty(name="準備情報", default="")
    error_detail: StringProperty(name="エラー詳細", default="")
    parameter_history: StringProperty(name="パラメータ変更履歴", default="")


def _notify_codex():
    """Send a best-effort local notification without waiting for a reply."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.sendto(b"PING", ("127.0.0.1", 8765))
    except OSError:
        pass
    return None


def _append_notification_error(settings, notification_error):
    if not notification_error:
        return
    if settings.error_detail:
        settings.error_detail += "\n"
    settings.error_detail += notification_error


def _settings_snapshot(settings):
    return {name: getattr(settings, name) for name in _PARAMETER_INFO}


def _parameter_changes(previous, current):
    changes = []
    for name, (label, category) in _PARAMETER_INFO.items():
        before = previous.get(name)
        after = current.get(name)
        if before != after:
            changes.append((name, label, category, before, after))
    return changes


def _parameter_value_text(name, value):
    if name in _DISTANCE_PARAMETERS:
        return f"{float(value) * 1000.0:.6g} mm"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parameter_history_line(resume_frame, changes):
    detail = "、".join(
        f"{label} {_parameter_value_text(name, before)}→{_parameter_value_text(name, after)}"
        for name, label, _category, before, after in changes)
    return f"フレーム{resume_frame}から再開: {detail}"


def _vec_text(value):
    return f"({value[0]:.5f}, {value[1]:.5f}, {value[2]:.5f}) m"


def _progress_snapshot(solver):
    try:
        progress = solver.progress()
        snapshot = {
            "phase": int(progress.phase),
            "substep": int(progress.substep),
            "substep_count": int(progress.substep_count),
            "iteration": int(progress.nonlinear_iteration),
            "iteration_limit": int(progress.nonlinear_iteration_limit),
            "attempted_substeps": int(progress.attempted_substeps),
            "accepted_substeps": int(progress.accepted_substeps),
            "sweep_guard_reductions": int(progress.sweep_guard_reductions),
            "soft_attempts": int(progress.soft_collider_attempts),
        }
        stats = solver.stats()
        snapshot["soft_retry_attempts"] = int(stats.soft_collider_retry_attempts)
        snapshot["soft_completed"] = int(stats.soft_collider_substeps)
        snapshot["hard_iteration_limit_retries"] = int(
            stats.hard_iteration_limit_retries)
        snapshot["moving_sweep_candidates"] = int(
            stats.moving_sweep_candidate_count)
        snapshot["maximum_predicted_displacement"] = float(
            stats.maximum_predicted_displacement)
        snapshot["sweep_displacement_limit"] = float(stats.sweep_displacement_limit)
        diagnostics = solver.failure_diagnostics()
        if diagnostics.kind:
            snapshot["failure"] = {
                "kind": int(diagnostics.kind),
                "frame_index": int(diagnostics.frame_index),
                "substep": int(diagnostics.substep),
                "requested_substeps": int(diagnostics.requested_substeps),
                "attempted_substeps": int(diagnostics.attempted_substeps),
                "maximum_substeps": int(diagnostics.maximum_substeps),
                "adaptive_attempt_count": int(diagnostics.adaptive_attempt_count),
                "strand_index": int(diagnostics.strand_index),
                "element_index": int(diagnostics.element_index),
                "triangle_index": int(diagnostics.collider_triangle_index),
                "distance": float(diagnostics.distance),
                "required_distance": float(diagnostics.required_distance),
                "clearance": float(diagnostics.clearance),
                "collider_substep_displacement": float(
                    diagnostics.collider_substep_displacement),
                "collider_frame_displacement": float(
                    diagnostics.collider_frame_displacement),
                "hair_start": (
                    float(diagnostics.hair_start.x), float(diagnostics.hair_start.y),
                    float(diagnostics.hair_start.z)),
                "hair_end": (
                    float(diagnostics.hair_end.x), float(diagnostics.hair_end.y),
                    float(diagnostics.hair_end.z)),
                "collider_point": (
                    float(diagnostics.collider_point.x), float(diagnostics.collider_point.y),
                    float(diagnostics.collider_point.z)),
            }
        return snapshot
    except Exception:
        return None


def _failure_detail(stage, exception, *, completed_frame=None, failed_frame=None, progress=None):
    parts = [stage]
    if completed_frame is not None:
        parts.append(f"フレーム{completed_frame}まで完了")
    if failed_frame is not None:
        parts.append(f"フレーム{failed_frame}で停止")
    if progress:
        phase = _PROGRESS_PHASE_NAMES.get(progress["phase"], f"フェーズ{progress['phase']}")
        parts.append(phase)
        if progress["substep_count"]:
            parts.append(
                f"可変時間ステップ 成功{progress.get('accepted_substeps', 0)} / "
                f"試行{progress.get('attempted_substeps', progress['substep'])} / "
                f"上限{progress['substep_count']}")
        if progress.get("sweep_guard_reductions"):
            parts.append(f"sweep事前縮小 {progress['sweep_guard_reductions']}回")
            parts.append(
                f"最大予測移動 {progress.get('maximum_predicted_displacement', 0.0) * 1000.0:.3f} mm / "
                f"上限 {progress.get('sweep_displacement_limit', 0.0) * 1000.0:.3f} mm / "
                f"候補 {progress.get('moving_sweep_candidates', 0):,}")
        if progress.get("hard_iteration_limit_retries"):
            parts.append(
                f"hard Newton上限後soft再試行 "
                f"{progress['hard_iteration_limit_retries']}回")
        if progress["iteration"]:
            parts.append(f"Newton反復 {progress['iteration']}/{progress['iteration_limit']}")
        if progress.get("soft_attempts"):
            parts.append(
                f"soft試行 {progress['soft_attempts']} / "
                f"hard失敗後 {progress.get('soft_retry_attempts', 0)} / "
                f"成功 {progress.get('soft_completed', 0)}")
        failure = progress.get("failure")
        if failure:
            kind = _FAILURE_KIND_NAMES.get(failure["kind"], f"失敗種別{failure['kind']}")
            parts.append(kind)
            parts.append(
                f"可変時間ステップ 試行{failure['attempted_substeps']} / "
                f"TOI制限{failure['adaptive_attempt_count']} / "
                f"基本{failure['requested_substeps']} / 上限{failure['maximum_substeps']}")
            if failure["strand_index"] != _UINT32_MAX:
                parts.append(
                    f"髪ストランド{failure['strand_index'] + 1} / "
                    f"内部要素{failure['element_index']} / "
                    f"コライダー三角形{failure['triangle_index']}")
                parts.append(
                    f"検出距離 {failure['distance'] * 1000.0:.4f} mm / "
                    f"必要距離 {failure['required_distance'] * 1000.0:.4f} mm / "
                    f"余裕 {failure['clearance'] * 1000.0:.4f} mm")
                parts.append(
                    f"対象面安全移動 {failure['collider_substep_displacement'] * 1000.0:.4f} mm/時間区間 / "
                    f"{failure['collider_frame_displacement'] * 1000.0:.3f} mm/フレーム")
                parts.append(
                    f"髪区間 {_vec_text(failure['hair_start'])}→{_vec_text(failure['hair_end'])} / "
                    f"コライダー位置 {_vec_text(failure['collider_point'])}")
            if failure["kind"] == 1:
                parts.append("推奨: 可変時間ステップ上限を増やすか、数フレーム戻して接触パラメータを調整してください")
    parts.append(f"{type(exception).__name__}: {exception}")
    return " / ".join(parts)


def _evaluated_hair(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    data = evaluated.data
    sizes = [curve.points_length for curve in data.curves]
    if not sizes or sum(sizes) == 0:
        raise RuntimeError("入力する髪にストランドがありません。")
    flat = np.empty(len(data.points) * 3, dtype=np.float32)
    data.points.foreach_get("position", flat)
    coordinates = flat.reshape((-1, 3)).astype(np.float64, copy=False)
    matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
    points = (coordinates @ matrix[:3, :3].T + matrix[:3, 3]).tolist()
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    radii = [0.0] * len(data.points)
    try:
        data.points.foreach_get("radius", radii)
    except (TypeError, AttributeError):
        radii = [0.01] * len(data.points)
    return points, offsets, sizes, radii


def _mesh_topology_signature(mesh):
    """Return a stable signature of the evaluated polygon connectivity.

    Loop triangles are deliberately excluded: Blender may flip the diagonal of
    a deforming non-planar quad without changing the mesh topology.
    """
    loop_vertices = array("I", [0]) * len(mesh.loops)
    polygon_sizes = array("I", [0]) * len(mesh.polygons)
    if loop_vertices:
        mesh.loops.foreach_get("vertex_index", loop_vertices)
    if polygon_sizes:
        mesh.polygons.foreach_get("loop_total", polygon_sizes)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(struct.pack("<QQQ", len(mesh.vertices), len(mesh.polygons), len(mesh.loops)))
    digest.update(loop_vertices.tobytes())
    digest.update(polygon_sizes.tobytes())
    return len(mesh.vertices), len(mesh.polygons), len(mesh.loops), digest.digest()


def _evaluated_collider(obj, depsgraph, *, triangulate=True):
    if obj is None:
        return [], [], (0, 0, 0, b"")
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", flat)
        coordinates = flat.reshape((-1, 3)).astype(np.float64, copy=False)
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        vertices = (coordinates @ matrix[:3, :3].T + matrix[:3, 3]).tolist()
        topology = _mesh_topology_signature(mesh)
        triangles = []
        if triangulate:
            mesh.calc_loop_triangles()
            triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
        return vertices, triangles, topology
    finally:
        evaluated.to_mesh_clear()


def _set_hair_positions(obj, positions):
    if obj is None or obj.type != "CURVES" or len(obj.data.points) != len(positions):
        return
    flat = [coordinate for point in positions for coordinate in point]
    obj.data.points.foreach_set("position", flat)
    obj.data.update_tag()


def _new_result_object(scene, source, sizes, positions, radii):
    settings = scene.kami_hair_3
    old = settings.result
    if old and old.name in bpy.data.objects:
        old_data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_data and old_data.users == 0:
            bpy.data.hair_curves.remove(old_data)
    data = bpy.data.hair_curves.new("髪_計算結果")
    data.add_curves(sizes)
    result = bpy.data.objects.new("髪_計算結果", data)
    scene.collection.objects.link(result)
    result.matrix_world = Matrix.Identity(4)
    _set_hair_positions(result, positions)
    try:
        data.points.foreach_set("radius", radii)
    except (TypeError, AttributeError):
        pass
    if hasattr(source.data, "materials"):
        for material in source.data.materials:
            data.materials.append(material)
    settings.result = result
    return result


def _set_mesh_positions(obj, positions):
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) != len(positions):
        return
    flat = [coordinate for point in positions for coordinate in point]
    obj.data.vertices.foreach_set("co", flat)
    obj.data.update()


def _new_collider_result_object(scene, source, vertices, triangles):
    settings = scene.kami_hair_3
    old = settings.collider_result
    if old and old.name in bpy.data.objects:
        old_data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if old_data and old_data.users == 0:
            bpy.data.meshes.remove(old_data)
    if not vertices or not triangles:
        settings.collider_result = None
        return None
    data = bpy.data.meshes.new("softコライダー_計算結果")
    data.from_pydata(vertices, [], triangles)
    data.update()
    result = bpy.data.objects.new("softコライダー_計算結果", data)
    scene.collection.objects.link(result)
    result.matrix_world = Matrix.Identity(4)
    if source is not None and hasattr(source.data, "materials"):
        for material in source.data.materials:
            data.materials.append(material)
    settings.collider_result = result
    return result


def _apply_solver_settings(settings, solver):
    solver.desc.substeps = settings.substeps
    solver.desc.maximum_substeps = settings.maximum_substeps
    solver.desc.newton_iterations = settings.newton_iterations
    solver.desc.maximum_element_length = settings.maximum_element_length
    solver.desc.minimum_dynamic_length = settings.minimum_dynamic_length
    solver.desc.fixed_root_nodes = settings.fixed_root_nodes
    solver.desc.collider_anchor_stiffness = (
        settings.collider_anchor_stiffness if settings.soft_collider else 0.0)
    solver.material.density = settings.density
    solver.material.radius = settings.radius
    solver.material.young_modulus = settings.young_modulus
    solver.material.poisson_ratio = settings.poisson_ratio
    solver.material.mass_damping = settings.mass_damping
    solver.material.contact_stiffness = settings.contact_stiffness
    solver.material.barrier_distance = settings.barrier_distance
    solver.material.friction = settings.friction
    solver.material.collider_offset = settings.collider_offset


def _configure_solver(settings):
    solver = HairSolver()
    _apply_solver_settings(settings, solver)
    solver.create()
    return solver


def _begin_prepare_scene(scene):
    settings = scene.kami_hair_3
    settings.error_detail = ""
    if settings.hair is None or settings.hair.type != "CURVES":
        raise RuntimeError("入力する Hair Curves を指定してください。")
    if settings.collider is not None and settings.collider.type != "MESH":
        raise RuntimeError("衝突メッシュには Mesh オブジェクトを指定してください。")
    if settings.frame_end < settings.frame_start:
        raise RuntimeError("終了フレームは開始フレーム以降にしてください。")
    if settings.maximum_substeps < settings.substeps:
        raise RuntimeError("可変時間ステップ上限は基本サブステップ以上にしてください。")
    original_frame = scene.frame_current
    scene.frame_set(settings.frame_start)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    solver = None
    try:
        points, offsets, sizes, radii = _evaluated_hair(settings.hair, depsgraph)
        collider_vertices, collider_triangles, collider_topology = _evaluated_collider(
            settings.collider, depsgraph)
        solver = _configure_solver(settings)
        solver.set_hair(points, offsets)
        solver.set_collider(collider_vertices, collider_triangles)
        stats = solver.build()
        frame_count = settings.frame_end - settings.frame_start + 1
        solver.allocate_animation(frame_count)
        solver.set_root_animation_frame(0, points)
        solver.set_collider_animation_frame(0, collider_vertices)
        settings.status = f"CUDAへ転送中 1/{frame_count}"
        return {
            "solver": solver,
            "stats": stats,
            "points": points,
            "offsets": offsets,
            "sizes": sizes,
            "radii": radii,
            "collider_vertices": collider_vertices,
            "collider_triangles": collider_triangles,
            "collider_topology": collider_topology,
            "frame_count": frame_count,
            "next_frame_index": 1,
            "original_frame": original_frame,
            "finished": False,
        }
    except Exception:
        if solver is not None:
            solver.close()
        scene.frame_set(original_frame)
        raise


def _advance_prepare_scene(scene, state):
    settings = scene.kami_hair_3
    frame_index = state["next_frame_index"]
    if frame_index >= state["frame_count"]:
        return False
    frame = settings.frame_start + frame_index
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    root_points, frame_offsets, _frame_sizes, _frame_radii = _evaluated_hair(settings.hair, depsgraph)
    if frame_offsets != state["offsets"]:
        raise RuntimeError(f"フレーム{frame}: 髪のストランドトポロジーが変化しました。")
    frame_collider, _frame_triangles, frame_topology = _evaluated_collider(
        settings.collider, depsgraph, triangulate=False)
    if frame_topology != state["collider_topology"]:
        raise RuntimeError(f"フレーム{frame}: コライダーのトポロジーが変化しました。")
    state["solver"].set_root_animation_frame(frame_index, root_points)
    state["solver"].set_collider_animation_frame(frame_index, frame_collider)
    state["next_frame_index"] += 1
    settings.status = f"CUDAへ転送中 {state['next_frame_index']}/{state['frame_count']}"
    return True


def _finish_prepare_scene(scene, state):
    settings = scene.kami_hair_3
    solver = state["solver"]
    solver.finalize_animation()
    gpu_info = solver.gpu_info()
    gpu_stats = solver.gpu_stats()
    result = _new_result_object(scene, settings.hair, state["sizes"], state["points"], state["radii"])
    collider_result = _new_collider_result_object(
        scene, settings.collider,
        state["collider_vertices"] if settings.soft_collider else [],
        state["collider_triangles"] if settings.soft_collider else [])
    scene_key = scene.as_pointer()
    old_resume = _RESUMABLE_BAKES.pop(scene_key, None)
    if old_resume:
        old_temporary = old_resume["path"].with_suffix(old_resume["path"].suffix + ".未完成")
        try:
            if old_temporary.exists():
                old_temporary.unlink()
            old_collider_path = _collider_cache_path(old_resume["path"])
            old_collider_temporary = old_collider_path.with_suffix(
                old_collider_path.suffix + ".未完成")
            if old_collider_temporary.exists():
                old_collider_temporary.unlink()
        except OSError:
            pass
    old_session = _SESSIONS.pop(scene_key, None)
    if old_session:
        old_session["solver"].close()
    session = {
        "solver": solver,
        "offsets": state["offsets"],
        "sizes": state["sizes"],
        "collider_triangles": state["collider_triangles"],
        "collider_topology": state["collider_topology"],
        "frame_count": state["frame_count"],
        "gpu_stats": gpu_stats,
        "point_count": len(state["points"]),
        "result": result,
        "collider_result": collider_result,
        "soft_collider": bool(settings.soft_collider and state["collider_vertices"]),
        "collider_count": len(state["collider_vertices"]),
        "hair": settings.hair,
        "collider": settings.collider,
        "frame_start": settings.frame_start,
        "frame_end": settings.frame_end,
        "checkpoint_capacity": settings.checkpoint_frames + 1,
        "initial_settings_snapshot": _settings_snapshot(settings),
        "parameter_history": [],
    }
    _SESSIONS[scene_key] = session
    stats = state["stats"]
    memory_mb = stats.estimated_bytes / (1024.0 * 1024.0)
    gpu_mb = gpu_stats.resident_bytes / (1024.0 * 1024.0)
    gpu_name = bytes(gpu_info.device_name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    checkpoint_mb = solver.checkpoint_size() / (1024.0 * 1024.0)
    settings.summary = (
        f"{stats.strand_count}本 / 元{stats.original_point_count}点 / "
        f"内部{stats.internal_node_count}節点 / {stats.element_count}要素 / "
        f"soft自由度 {stats.soft_collider_degree_of_freedom_count} / "
        f"CPU {memory_mb:.1f} MiB / CUDA {gpu_mb:.1f} MiB / "
        f"巻戻し最大約{checkpoint_mb * session['checkpoint_capacity']:.1f} MiB / {gpu_name}")
    settings.parameter_history = ""
    diagnostics = []
    if stats.virtual_extension_strand_count:
        diagnostics.append(
            f"非表示延長 {stats.virtual_extension_strand_count}本 / "
            f"{stats.virtual_extension_node_count}節点 / "
            f"計{stats.virtual_extension_rest_length:.3f} m")
    if stats.merged_zero_length_segment_count:
        diagnostics.append(f"ゼロ長統合 {stats.merged_zero_length_segment_count}区間")
    if stats.excluded_collider_triangle_count:
        diagnostics.append(f"除外面 {stats.excluded_collider_triangle_count}")
    if stats.collider_boundary_edge_count:
        diagnostics.append(f"境界辺 {stats.collider_boundary_edge_count}")
    if stats.collider_nonmanifold_edge_count:
        diagnostics.append(f"非多様体辺 {stats.collider_nonmanifold_edge_count}")
    if stats.collider_inconsistent_edge_count:
        diagnostics.append(f"向き不整合辺 {stats.collider_inconsistent_edge_count}")
    if stats.collider_inverted_closed_component_count:
        diagnostics.append("閉コライダー全体の面向きが反転")
    settings.status = "準備完了" + ("（" + "、".join(diagnostics) + "）" if diagnostics else "")
    state["finished"] = True
    scene.frame_set(state["original_frame"])
    return stats


def _abort_prepare_scene(scene, state):
    if state is None or state.get("finished"):
        return
    state["solver"].close()
    scene.frame_set(state["original_frame"])


def prepare_scene(scene, upload_progress=None):
    state = _begin_prepare_scene(scene)
    try:
        if upload_progress:
            upload_progress(1, state["frame_count"])
        while state["next_frame_index"] < state["frame_count"]:
            _advance_prepare_scene(scene, state)
            if upload_progress:
                upload_progress(state["next_frame_index"], state["frame_count"])
        return _finish_prepare_scene(scene, state)
    except Exception:
        _abort_prepare_scene(scene, state)
        raise


def _write_cache_frame(file, positions):
    values = array("d", (coordinate for point in positions for coordinate in point))
    if values.itemsize != 8:
        raise RuntimeError("この環境では髪キャッシュの倍精度形式を使用できません。")
    values.tofile(file)


def _read_cache_frame(path, frame):
    with path.open("rb") as file:
        raw = file.read(_CACHE_HEADER.size)
        if len(raw) != _CACHE_HEADER.size:
            return None
        magic, start, end, point_count = _CACHE_HEADER.unpack(raw)
        if magic != _CACHE_MAGIC or frame < start or frame > end:
            return None
        stride = point_count * 3 * 8
        file.seek(_CACHE_HEADER.size + (frame - start) * stride)
        values = array("d")
        try:
            values.fromfile(file, point_count * 3)
        except EOFError:
            return None
        return [tuple(values[3 * i:3 * i + 3]) for i in range(point_count)]


def _collider_cache_path(hair_cache_path):
    return hair_cache_path.with_suffix(hair_cache_path.suffix + ".soft-collider")


def _write_collider_cache_frame(file, positions):
    values = array("f", (coordinate for point in positions for coordinate in point))
    if values.itemsize != 4:
        raise RuntimeError("この環境ではsoftコライダーキャッシュの単精度形式を使用できません。")
    values.tofile(file)


def _read_collider_cache_frame(path, frame):
    with path.open("rb") as file:
        raw = file.read(_CACHE_HEADER.size)
        if len(raw) != _CACHE_HEADER.size:
            return None
        magic, start, end, point_count = _CACHE_HEADER.unpack(raw)
        if magic != _COLLIDER_CACHE_MAGIC or frame < start or frame > end:
            return None
        stride = point_count * 3 * 4
        file.seek(_CACHE_HEADER.size + (frame - start) * stride)
        values = array("f")
        try:
            values.fromfile(file, point_count * 3)
        except EOFError:
            return None
        return [tuple(values[3 * i:3 * i + 3]) for i in range(point_count)]


def _duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分"
    if minutes:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def _new_calculation_state(start, *, resume=None, checkpoint_capacity=11,
                           settings_snapshot=None, parameter_history=None):
    resume = resume or {}
    return {
        "done": False,
        "error": None,
        "error_stage": None,
        "error_progress": None,
        "failed_frame": None,
        "completed_frame": resume.get("completed_frame", start - 1),
        "solver_frame": resume.get("solver_frame", start),
        "pending_positions": resume.get("pending_positions"),
        "frame_seconds": list(resume.get("frame_seconds", [])),
        "final_stats": resume.get("final_stats"),
        "final_positions": resume.get("final_positions"),
        "final_collider_positions": resume.get("final_collider_positions"),
        "cancelled": False,
        "resume": bool(resume),
        "checkpoints": dict(resume.get("checkpoints", {})),
        "checkpoint_capacity": int(resume.get("checkpoint_capacity", checkpoint_capacity)),
        "settings_snapshot": dict(
            resume.get("settings_snapshot", settings_snapshot or {})),
        "parameter_history": list(
            resume.get("parameter_history", parameter_history or [])),
    }


def _remember_checkpoint(state, frame, solver):
    checkpoints = state["checkpoints"]
    checkpoints[frame] = solver.save_checkpoint()
    while len(checkpoints) > state["checkpoint_capacity"]:
        del checkpoints[next(iter(checkpoints))]


def _resume_frame_range(record):
    checkpoints = record.get("checkpoints", {})
    failed = record.get("failed_frame")
    if not checkpoints or failed is None:
        return None
    minimum = min(checkpoints) + 1
    maximum = min(failed, max(checkpoints) + 1)
    return minimum, maximum


def _record_for_resume_frame(record, resume_frame):
    valid_range = _resume_frame_range(record)
    if valid_range is None or not valid_range[0] <= resume_frame <= valid_range[1]:
        if valid_range:
            raise RuntimeError(
                f"再開フレームは{valid_range[0]}〜{valid_range[1]}を指定してください。")
        raise RuntimeError("巻き戻し用CUDAチェックポイントがありません。")
    checkpoint_frame = resume_frame - 1
    checkpoint = record["checkpoints"].get(checkpoint_frame)
    if checkpoint is None:
        raise RuntimeError(f"フレーム{checkpoint_frame}終了時のCUDA状態がありません。")
    record["session"]["solver"].restore_checkpoint(checkpoint)
    resumed = dict(record)
    resumed["completed_frame"] = checkpoint_frame
    resumed["solver_frame"] = checkpoint_frame
    resumed["pending_positions"] = None
    resumed["failed_frame"] = resume_frame
    resumed["error_stage"] = None
    resumed["error_progress"] = None
    completed_count = max(0, checkpoint_frame - record["start"] + 1)
    resumed["frame_seconds"] = list(record.get("frame_seconds", []))[:completed_count]
    resumed["checkpoints"] = {
        frame: value for frame, value in record["checkpoints"].items()
        if frame <= checkpoint_frame
    }
    temporary = record["path"].with_suffix(record["path"].suffix + ".未完成")
    resumed["final_positions"] = _read_cache_frame(temporary, checkpoint_frame)
    if record["session"].get("soft_collider"):
        collider_temporary = _collider_cache_path(record["path"]).with_suffix(
            _collider_cache_path(record["path"]).suffix + ".未完成")
        resumed["final_collider_positions"] = _read_collider_cache_frame(
            collider_temporary, checkpoint_frame)
    resumed["final_stats"] = None
    return resumed


def _resume_parameter_advice(record, settings, resume_frame=None):
    current = _settings_snapshot(settings)
    changes = _parameter_changes(record.get("settings_snapshot", {}), current)
    if any(change[2] == "restart" for change in changes):
        labels = "、".join(change[1] for change in changes if change[2] == "restart")
        return f"{labels}の変更は最初から再計算が必要です。", changes
    selected = settings.resume_frame if resume_frame is None else resume_frame
    if any(change[2] == "rewind" for change in changes) and selected >= record["failed_frame"]:
        return "材料・接触パラメータの変更は1フレーム以上戻して再開することを推奨します。", changes
    if changes:
        return "変更したパラメータを適用して再開します。", changes
    return "保存済みのCUDA状態から再開します。", changes


def _open_cache_for_calculation(temporary, start, end, point_count, state,
                                *, magic=_CACHE_MAGIC, item_size=8):
    if not state["resume"]:
        file = temporary.open("wb")
        file.write(_CACHE_HEADER.pack(magic, start, end, point_count))
        return file

    file = temporary.open("r+b")
    raw = file.read(_CACHE_HEADER.size)
    expected_header = (magic, start, end, point_count)
    if len(raw) != _CACHE_HEADER.size or _CACHE_HEADER.unpack(raw) != expected_header:
        file.close()
        raise RuntimeError("未完成キャッシュのヘッダーが再開対象と一致しません。")
    completed_count = max(0, state["completed_frame"] - start + 1)
    expected_size = _CACHE_HEADER.size + completed_count * point_count * 3 * item_size
    file.seek(0, os.SEEK_END)
    if file.tell() < expected_size:
        file.close()
        raise RuntimeError("未完成キャッシュが完了フレームより短いため再開できません。")
    file.truncate(expected_size)
    file.seek(expected_size)
    return file


def _calculate_preloaded(session, start, end, fps, path, state, frame_callback=None):
    solver = session["solver"]
    temporary = path.with_suffix(path.suffix + ".未完成")
    collider_path = _collider_cache_path(path)
    collider_temporary = collider_path.with_suffix(collider_path.suffix + ".未完成")
    try:
        state["error_stage"] = "未完成キャッシュを開く処理"
        with ExitStack() as stack:
            file = stack.enter_context(_open_cache_for_calculation(
                temporary, start, end, session["point_count"], state))
            collider_file = None
            if session.get("soft_collider"):
                collider_file = stack.enter_context(_open_cache_for_calculation(
                    collider_temporary, start, end, session["collider_count"], state,
                    magic=_COLLIDER_CACHE_MAGIC, item_size=4))
            for frame in range(state["completed_frame"] + 1, end + 1):
                state["failed_frame"] = frame
                if state.get("cancelled"):
                    state["error_stage"] = "中止処理"
                    raise RuntimeError("CUDA計算を中止しました。")
                frame_began = time.perf_counter()
                if state["pending_positions"] is not None and state["solver_frame"] == frame:
                    positions = state["pending_positions"]
                elif frame == start and state["solver_frame"] == start:
                    state["error_stage"] = f"フレーム{frame}の初期状態取得"
                    positions = solver.positions()
                    state["pending_positions"] = positions
                else:
                    state["error_stage"] = f"フレーム{frame}のCUDA求解"
                    positions, state["final_stats"] = solver.step_animation_frame(
                        frame - start, 1.0 / fps)
                    state["solver_frame"] = frame
                    state["pending_positions"] = positions

                state["error_stage"] = f"フレーム{frame}のキャッシュ書込"
                _write_cache_frame(file, positions)
                file.flush()
                collider_positions = None
                if collider_file is not None:
                    collider_positions = solver.collider_positions()
                    _write_collider_cache_frame(collider_file, collider_positions)
                    collider_file.flush()
                state["error_stage"] = f"フレーム{frame}の巻き戻し状態保存"
                _remember_checkpoint(state, frame, solver)
                state["final_positions"] = positions
                state["final_collider_positions"] = collider_positions
                state["completed_frame"] = frame
                state["pending_positions"] = None
                state["frame_seconds"].append(time.perf_counter() - frame_began)
                if frame_callback:
                    frame_callback(frame, positions, collider_positions)
        state["error_stage"] = "完成キャッシュの確定"
        os.replace(temporary, path)
        if session.get("soft_collider"):
            os.replace(collider_temporary, collider_path)
        state["failed_frame"] = None
        state["error_stage"] = None
    except Exception as exception:
        state["error"] = exception
        state["error_progress"] = _progress_snapshot(solver)
    finally:
        state["done"] = True


def _resume_record(session, start, end, path, state):
    return {
        "session": session,
        "start": start,
        "end": end,
        "path": path,
        "completed_frame": state["completed_frame"],
        "solver_frame": state["solver_frame"],
        "pending_positions": state["pending_positions"],
        "frame_seconds": state["frame_seconds"],
        "final_stats": state["final_stats"],
        "final_positions": state["final_positions"],
        "final_collider_positions": state["final_collider_positions"],
        "failed_frame": state["failed_frame"],
        "error_stage": state["error_stage"],
        "error_progress": state["error_progress"],
        "checkpoints": dict(state["checkpoints"]),
        "checkpoint_capacity": state["checkpoint_capacity"],
        "settings_snapshot": dict(state["settings_snapshot"]),
        "parameter_history": list(state["parameter_history"]),
    }


def _store_resume_record(scene_key, session, start, end, path, state):
    temporary = path.with_suffix(path.suffix + ".未完成")
    try:
        with temporary.open("rb") as file:
            raw = file.read(_CACHE_HEADER.size)
        if (len(raw) != _CACHE_HEADER.size or
                _CACHE_HEADER.unpack(raw) != (_CACHE_MAGIC, start, end, session["point_count"])):
            raise RuntimeError("未完成キャッシュのヘッダーが不正です。")
        completed_count = max(0, state["completed_frame"] - start + 1)
        minimum_size = _CACHE_HEADER.size + completed_count * session["point_count"] * 3 * 8
        if temporary.stat().st_size < minimum_size:
            raise RuntimeError("未完成キャッシュが完了フレームより短いです。")
        if session.get("soft_collider"):
            collider_path = _collider_cache_path(path)
            collider_temporary = collider_path.with_suffix(collider_path.suffix + ".未完成")
            with collider_temporary.open("rb") as file:
                collider_raw = file.read(_CACHE_HEADER.size)
            expected = (_COLLIDER_CACHE_MAGIC, start, end, session["collider_count"])
            if len(collider_raw) != _CACHE_HEADER.size or _CACHE_HEADER.unpack(collider_raw) != expected:
                raise RuntimeError("未完成softコライダーキャッシュのヘッダーが不正です。")
            collider_minimum = (
                _CACHE_HEADER.size + completed_count * session["collider_count"] * 3 * 4)
            if collider_temporary.stat().st_size < collider_minimum:
                raise RuntimeError("未完成softコライダーキャッシュが完了フレームより短いです。")
    except (OSError, RuntimeError, struct.error):
        _RESUMABLE_BAKES.pop(scene_key, None)
        return False
    _RESUMABLE_BAKES[scene_key] = _resume_record(session, start, end, path, state)
    return True


def _finish_bake_result(session, path, start, end, positions, collider_positions=None):
    result = session["result"]
    result["髪キャッシュ"] = str(path)
    result["髪開始フレーム"] = start
    result["髪終了フレーム"] = end
    result["髪点数"] = session["point_count"]
    result["髪解法"] = "CUDA・Cosserat FEM・implicit Euler・Gauss-Newton・バリア接触・Coulomb摩擦"
    if positions is not None:
        _set_hair_positions(result, positions)
    collider_result = session.get("collider_result")
    if collider_result is not None:
        collider_result["softコライダーキャッシュ"] = str(_collider_cache_path(path))
        collider_result["softコライダー開始フレーム"] = start
        collider_result["softコライダー終了フレーム"] = end
        collider_result["softコライダー頂点数"] = session["collider_count"]
        collider_result["softコライダー解法"] = "面積集中アンカーばね・髪接触連成・3x3ブロックGauss-Newton"
        if collider_positions is not None:
            _set_mesh_positions(collider_result, collider_positions)
    return result


def bake_scene(scene, progress=None):
    settings = scene.kami_hair_3
    scene_key = scene.as_pointer()
    original_frame = scene.frame_current
    state = None
    notified = False
    try:
        prepare_scene(scene)
        session = _SESSIONS[scene_key]
        start = settings.frame_start
        end = settings.frame_end
        path = Path(bpy.path.abspath(settings.cache_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = scene.render.fps / scene.render.fps_base
        began = time.perf_counter()
        state = _new_calculation_state(
            start, checkpoint_capacity=session["checkpoint_capacity"],
            settings_snapshot=_settings_snapshot(settings),
            parameter_history=session["parameter_history"])

        def frame_finished(frame, positions, collider_positions):
            _set_hair_positions(session["result"], positions)
            if collider_positions is not None:
                _set_mesh_positions(session.get("collider_result"), collider_positions)
            done = frame - start + 1
            total = end - start + 1
            remaining = (time.perf_counter() - began) / done * (total - done)
            settings.status = f"計算中 {frame}/{end}（残り約{remaining:.0f}秒）"
            if progress:
                progress(frame, start, end)

        _calculate_preloaded(session, start, end, fps, path, state, frame_finished)
        if state["error"] is not None:
            if _store_resume_record(scene_key, session, start, end, path, state):
                settings.resume_frame = state["failed_frame"]
            settings.error_detail = _failure_detail(
                state["error_stage"], state["error"],
                completed_frame=(state["completed_frame"]
                                 if state["completed_frame"] >= start else None),
                failed_frame=state["failed_frame"],
                progress=state["error_progress"])
            settings.status = f"計算失敗: {settings.error_detail}"
            _append_notification_error(settings, _notify_codex())
            notified = True
            raise state["error"]

        _RESUMABLE_BAKES.pop(scene_key, None)
        result = _finish_bake_result(
            session, path, start, end, state["final_positions"],
            state["final_collider_positions"])
        final_stats = state["final_stats"]
        if final_stats:
            gap_text = (f"最小ギャップ {final_stats.minimum_gap:.3e} m"
                        if final_stats.minimum_gap != float("inf") else "接触候補なし")
            settings.status = (
                f"計算完了 {end - start + 1}フレーム / "
                f"最終残差 {final_stats.final_residual_norm:.3e} / {gap_text}")
        else:
            settings.status = "計算完了（開始フレームのみ）"
        settings.error_detail = ""
        _append_notification_error(settings, _notify_codex())
        notified = True
        return result
    except Exception as exception:
        if state is None:
            settings.error_detail = _failure_detail("CUDA準備", exception)
            settings.status = f"計算失敗: {settings.error_detail}"
        if not notified:
            _append_notification_error(settings, _notify_codex())
        raise
    finally:
        scene.frame_set(original_frame)


@persistent
def _apply_hair_cache(scene, _depsgraph=None):
    for obj in scene.objects:
        cache = obj.get("髪キャッシュ")
        if cache and obj.type == "CURVES":
            positions = _read_cache_frame(Path(cache), scene.frame_current)
            if positions is not None:
                _set_hair_positions(obj, positions)
        collider_cache = obj.get("softコライダーキャッシュ")
        if collider_cache and obj.type == "MESH":
            positions = _read_collider_cache_frame(
                Path(collider_cache), scene.frame_current)
            if positions is not None:
                _set_mesh_positions(obj, positions)


class KAMI3_OT_prepare(Operator):
    bl_idname = "kami_hair_3.prepare"
    bl_label = "髪を準備"
    bl_description = "入力検査、内部有限要素メッシュ、初期接触可能性を準備します"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            prepare_scene(context.scene)
        except Exception as exception:
            self.report({"ERROR"}, str(exception))
            return {"CANCELLED"}
        self.report({"INFO"}, context.scene.kami_hair_3.status)
        return {"FINISHED"}


class KAMI3_OT_bake(Operator):
    bl_idname = "kami_hair_3.bake"
    bl_label = "髪を計算"
    bl_description = "全フレームを非線形有限要素接触ソルバーで計算します"
    bl_options = {"REGISTER"}
    resume: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    def _start_modal(self, context):
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        _ACTIVE_BAKES.add(self._scene_key)
        return {"RUNNING_MODAL"}

    def _execute_resume(self, context, scene, scene_key):
        settings = scene.kami_hair_3
        record = _RESUMABLE_BAKES.get(scene_key)
        if record is None:
            message = "再開できる未完成の計算がありません。"
            settings.error_detail = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}
        session = record["session"]
        try:
            if _SESSIONS.get(scene_key) is not session:
                raise RuntimeError("再開対象のCUDAセッションが失われました。最初から計算してください。")
            if settings.hair != session["hair"] or settings.collider != session["collider"]:
                raise RuntimeError("再開中は入力する髪と衝突メッシュを変更できません。")
            if settings.frame_start != record["start"] or settings.frame_end != record["end"]:
                raise RuntimeError("再開中は開始フレームと終了フレームを変更できません。")
            if settings.maximum_substeps < settings.substeps:
                raise RuntimeError("可変時間ステップ上限は基本サブステップ以上にしてください。")
            path = Path(bpy.path.abspath(settings.cache_path))
            if path != record["path"]:
                raise RuntimeError("再開中は髪キャッシュの保存先を変更できません。")
            advice, changes = _resume_parameter_advice(
                record, settings, settings.resume_frame)
            restart_changes = [change[1] for change in changes if change[2] == "restart"]
            if restart_changes:
                raise RuntimeError(
                    "、".join(restart_changes) + "は構造を変えるため最初から計算してください。")
            current_settings = _settings_snapshot(settings)
            _apply_solver_settings(settings, session["solver"])
            session["solver"].update_runtime_parameters()
            resumed_record = _record_for_resume_frame(record, settings.resume_frame)
            resumed_record["settings_snapshot"] = current_settings
            history = list(record.get("parameter_history", []))
            if changes:
                history.append(_parameter_history_line(settings.resume_frame, changes))
            resumed_record["parameter_history"] = history
            session["parameter_history"] = history
            settings.parameter_history = "\n".join(history)
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exception:
            settings.error_detail = _failure_detail("再開準備", exception)
            settings.status = f"再開失敗: {settings.error_detail}"
            self.report({"ERROR"}, str(exception))
            _append_notification_error(settings, _notify_codex())
            return {"CANCELLED"}

        self._scene = scene
        self._scene_key = scene_key
        self._original_frame = scene.frame_current
        self._phase = "solve"
        self._session = session
        self._path = path
        self._start = record["start"]
        self._end = record["end"]
        self._state = _new_calculation_state(self._start, resume=resumed_record)
        if self._state["final_positions"] is not None:
            _set_hair_positions(session["result"], self._state["final_positions"])
        if self._state["final_collider_positions"] is not None:
            _set_mesh_positions(
                session.get("collider_result"), self._state["final_collider_positions"])
        self._began = time.perf_counter()
        fps = scene.render.fps / scene.render.fps_base
        self._thread = threading.Thread(
            target=_calculate_preloaded,
            args=(session, self._start, self._end, fps, path, self._state),
            name="髪CUDA再開計算", daemon=True)
        self._thread.start()
        settings.error_detail = ""
        settings.status = (
            f"CUDA計算をフレーム{self._state['completed_frame'] + 1}から再開 / {advice}")
        return self._start_modal(context)

    def execute(self, context):
        scene = context.scene
        scene_key = scene.as_pointer()
        if scene_key in _ACTIVE_BAKES:
            self.report({"ERROR"}, "このシーンはCUDA計算中です。")
            return {"CANCELLED"}
        if self.resume:
            return self._execute_resume(context, scene, scene_key)
        self._began = time.perf_counter()
        try:
            prepare_state = _begin_prepare_scene(scene)
        except Exception as exception:
            settings = scene.kami_hair_3
            settings.error_detail = _failure_detail("CUDA準備開始", exception)
            settings.status = f"計算失敗: {settings.error_detail}"
            self.report({"ERROR"}, str(exception))
            _append_notification_error(settings, _notify_codex())
            return {"CANCELLED"}
        self._scene = scene
        self._scene_key = scene_key
        self._prepare_state = prepare_state
        self._original_frame = prepare_state["original_frame"]
        self._phase = "prepare"
        return self._start_modal(context)

    def _redraw(self, context):
        if context.screen is not None:
            for area in context.screen.areas:
                area.tag_redraw()

    def _stop_modal(self, context):
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _ACTIVE_BAKES.discard(self._scene_key)

    def _start_worker(self):
        _finish_prepare_scene(self._scene, self._prepare_state)
        settings = self._scene.kami_hair_3
        session = _SESSIONS[self._scene_key]
        path = Path(bpy.path.abspath(settings.cache_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = self._scene.render.fps / self._scene.render.fps_base
        self._session = session
        self._path = path
        self._start = settings.frame_start
        self._end = settings.frame_end
        self._state = _new_calculation_state(
            self._start, checkpoint_capacity=session["checkpoint_capacity"],
            settings_snapshot=_settings_snapshot(settings),
            parameter_history=session["parameter_history"])
        self._thread = threading.Thread(
            target=_calculate_preloaded,
            args=(session, self._start, self._end, fps, path, self._state),
            name="髪CUDA計算", daemon=True)
        self._thread.start()
        self._phase = "solve"
        settings.status = "CUDA計算開始"

    def modal(self, context, event):
        settings = self._scene.kami_hair_3
        if event.type == "ESC":
            if self._phase == "prepare":
                _abort_prepare_scene(self._scene, self._prepare_state)
                self._stop_modal(context)
                settings.status = "CUDA転送を中止しました"
                self.report({"INFO"}, settings.status)
                return {"CANCELLED"}
            try:
                self._state["cancelled"] = True
                self._session["solver"].cancel()
                settings.status = "CUDA計算の中止を要求しました"
            except Exception as exception:
                settings.status = f"中止要求エラー: {exception}"
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        elapsed = time.perf_counter() - self._began
        if self._phase == "prepare":
            try:
                if self._prepare_state["next_frame_index"] < self._prepare_state["frame_count"]:
                    _advance_prepare_scene(self._scene, self._prepare_state)
                settings.status = (
                    f"CUDAへ転送中 {self._prepare_state['next_frame_index']}/"
                    f"{self._prepare_state['frame_count']} / 経過{_duration_text(elapsed)}")
                self._redraw(context)
                if self._prepare_state["next_frame_index"] < self._prepare_state["frame_count"]:
                    return {"RUNNING_MODAL"}
                self._start_worker()
                return {"RUNNING_MODAL"}
            except Exception as exception:
                _abort_prepare_scene(self._scene, self._prepare_state)
                self._stop_modal(context)
                last_uploaded = (
                    settings.frame_start + self._prepare_state["next_frame_index"] - 1)
                settings.error_detail = _failure_detail(
                    "CUDAアニメーション転送", exception, completed_frame=last_uploaded)
                settings.status = f"計算失敗: {settings.error_detail}"
                self.report({"ERROR"}, str(exception))
                _append_notification_error(settings, _notify_codex())
                return {"CANCELLED"}

        completed = max(0, self._state["completed_frame"] - self._start + 1)
        total = self._end - self._start + 1
        try:
            native_progress = self._session["solver"].progress()
            phase = _PROGRESS_PHASE_NAMES.get(
                int(native_progress.phase), f"フェーズ{int(native_progress.phase)}")
            detail = (
                f"{phase} / 区間 成功{native_progress.accepted_substeps}・"
                f"試行{native_progress.attempted_substeps}/{native_progress.substep_count} / "
                f"sweep縮小{native_progress.sweep_guard_reductions} / "
                f"soft試行{native_progress.soft_collider_attempts} / "
                f"反復 {native_progress.nonlinear_iteration}/{native_progress.nonlinear_iteration_limit}")
        except Exception:
            detail = "GPU状態取得中"
        dynamic_times = self._state["frame_seconds"][1:]
        if dynamic_times:
            remaining = sum(dynamic_times) / len(dynamic_times) * (total - completed)
            remaining_text = f" / 残り約{_duration_text(remaining)}"
        else:
            remaining_text = " / 残り時間を計測中"
        settings.status = (
            f"CUDA計算中 {min(self._state['completed_frame'] + 1, self._end)}/{self._end} / "
            f"{detail} / 経過{_duration_text(elapsed)}{remaining_text}")
        self._redraw(context)

        if not self._state["done"]:
            return {"RUNNING_MODAL"}

        self._thread.join()
        self._stop_modal(context)
        self._scene.frame_set(self._original_frame)
        if self._state["error"] is not None:
            if _store_resume_record(
                    self._scene_key, self._session, self._start, self._end,
                    self._path, self._state):
                settings.resume_frame = self._state["failed_frame"]
            settings.error_detail = _failure_detail(
                self._state["error_stage"], self._state["error"],
                completed_frame=(self._state["completed_frame"]
                                 if self._state["completed_frame"] >= self._start else None),
                failed_frame=self._state["failed_frame"],
                progress=self._state["error_progress"])
            if self._state["cancelled"]:
                settings.status = f"CUDA計算を中止: {settings.error_detail}"
                self.report({"WARNING"}, settings.status)
            else:
                settings.status = f"計算失敗: {settings.error_detail}"
                self.report({"ERROR"}, str(self._state["error"]))
                _append_notification_error(settings, _notify_codex())
            return {"CANCELLED"}

        try:
            _RESUMABLE_BAKES.pop(self._scene_key, None)
            _finish_bake_result(
                self._session, self._path, self._start, self._end,
                self._state["final_positions"], self._state["final_collider_positions"])
            final_stats = self._state["final_stats"]
            if final_stats:
                gpu_stats = self._session["solver"].gpu_stats()
                settings.status = (
                    f"CUDA計算完了 {total}フレーム / {_duration_text(elapsed)} / "
                    f"最終残差 {final_stats.final_residual_norm:.3e} / "
                    f"soft成功 {final_stats.soft_collider_substeps}/"
                    f"{final_stats.soft_collider_attempts}（hard失敗後"
                    f"{final_stats.soft_collider_retry_attempts}） / "
                    f"sweep縮小 {final_stats.sweep_guard_reductions} / "
                    f"hard上限後 {final_stats.hard_iteration_limit_retries} / "
                    f"soft変形 {final_stats.collider_maximum_displacement * 1000.0:.3f} mm / "
                    f"最終GPUフレーム {gpu_stats.last_frame_milliseconds / 1000.0:.2f}秒")
            else:
                settings.status = "CUDA計算完了（開始フレームのみ）"
            settings.error_detail = ""
            _append_notification_error(settings, _notify_codex())
            self.report({"INFO"}, settings.status)
            return {"FINISHED"}
        except Exception as exception:
            settings.error_detail = _failure_detail("計算完了処理", exception)
            settings.status = f"計算終了後エラー: {settings.error_detail}"
            self.report({"ERROR"}, str(exception))
            _append_notification_error(settings, _notify_codex())
            return {"CANCELLED"}


class KAMI3_OT_set_resume_frame(Operator):
    bl_idname = "kami_hair_3.set_resume_frame"
    bl_label = "再開フレームを設定"
    bl_options = {"INTERNAL"}
    target_frame: IntProperty(options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        record = _RESUMABLE_BAKES.get(context.scene.as_pointer())
        valid_range = _resume_frame_range(record) if record else None
        if valid_range is None:
            self.report({"ERROR"}, "再開できるCUDAチェックポイントがありません。")
            return {"CANCELLED"}
        context.scene.kami_hair_3.resume_frame = min(
            valid_range[1], max(valid_range[0], self.target_frame))
        return {"FINISHED"}


class KAMI3_OT_restore_parameters(Operator):
    bl_idname = "kami_hair_3.restore_parameters"
    bl_label = "開始時設定に戻す"
    bl_description = "この計算を開始した時点の物理パラメータへ戻します"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        if scene.as_pointer() in _ACTIVE_BAKES:
            self.report({"ERROR"}, "CUDA計算中は設定を戻せません。")
            return {"CANCELLED"}
        session = _SESSIONS.get(scene.as_pointer())
        if session is None:
            self.report({"ERROR"}, "開始時設定がありません。")
            return {"CANCELLED"}
        settings = scene.kami_hair_3
        for name, value in session["initial_settings_snapshot"].items():
            setattr(settings, name, value)
        settings.status = "計算開始時の物理パラメータへ戻しました"
        return {"FINISHED"}


class KAMI3_OT_clear(Operator):
    bl_idname = "kami_hair_3.clear"
    bl_label = "髪の計算結果を消去"
    bl_description = "このシーンの髪キャッシュと計算結果オブジェクトを消去します"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene_key = scene.as_pointer()
        if scene_key in _ACTIVE_BAKES:
            self.report({"ERROR"}, "CUDA計算中です。Escで中止してから消去してください。")
            return {"CANCELLED"}
        settings = scene.kami_hair_3
        resume = _RESUMABLE_BAKES.pop(scene_key, None)
        if resume:
            temporary = resume["path"].with_suffix(resume["path"].suffix + ".未完成")
            if temporary.exists():
                temporary.unlink()
            collider_path = _collider_cache_path(resume["path"])
            collider_temporary = collider_path.with_suffix(collider_path.suffix + ".未完成")
            if collider_temporary.exists():
                collider_temporary.unlink()
        session = _SESSIONS.pop(scene_key, None)
        if session:
            session["solver"].close()
        result = settings.result
        cache_path = None
        if result:
            cache_path = result.get("髪キャッシュ")
            data = result.data
            bpy.data.objects.remove(result, do_unlink=True)
            if data and data.users == 0:
                bpy.data.hair_curves.remove(data)
        if cache_path:
            path = Path(cache_path)
            if path.exists():
                path.unlink()
        collider_result = settings.collider_result
        collider_cache_path = None
        if collider_result:
            collider_cache_path = collider_result.get("softコライダーキャッシュ")
            data = collider_result.data
            bpy.data.objects.remove(collider_result, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
        if collider_cache_path:
            path = Path(collider_cache_path)
            if path.exists():
                path.unlink()
        settings.result = None
        settings.collider_result = None
        settings.summary = ""
        settings.error_detail = ""
        settings.parameter_history = ""
        settings.status = "未準備"
        return {"FINISHED"}


class KAMI3_PT_panel(Panel):
    bl_idname = "KAMI3_PT_hair_solver"
    bl_label = "髪3・soft実験"
    bl_category = "髪3"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.kami_hair_3
        layout.prop(settings, "hair")
        layout.prop(settings, "collider")
        soft_box = layout.box()
        soft_box.prop(settings, "soft_collider")
        if settings.soft_collider:
            soft_box.prop(settings, "collider_anchor_stiffness")
            soft_box.label(text="実験機能: 結果メッシュと単精度キャッシュを作成", icon="INFO")
        row = layout.row(align=True)
        row.prop(settings, "frame_start")
        row.prop(settings, "frame_end")
        layout.prop(settings, "maximum_element_length")
        layout.prop(settings, "minimum_dynamic_length")
        layout.prop(settings, "fixed_root_nodes")
        layout.prop(settings, "substeps")
        layout.prop(settings, "maximum_substeps")
        layout.prop(settings, "newton_iterations")
        layout.prop(settings, "checkpoint_frames")
        layout.prop(settings, "cache_path")
        layout.prop(settings, "show_advanced", toggle=True)
        if settings.show_advanced:
            box = layout.box()
            box.label(text="髪材料")
            box.prop(settings, "density")
            box.prop(settings, "radius")
            box.prop(settings, "young_modulus")
            box.prop(settings, "poisson_ratio")
            box.prop(settings, "mass_damping")
            box.label(text="接触と摩擦")
            box.prop(settings, "contact_stiffness")
            box.prop(settings, "barrier_distance")
            box.prop(settings, "friction")
            box.prop(settings, "collider_offset")
        row = layout.row(align=True)
        row.operator("kami_hair_3.prepare", icon="MOD_PARTICLES")
        row.operator("kami_hair_3.bake", icon="PHYSICS")
        scene_key = context.scene.as_pointer()
        record = _RESUMABLE_BAKES.get(scene_key)
        if record:
            resume_box = layout.box()
            resume_box.label(text="デバッグ再開", icon="RECOVER_LAST")
            valid_range = _resume_frame_range(record)
            if valid_range:
                resume_box.label(text=f"再開可能: {valid_range[0]}〜{valid_range[1]}")
                resume_box.prop(settings, "resume_frame")
                shortcut_row = resume_box.row(align=True)
                failed = record["failed_frame"]
                for label, offset in (("失敗位置", 0), ("-1", -1), ("-5", -5), ("-10", -10)):
                    operator = shortcut_row.operator(
                        "kami_hair_3.set_resume_frame", text=label)
                    operator.target_frame = failed + offset
                if not valid_range[0] <= settings.resume_frame <= valid_range[1]:
                    resume_box.label(text="再開フレームが保存範囲外です", icon="ERROR")
                advice, _changes = _resume_parameter_advice(record, settings)
                for line in textwrap.wrap(advice, width=38):
                    resume_box.label(text=line, icon="INFO")
                resume_row = resume_box.row()
                resume_row.enabled = (
                    scene_key not in _ACTIVE_BAKES and
                    valid_range[0] <= settings.resume_frame <= valid_range[1])
                resume_operator = resume_row.operator(
                    "kami_hair_3.bake", text="指定フレームから再開", icon="PLAY")
                resume_operator.resume = True
                restore_row = resume_box.row()
                restore_row.enabled = scene_key not in _ACTIVE_BAKES
                restore_row.operator("kami_hair_3.restore_parameters", icon="FILE_REFRESH")
            else:
                resume_box.label(text="再開用CUDA状態がありません", icon="ERROR")
        layout.operator("kami_hair_3.clear", icon="TRASH")
        layout.separator()
        layout.label(text=f"状態: {settings.status}")
        if settings.error_detail:
            box = layout.box()
            box.label(text="エラー詳細", icon="ERROR")
            for line in textwrap.wrap(settings.error_detail, width=42):
                box.label(text=line)
        if settings.summary:
            layout.label(text=settings.summary)
        if settings.parameter_history:
            box = layout.box()
            box.label(text="パラメータ変更履歴", icon="TIME")
            for entry in settings.parameter_history.splitlines():
                for line in textwrap.wrap(entry, width=42):
                    box.label(text=line)


_CLASSES = (
    HairSettings3, KAMI3_OT_prepare, KAMI3_OT_bake, KAMI3_OT_set_resume_frame,
    KAMI3_OT_restore_parameters, KAMI3_OT_clear, KAMI3_PT_panel)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kami_hair_3 = PointerProperty(type=HairSettings3)
    if _apply_hair_cache not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_apply_hair_cache)


def unregister():
    if _apply_hair_cache in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_apply_hair_cache)
    for session in _SESSIONS.values():
        session["solver"].close()
    _SESSIONS.clear()
    _RESUMABLE_BAKES.clear()
    if hasattr(bpy.types.Scene, "kami_hair_3"):
        del bpy.types.Scene.kami_hair_3
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
