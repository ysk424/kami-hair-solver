from __future__ import annotations

from array import array
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
}


def _hair_poll(_self, obj):
    return obj is not None and obj.type == "CURVES"


def _mesh_poll(_self, obj):
    return obj is not None and obj.type == "MESH"


class HairSettings(PropertyGroup):
    hair: PointerProperty(name="入力する髪", type=bpy.types.Object, poll=_hair_poll)
    collider: PointerProperty(name="衝突メッシュ", type=bpy.types.Object, poll=_mesh_poll)
    result: PointerProperty(name="計算結果の髪", type=bpy.types.Object)
    frame_start: IntProperty(name="開始フレーム", default=1, min=-1048574, max=1048574)
    frame_end: IntProperty(name="終了フレーム", default=30, min=-1048574, max=1048574)
    substeps: IntProperty(name="サブステップ", default=8, min=1, max=256)
    newton_iterations: IntProperty(name="Newton反復上限", default=24, min=2, max=100)
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
    contact_stiffness: FloatProperty(name="接触バリア剛性", default=1.0e4, min=1.0e-3, soft_max=1.0e6)
    barrier_distance: FloatProperty(
        name="バリア活性距離", default=2.0e-4, min=1.0e-7, soft_max=0.01,
        subtype="DISTANCE", unit="LENGTH")
    friction: FloatProperty(name="摩擦係数", default=0.35, min=0.0, soft_max=1.5)
    collider_offset: FloatProperty(
        name="コライダー間隔", default=0.0, min=0.0, soft_max=0.01,
        subtype="DISTANCE", unit="LENGTH")
    cache_path: StringProperty(name="髪キャッシュ", default="//髪キャッシュ.khc", subtype="FILE_PATH")
    show_advanced: BoolProperty(name="詳細設定", default=False)
    status: StringProperty(name="状態", default="未準備")
    summary: StringProperty(name="準備情報", default="")
    error_detail: StringProperty(name="エラー詳細", default="")


def _notify_codex():
    """Notify the local terminal helper without changing the simulation result."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1.0)
            client.sendto(b"PING", ("127.0.0.1", 8765))
            response, _remote = client.recvfrom(64)
        if response.strip().upper() != b"PONG":
            return f"8765/UDPから予期しない応答を受信しました: {response!r}"
    except OSError as exception:
        return f"8765/UDP通知に失敗しました: {exception}"
    return None


def _append_notification_error(settings, notification_error):
    if not notification_error:
        return
    if settings.error_detail:
        settings.error_detail += "\n"
    settings.error_detail += notification_error


def _progress_snapshot(solver):
    try:
        progress = solver.progress()
        return {
            "phase": int(progress.phase),
            "substep": int(progress.substep),
            "substep_count": int(progress.substep_count),
            "iteration": int(progress.nonlinear_iteration),
            "iteration_limit": int(progress.nonlinear_iteration_limit),
        }
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
            parts.append(f"サブステップ {progress['substep']}/{progress['substep_count']}")
        if progress["iteration_limit"]:
            parts.append(f"Newton反復 {progress['iteration']}/{progress['iteration_limit']}")
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
    settings = scene.kami_hair
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


def _apply_solver_settings(settings, solver):
    solver.desc.substeps = settings.substeps
    solver.desc.newton_iterations = settings.newton_iterations
    solver.desc.maximum_element_length = settings.maximum_element_length
    solver.desc.minimum_dynamic_length = settings.minimum_dynamic_length
    solver.desc.fixed_root_nodes = settings.fixed_root_nodes
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
    settings = scene.kami_hair
    settings.error_detail = ""
    if settings.hair is None or settings.hair.type != "CURVES":
        raise RuntimeError("入力する Hair Curves を指定してください。")
    if settings.collider is not None and settings.collider.type != "MESH":
        raise RuntimeError("衝突メッシュには Mesh オブジェクトを指定してください。")
    if settings.frame_end < settings.frame_start:
        raise RuntimeError("終了フレームは開始フレーム以降にしてください。")
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
    settings = scene.kami_hair
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
    settings = scene.kami_hair
    solver = state["solver"]
    solver.finalize_animation()
    gpu_info = solver.gpu_info()
    gpu_stats = solver.gpu_stats()
    result = _new_result_object(scene, settings.hair, state["sizes"], state["points"], state["radii"])
    scene_key = scene.as_pointer()
    old_resume = _RESUMABLE_BAKES.pop(scene_key, None)
    if old_resume:
        old_temporary = old_resume["path"].with_suffix(old_resume["path"].suffix + ".未完成")
        try:
            if old_temporary.exists():
                old_temporary.unlink()
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
        "hair": settings.hair,
        "collider": settings.collider,
        "frame_start": settings.frame_start,
        "frame_end": settings.frame_end,
    }
    _SESSIONS[scene_key] = session
    stats = state["stats"]
    memory_mb = stats.estimated_bytes / (1024.0 * 1024.0)
    gpu_mb = gpu_stats.resident_bytes / (1024.0 * 1024.0)
    gpu_name = bytes(gpu_info.device_name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    settings.summary = (
        f"{stats.strand_count}本 / 元{stats.original_point_count}点 / "
        f"内部{stats.internal_node_count}節点 / {stats.element_count}要素 / "
        f"CPU {memory_mb:.1f} MiB / CUDA {gpu_mb:.1f} MiB / {gpu_name}")
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


def _duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}時間{minutes:02d}分"
    if minutes:
        return f"{minutes}分{seconds:02d}秒"
    return f"{seconds}秒"


def _new_calculation_state(start, *, resume=None):
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
        "cancelled": False,
        "resume": bool(resume),
    }


def _open_cache_for_calculation(temporary, start, end, point_count, state):
    if not state["resume"]:
        file = temporary.open("wb")
        file.write(_CACHE_HEADER.pack(_CACHE_MAGIC, start, end, point_count))
        return file

    file = temporary.open("r+b")
    raw = file.read(_CACHE_HEADER.size)
    expected_header = (_CACHE_MAGIC, start, end, point_count)
    if len(raw) != _CACHE_HEADER.size or _CACHE_HEADER.unpack(raw) != expected_header:
        file.close()
        raise RuntimeError("未完成キャッシュのヘッダーが再開対象と一致しません。")
    completed_count = max(0, state["completed_frame"] - start + 1)
    expected_size = _CACHE_HEADER.size + completed_count * point_count * 3 * 8
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
    try:
        state["error_stage"] = "未完成キャッシュを開く処理"
        with _open_cache_for_calculation(
                temporary, start, end, session["point_count"], state) as file:
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
                state["final_positions"] = positions
                state["completed_frame"] = frame
                state["pending_positions"] = None
                state["frame_seconds"].append(time.perf_counter() - frame_began)
                if frame_callback:
                    frame_callback(frame, positions)
        state["error_stage"] = "完成キャッシュの確定"
        os.replace(temporary, path)
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
        "failed_frame": state["failed_frame"],
        "error_stage": state["error_stage"],
        "error_progress": state["error_progress"],
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
    except (OSError, RuntimeError, struct.error):
        _RESUMABLE_BAKES.pop(scene_key, None)
        return False
    _RESUMABLE_BAKES[scene_key] = _resume_record(session, start, end, path, state)
    return True


def _finish_bake_result(session, path, start, end, positions):
    result = session["result"]
    result["髪キャッシュ"] = str(path)
    result["髪開始フレーム"] = start
    result["髪終了フレーム"] = end
    result["髪点数"] = session["point_count"]
    result["髪解法"] = "CUDA・Cosserat FEM・implicit Euler・Gauss-Newton・バリア接触・Coulomb摩擦"
    if positions is not None:
        _set_hair_positions(result, positions)
    return result


def bake_scene(scene, progress=None):
    settings = scene.kami_hair
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
        state = _new_calculation_state(start)

        def frame_finished(frame, positions):
            _set_hair_positions(session["result"], positions)
            done = frame - start + 1
            total = end - start + 1
            remaining = (time.perf_counter() - began) / done * (total - done)
            settings.status = f"計算中 {frame}/{end}（残り約{remaining:.0f}秒）"
            if progress:
                progress(frame, start, end)

        _calculate_preloaded(session, start, end, fps, path, state, frame_finished)
        if state["error"] is not None:
            _store_resume_record(scene_key, session, start, end, path, state)
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
            session, path, start, end, state["final_positions"])
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
        if not cache or obj.type != "CURVES":
            continue
        positions = _read_cache_frame(Path(cache), scene.frame_current)
        if positions is not None:
            _set_hair_positions(obj, positions)


class KAMI_OT_prepare(Operator):
    bl_idname = "kami_hair.prepare"
    bl_label = "髪を準備"
    bl_description = "入力検査、内部有限要素メッシュ、初期接触可能性を準備します"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            prepare_scene(context.scene)
        except Exception as exception:
            self.report({"ERROR"}, str(exception))
            return {"CANCELLED"}
        self.report({"INFO"}, context.scene.kami_hair.status)
        return {"FINISHED"}


class KAMI_OT_bake(Operator):
    bl_idname = "kami_hair.bake"
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
        settings = scene.kami_hair
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
            path = Path(bpy.path.abspath(settings.cache_path))
            if path != record["path"]:
                raise RuntimeError("再開中は髪キャッシュの保存先を変更できません。")
            _apply_solver_settings(settings, session["solver"])
            session["solver"].update_runtime_parameters()
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
        self._state = _new_calculation_state(self._start, resume=record)
        self._began = time.perf_counter()
        fps = scene.render.fps / scene.render.fps_base
        self._thread = threading.Thread(
            target=_calculate_preloaded,
            args=(session, self._start, self._end, fps, path, self._state),
            name="髪CUDA再開計算", daemon=True)
        self._thread.start()
        settings.error_detail = ""
        settings.status = f"CUDA計算をフレーム{self._state['completed_frame'] + 1}から再開"
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
            settings = scene.kami_hair
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
        settings = self._scene.kami_hair
        session = _SESSIONS[self._scene_key]
        path = Path(bpy.path.abspath(settings.cache_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = self._scene.render.fps / self._scene.render.fps_base
        self._session = session
        self._path = path
        self._start = settings.frame_start
        self._end = settings.frame_end
        self._state = _new_calculation_state(self._start)
        self._thread = threading.Thread(
            target=_calculate_preloaded,
            args=(session, self._start, self._end, fps, path, self._state),
            name="髪CUDA計算", daemon=True)
        self._thread.start()
        self._phase = "solve"
        settings.status = "CUDA計算開始"

    def modal(self, context, event):
        settings = self._scene.kami_hair
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
            detail = (
                f"サブステップ {native_progress.substep}/{native_progress.substep_count} / "
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
            _store_resume_record(
                self._scene_key, self._session, self._start, self._end,
                self._path, self._state)
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
                self._state["final_positions"])
            final_stats = self._state["final_stats"]
            if final_stats:
                gpu_stats = self._session["solver"].gpu_stats()
                settings.status = (
                    f"CUDA計算完了 {total}フレーム / {_duration_text(elapsed)} / "
                    f"最終残差 {final_stats.final_residual_norm:.3e} / "
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


class KAMI_OT_clear(Operator):
    bl_idname = "kami_hair.clear"
    bl_label = "髪の計算結果を消去"
    bl_description = "このシーンの髪キャッシュと計算結果オブジェクトを消去します"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        scene_key = scene.as_pointer()
        if scene_key in _ACTIVE_BAKES:
            self.report({"ERROR"}, "CUDA計算中です。Escで中止してから消去してください。")
            return {"CANCELLED"}
        settings = scene.kami_hair
        resume = _RESUMABLE_BAKES.pop(scene_key, None)
        if resume:
            temporary = resume["path"].with_suffix(resume["path"].suffix + ".未完成")
            if temporary.exists():
                temporary.unlink()
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
        settings.result = None
        settings.summary = ""
        settings.error_detail = ""
        settings.status = "未準備"
        return {"FINISHED"}


class KAMI_PT_panel(Panel):
    bl_idname = "KAMI_PT_hair_solver"
    bl_label = "髪"
    bl_category = "髪"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.kami_hair
        layout.prop(settings, "hair")
        layout.prop(settings, "collider")
        row = layout.row(align=True)
        row.prop(settings, "frame_start")
        row.prop(settings, "frame_end")
        layout.prop(settings, "maximum_element_length")
        layout.prop(settings, "minimum_dynamic_length")
        layout.prop(settings, "fixed_root_nodes")
        layout.prop(settings, "substeps")
        layout.prop(settings, "newton_iterations")
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
        row.operator("kami_hair.prepare", icon="MOD_PARTICLES")
        row.operator("kami_hair.bake", icon="PHYSICS")
        resume_row = layout.row()
        resume_row.enabled = (
            context.scene.as_pointer() in _RESUMABLE_BAKES and
            context.scene.as_pointer() not in _ACTIVE_BAKES)
        resume_operator = resume_row.operator("kami_hair.bake", text="失敗フレームから再開", icon="PLAY")
        resume_operator.resume = True
        layout.operator("kami_hair.clear", icon="TRASH")
        layout.separator()
        layout.label(text=f"状態: {settings.status}")
        if settings.error_detail:
            box = layout.box()
            box.label(text="エラー詳細", icon="ERROR")
            for line in textwrap.wrap(settings.error_detail, width=42):
                box.label(text=line)
        if settings.summary:
            layout.label(text=settings.summary)


_CLASSES = (HairSettings, KAMI_OT_prepare, KAMI_OT_bake, KAMI_OT_clear, KAMI_PT_panel)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kami_hair = PointerProperty(type=HairSettings)
    if _apply_hair_cache not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_apply_hair_cache)


def unregister():
    if _apply_hair_cache in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_apply_hair_cache)
    for session in _SESSIONS.values():
        session["solver"].close()
    _SESSIONS.clear()
    _RESUMABLE_BAKES.clear()
    if hasattr(bpy.types.Scene, "kami_hair"):
        del bpy.types.Scene.kami_hair
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
