from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-stage", required=True)
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(values)


def point_positions(obj):
    flat = [0.0] * (len(obj.data.points) * 3)
    obj.data.points.foreach_get("position", flat)
    return [tuple(flat[3 * i:3 * i + 3]) for i in range(len(obj.data.points))]


def test_incomplete_cache_resume(addon, directory):
    class FakeSolver:
        def __init__(self):
            self.failed_once = False

        def positions(self):
            return [(1.0, 0.0, 0.0)]

        def step_animation_frame(self, frame, _dt):
            if frame == 2 and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("意図した再開試験エラー")
            return [(float(frame + 1), 0.0, 0.0)], None

    path = directory / "再開試験.khc"
    temporary = path.with_suffix(path.suffix + ".未完成")
    path.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    solver = FakeSolver()
    session = {"solver": solver, "point_count": 1}
    state = addon._new_calculation_state(1)
    addon._calculate_preloaded(session, 1, 3, 24.0, path, state)
    assert state["error"] is not None
    assert state["completed_frame"] == 2
    assert temporary.exists() and not path.exists()

    record = addon._resume_record(session, 1, 3, path, state)
    resumed = addon._new_calculation_state(1, resume=record)
    addon._calculate_preloaded(session, 1, 3, 24.0, path, resumed)
    assert resumed["error"] is None
    assert resumed["completed_frame"] == 3
    assert path.exists() and not temporary.exists()
    assert addon._read_cache_frame(path, 1) == [(1.0, 0.0, 0.0)]
    assert addon._read_cache_frame(path, 2) == [(2.0, 0.0, 0.0)]
    assert addon._read_cache_frame(path, 3) == [(3.0, 0.0, 0.0)]
    path.unlink()


def main():
    args = arguments()
    stage = Path(args.extension_stage).resolve()
    sys.path.insert(0, str(stage.parent))
    import kami_hair_solver
    from kami_hair_solver import addon

    notifications = []
    addon._notify_codex = lambda: notifications.append("PING") or None

    kami_hair_solver.register()
    scene = bpy.context.scene

    hair_data = bpy.data.hair_curves.new("髪_入力データ")
    hair_data.add_curves([5])
    hair = bpy.data.objects.new("髪_入力", hair_data)
    scene.collection.objects.link(hair)
    source = [(0.00, 0.0, 0.20), (0.01, 0.0, 0.20), (0.02, 0.0, 0.20),
              (0.03, 0.0, 0.20), (0.04, 0.0, 0.20)]
    hair_data.points.foreach_set("position", [v for p in source for v in p])
    source_before = point_positions(hair)

    mesh = bpy.data.meshes.new("髪_衝突メッシュデータ")
    collider_xy = [(0.0, 0.0), (2.0, 0.0), (3.0, 1.0),
                   (1.0, 2.0), (-1.0, 1.0)]
    mesh.from_pydata([(x, y, -10.0) for x, y in collider_xy], [],
                     [(0, 1, 2, 3, 4)])
    collider = bpy.data.objects.new("髪_衝突メッシュ", mesh)
    scene.collection.objects.link(collider)
    collider.shape_key_add(name="Basis")
    deform = collider.shape_key_add(name="変形")
    z_offsets = [1.3942679845788373, -4.74989244777333, -2.2497068163088074,
                 -2.7678926185117723, 2.3647121416401244]
    for point, z_offset in zip(deform.data, z_offsets):
        point.co.z = -10.0 + z_offset
    deform.value = 0.0
    deform.keyframe_insert("value", frame=1)
    deform.value = 1.0
    deform.keyframe_insert("value", frame=2)

    settings = scene.kami_hair
    settings.hair = hair
    settings.collider = collider
    settings.frame_start = 1
    settings.frame_end = 2
    settings.substeps = 8
    settings.newton_iterations = 24
    settings.maximum_element_length = 0.021
    settings.minimum_dynamic_length = 0.10
    settings.fixed_root_nodes = 2
    settings.young_modulus = 1.0e7
    settings.cache_path = str(stage.parent / "髪_試験キャッシュ.khc")

    scene.frame_set(1)
    _vertices, frame_1_triangles, frame_1_topology = addon._evaluated_collider(
        collider, bpy.context.evaluated_depsgraph_get())
    scene.frame_set(2)
    _vertices, frame_2_triangles, frame_2_topology = addon._evaluated_collider(
        collider, bpy.context.evaluated_depsgraph_get())
    assert frame_1_triangles != frame_2_triangles
    assert frame_1_topology == frame_2_topology

    stats = addon.prepare_scene(scene)
    assert stats.strand_count == 1
    assert stats.original_point_count == 5
    assert stats.virtual_extension_strand_count == 1
    assert stats.virtual_extension_node_count == 3
    assert abs(stats.virtual_extension_rest_length - 0.06) < 1.0e-6
    assert settings.result is not hair
    assert settings.result.name.startswith("髪_計算結果")
    solver = addon._SESSIONS[scene.as_pointer()]["solver"]
    solver.desc.substeps = 9
    solver.update_runtime_parameters()
    original_element_length = solver.desc.maximum_element_length
    solver.desc.maximum_element_length *= 1.1
    try:
        solver.update_runtime_parameters()
        raise AssertionError("構造パラメーター変更が拒否されませんでした")
    except RuntimeError as exception:
        assert "最初から計算" in str(exception)
    solver.desc.maximum_element_length = original_element_length
    solver.update_runtime_parameters()
    try:
        result = addon.bake_scene(scene)
    except Exception:
        native_stats = addon._SESSIONS[scene.as_pointer()]["solver"].stats()
        print("FAILED_STATS", {name: getattr(native_stats, name) for name, _kind in native_stats._fields_})
        raise
    assert Path(settings.cache_path).exists()
    assert result.get("髪解法")
    assert point_positions(hair) == source_before
    scene.frame_set(1)
    first = point_positions(result)
    scene.frame_set(2)
    second = point_positions(result)
    assert len(first) == len(second) == 5
    assert first[0] == second[0]
    assert settings.status.startswith("計算完了")
    assert addon.KAMI_PT_panel.bl_label == "髪"
    assert notifications == ["PING"]
    test_incomplete_cache_resume(addon, stage.parent)
    successful_status = settings.status
    settings.hair = None
    try:
        addon.bake_scene(scene)
        raise AssertionError("入力エラー試験が失敗しませんでした")
    except RuntimeError as exception:
        assert "Hair Curves" in str(exception)
    assert notifications == ["PING", "PING"]
    assert settings.status.startswith("計算失敗: CUDA準備")
    assert "RuntimeError" in settings.error_detail
    settings.hair = hair
    print({
        "status": successful_status,
        "summary": settings.summary,
        "cache": settings.cache_path,
        "source_unchanged": point_positions(hair) == source_before,
    })
    kami_hair_solver.unregister()


if __name__ == "__main__":
    main()
