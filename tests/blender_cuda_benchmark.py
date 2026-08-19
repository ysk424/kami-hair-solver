from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import bpy


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-stage", required=True)
    parser.add_argument("--frame-end", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--friction", type=float, default=0.35)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e4)
    parser.add_argument("--barrier-distance", type=float, default=2.0e-4)
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(values)


def main():
    args = arguments()
    stage = Path(args.extension_stage).resolve()
    sys.path.insert(0, str(stage.parent))
    import kami_hair_solver
    from kami_hair_solver import addon

    kami_hair_solver.register()
    scene = bpy.context.scene
    settings = scene.kami_hair
    settings.hair = bpy.data.objects["カーブ"]
    settings.collider = bpy.data.objects["CC_Base_Body"]
    settings.frame_start = 1
    settings.frame_end = args.frame_end
    settings.maximum_element_length = 0.01
    settings.fixed_root_nodes = 2
    settings.substeps = args.substeps
    settings.newton_iterations = args.iterations
    settings.friction = args.friction
    settings.contact_stiffness = args.contact_stiffness
    settings.barrier_distance = args.barrier_distance

    began = time.perf_counter()
    stats = addon.prepare_scene(scene)
    prepared = time.perf_counter()
    session = addon._SESSIONS[scene.as_pointer()]
    gpu_before = session["solver"].gpu_stats()
    print("CUDA_PREPARED", {
        "seconds": prepared - began,
        "resident_gib": gpu_before.resident_bytes / (1024.0 ** 3),
    }, flush=True)
    step_times = []
    positions = session["solver"].positions()
    step = None
    for frame_index in range(1, args.frame_end):
        step_began = time.perf_counter()
        try:
            positions, step = session["solver"].step_animation_frame(frame_index, 1.0 / 24.0)
        except Exception:
            failed = session["solver"].stats()
            print("CUDA_FRAME_FAILED", {
                "frame": frame_index + 1,
                "seconds": time.perf_counter() - step_began,
                "iterations": failed.newton_iterations,
                "candidates": failed.contact_candidate_count,
                "active": failed.active_contact_count,
                "minimum_gap": failed.minimum_gap,
                "residual": failed.final_residual_norm,
                "relative": failed.relative_residual_norm,
                "increment": failed.increment_norm,
                "accepted_step": failed.accepted_step_length,
                "ccd_limit": failed.ccd_step_limit,
                "kinetic": failed.kinetic_energy,
                "elastic": failed.elastic_energy,
                "contact": failed.contact_energy,
                "friction": failed.friction_energy,
            }, flush=True)
            raise
        frame_seconds = time.perf_counter() - step_began
        step_times.append(frame_seconds)
        print("CUDA_FRAME", {
            "frame": frame_index + 1,
            "seconds": frame_seconds,
            "iterations": step.newton_iterations,
            "candidates": step.contact_candidate_count,
            "active": step.active_contact_count,
            "minimum_gap": step.minimum_gap,
            "residual": step.final_residual_norm,
        }, flush=True)
    finished = time.perf_counter()
    gpu_after = session["solver"].gpu_stats()
    print("CUDA_BENCHMARK", {
        "prepare_seconds": prepared - began,
        "step_seconds": finished - prepared,
        "average_frame_seconds": sum(step_times) / len(step_times) if step_times else 0.0,
        "maximum_frame_seconds": max(step_times) if step_times else 0.0,
        "gpu_step_seconds": gpu_after.last_frame_milliseconds / 1000.0,
        "strands": stats.strand_count,
        "nodes": stats.internal_node_count,
        "elements": stats.element_count,
        "resident_gib": gpu_before.resident_bytes / (1024.0 ** 3),
        "iterations": step.newton_iterations if step else 0,
        "linear_solves": step.linear_solves if step else 0,
        "candidates": step.contact_candidate_count if step else 0,
        "active": step.active_contact_count if step else 0,
        "residual": step.final_residual_norm if step else 0.0,
        "positions": len(positions),
    })
    kami_hair_solver.unregister()


if __name__ == "__main__":
    main()
