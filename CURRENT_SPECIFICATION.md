# Kami Hair Solver — Current Specification

**Software version:** 0.4.0<br>
**C ABI version:** 4<br>
**Document status:** Current, as-built specification<br>
**License:** GPL-3.0-or-later

## 1. Purpose and authority

This document describes the behavior implemented in Kami Hair Solver version 0.4.0. It is an as-built record of the current research software, not an initial requirements document or a promise of future behavior. Earlier research plans and design specifications are superseded by this document. If this document and the version 0.4.0 source disagree, the source is authoritative and this document must be corrected.

Kami Hair Solver computes Blender Hair Curves as geometrically nonlinear Cosserat rods with rotational degrees of freedom, implicit time integration, finite-element elasticity, barrier contact, and post-solve Coulomb friction. The production solver is CUDA-only. It does not use PBD, XPBD, position-constraint projection, or a CPU solver fallback.

## 2. Supported environment

- Host platform: Windows x64.
- Blender integration: Blender 5.2 or newer through a Blender Extension.
- Native build: C++17 and CUDA C++17.
- CUDA toolkit used by the release build: CUDA 12.9.
- GPU code target: Compute Capability 12.0 (`sm_120`).
- Tested GPU: NVIDIA GeForce RTX 5070 Ti with 16 GiB VRAM.
- Linear algebra dependency: Eigen 3.3 or newer on the host and cuBLAS on the GPU.
- Numerical precision: double precision for solver state, geometry, cache coordinates, and CUDA computation.

The release build produces `kami_hair_solver.dll` and the Blender Extension archive `kami_hair_solver-0.4.0-windows-x64.zip`.

## 3. Components

- `include/kami_hair_solver.h`: public C ABI.
- `src/solver.cpp`: input validation, curve preprocessing, material/mass construction, collider diagnostics, and CUDA backend setup.
- `src/cuda_backend.cu`: GPU-resident animation state, incremental potential evaluation, nonlinear solve, BVH refit, contact, CCD, and friction.
- `blender_extension/kami_hair_solver/`: Blender UI, evaluated-scene extraction, animation upload, modal execution, result object management, and `.khc` cache I/O.
- `tests/`: C ABI, core solver, CUDA contact, virtual-extension, Blender smoke, and scene benchmark tests.

## 4. Coordinate system, units, and input ownership

- The native API has no coordinate-space metadata; the supported Blender integration supplies Cartesian world-space coordinates and uses meters.
- Time is supplied in seconds.
- Density is in kg/m³, Young's modulus is in Pa, and all contact distances are in meters.
- The Blender Extension evaluates the source Hair Curves and collider through Blender's dependency graph and converts both to world space.
- The source Hair Curves object is never modified. A separate world-space Hair Curves result object is created.
- Each input strand is described by a shared point array and an offset array of `strand_count + 1` entries.
- Hair and collider topology must remain constant over the uploaded animation range.

## 5. Internal hair discretization

### 5.1 Nodes and elements

Each internal rod node has six degrees of freedom: three translations and a three-component rotation vector. Each nonzero source segment is divided into

`ceil(segment_length / maximum_element_length)`

elements, with a minimum of one element. Zero-length source segments are merged, and duplicate source points map to the same internal node. A strand whose points are all coincident is rejected.

Every source point has a mapping to an internal node. Public output and Blender cache output gather only these original mapped positions; subdivision nodes never change the visible point count.

Reference frames are initialized from strand tangents and propagated by parallel transport. Element rest data contains rest length, shear/stretch strain, and curvature/twist strain. Node translational and rotational masses are assembled from cylindrical cross-section properties and distributed equally from adjacent elements.

### 5.2 Fixed roots

The first `fixed_root_nodes` internal nodes of every strand, limited by the strand's node count, are Dirichlet nodes. Their animated positions are evaluated from the original Hair Curves through stored source-point bindings. Their orientations are reconstructed from animated tangents and parallel-transported frames.

An element whose left node is fixed is excluded from barrier contact, moving-collider sweep tests, and line-search CCD. Friction skips elements only when both endpoints are fixed. This behavior keeps the scalp-embedded fixed portion out of the nonlinear contact solve while allowing a boundary element with one free endpoint to receive a friction impulse.

### 5.3 Minimum dynamic length and invisible extension

`minimum_dynamic_length == 0` disables invisible extension.

When a strand's source rest length is shorter than a positive `minimum_dynamic_length`, the solver appends a straight invisible section from the visible tip along the direction of the last nonzero source segment. The appended section reaches the requested total rest length exactly and is subdivided using `maximum_element_length`.

The invisible section is a deliberate surrogate model, not the physical model of an actual short free-ended hair. Its nodes and elements retain:

- translational mass;
- rotational inertia;
- Cosserat shear, stretch, bend, and twist elasticity;
- damping;
- gravity; and
- dynamic coupling to the visible prefix.

The invisible section does not participate in collider interaction. This exclusion applies to initial inside/intersection checks, barrier energy, contact Hessian approximation, moving-collider sweep tests, line-search CCD, and Coulomb friction. The boundary element from the visible tip to the first invisible node is also excluded. The preceding fully visible element still supplies contact at the visible tip.

Invisible nodes are never written to the Blender result object or `.khc` cache.

## 6. Cosserat rod dynamics

The time step is formulated as an incremental potential containing:

- implicit-Euler inertia around a velocity-and-gravity prediction;
- mass-proportional damping relative to the previous position;
- geometrically nonlinear Cosserat rod elastic energy; and
- collider barrier energy for contact-enabled elements.

The material model uses density, physical radius, Young's modulus, Poisson ratio, and a shear-correction factor. Circular cross-section area, second moment, and polar moment are derived from the physical radius. The model has shear/stretch, bend, and twist response. It has no XPBD compliance, constraint projection, plasticity, or strand-level shape matching.

Gravity enters the implicit prediction. Friction is not included in the nonlinear potential; it is applied after a converged substep as a smoothed Coulomb impulse.

## 7. Nonlinear solution

For each substep, the CUDA backend:

1. predicts free-node motion and imposes interpolated root targets;
2. interpolates and refits the animated collider;
3. verifies that the moving collider did not sweep across contact-enabled hair;
4. assembles the incremental potential, gradient, and a positive Gauss-Newton block approximation;
5. solves a per-strand block-tridiagonal linear system;
6. falls back to a diagonally preconditioned steepest-descent direction if the block direction fails or is not a descent direction;
7. limits excessive translation by a trust radius;
8. limits the step with segment-triangle CCD;
9. performs Armijo backtracking line search; and
10. updates velocity and applies friction.

The nonlinear solve fails when it starts outside the barrier-feasible region or cannot find an acceptable line-search step. Reaching the configured Newton iteration budget by itself accepts the last feasible iterate. Cancellation is checked during nonlinear iterations.

### 7.1 Adaptive substeps and rollback

Every animation frame starts with the configured base substep count. The frame's positions and velocities are snapshotted before solving. If a solve attempt fails or a moving collider crosses contact-enabled hair, the entire frame is restored and retried with twice as many substeps.

With a collider, the retry ceiling is four times the configured base count when the base is at most 64, capped globally at 256. Without a collider, no adaptive increase is used. Failure at the ceiling returns an error and leaves the frame rolled back.

## 8. Collider and contact model

### 8.1 Collider preprocessing

The collider is a triangle mesh. Invalid indices, repeated triangle indices, non-finite vertices, and triangles with squared area at or below `1e-24` are excluded. At least one valid triangle is required when a collider is enabled.

The host records boundary edges, nonmanifold edges, inconsistent edge orientation, and inverted orientation of a closed mesh. These are diagnostics; the solver does not repair the mesh. A closed consistently oriented collider enables an initial point-inside test for free, visible hair nodes.

The Blender Extension fixes the first-frame triangle connectivity for the full bake. Evaluated vertex count and polygon/loop topology must remain constant at every uploaded frame. Collider object transforms and modifier/armature deformation are both baked into the uploaded world-space vertices.

### 8.2 BVH and proximity

Collider triangles are organized in a median-split BVH. The topology is built once and its bounds are refitted on the GPU for each collider state. Swept bounds are used for moving-collider checks.

Contact is unsigned segment-triangle proximity. For each contact-enabled rod element, BVH candidates inside the barrier padding are examined and the closest triangle is used for that element's contact contribution during that evaluation.

The gap is

`segment_triangle_distance - hair_radius - collider_offset`.

The state is infeasible when the gap is not greater than `minimum_gap`. For a feasible gap below `barrier_distance`, a logarithmic barrier contributes energy, gradient, and a rank-one positive curvature approximation.

### 8.3 Continuous checks and friction

- Moving-collider conservative advancement rejects a collider surface sweep through stationary hair at the start of a substep.
- Line-search CCD limits hair displacement against the current collider.
- Both checks ignore fixed-root elements and invisible-extension elements.
- Friction uses hair velocity relative to the animated collider surface velocity.
- The friction impulse is bounded by the Coulomb coefficient and an estimate of the barrier normal force.
- Hair-hair and strand self-collision are not implemented.

## 9. GPU-resident animation

Before simulation, the Blender Extension evaluates every frame from `frame_start` through `frame_end` and uploads:

- root target positions and orientations for all internal nodes; and
- collider vertex positions for all frames.

The arrays remain resident on the GPU. Animation frames must be stepped sequentially. Within each frame, root targets and collider vertices are linearly interpolated over the substeps. The frame time is derived from Blender's render FPS and FPS base.

The GPU backend keeps positions, rotations, velocities, rollback snapshots, solver vectors, strand block matrices, animation arrays, collider geometry, BVH data, and output buffers resident. Copying visible results to the host gathers only the original source-point mapping.

There is no CPU simulation path.

## 10. Blender Extension behavior

The Extension is a Japanese-language panel in the 3D View sidebar under the tab `髪`. It exposes source hair, collider, frame range, maximum element length, minimum dynamic length, fixed-root node count, substeps, Newton iteration limit, cache path, and advanced material/contact parameters.

The prepare phase:

1. evaluates the start-frame hair and collider;
2. validates and builds the internal model;
3. allocates animation storage;
4. evaluates and uploads all remaining frames on Blender's main thread;
5. verifies stable hair and collider topology; and
6. creates or replaces the separate result Hair Curves object.

The solve phase runs the native CUDA calculation on a worker thread. The UI reports upload progress, current frame, substep, nonlinear iteration, elapsed time, and an estimated remaining time. Escape requests cancellation. Blender data is not edited from the worker thread.

On a failed solve, the panel reports the last completed frame, failed frame, CUDA phase, substep, Newton iteration, and native exception. The rolled-back GPU state and the complete prefix of the `.未完成` cache are retained. The `失敗フレームから再開` action reapplies current numerical and material/contact parameters and retries the failed frame before continuing sequentially. Source objects, frame range, cache path, maximum element length, minimum dynamic length, and fixed-root node count cannot change during a resume because they define the preloaded animation or internal topology.

Successful completion and non-cancellation errors send `PING` to `127.0.0.1:8765/UDP` and expect `PONG`. Notification failure is displayed without changing the simulation result.

The removed collider-inspection-copy operator is not part of version 0.4.0.

## 11. Default parameters

### 11.1 Solver defaults

| Parameter | Default |
| --- | ---: |
| Gravity | `(0, 0, -9.81)` m/s² |
| Base substeps | 8 |
| Newton iterations | 24 |
| Line-search iterations | 20 |
| Absolute tolerance | `1e-8` |
| Relative tolerance | `1e-5` |
| Increment tolerance | `1e-8` |
| Minimum line-search step | `1e-9` |
| Minimum feasible gap | `1e-7` m |
| Maximum element length | `0.01` m |
| Minimum dynamic length | `0` m (disabled) |
| Fixed root nodes | 2 |

`thread_count` is present in ABI version 4 but is not used by the CUDA backend.

### 11.2 Material and contact defaults

| Parameter | Default |
| --- | ---: |
| Density | `1300` kg/m³ |
| Physical radius | `4e-5` m |
| Young's modulus | `4e9` Pa |
| Poisson ratio | `0.38` |
| Shear correction | `0.9` |
| Mass damping | `8.0` |
| Contact barrier stiffness | `1e4` |
| Barrier activation distance | `2e-4` m |
| Friction coefficient | `0.35` |
| Friction smoothing | `1e-6` |
| Collider offset | `0` m |

## 12. Result and cache format

The result object preserves the source strand count, per-strand visible point count, point radii where available, and material slots. Its point coordinates are world-space simulation output.

The `.khc` cache is a little-endian binary stream:

1. 8-byte magic: `KAMIHC1\0`;
2. unsigned 32-bit frame start;
3. unsigned 32-bit frame end, inclusive;
4. 32-bit visible point count; and
5. for each frame, `point_count * 3` consecutive IEEE-754 64-bit coordinates in XYZ order.

The supported Windows build writes the header with the layout `<8sIII` and writes native 64-bit doubles, which are little-endian on the supported platform.

The bake writes to a sibling file with the suffix `.未完成`. The final cache path is atomically replaced only after every requested frame succeeds. On cancellation or error, the incomplete file and its complete frame prefix are retained for same-session resume; the previous completed cache is left untouched. Resume validates the header and byte length, truncates any partial frame bytes, and appends from the first incomplete frame. Frame-change playback applies only the finalized cache and only when the current frame is inside the cache header's inclusive range.

## 13. C API contract

- Public ABI version: `KHS_ABI_VERSION == 4`.
- Callers obtain defaults, create a solver, set hair and optional collider data, build, then either use per-frame updates or allocate/upload a full animation.
- Full-animation frames must be uploaded completely before finalization and stepped in increasing sequential order.
- Output APIs expose visible/original positions, all internal positions, and original-to-internal mapping.
- Build, step, GPU, and progress structures expose diagnostics and counters.
- `khsGetLastError` returns the solver's most recent diagnostic string.
- `khsRequestCancel` is cooperative and is observed during nonlinear solution.
- `khsUpdateRuntimeParameters` preserves the current simulation state while updating numerical and material/contact parameters. Topology-defining parameters are rejected if changed.

Some ABI fields are reserved by the present implementation: `objective_change`, `friction_energy`, `peak_temporary_bytes`, and the assembly/collision/optimization timing breakdown are not populated with independent measurements in version 0.4.0.

## 14. Failure behavior and limitations

- Visible hair must begin outside the collider with a gap greater than `minimum_gap`; the solver does not project an invalid initial state outward.
- Invisible extension is intentionally excluded from all collider contact and feasibility tests.
- A sufficiently fast or deforming collider may still cross visible hair after the adaptive retry limit.
- A difficult visible contact state may fail Gauss-Newton line search.
- Resume is available only while the Blender session and its GPU solver remain alive; reopening Blender requires a new bake.
- Although the Blender frame properties currently permit negative values, the version 0.4.0 cache header stores frame indices as unsigned 32-bit integers; attempting to bake a negative frame range fails during cache-header encoding.
- Hair-hair collision, self-collision, aerodynamic drag, wind, plasticity, cutting, remeshing during animation, and topology changes are not implemented.
- Contact uses the closest collider triangle per rod element in an evaluation rather than assembling multiple simultaneous triangle contacts for that element.
- The release has no CPU fallback and no binaries for architectures other than Windows x64 / `sm_120`.
- Minimum dynamic length deliberately changes the dynamics of short visible hair and must be treated as an artist-selected surrogate parameter.

## 15. Version 0.4.0 verification record

The following checks passed for the source represented by this specification:

- native core test suite;
- public C API test suite;
- Blender Extension smoke test using Blender 5.2;
- static intersection and moving-collider regression tests proving that invisible extension creates no collider candidates while visible contact tests remain active; and
- a saved production scene benchmark covering frames 1–30.

The production-scene verification used 6,757 strands, 74,327 visible points, 278,919 internal nodes, 272,162 elements, and `minimum_dynamic_length = 0.2 m`. It created 13,811 invisible nodes on 1,408 strands with 131.016 m total invisible rest length. With 8 base substeps and 24 Newton iterations on an RTX 5070 Ti, all 30 frames completed. Preparation took approximately 17.95 seconds and dynamic stepping took approximately 194.65 seconds; resident GPU allocation was approximately 1.003 GiB. These values are a verification snapshot, not a performance guarantee.

## 16. Licensing

Kami Hair Solver is distributed under GPL-3.0-or-later. The Blender Extension package includes the GPL license. Eigen remains under MPL-2.0, and NVIDIA CUDA components remain governed by NVIDIA's applicable licenses. See `THIRD_PARTY_NOTICES.md` for dependency notices.
