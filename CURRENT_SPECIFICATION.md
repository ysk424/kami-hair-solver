# Kami Hair Solver — Current Specification

**Software version:** 0.7.2 experimental<br>
**C ABI version:** 8<br>
**Document status:** Current, as-built specification<br>
**License:** GPL-3.0-or-later

## 1. Purpose and authority

This document describes the behavior implemented in the experimental Kami Hair Solver 3 version 0.7.2. It is an as-built record of the current research software, not an initial requirements document or a promise of future behavior. Earlier research plans and design specifications are superseded by this document. If this document and the version 0.7.2 source disagree, the source is authoritative and this document must be corrected.

Kami Hair Solver computes Blender Hair Curves as geometrically nonlinear Cosserat rods with rotational degrees of freedom, implicit time integration, finite-element elasticity, barrier contact, and post-solve normal/Coulomb contact impulses. The production solver is CUDA-only. It does not use PBD, XPBD, position-constraint projection, or a CPU solver fallback.

## 2. Supported environment

- Host platform: Windows x64.
- Blender integration: Blender 5.2 or newer through a Blender Extension.
- Native build: C++17 and CUDA C++17.
- CUDA toolkit used by the release build: CUDA 12.9.
- GPU code target: Compute Capability 12.0 (`sm_120`).
- Tested GPU: NVIDIA GeForce RTX 5070 Ti with 16 GiB VRAM.
- Linear algebra dependency: Eigen 3.3 or newer on the host and cuBLAS on the GPU.
- Numerical precision: double precision for solver state, geometry, hair-cache coordinates, and CUDA computation. The optional soft-collider display sidecar is 32-bit float.

The release build produces `kami_hair_solver.dll` and the Blender Extension archive `kami_hair_solver_3-0.7.2-windows-x64.zip`.

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

An element whose left node is fixed is excluded from barrier contact, relative-motion sweep tests, and line-search CCD. Contact impulses skip elements only when both endpoints are fixed. This behavior keeps the scalp-embedded fixed portion out of the nonlinear contact solve while allowing a boundary element with one free endpoint to receive a normal/friction impulse.

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

The invisible section does not participate in collider interaction. This exclusion applies to initial inside/intersection checks, barrier energy, contact Hessian approximation, relative-motion sweep tests, line-search CCD, and normal/Coulomb contact impulses. The boundary element from the visible tip to the first invisible node is also excluded. The preceding fully visible element still supplies contact at the visible tip.

Invisible nodes are never written to the Blender result object or `.khc` cache.

## 6. Cosserat rod dynamics

The time step is formulated as an incremental potential containing:

- implicit-Euler inertia around a velocity-and-gravity prediction;
- mass-proportional damping relative to the previous position;
- geometrically nonlinear Cosserat rod elastic energy; and
- collider barrier energy for contact-enabled elements.

The material model uses density, physical radius, Young's modulus, Poisson ratio, and a shear-correction factor. Circular cross-section area, second moment, and polar moment are derived from the physical radius. The model has shear/stretch, bend, and twist response. It has no XPBD compliance, constraint projection, plasticity, or strand-level shape matching.

Gravity enters the implicit prediction. Contact impulses are not included in the nonlinear potential. After a converged interval, a normal impulse removes closing hair-to-collider velocity and a smoothed Coulomb impulse limits tangential slip.

## 7. Nonlinear solution

For each time interval, the CUDA backend:

1. proposes the remaining span of the configured base interval;
2. predicts hair translation from velocity and gravity and reduces the proposal before BVH traversal when the maximum translation exceeds the sweep displacement limit;
3. uses batched conservative advancement to limit the relative hair-collider trajectory to a safe TOI, reducing and retrying a proposal when one hair element would enumerate more than 512 collider candidates;
4. advances the animated roots and collider to the same limited end time;
5. initializes free nodes at the same collision-checked velocity-and-gravity prediction using the corresponding physical `dt`;
6. assembles the incremental potential, gradient, and a positive Gauss-Newton block approximation;
7. solves a per-strand block-tridiagonal linear system;
8. falls back to a diagonally preconditioned steepest-descent direction if the block direction fails or is not a descent direction;
9. limits excessive translation by a trust radius;
10. limits the hair search direction with segment-triangle CCD;
11. performs Armijo backtracking line search; and
12. updates velocity, removes closing normal relative velocity, applies friction, and continues the unconsumed frame time.

The nonlinear solve fails when it starts outside the barrier-feasible region or cannot find an acceptable line-search step. With hybrid soft mode disabled, the established hard-collider path still accepts the last feasible iterate at the configured Newton iteration budget. With hybrid mode enabled, a hard solve that reaches that budget reports non-convergence and is retried in soft mode; a soft solve reaching the budget is rolled back and retried at a smaller physical interval. Cancellation is checked during nonlinear iterations and between batches of the relative-motion sweep.

### 7.1 TOI-limited variable time steps and rollback

The configured base substep count partitions each animation frame into nominal intervals. Before conservative advancement, a linear-time GPU reduction measures the maximum free-node translation predicted by velocity and gravity. A proposal above `max(maximum rest element length, 4 * (radius + collider offset + barrier distance))` is reduced without entering the collider BVH. The moving sweep then runs in batches of 8,192 hair elements. Enumeration stops for an element at 512 candidate triangles; such an over-broad proposal is halved and retried rather than launching an effectively unbounded traversal.

Inside an accepted-size proposal, conservative advancement evaluates the collider trajectory together with the hair trajectory predicted from its current velocity and gravity, and computes a safe temporal fraction before contact. Its safety clearance preserves 10% of the clearance present at the beginning of that proposal, capped by the larger of four minimum gaps and one quarter of the barrier distance; it therefore remains positive without demanding clearance that an already-active contact does not have. The collision-checked hair prediction is also the Newton initial iterate. The collider, root constraints, implicit prediction, velocity update, and contact impulses all use that same fraction, so prescribed and simulated motion remain on one physical clock. The unconsumed portion is then proposed again instead of uniformly subdividing the whole frame.

Positions, velocities, and the collider pose are snapshotted before each accepted variable interval. A nonlinear failure restores only that interval and retries a half-sized span. A fatal error restores the complete frame snapshot. With a collider, the separately configured variable-time-step maximum bounds accepted intervals and local retry attempts; without a collider, the base partition is used directly. The Blender default is 8 base intervals and a maximum of 128; both fields accept explicit values up to 4096. Exhausting the budget or collapsing below the minimum temporal fraction returns an error and leaves the frame rolled back.

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

The state is infeasible when the gap is not greater than `minimum_gap`. For a feasible gap below `barrier_distance`, a logarithmic barrier contributes energy, gradient, and a rank-one positive curvature approximation. The user contact stiffness is a lower bound; the active stiffness is also projected from the contact-point effective mass, physical time interval, and current gap. The mass-over-gap-squared term follows the PPF contact scaling and prevents successively shorter TOI intervals from making contact too weak to advance the hair.

### 8.3 Continuous checks and contact impulses

- Relative-motion conservative advancement sweeps both the predicted hair and animated collider and limits each proposed interval to a safe TOI before solving that interval.
- Line-search CCD limits hair displacement against the current collider. In soft mode, hair and collider directions are scaled together by the same trust-radius factor, and broad phase uses a BVH refitted to the exact swept collider direction rather than applying the largest collider-vertex direction as global padding.
- Both checks ignore fixed-root elements and invisible-extension elements.
- The post-solve normal impulse prevents a contacting hair point from retaining a velocity into the collider and transfers the collider's normal surface velocity to the hair.
- Friction uses hair velocity relative to the animated collider surface velocity.
- The friction impulse is bounded by the Coulomb coefficient and the larger of the normal collision impulse and an estimate of the barrier normal impulse.
- Hair-hair and strand self-collision are not implemented.

### 8.4 Experimental hybrid soft collider

`collider_anchor_stiffness == 0` selects the established prescribed collider. A positive value enables the experimental hybrid mode and is interpreted as an area-density stiffness in N/m³. Each valid input triangle contributes one third of its initial area to each incident vertex. For collider vertex `i`, actual position `x_i`, animation target `x_i*`, and lumped area `A_i`, the added potential is

`E_anchor = 1/2 * collider_anchor_stiffness * sum_i A_i ||x_i - x_i*||²`.

The mode first tries the established prescribed moving-collider sweep for every proposed interval. If the full target motion is safe, the actual collider is set exactly to the animation target and the ordinary hard-collider nonlinear path runs. If that sweep would be TOI-limited, the collider is restored to its last actual state and all collider vertex translations become nonlinear unknowns for that complete proposed interval. If a sweep-safe hard nonlinear solve nevertheless fails because it begins outside the barrier, cannot complete its line search, reaches the Newton iteration budget, or otherwise reports non-convergence, the complete interval is rolled back and retried once in soft mode before the interval is halved.

A soft attempt initializes every free hair node at the preceding feasible position rather than directly at its velocity-and-gravity prediction. The prediction remains the center of the implicit inertial potential, so velocity and gravity still drive the solve while the initial line-search state remains barrier-feasible. Fixed roots are advanced to the interval target as usual. Contact gradients are applied to the hair and, with opposite sign and triangle barycentric weights, to the three collider vertices. Hair and collider directions share one CCD-limited Armijo line search and the collider BVH is refitted for every trial state.

The linear approximation deliberately remains block diagonal between the hair strand solve and independent collider-vertex 3x3 solves; hair-collider off-diagonal blocks and collider vertex-to-vertex blocks are omitted. The converged actual collider persists into following intervals and is included in checkpoints. `soft_collider_attempts` counts all soft nonlinear attempts, `soft_collider_retry_attempts` counts those entered after a failed hard solve, and `soft_collider_substeps` counts completed soft intervals. `collider_anchor_energy` and `collider_maximum_displacement` report the last evaluated state.

This is a quasi-static escape mechanism, not a material soft-body model. Collider inertia, velocity state, membrane/stretch energy, bending energy, volume preservation, plasticity, and collider self-collision are absent. The only internal resistance is the spatially independent area-lumped anchor, so it permits local dents and does not preserve the shape of the body. Contact impulses update hair velocity only; actual collider motion is nevertheless used as the contact surface's relative velocity.

## 9. GPU-resident animation

Before simulation, the Blender Extension evaluates every frame from `frame_start` through `frame_end` and uploads:

- root target positions and orientations for all internal nodes; and
- collider vertex positions for all frames.

The arrays remain resident on the GPU. Animation frames must be stepped sequentially. Within each frame, root targets and collider vertices are linearly interpolated over the substeps. The frame time is derived from Blender's render FPS and FPS base.

The GPU backend keeps positions, rotations, velocities, rollback snapshots, solver vectors, strand block matrices, animation arrays, collider geometry, BVH data, and output buffers resident. Copying visible results to the host gathers only the original source-point mapping.

There is no CPU simulation path.

## 10. Blender Extension behavior

The experimental Extension is a Japanese-language panel in the 3D View sidebar under the tab `髪3`. It has a separate extension ID, Scene property namespace, operators, and panel classes from Kami Hair Solver 2, so both versions can be enabled together. It exposes source hair, collider, frame range, maximum element length, minimum dynamic length, fixed-root node count, base intervals and the variable-time-step maximum, Newton iteration limit, rollback-history length, cache path, advanced material/contact parameters, and the soft-collider switch and anchor density.

The prepare phase:

1. evaluates the start-frame hair and collider;
2. validates and builds the internal model;
3. allocates animation storage;
4. evaluates and uploads all remaining frames on Blender's main thread;
5. verifies stable hair and collider topology; and
6. creates or replaces the separate result Hair Curves object and, when soft mode is enabled, a separate result collider mesh.

The solve phase runs the native CUDA calculation on a worker thread. The UI reports upload progress, current frame, accepted and attempted intervals, sweep reductions, soft attempts, nonlinear iteration, elapsed time, and an estimated remaining time. It distinguishes sweep preflight, relative-motion sweep, line-search CCD, assembly, linear solve, and line search. Escape requests cancellation. Blender data is not edited from the worker thread.

At every completed frame boundary, the Extension stores an opaque native checkpoint containing all internal generalized coordinates, rotations, velocities, the actual collider position, and the current animation index. A bounded in-memory history retains `checkpoint_frames + 1` states; the extra state makes the configured number of backward frame steps available. Checkpoints are valid only for the same live, built CUDA solver.

On a failed solve, the panel reports the last completed frame, failed frame, CUDA phase, attempted variable intervals, TOI-limited attempts, configured base count and maximum, and the native exception. Moving-collider failures additionally report an offending strand, internal element, collider triangle, measured and required distance, safe collider displacement per interval and total frame displacement, and world-space detection coordinates. The complete prefix of the `.未完成` display cache is retained.

The debug-resume box accepts an absolute resume frame inside the retained checkpoint range and provides failed-frame, -1, -5, and -10 shortcuts. Selecting frame `F` restores the complete state at the end of `F - 1`, truncates the incomplete display cache after `F - 1`, and recomputes from `F`. Numerical parameters can retry the same frame. Material and contact changes produce a recommendation to rewind at least one frame. Topology-defining changes are rejected and require a fresh bake. Parameter changes are recorded in the panel, and the calculation's initial parameter values can be restored.

Successful completion and non-cancellation errors send a best-effort `PING` to `127.0.0.1:8765/UDP`. The extension does not wait for a reply and silently ignores notification failure.

## 11. Default parameters

### 11.1 Solver defaults

| Parameter | Default |
| --- | ---: |
| Gravity | `(0, 0, -9.81)` m/s² |
| Base substeps | 8 |
| Maximum variable time steps | 128 |
| Newton iterations | 32 |
| Line-search iterations | 20 |
| Absolute tolerance | `1e-8` |
| Relative tolerance | `1e-5` |
| Increment tolerance | `1e-8` |
| Minimum line-search step | `1e-9` |
| Minimum feasible gap | `1e-7` m |
| Maximum element length | `0.01` m |
| Minimum dynamic length | `0` m (disabled) |
| Fixed root nodes | 2 |
| Collider anchor density | `0` N/m³ (native API disabled) |

`thread_count` is present in ABI version 8 but is not used by the CUDA backend. The Blender Extension exposes soft mode as an opt-in switch and uses `1e8` N/m³ as its initial anchor-density value when enabled.

### 11.2 Material and contact defaults

| Parameter | Default |
| --- | ---: |
| Density | `1300` kg/m³ |
| Physical radius | `4e-5` m |
| Young's modulus | `4e9` Pa |
| Poisson ratio | `0.38` |
| Shear correction | `0.9` |
| Mass damping | `8.0` |
| Contact barrier stiffness | `1e5` |
| Barrier activation distance | `7e-4` m |
| Friction coefficient | `0.35` |
| Friction smoothing | `1e-6` |
| Collider offset | `5e-4` m |

The Blender extension additionally defaults to frames 1 through 100 and retains ten completed frame checkpoints for debugging rollback.

## 12. Result and cache format

The result object preserves the source strand count, per-strand visible point count, point radii where available, and material slots. Its point coordinates are world-space simulation output.

The `.khc` cache is a little-endian binary stream:

1. 8-byte magic: `KAMIHC1\0`;
2. unsigned 32-bit frame start;
3. unsigned 32-bit frame end, inclusive;
4. 32-bit visible point count; and
5. for each frame, `point_count * 3` consecutive IEEE-754 64-bit coordinates in XYZ order.

The supported Windows build writes the header with the layout `<8sIII` and writes native 64-bit doubles, which are little-endian on the supported platform.

The bake writes to a sibling file with the suffix `.未完成`. The final cache path is atomically replaced only after every requested frame succeeds. On cancellation or error, the incomplete file and its complete frame prefix are retained for same-session resume; the previous completed cache is left untouched. Resume validates the header and byte length, truncates complete or partial data after the frame immediately preceding the selected resume frame, and appends recomputed output. The `.khc` stream stores display positions only; safe rewind is provided by the separate same-session opaque CUDA checkpoints. Frame-change playback applies only the finalized cache and only when the current frame is inside the cache header's inclusive range.

Soft mode additionally creates `softコライダー_計算結果` and writes `<hair-cache>.soft-collider`. The sidecar header uses magic `KAMISC1`, the same inclusive frame range, and the collider vertex count, followed by three little-endian IEEE-754 32-bit coordinates per vertex per frame. It is display output rather than restart state; restart uses the opaque double-precision native checkpoint.

## 13. C API contract

- Public ABI version: `KHS_ABI_VERSION == 8`.
- Callers obtain defaults, create a solver, set hair and optional collider data, build, then either use per-frame updates or allocate/upload a full animation.
- Full-animation frames must be uploaded completely before finalization and stepped in increasing sequential order.
- Output APIs expose visible/original positions, all internal positions, original-to-internal mapping, and actual collider positions.
- Build, step, GPU, and progress structures expose diagnostics and counters.
- `khsGetLastError` returns the solver's most recent diagnostic string.
- `khsRequestCancel` is cooperative and is observed during nonlinear solution.
- `khsUpdateRuntimeParameters` preserves the current simulation state while updating numerical and material/contact parameters. Topology-defining parameters are rejected if changed.
- Animation checkpoint APIs return the required opaque byte count and save or restore a frame-boundary state in the same live solver.
- Failure diagnostics identify moving-collider sweep failures and expose attempted variable intervals, TOI-limited attempts, and geometric detection data.

Some ABI fields are reserved by the present implementation: `objective_change`, `friction_energy`, `peak_temporary_bytes`, and the assembly/collision/optimization timing breakdown are not populated with independent measurements in version 0.7.2.

## 14. Failure behavior and limitations

- Visible hair must begin outside the collider with a gap greater than `minimum_gap`; the solver does not project an invalid initial state outward.
- Invisible extension is intentionally excluded from all collider contact and feasibility tests.
- A sufficiently fast or deforming prescribed collider may exhaust the variable-time-step budget before completing a frame. Soft mode avoids some such failures but may itself fail to converge when the anchor/contact compromise has no feasible solution.
- A difficult visible contact state may fail Gauss-Newton line search.
- Resume is available only inside the bounded in-memory checkpoint range and while the Blender session and its GPU solver remain alive; reopening Blender requires a new bake.
- Checkpoint memory is approximately two arrays of six doubles per internal node for every retained state. The panel reports an estimate for the configured history.
- Although the Blender frame properties currently permit negative values, the version 0.7.2 cache headers store frame indices as unsigned 32-bit integers; attempting to bake a negative frame range fails during cache-header encoding.
- Hair-hair collision, self-collision, aerodynamic drag, wind, plasticity, cutting, remeshing during animation, and topology changes are not implemented.
- Contact uses the closest collider triangle per rod element in an evaluation rather than assembling multiple simultaneous triangle contacts for that element.
- The release has no CPU fallback and no binaries for architectures other than Windows x64 / `sm_120`.
- Minimum dynamic length deliberately changes the dynamics of short visible hair and must be treated as an artist-selected surrogate parameter.
- Soft mode is not a substitute for correcting an intersecting initial state. The first frame must still be feasible.
- Soft fallback cost depends on the number of affected intervals and nonlinear/line-search iterations. Each active trial adds O(collider vertices) anchor, direction, position-update, and BVH-refit work.

## 15. Version 0.7.2 verification record

The following checks passed for the source represented by this specification:

- native core test suite;
- public C API test suite;
- Blender Extension smoke test using Blender 5.2;
- opaque CUDA checkpoint save/restore determinism and backward-cache truncation tests;
- a prescribed-crossing regression in which hard motion stops but soft motion remains feasible, deforms the collider, reports a fallback interval, and reproduces the collider state bit-for-bit after checkpoint restore;
- a hair-velocity crossing regression proving that a soft attempt begins at the preceding feasible hair state, plus a forced hard-failure regression proving that the same interval is retried in soft mode before temporal subdivision;
- a hard Newton-budget regression proving that hybrid mode retries the interval in soft mode, and a large-predicted-motion regression proving that the sweep is reduced before any collider candidate traversal;
- moving-collider failure diagnostics, explicit variable-time-step ceiling tests, a 20 cm moving-collider crossing regression using ordinary material values, an active-contact regression whose existing clearance is smaller than the maximum TOI safety clearance, and a multi-frame 0.44 mm/frame normal-velocity transfer regression using the Blender default material scale;
- static intersection and moving-collider regression tests proving that invisible extension creates no collider candidates while visible contact tests remain active; and
- Blender 5.2 extension packaging and installation smoke tests.

The production-scene version-0.7.0 frame 1-to-2 comparison used 6,757 strands, 74,327 visible points, 265,108 internal nodes, 258,351 elements, and a 225,184-vertex body collider. Hard mode used 551,329,796 resident bytes and 266.468 ms GPU frame time. Soft mode with `1e8` N/m³ used 596,366,596 bytes and 260.504 ms. No fallback was needed in that frame, so the actual collider matched its target and the timing difference is measurement noise; the measured fixed memory cost was 45,036,800 bytes (42.95 MiB, 8.17%). Preparation wall time changed from 2.858 s to 3.002 s. This single-frame comparison measures the hybrid fast path, not the cost of an active soft fallback.

In the production scene that originally failed at frame 13, version 0.7.1 with `1e8` N/m³ and a 64-iteration setting completed the frame in 164.10 GPU seconds. It accepted 31 variable intervals, of which 20 were soft, from 62 soft attempts. Maximum final collider displacement was 0.077 mm and minimum feasible gap was 2.25 µm. The preceding frames 11 and 12 took 31.64 and 66.59 wall seconds, compared with 12.07 and 17.89 seconds before feasible-start soft activation. This is the pre-0.7.2 reference measurement. With the original 32-iteration setting frame 13 still exhausted 128 interval attempts; 64 iterations were required for that recorded 0.7.1 case.

The same scene was retested with version 0.7.2, 32 Newton iterations, eight base intervals, and soft density `1e8` N/m³. At the default maximum of 128 variable attempts, frame 13 terminated explicitly after about four minutes with 40 accepted intervals, nine sweep preflight reductions, 11 hard-Newton-budget soft retries, a 12.212 mm maximum prediction, and a 10.000 mm sweep limit. Raising only the maximum to 256 and resuming frame 13 in the same session completed through frame 18. The resumed frame-13-to-18 wall time was 20 minutes 08 seconds. Frame 18 passed the attempt number 45 where version 0.7.1 had remained in one CUDA operation for more than two hours; version 0.7.2 passed it in about 70 seconds and completed the frame in 176.28 GPU seconds with 45 accepted intervals from 125 attempts, 19 preflight reductions, 26,126,515 moving-sweep candidates, four successful soft intervals from 65 soft attempts, and 11 hard-budget retries. Its maximum predicted translation was 17.884 mm against the 10.000 mm guard. Comparison of the soft-collider display cache with animated targets found a peak displacement of 0.174 mm at frame 11 and target agreement to float-output precision at frame 18. This verifies bounded progress and parameter-only recovery, not low cost in severe-contact frames.

The 0.6.2 worktree production scene used 6,757 strands, 74,327 visible points, 265,108 internal nodes, 258,351 elements, and a 225,184-vertex body collider. With the contact defaults adopted by 0.6.3, frames 1 through 30 completed in 6 minutes 36 seconds; the final GPU frame took 11.79 seconds and resident CUDA allocation was approximately 0.986 GiB on an RTX 5070 Ti. This is a stability record, not a performance guarantee.

The preceding 0.5.1 production-scene baseline used 6,757 strands, 74,327 visible points, 278,919 internal nodes, 272,162 elements, and `minimum_dynamic_length = 0.2 m`. It created 13,811 invisible nodes on 1,408 strands with 131.016 m total invisible rest length. With 8 base substeps and 24 Newton iterations on an RTX 5070 Ti, all 30 frames completed. Preparation took approximately 17.95 seconds and dynamic stepping took approximately 194.65 seconds; resident GPU allocation was approximately 1.003 GiB. This historical baseline predates the variable-time-step algorithm and is not a performance guarantee.

## 16. Licensing

Kami Hair Solver is distributed under GPL-3.0-or-later. The Blender Extension package includes the GPL license. Eigen remains under MPL-2.0, and NVIDIA CUDA components remain governed by NVIDIA's applicable licenses. See `THIRD_PARTY_NOTICES.md` for dependency notices.
