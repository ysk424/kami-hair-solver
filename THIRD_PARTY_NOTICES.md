# Third-party notices

## Eigen

The native solver uses the Eigen C++ template library.

- Project: https://eigen.tuxfamily.org/
- License: Mozilla Public License 2.0
- Source: https://gitlab.com/libeigen/eigen

Eigen is used through its public headers. The project does not modify Eigen. The MPL-2.0 text is available at https://www.mozilla.org/MPL/2.0/.

## NVIDIA CUDA Toolkit and cuBLAS

The Windows CUDA build uses the CUDA Runtime and dynamically links to cuBLAS from the user's CUDA 12.9 installation. NVIDIA components are governed by the NVIDIA CUDA Toolkit End User License Agreement and are not relicensed by this project.

## Acknowledgement (not incorporated code)

The design study for Kami Hair Solver consulted ZOZO, Inc.'s publicly available [ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver), particularly its approaches to contact-barrier stiffness and TOI-limited time advancement. Kami Hair Solver does not incorporate source code from ppf-contact-solver and is independently implemented for hair simulation.

This voluntary acknowledgement does not state or imply endorsement, affiliation, authorship, ownership, or other rights in Kami Hair Solver by ZOZO, Inc. It does not alter the license or copyright ownership of either project. Accordingly, ppf-contact-solver is acknowledged here as a design reference, not listed as a bundled third-party software component.
