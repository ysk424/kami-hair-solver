# Third-party notices

## Eigen

The native solver uses the Eigen C++ template library.

- Project: https://eigen.tuxfamily.org/
- License: Mozilla Public License 2.0
- Source: https://gitlab.com/libeigen/eigen

Eigen is used through its public headers. The project does not modify Eigen. The MPL-2.0 text is available at https://www.mozilla.org/MPL/2.0/.

## NVIDIA CUDA Toolkit and cuBLAS

The Windows CUDA build uses the CUDA Runtime and dynamically links to cuBLAS from the user's CUDA 12.9 installation. NVIDIA components are governed by the NVIDIA CUDA Toolkit End User License Agreement and are not relicensed by this project.
