#include "kami_hair_solver.h"

#include <assert.h>

int main(void)
{
    KhsSolverDesc desc;
    KhsHairMaterial material;
    khsDefaultSolverDesc(&desc);
    khsDefaultHairMaterial(&material);
    assert(khsGetAbiVersion() == KHS_ABI_VERSION);
    assert(desc.struct_size == sizeof(desc));
    assert(material.struct_size == sizeof(material));
    KhsGpuInfo gpu = {0};
    gpu.struct_size = sizeof(gpu);
    assert(khsGetGpuInfo(&gpu) == KHS_OK);
    assert(gpu.available == 1);
    assert(gpu.compute_capability_major >= 12);
    assert(gpu.total_vram_bytes > 0);
    KhsSolver *solver = khsCreate(&desc);
    assert(solver != 0);
    khsDestroy(solver);
    return 0;
}
