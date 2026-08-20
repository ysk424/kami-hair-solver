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
    assert(desc.minimum_dynamic_length == 0.0);
    assert(desc.maximum_substeps == 128);
    assert(desc.newton_iterations == 32);
    assert(material.contact_stiffness == 1.0e5);
    assert(material.barrier_distance == 7.0e-4);
    assert(material.collider_offset == 5.0e-4);
    desc.maximum_substeps = desc.substeps - 1;
    assert(khsCreate(&desc) == 0);
    desc.maximum_substeps = 128;
    desc.minimum_dynamic_length = -1.0;
    assert(khsCreate(&desc) == 0);
    desc.minimum_dynamic_length = 0.0;
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
