#include <stdio.h>

int main(void){
    int counter, i;
    cudaDeviceProp properties;

    cudaGetDeviceCount(&counter);
    printf("Device count:%d\n", counter);

    for(i=0; i<counter; i++){
        cudaGetDeviceProperties(&properties, i);
        printf("\n\nDEVICE %d: \n",i);
        printf("name: %s\ntotalGlobalMam: %d\nsharedMemPerBlock: %d\nregsPerBlock: %d\nwarpSize: %d\nmemPitch: %d\nmaxThreadsPerBlock: %d\nmaxThreadsDim[3]: %d\nmaxGridSize[3]: %d\ntotalConstMem: %d\nmajor: %d\nminor: %d\nclockRate: %d\ntextureAlignment: %d\ndeviceOverlap: %d\nmultiProcessorCount: %d\nkernelExecTimeoutEnabled: %d\nintegrated: %d\ncanMapHostMemory: %d\ncomputeMode: %d\nconcurrentKernels: %d\nECCEnabled: %d\npciBusID: %d\npciDeviceID: %d\ntccDriver: %d\n",
            properties.name,
            properties.totalGlobalMem,
            properties.sharedMemPerBlock,
            properties.regsPerBlock,
            properties.warpSize,
            properties.memPitch,
            properties.maxThreadsPerBlock,
            properties.maxThreadsDim[3],
            properties.maxGridSize[3],
            properties.totalConstMem,
            properties.major,
            properties.minor,
            properties.clockRate,
            properties.textureAlignment,
            properties.deviceOverlap,
            properties.multiProcessorCount,
            properties.kernelExecTimeoutEnabled,
            properties.integrated,
            properties.canMapHostMemory,
            properties.computeMode,
            properties.concurrentKernels,
            properties.ECCEnabled,
            properties.pciBusID,
            properties.pciDeviceID,
            properties.tccDriver
        );
    }
}