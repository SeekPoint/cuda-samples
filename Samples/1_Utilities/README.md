# 1. Utilities


### [bandwidthTest](./bandwidthTest)
This is a simple test program to measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e. This test application is capable of measuring device to device copy bandwidth, host to device copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for pageable and page-locked memory.

### [deviceQuery](./deviceQuery)
This sample enumerates the properties of the CUDA devices present in the system.

### [deviceQueryDrv](./deviceQueryDrv)
This sample enumerates the properties of the CUDA devices present using CUDA Driver API calls

### [topologyQuery](./topologyQuery)
A simple exemple on how to query the topology of a system with multiple GPU



涨姿势：
● Runtime API 比 Driver API 写起来更简单，且能直接检测的内容不少于 Driver API。
● 用到的 Runtime API 函数和 Driver API 函数。
 1 // cuda_runtime_api.h
 2 extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion);
 3 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion);
 4 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
 5 
 6 // cuda_device_runtime_api.h
 7 #define __NV_WEAK__ __declspec(nv_weak)
 8 __device__ __NV_WEAK__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
 9 
10 // cuda.h
11 CUresult CUDAAPI cuDeviceGetCount(int *count);
12 CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
13 CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
14 CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev);