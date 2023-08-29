# deviceQuery - Device Query

## Description

This sample enumerates the properties of the CUDA devices present in the system.

## Key Concepts

CUDA Runtime API, Device Query

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l, aarch64

## CUDA APIs involved

### [CUDA Driver API](http://docs.nvidia.com/cuda/cuda-driver-api/index.html)
cuDeviceGetAttribute, cuSafeCallNoSync

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaRuntimeGetVersion, cudaGetErrorString, cudaDeviceCanAccessPeer, cudaSetDevice, cudaGetDeviceCount, cudaDriverGetVersion, cudaGetDeviceProperties

## Prerequisites

Download and install the [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l, aarch64.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/> `$ make TARGET_ARCH=aarch64` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```

## References (for more details)

1_Utilities__deviceQuery + 1_Utilities__deviceQueryDrv + 1_Utilities__topologyQuery
https://www.cnblogs.com/cuancuancuanhao/p/8040034.html
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