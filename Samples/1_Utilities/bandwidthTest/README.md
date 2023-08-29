# bandwidthTest - Bandwidth Test

## Description

This is a simple test program to measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e. This test application is capable of measuring device to device copy bandwidth, host to device copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for pageable and page-locked memory.

## Key Concepts

CUDA Streams and Events, Performance Strategies

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaHostAlloc, cudaMemcpy, cudaMalloc, cudaMemcpyAsync, cudaFree, cudaGetErrorString, cudaMallocHost, cudaSetDevice, cudaGetDeviceProperties, cudaDeviceSynchronize, cudaEventRecord, cudaFreeHost, cudaEventDestroy, cudaEventElapsedTime, cudaGetDeviceCount, cudaEventCreate

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
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
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

1_Utilities__bandwidthTest  https://www.cnblogs.com/cuancuancuanhao/p/8039654.html
使用三种模式（QUICK_MODE，RANGE_MODE，SHMOO_MODE），测试三种拷贝情况下的显存带宽（HostToDevice，DeviceToHost，DeviceToDevice）
涨姿势
● 申请全局内存时使用新版本的函数 cudaHostAlloc() 与老版本的函数 cudaMallocHost() 性能没有明显差距。
● 申请全局内存时是否指明合并写入标志，性能没有明显差距。
● 对不同大小的内存随便进行拷贝，可以发现带宽随碎片大小的增大而增大，并逐渐趋于稳定。设备之间的内存拷贝比设备与主机之间的拷贝速度高一个量级。
● 从主机内存向设备进行内存拷贝的时候使用了占位内存 h_cacheClear1 和 h_cacheClear2（共 32M 大小），这是为了消除主机内存缓存对内存拷贝的加速作用。