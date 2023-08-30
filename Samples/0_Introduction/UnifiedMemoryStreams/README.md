# UnifiedMemoryStreams - Unified Memory Streams

## Description

This sample demonstrates the use of OpenMP and streams with Unified Memory on a single GPU.

## Key Concepts

CUDA Systems Integration, OpenMP, CUBLAS, Multithreading, Unified Memory, CUDA Streams and Events

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamDestroy, cudaFree, cudaMallocManaged, cudaStreamAttachMemAsync, cudaSetDevice, cudaDeviceSynchronize, cudaStreamSynchronize, cudaStreamCreate, cudaGetDeviceProperties

## Dependencies needed to build/run
[OpenMP](../../../README.md#openmp), [UVM](../../../README.md#uvm), [CUBLAS](../../../README.md#cublas)

## Prerequisites

Download and install the [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

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


https://www.cnblogs.com/cuancuancuanhao/p/8012435.html
ʹ�� OpenMP �� pthreads ���ֻ���������ʵ��ͳһ�ڴ��ַ����������ľ���˷� result = �� * A * x + �� * result ��

 �����ƣ�

�� ʹ�� C++ �ṹ�������������ķ��������ڽṹ���ж��幹�캯������������������������

�� ʹ���� cuBLAS �⣬ע������ʹ�úͿ⺯���ĵ��á�

�� �õ��������ڴ�ĺ���


 1 // driver_types.h
 2 #define cudaMemAttachGlobal 0x01  // �ɷ����ڴ�
 3 #define cudaMemAttachHost   0x02  // ���ɷ����ڴ�
 4 #define cudaMemAttachSingle 0x04  // ���߳̿ɷ����ڴ�
 5 
 6 // cuda_runtime.h
 7 template<class T> static __inline__ __host__ cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, T *devPtr, size_t length = 0, unsigned int flags = cudaMemAttachSingle)
 8 {
 9     return ::cudaStreamAttachMemAsync(stream, (void*)devPtr, length, flags);
10 }
11 
12 // cuda_runtime_api.h
13 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags __dv(cudaMemAttachSingle));
