# simpleStreams - simpleStreams

## Description

This sample uses CUDA streams to overlap kernel executions with memory copies between the host and a GPU device.  This sample uses a new CUDA 4.0 feature that supports pinning of generic host memory.  Requires Compute Capability 2.0 or higher.

## Key Concepts

Asynchronous Data Transfers, CUDA Streams and Events

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaMemcpy, cudaSetDeviceFlags, cudaSetDevice, cudaEventDestroy, cudaStreamCreate, cudaMallocHost, cudaEventCreateWithFlags, cudaFreeHost, cudaMemcpyAsync, cudaGetDeviceCount, cudaStreamDestroy, cudaMemset, cudaEventElapsedTime, cudaHostAlloc, cudaFree, cudaHostRegister, cudaEventSynchronize, cudaEventRecord, cudaMalloc, cudaGetDeviceProperties, cudaHostUnregister

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

▶ 涨姿势

● 涉及的宏和内部函数原型

复制代码
 1 // driver types.h
 2 #define cudaStreamPerThread                 ((cudaStream_t)0x2)
 3 
 4 #define cudaEventDefault                    0x00  // Default event flag 
 5 #define cudaEventBlockingSync               0x01  // Event uses blocking synchronization 
 6 #define cudaEventDisableTiming              0x02  // Event will not record timing data 
 7 #define cudaEventInterprocess               0x04  // Event is suitable for interprocess use. cudaEventDisableTiming must be set 
 8 
 9 #define cudaDeviceScheduleAuto              0x00  // Device flag - Automatic scheduling 
10 #define cudaDeviceScheduleSpin              0x01  // Device flag - Spin default scheduling 
11 #define cudaDeviceScheduleYield             0x02  // Device flag - Yield default scheduling 
12 #define cudaDeviceScheduleBlockingSync      0x04  // Device flag - Use blocking synchronization 
13 #define cudaDeviceBlockingSync              0x04  // Device flag - Use blocking synchronization 
14                                                      deprecated This flag was deprecated as of CUDA 4.0 and
15                                                      replaced with ::cudaDeviceScheduleBlockingSync. 
16 #define cudaDeviceScheduleMask              0x07  // Device schedule flags mask 
17 #define cudaDeviceMapHost                   0x08  // Device flag - Support mapped pinned allocations 
18 #define cudaDeviceLmemResizeToMax           0x10  // Device flag - Keep local memory allocation after launch 
19 #define cudaDeviceMask                      0x1f  // Device flags mask 
20 
21 #define cudaArrayDefault                    0x00  // Default CUDA array allocation flag 
22 #define cudaArrayLayered                    0x01  // Must be set in cudaMalloc3DArray to create a layered CUDA array 
23 #define cudaArraySurfaceLoadStore           0x02  // Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array 
24 #define cudaArrayCubemap                    0x04  // Must be set in cudaMalloc3DArray to create a cubemap CUDA array 
25 #define cudaArrayTextureGather              0x08  // Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array 
26 
27 #define cudaIpcMemLazyEnablePeerAccess      0x01  // Automatically enable peer access between remote devices as needed 
28 
29 #define cudaMemAttachGlobal                 0x01  // Memory can be accessed by any stream on any device
30 #define cudaMemAttachHost                   0x02  // Memory cannot be accessed by any stream on any device 
31 #define cudaMemAttachSingle                 0x04  // Memory can only be accessed by a single stream on the associated device 
32 
33 #define cudaOccupancyDefault                0x00  // Default behavior 
34 #define cudaOccupancyDisableCachingOverride 0x01  // Assume global caching is enabled and cannot be automatically turned off 
35 
36 #define cudaCpuDeviceId                     ((int)-1) // Device id that represents the CPU 
37 #define cudaInvalidDeviceId                 ((int)-2) // Device id that represents an invalid device 
38 
39 // cuda_runtime_api.h
40 extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags );
41 
42 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
43 
44 extern __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags);
45 
46 extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr);
47 
48 
49 // memoryapi.h
50 WINBASEAPI _Ret_maybenull_ _Post_writable_byte_size_(dwSize) LPVOID WINAPI VirtualAlloc                 \
51 (                                                                                                       \
52     _In_opt_ LPVOID lpAddress, _In_ SIZE_T dwSize, _In_ DWORD flAllocationType, _In_ DWORD flProtect    \
53 );
54 
55 WINBASEAPI BOOL WINAPI VirtualFree  \
56 (
57     _Pre_notnull_ _When_(dwFreeType == MEM_DECOMMIT, _Post_invalid_) _When_(dwFreeType == MEM_RELEASE, _Post_ptr_invalid_) LPVOID lpAddress,
58     _In_ SIZE_T dwSize,
59     _In_ DWORD dwFreeType
60 );
61 
62 // winnt.h
63 #define PAGE_READWRITE  0x04
64 #define MEM_COMMIT      0x1000      
65 #define MEM_RESERVE     0x2000
复制代码
 

● 使用原生页对齐锁定内存的步骤

复制代码
 1 #define CEIL(x,y) (((x) - 1) / (y) + 1)
 2 
 3 int sizeByte = sizeof(int) * 16 * 1024 * 1024;
 4 int align = 4096;
 5 int *p, *pAlign;
 6 p= (int *)VirtualAlloc(NULL, (sizeByte + align), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
 7 pAlign = (int *)CEIL(*p, align);
 8 cudaHostRegister(pAlign, sizeByte, cudaHostRegisterMapped);
 9 
10 ...
11 
12 cudaHostUnregister(pAlign);
13 VirtualFree(p, 0, MEM_RELEASE);
复制代码
 

● 使用函数 cudaEventCreateWithFlags() 相关来计时，与之前的函数 cudaEventCreate() 稍有不同。

复制代码
 1 float elapsed_time = 0.0f;
 2 cudaEvent_t start_event, stop_event;
 3 cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
 4 cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
 5 cudaEventRecord(start_event, 0);
 6     
 7 ...
 8 
 9 cudaEventRecord(stop_event, 0);
10 cudaEventSynchronize(stop_event);
11 cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
12 
13 cudaEventDestroy(start_event);
14 cudaEventDestroy(stop_event);
复制代码