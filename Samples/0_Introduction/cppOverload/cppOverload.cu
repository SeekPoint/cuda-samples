/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define THREAD_N 256 //每个block里设定256个线程
#define N 1024	//总共计算1024个数
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))//定义函数，这是标准的计算block个数的函数

 //以下引用一段cuda系列第三篇的讲解：
 /*设定block数，线程数；保证总线程数大于矢量的长度，这样每个矢量中的元素都会被计算到
 每个线程计算矢量中的一个元素
 这里设定线程与block数是一个常用的方法：
 先设定threadsPerBlock为某个值，然后计算blocksPerGrid，公式为：
 blocksPerGrid = （要计算的矢量长度+threadsPerBlock-1）/threadsPerBlock
 这样可以保证总共可调用线程数>=要计算的矢量长度 */


 // Includes, system
#include <stdio.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_math.h>
#include "cppOverload_kernel.cuh"

const char* sampleName = "C++ Function Overloading";

#define OUTPUT_ATTR(attr)                                         \
  printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);       \
  printf("Constant Size: %d\n", (int)attr.constSizeBytes);        \
  printf("Local Size:    %d\n", (int)attr.localSizeBytes);        \
  printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock); \
  printf("Number of Registers: %d\n", attr.numRegs);              \
  printf("PTX Version: %d\n", attr.ptxVersion);                   \
  printf("Binary Version: %d\n", attr.binaryVersion);

//三个cpu检查函数，用来检查gpu核函数的运算结果是否足够准确
bool check_func1(int* hInput, int* hOutput, int a)
{
    for (int i = 0; i < N; ++i)
    {
        int cpuRes = hInput[i] * a + i;

        if (hOutput[i] != cpuRes)
        {
            return false;
        }
    }

    return true;
}

bool check_func2(int2* hInput, int* hOutput, int a) {
    for (int i = 0; i < N; i++) {
        int cpuRes = (hInput[i].x + hInput[i].y) * a + i;

        if (hOutput[i] != cpuRes) {
            return false;
        }
    }

    return true;
}

bool check_func3(int* hInput1, int* hInput2, int* hOutput, int a) {
    for (int i = 0; i < N; i++) {
        if (hOutput[i] != (hInput1[i] + hInput2[i]) * a + i) {
            return false;
        }
    }

    return true;
}

int main(int argc, const char* argv[]) {
    int* hInput = NULL;
    int* hOutput = NULL;
    int* dInput = NULL;
    int* dOutput = NULL;

    printf("%s starting...\n", sampleName);

    //获取可用的GPU设备信息
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    printf("Device Count: %d\n", deviceCount);

    int deviceID = findCudaDevice(argc, argv);
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, deviceID));
    if (prop.major < 2) {
        printf(
            "ERROR: cppOverload requires GPU devices with compute SM 2.0 or "
            "higher.\n");
        printf("Current GPU device has compute SM%d.%d, Exiting...", prop.major,
            prop.minor);
        exit(EXIT_WAIVED);
    }

    checkCudaErrors(cudaSetDevice(deviceID));

    // 分配内存空间
  // Allocate device memory
    checkCudaErrors(cudaMalloc(&dInput, sizeof(int) * N * 2));
    checkCudaErrors(cudaMalloc(&dOutput, sizeof(int) * N));

    // Allocate host memory
    checkCudaErrors(cudaMallocHost(&hInput, sizeof(int) * N * 2));
    checkCudaErrors(cudaMallocHost(&hOutput, sizeof(int) * N));

    // 设定数据的数值1-2048
    for (int i = 0; i < N * 2; i++)
    {
        hInput[i] = i;
    }

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(dInput, hInput, sizeof(int) * N * 2, cudaMemcpyHostToDevice));

    // Test C++ overloading
    bool testResult = true;
    bool funcResult = true;
    int a = 1;

    // 3个函数指针
    void (*func1)(const int*, int*, int);
    void (*func2)(const int2*, int*, int);
    void (*func3)(const int*, const int*, int*, int);
    struct cudaFuncAttributes attr;//记录函数属性的结构体

    // 重载第一个函数
    func1 = simple_kernel;//由于func的参数类型与第一个simple_kernel吻合，这里将会与之绑定
    memset(&attr, 0, sizeof(attr));
    //CacheConfig设定为优先使用共享内存
    checkCudaErrors(cudaFuncSetCacheConfig(*func1, cudaFuncCachePreferShared));
    //获取函数的属性
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func1));
    OUTPUT_ATTR(attr);//输出属性信息
    (*func1) << <DIV_UP(N, THREAD_N), THREAD_N >> > (dInput, dOutput, a);
    checkCudaErrors(cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    funcResult = check_func1(hInput, hOutput, a);//cpu检查一下gpu计算的是否够准确
    printf("simple_kernel(const int *pIn, int *pOut, int a) %s\n\n", funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;

    // overload function 2
    func2 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    checkCudaErrors(cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func2));
    OUTPUT_ATTR(attr);
    (*func2) << <DIV_UP(N, THREAD_N), THREAD_N >> > ((int2*)dInput, dOutput, a);
    checkCudaErrors(
        cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    funcResult = check_func2(reinterpret_cast<int2*>(hInput), hOutput, a);
    printf("simple_kernel(const int2 *pIn, int *pOut, int a) %s\n\n",
        funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;

    // overload function 3
    func3 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    checkCudaErrors(cudaFuncSetCacheConfig(*func3, cudaFuncCachePreferShared));
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func3));
    OUTPUT_ATTR(attr);
    (*func3) << <DIV_UP(N, THREAD_N), THREAD_N >> > (dInput, dInput + N, dOutput, a);
    checkCudaErrors(
        cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    funcResult = check_func3(&hInput[0], &hInput[N], hOutput, a);
    printf(
        "simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) "
        "%s\n\n",
        funcResult ? "PASSED" : "FAILED");
    testResult &= funcResult;

    checkCudaErrors(cudaFree(dInput));
    checkCudaErrors(cudaFree(dOutput));
    checkCudaErrors(cudaFreeHost(hOutput));
    checkCudaErrors(cudaFreeHost(hInput));

    checkCudaErrors(cudaDeviceSynchronize());

    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

/*
这份代码介绍了核函数重载的方法，先介绍一些需要的基础知识。

int与int2类型
int是四个字节，32位；
int2类型是2个int，可以分为2个int：

int2 position；
position.x = 1;
position.y = 3;

强制类型转换
reinterpret_cast为强制类型转换符，他只是把指针以新的方式进行解析，并不会做更多的动作。

int *hInput  = NULL;
cudaMallocHost(&hInput , sizeof(int)*N*2);
check_func2(reinterpret_cast<int2 *>(hInput), hOutput, a)；

上面这里只是把hInput以int2进行解析，原来*（hInput+1）向前前进一个int，现在向前前进2个int。这段代码里，最终传进check_func2()的第一个参数，就是 int2*类型了。

函数指针
函数指针的用法为：

int func(int x); /* 声明一个函数，记得定义 */
int (*f) (int x); /* 声明一个函数指针 */
f = func; /* 将func函数的首地址赋给指针f */
(*f) （３）；／／(*f)取地址解析出这个函数，然后直接调用这个函数
————————————————




函数重载
写几个相同函数名，但是不同函数参数的函数就是了。
源代码中有这么几个simple_kernel函数，都是核函数（在GPU上运行的函数）：

__global__ void simple_kernel(const int* pIn, int* pOut, int a)；
__global__ void simple_kernel(const int2* pIn, int* pOut, int a)；
__global__ void simple_kernel(const int* pIn1, const int* pIn2, int* pOut, int a)；

与之配合，定义了3个函数指针：

void (*func1)(const int*, int*, int);
void (*func2)(const int2*, int*, int);
void (*func3)(const int*, const int*, int*, int);

设定核函数属性
用到了一个函数，设定核函数运行时，更多的使用什么内存

cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared);

官方文档介绍了这个函数。


获取函数属性：

memset(&attr, 0, sizeof(attr));
checkCudaErrors(cudaFuncGetAttributes(&attr, *func2));
OUTPUT_ATTR(attr);

输出显示属性的函数如下：

#define OUTPUT_ATTR(attr)  \
    printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);   \
    printf("Constant Size: %d\n", (int)attr.constSizeBytes);                 \
    printf("Local Size:    %d\n", (int)attr.localSizeBytes);                 \
    printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock);          \
    printf("Number of Registers: %d\n", attr.numRegs);                       \
    printf("PTX Version: %d\n", attr.ptxVersion);                            \
    printf("Binary Version: %d\n", attr.binaryVersion);                      \

ShareMemory
之前，每每说到，拷贝到GPU内存空间，都指的是全局内存；
现在介绍下共享内存ShareMemory，他的空间非常小，每个block有一块几十kB的空间，但是速度非常快；不可以用cudaMemcpy来直接复制数据过去，他仅仅是在计算时作为一个储物台存在的。他最大的特点是：可以被同一块block里的任意进程访问到。

这是GPU的硬件逻辑图，其中的SM就是一块block，shared memory非常小，非常快，而且同一个SM里的所有的sp都可以访问到。
比如下面这个核函数：

__global__ void simple_kernel(const int* pIn, int* pOut, int a)
{
    __shared__ int sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = sData[threadIdx.x] * a + tid;;
}
\
这是使用共享内存的方法，直接在核函数里定义一个大小为THREAD_N个int的空间，THREAD_N是设定的一个block有多少个线程，源代码中为#define THREAD_N 256 。

*/