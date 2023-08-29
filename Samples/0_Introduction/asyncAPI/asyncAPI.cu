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

/*
 * This sample illustrates the usage of CUDA events for both GPU timing and
 * overlapping CPU and GPU execution.  Events are inserted into a stream
 * of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
 * perform computations while GPU is executing (including DMA memcopies
 * between the host and device).  CPU can query CUDA events to determine
 * whether GPU has completed tasks.
 */
// https://zhuanlan.zhihu.com/p/598109614  CUDA Samples学习笔记: 0_Sample/asyncAPI
// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

__global__ void increment_kernel(int *g_data, int inc_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x) {
  for (int i = 0; i < n; i++)
    if (data[i] != x) {
      printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
      return false;
    }

  return true;
}

int main(int argc, char *argv[]) {
  int devID;
  cudaDeviceProp deviceProps;

  printf("[%s] - Starting...\n", argv[0]);

  // This will pick the best possible CUDA capable device  返回最合适的cuda设备；在main函数的开头调用，返回deviceID
  devID = findCudaDevice(argc, (const char **)argv);

  // get device name
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);

  int n = 16 * 1024 * 1024;
  int nbytes = n * sizeof(int);
  int value = 26;

  // allocate host memory
  int *a = 0;
  checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
  memset(a, 0, nbytes);

  // allocate device memory
  int *d_a = 0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks = dim3(n / threads.x, 1);

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU (all to stream 0)
  checkCudaErrors(cudaProfilerStart());
  sdkStartTimer(&timer);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);
  checkCudaErrors(cudaProfilerStop());

  // have CPU do some work while waiting for stage 1 to finish
  unsigned long int counter = 0;

  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    counter++;
  }

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

  // print the cpu and gpu times
  printf("time spent executing by the GPU: %.2f\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
  printf("CPU executed %lu iterations while waiting for GPU to finish\n",
         counter);

  // check the output for correctness
  bool bFinalResults = correct_output(a, n, value);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));

  exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

/*
本例子通过CUDA event来展示GPU及CPU+GPU重合情况下的程序及时。通过在CUDA调用中插入事件(Event), 
在CPU上可以查询CUDA Event的方式来确定GPU上的操作是否已经执行完成。

本例子展示异步API的使用方法。使用异步API及GPU内核操作，通过cudaEvent来来记录GPU的执行时间。
使用的异步操作有： 1)从Host向device拷贝内存； 2)kernel自增操作；3)device上的数据copy回host。 
最后通过事件状态检查的方式记录GPU上操作的时间；

核心概念：

异步数据传输：cudaMemcpyAsync
CUDA Stream: A sequence of CUDA commands
Event: cudaEventCreate
Asynchronous Data Transfers, CUDA Streams and Events

核心函数
cudaEventCreate(&start)： 
本质是一个GPU时间戳，这个时间戳是在用户指定的时间点上记录的。
该函数可以直接在GPU上记录时间；比如在异步操作开始时使用cudaEventCreate(&start)来记录开始时间，
在异步操作代码结束时通过cudaEventCreate(&stop)记录结束时间；之后start和stop会被GPU填充时间戳，
并且start/stop支持通过cudaEventQuery来查询状态。

cudaEventQuery:通过此API查询event是否ready来看看此event是否被执行来确定stop事件是否被GPU记录。

代码分析
事件：cudaEvent_t

事件相关：通过cudaEventCreate来创建开始及结束Event，Event的类型为cudaEvent_t；

    // create cuda event handles
 cudaEvent_t start, stop;
 checkCudaErrors(cudaEventCreate(&start));
 checkCudaErrors(cudaEventCreate(&stop));
对应的event清理操作： cudaEventDestroy

同步：
cudaDeviceSynchronize: 本例中在GPU操作开始之前调用此函数，
吃函数功能为阻塞当前操作直到CUDA device ready，
在此相当于等待直到GPU状态可用；

内存相关：
本例中内存使用多个cuda kernel来对每个内存地址增加一个值，内存先在host上分配好，然后设置初值为255； 
之后在host上申请相同大小的内容，再将host上初值为255的内存快copy到GPU上；
之后开始kernel的操作，即将每个内存快加固定值26，完成后将数据再copy回host，调用correct_output来逐字检查。

在host上分配内存 cudaAllocHost;
在设备上分配内存&设置初值：cudaMalloc; cudaMemset ;
内存copy：cudaMemcpyAsync，通过cudaMemcpyHostToDevice或cudaMemcpyDeviceToHost指明copy的方向。
内存创建与清理：
cudaAllocHost -> cudaFreeHost (host上创建于释放）
cudaAlloc -> CudaFree （device上创建于释放）
 cudaEventRecord(start, 0);
 cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
 increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
 cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
 cudaEventRecord(stop, 0);
在异步操作API执行完以后通过查询cuda事件下边代码来等待执行的完成：

  while (cudaEventQuery(stop) == cudaErrorNotReady)   {  counter++;}
执行时间的记录

可以通过下边方式来记录GPU及CPU上的执行时间

GPU上执行时间： （cudaEventElapsedTime返回的时间单位为ms）

cudaEventElapsedTime(&gpu_time, start, stop)
printf("time spent executing by the GPU: %.2f\n", gpu_time);
CPU执行时间记录，返回单位也是ms

 StopWatchInterface *timer = NULL;
 sdkCreateTimer(&timer);
 sdkResetTimer(&timer);

 sdkStartTimer(&timer);
// 各种操作
 sdkStopTimer(&timer);
 printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
 
 */