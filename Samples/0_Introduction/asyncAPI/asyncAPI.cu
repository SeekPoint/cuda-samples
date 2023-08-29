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
// https://zhuanlan.zhihu.com/p/598109614  CUDA Samplesѧϰ�ʼ�: 0_Sample/asyncAPI
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

  // This will pick the best possible CUDA capable device  ��������ʵ�cuda�豸����main�����Ŀ�ͷ���ã�����deviceID
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
������ͨ��CUDA event��չʾGPU��CPU+GPU�غ�����µĳ���ʱ��ͨ����CUDA�����в����¼�(Event), 
��CPU�Ͽ��Բ�ѯCUDA Event�ķ�ʽ��ȷ��GPU�ϵĲ����Ƿ��Ѿ�ִ����ɡ�

������չʾ�첽API��ʹ�÷�����ʹ���첽API��GPU�ں˲�����ͨ��cudaEvent������¼GPU��ִ��ʱ�䡣
ʹ�õ��첽�����У� 1)��Host��device�����ڴ棻 2)kernel����������3)device�ϵ�����copy��host�� 
���ͨ���¼�״̬���ķ�ʽ��¼GPU�ϲ�����ʱ�䣻

���ĸ��

�첽���ݴ��䣺cudaMemcpyAsync
CUDA Stream: A sequence of CUDA commands
Event: cudaEventCreate
Asynchronous Data Transfers, CUDA Streams and Events

���ĺ���
cudaEventCreate(&start)�� 
������һ��GPUʱ��������ʱ��������û�ָ����ʱ����ϼ�¼�ġ�
�ú�������ֱ����GPU�ϼ�¼ʱ�䣻�������첽������ʼʱʹ��cudaEventCreate(&start)����¼��ʼʱ�䣬
���첽�����������ʱͨ��cudaEventCreate(&stop)��¼����ʱ�䣻֮��start��stop�ᱻGPU���ʱ�����
����start/stop֧��ͨ��cudaEventQuery����ѯ״̬��

cudaEventQuery:ͨ����API��ѯevent�Ƿ�ready��������event�Ƿ�ִ����ȷ��stop�¼��Ƿ�GPU��¼��

�������
�¼���cudaEvent_t

�¼���أ�ͨ��cudaEventCreate��������ʼ������Event��Event������ΪcudaEvent_t��

    // create cuda event handles
 cudaEvent_t start, stop;
 checkCudaErrors(cudaEventCreate(&start));
 checkCudaErrors(cudaEventCreate(&stop));
��Ӧ��event��������� cudaEventDestroy

ͬ����
cudaDeviceSynchronize: ��������GPU������ʼ֮ǰ���ô˺�����
�Ժ�������Ϊ������ǰ����ֱ��CUDA device ready��
�ڴ��൱�ڵȴ�ֱ��GPU״̬���ã�

�ڴ���أ�
�������ڴ�ʹ�ö��cuda kernel����ÿ���ڴ��ַ����һ��ֵ���ڴ�����host�Ϸ���ã�Ȼ�����ó�ֵΪ255�� 
֮����host��������ͬ��С�����ݣ��ٽ�host�ϳ�ֵΪ255���ڴ��copy��GPU�ϣ�
֮��ʼkernel�Ĳ���������ÿ���ڴ��ӹ̶�ֵ26����ɺ�������copy��host������correct_output�����ּ�顣

��host�Ϸ����ڴ� cudaAllocHost;
���豸�Ϸ����ڴ�&���ó�ֵ��cudaMalloc; cudaMemset ;
�ڴ�copy��cudaMemcpyAsync��ͨ��cudaMemcpyHostToDevice��cudaMemcpyDeviceToHostָ��copy�ķ���
�ڴ洴��������
cudaAllocHost -> cudaFreeHost (host�ϴ������ͷţ�
cudaAlloc -> CudaFree ��device�ϴ������ͷţ�
 cudaEventRecord(start, 0);
 cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
 increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
 cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
 cudaEventRecord(stop, 0);
���첽����APIִ�����Ժ�ͨ����ѯcuda�¼��±ߴ������ȴ�ִ�е���ɣ�

  while (cudaEventQuery(stop) == cudaErrorNotReady)   {  counter++;}
ִ��ʱ��ļ�¼

����ͨ���±߷�ʽ����¼GPU��CPU�ϵ�ִ��ʱ��

GPU��ִ��ʱ�䣺 ��cudaEventElapsedTime���ص�ʱ�䵥λΪms��

cudaEventElapsedTime(&gpu_time, start, stop)
printf("time spent executing by the GPU: %.2f\n", gpu_time);
CPUִ��ʱ���¼�����ص�λҲ��ms

 StopWatchInterface *timer = NULL;
 sdkCreateTimer(&timer);
 sdkResetTimer(&timer);

 sdkStartTimer(&timer);
// ���ֲ���
 sdkStopTimer(&timer);
 printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
 
 */