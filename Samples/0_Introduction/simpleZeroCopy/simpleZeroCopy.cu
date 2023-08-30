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

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// Allocate generic memory with malloc() and pin it laster instead of using
// cudaHostAlloc()
bool bPinGenericMemory = false;

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

int main(int argc, char **argv) {
  int n, nelem, deviceCount;
  int idev = 0;  // use default device 0
  char *device = NULL;
  unsigned int flags;
  size_t bytes;
  float *a, *b, *c;           // Pinned memory allocated on the CPU
  float *a_UA, *b_UA, *c_UA;  // Non-4K Aligned Pinned memory on the CPU
  float *d_a, *d_b, *d_c;     // Device pointers for mapped memory
  float errorNorm, refNorm, ref, diff;
  cudaDeviceProp deviceProp;

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Usage:  simpleZeroCopy [OPTION]\n\n");
    printf("Options:\n");
    printf("  --device=[device #]  Specify the device to be used\n");
    printf(
        "  --use_generic_memory (optional) use generic page-aligned for system "
        "memory\n");
    return EXIT_SUCCESS;
  }

  /* Get the device selected by the user or default to 0, and then set it. */
  if (getCmdLineArgumentString(argc, (const char **)argv, "device", &device)) {
    cudaGetDeviceCount(&deviceCount);
    idev = atoi(device);

    if (idev >= deviceCount || idev < 0) {
      fprintf(stderr,
              "Device number %d is invalid, will use default CUDA device 0.\n",
              idev);
      idev = 0;
    }
  }

  // if GPU found supports SM 1.2, then continue, otherwise we exit
  if (!checkCudaCapabilities(1, 2)) {
    exit(EXIT_SUCCESS);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "use_generic_memory")) {
#if defined(__APPLE__) || defined(MACOSX)  // MacOS 系统不支持将普通堆内存设置为页锁定内存
    bPinGenericMemory = false;  // Generic Pinning of System Paged memory is not
                                // currently supported on Mac OSX
#else
    bPinGenericMemory = true;
#endif
  }

  if (bPinGenericMemory) {
    printf("> Using Generic System Paged Memory (malloc)\n");
  } else {
    printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");
  }

  checkCudaErrors(cudaSetDevice(idev));

  /* Verify the selected device supports mapped memory and set the device
     flags for mapping host memory. */

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, idev));

#if CUDART_VERSION >= 2020

  if (!deviceProp.canMapHostMemory) {
    fprintf(stderr, "Device %d does not support mapping CPU host memory!\n",
            idev);

    exit(EXIT_SUCCESS);
  }

  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));  // MapHostFlag 功能正常，设置标志
#else
  fprintf(stderr,
          "CUDART version %d.%d does not support "
          "<cudaDeviceProp.canMapHostMemory> field\n",
          , CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);

  exit(EXIT_SUCCESS);
#endif
// 总体逻辑：
// CUDA Runtime version < 2200，不支持 MApHostMamory，退出
// CUDA Runtime version ∈[2200, 4000)，且为 MAcOS 系统，使用 cudaHostAlloc() + cudaHostAllocMapped
// CUDA Runtime version ∈[2200, 4000)，且不是 MAcOS 系统，退出
// CUDA Runtime version ≥ 4000，使用 malloc() + cudaHostRegister()
#if CUDART_VERSION < 4000

  if (bPinGenericMemory) {
    fprintf(
        stderr,
        "CUDART version %d.%d does not support <cudaHostRegister> function\n",
        CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);

    exit(EXIT_SUCCESS);
  }

#endif

  /* Allocate mapped CPU memory. */
  // 内存申请
  nelem = 1048576;
  bytes = nelem * sizeof(float);

  if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000
    a_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);  // 申请时多 4KB，用于滑动对齐，释放内存时以该指针为准
    b_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);
    c_UA = (float *)malloc(bytes + MEMORY_ALIGNMENT);

    // We need to ensure memory is aligned to 4K (so we will need to padd memory
    // accordingly)
    a = (float *)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);  // 指针指到 4K 对齐的位置上去，用于计算
    b = (float *)ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
    c = (float *)ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

    checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped));  // 设置页锁定内存
    checkCudaErrors(cudaHostRegister(b, bytes, cudaHostRegisterMapped));
    checkCudaErrors(cudaHostRegister(c, bytes, cudaHostRegisterMapped));
#endif
  } else {
#if CUDART_VERSION >= 2020
    flags = cudaHostAllocMapped;
    checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags));   // 使用函数 cudaHostAlloc() 一步到位
    checkCudaErrors(cudaHostAlloc((void **)&b, bytes, flags));
    checkCudaErrors(cudaHostAlloc((void **)&c, bytes, flags));
#endif
  }

  /* Initialize the vectors. */
  // 初始化和内存映射
  for (n = 0; n < nelem; n++) {
    a[n] = rand() / (float)RAND_MAX;
    b[n] = rand() / (float)RAND_MAX;
  }

    /* Get the device pointers for the pinned CPU memory mapped into the GPU
       memory space. */

#if CUDART_VERSION >= 2020
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
  checkCudaErrors(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));
#endif

  /* Call the GPU kernel using the CPU pointers residing in CPU mapped memory.
  * // 调用内核
   */
  printf("> vectorAddGPU kernel will add vectors using mapped CPU memory...\n");
  dim3 block(256);
  dim3 grid((unsigned int)ceil(nelem / (float)block.x));
  vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, nelem);
  checkCudaErrors(cudaDeviceSynchronize());
  getLastCudaError("vectorAddGPU() execution failed");

  /* Compare the results */

  printf("> Checking the results from vectorAddGPU() ...\n");
  // 检查结果
  errorNorm = 0.f;
  refNorm = 0.f;

  for (n = 0; n < nelem; n++) {
    // ref 为 CPU 计算的和，diff 为 GPU 计算结果与 CPU 计算结果的差          
    ref = a[n] + b[n];  
    diff = c[n] - ref;
    errorNorm += diff * diff;  // 向量 a + b 的两种计算结果的差的平方
    refNorm += ref * ref;      // 向量 a 与向量 b 的和的平方
  }

  errorNorm = (float)sqrt((double)errorNorm);
  refNorm = (float)sqrt((double)refNorm);

  /* Memory clean up */

  printf("> Releasing CPU memory...\n");

  if (bPinGenericMemory) {
#if CUDART_VERSION >= 4000  // 清理工作
    checkCudaErrors(cudaHostUnregister(a));
    checkCudaErrors(cudaHostUnregister(b));
    checkCudaErrors(cudaHostUnregister(c));
    free(a_UA);
    free(b_UA);
    free(c_UA);
#endif
  } else {
#if CUDART_VERSION >= 2020
    checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFreeHost(b));
    checkCudaErrors(cudaFreeHost(c));
#endif
  }

  exit(errorNorm / refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
