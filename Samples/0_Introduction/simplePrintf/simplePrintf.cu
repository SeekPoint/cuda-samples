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
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

__global__ void testKernel(int val) {
  printf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
         threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x,
         val);
}

int main(int argc, char **argv) {
  int devID;
  cudaDeviceProp props;

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  printf("printf() is called. Output:\n\n");

  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2, 2, 2);
  testKernel<<<dimGrid, dimBlock>>>(10);
  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}

/*
GPU Device 0: "GeForce GTX 1070" with compute capability 6.1

Device 0 : "GeForce GTX 1070" with Compute 6.1 capability
[2, 0] : Value is : 10
[2, 1] : Value is : 10
[2, 2] : Value is : 10
[2, 3] : Value is : 10
[2, 4] : Value is : 10
[2, 5] : Value is : 10
[2, 6] : Value is : 10
[2, 7] : Value is : 10
[3, 0] : Value is : 10
[3, 1] : Value is : 10
[3, 2] : Value is : 10
[3, 3] : Value is : 10
[3, 4] : Value is : 10
[3, 5] : Value is : 10
[3, 6] : Value is : 10
[3, 7] : Value is : 10
[1, 0] : Value is : 10
[1, 1] : Value is : 10
[1, 2] : Value is : 10
[1, 3] : Value is : 10
[1, 4] : Value is : 10
[1, 5] : Value is : 10
[1, 6] : Value is : 10
[1, 7] : Value is : 10
[0, 0] : Value is : 10
[0, 1] : Value is : 10
[0, 2] : Value is : 10
[0, 3] : Value is : 10
[0, 4] : Value is : 10
[0, 5] : Value is : 10
[0, 6] : Value is : 10
[0, 7] : Value is : 10
*/