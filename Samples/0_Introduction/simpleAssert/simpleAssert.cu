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
// https ://blog.csdn.net/zcy0xy/article/details/84424182  CUDA samples系列 0.2 simpleAssert

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

/*
在这个库中，有个结构utsname，里面可以获取这些信息。

struct utsname
  { char sysname[_UTSNAME_SYSNAME_LENGTH];//当前操作系统名
   char nodename[_UTSNAME_NODENAME_LENGTH];//网络上的名称
   char release[_UTSNAME_RELEASE_LENGTH];//当前发布级别
   char version[_UTSNAME_VERSION_LENGTH];//当前发布版本
   char machine[_UTSNAME_MACHINE_LENGTH];//当前硬件体系类型

*/


// Includes, system
#include <stdio.h>
#include <cassert>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

const char *sampleName = "simpleAssert";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Tests assert function.
//! Thread whose id > N will print assertion failed error message.
////////////////////////////////////////////////////////////////////////////////
/*
这是核函数，输入的60，计算出线程号,断言线程号<60后。开辟的线程数2*32=64，线程号从0开始，所以60,61,62,63这4个线程号会被断言错误
*/
__global__ void testKernel(int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(gtid < N);
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;
  cudaError_t error;

#ifndef _WIN32
  utsname OS_System_Type;
  uname(&OS_System_Type);

  printf("OS_System_Type.release = %s\n", OS_System_Type.release);

  if (!strcasecmp(OS_System_Type.sysname, "Darwin")) {
    printf("simpleAssert is not current supported on Mac OSX\n\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("OS Info: <%s>\n\n", OS_System_Type.version);
  }

#endif

  // This will pick the best possible CUDA capable device
  findCudaDevice(argc, (const char **)argv);

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  // dim3 这个类型，定义核函数的block个数，thread个数，基本都用这个类型。可以是1,2,3维的，这里是一维的。
  // 实际上，你如果只写一维的话，比如上面代码里dim3 dimGrid(Nblocks)， 它会看作是dim3 dimGrid(Nblocks，1，1)，最后总归是三维的。
  dim3 dimGrid(Nblocks);
  dim3 dimBlock(Nthreads);

  printf("Launch kernel to generate assertion failures\n");
  testKernel<<<dimGrid, dimBlock>>>(60);

  // Synchronize (flushes assert output).
  printf("\n-- Begin assert output\n\n");

  /*
  先看这句。cudaDeviceSynchronize() 会阻塞当前程序的执行，直到所有任务都处理完毕；
  也就是说，程序走到这里，会等待这句话之前的所有代码全部执行完毕了，所有的stream都执行完毕了，
  关于stream，上一篇0.1已经做了详细直观的解释。

  也有与stream绑定的阻塞等待函数：cudaStreamSynchronize(streamID)带有一个参数，cuda流ID，
  它只阻塞那些cuda流ID等于参数中指定ID的那些cuda例程，对于那些流ID不等的例程，还是异步执行的。
    */
  error = cudaDeviceSynchronize();
  printf("\n-- End assert output\n\n");

  // Check for errors and failed asserts in asynchronous kernel launch.
  /*
  这里有个cudaErrorAssert，是个没有定义的，应该是专有名词，最后再cuda官方教程中找到了
  在这里：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
  */
  if (error == cudaErrorAssert) {
    printf(
        "Device assert failed as expected, "
        "CUDA error message is: %s\n\n",
        cudaGetErrorString(error));
  }

  testResult = error == cudaErrorAssert;
}
