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

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */


/*
* https://blog.csdn.net/zcy0xy/article/details/84452266  CUDA samples系列 0.3 vectorAdd
这份代码非常的简单和基础，就把两个向量相加。

CPU与GPU同步方法详解
源代码中的同步
代码很traditional，完全按照五步走，第一篇提到过的：

开辟一块内存空间A（cudaMalloc或者cudaMallocHost，第二个函数开辟的空间可以在cpu和gpu同时访问到，第一个只能由gpu访问，但是第一个要快很多）
把需要运算的数据拷贝到A
GPU执行运算
把运算结果拷贝回到cpu
卸磨杀驴，过河拆桥，释放开辟的空间A（cudaFreeHost(A)，cudaFree(A)）
这里的关键是，2-3-4是怎么保证按照顺序运行的？就是说，我怎么保证2执行完毕了，才能够启动3呢？

要知道在第一篇当中，可是明确地把2-3-4绑定在0号stream上的，很清晰。这里我怎么保证呢？这里的代码没有涉及到stream的编号。那么具体是怎么执行的呢？

这篇官方文档对于stream同步问题做了很好地说明，可以看其中的“CUDA Streams”这一段。

下面开始解释：

这里的内存拷贝函数cudaMemcpy，核函数vectorAdd都没有指定stream，这种情况，所有的GPU相关的代码都被绑定在“stream0”上，所以对于GPU，这三步自然是依次执行的。

对于CPU而言，我们把CPU的这个stream命名为“streamMAX”，第一次内存拷贝是从CPU到GPU，因为牵扯了CPU，所以也隶属于“streamMAX”，第二次从GPU到CPU，同样的CPU也是当事人，所以也隶属于“streamMAX”。



好的，看下这两个stream都有哪些函数：



好的，很明确了，同时隶属于不同 stream的一个函数会等待两个stream完成后执行。

同步方法扩展
明白了这个矢量相加的，我们再来看刚才让大家看的这篇官方文档里的这段有点让人摸不着头脑的代码：



这里我用stream0标注GPU默认的stream，streamMAX表示CPU主程序的stream。

这样就可以画出他们的执行顺序图:



这样就一目了然了，注意我图像标注全黑的部分意思是阻塞等待，什么也不执行。





扩展一：vectorAdd_nvrtc
就是把gpu核函数写在cu文件里，其他代码在cpp里，这样就用系统自带的编译器（而不是cuda C/C++编译器）来编译cpp文件，用nvrtc库函数编译cu文件，加快编译速度。

可以参考第二篇

扩展二：vectorAddDrv
这个是使用cuda的driver api的相关内容，参考官方文档，大概的概念是，相比与runtime API，这是一种更加底层的控制cuda程序的方法，设置也更加复杂。

*/
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
	//计算线程号i，方法为:
	//block的ID*一个block内线程的个数+当前的线程ID
  int i = blockDim.x * blockIdx.x + threadIdx.x;

	//如果当前线程号小于矢量的长度，则进行运算
  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
    int numElements = 50000;//矢量的长度为5万
    size_t size = numElements * sizeof(float);//需要分配的，每个矢量的空间大小
  printf("[Vector addition of %d elements]\n", numElements);

//为A,B,C三个矢量分配内存空间，在cpu上分配
  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

	//A B的值设定为随机值
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
 
	//为ABC三个矢量分配空间，在GPU上分配
  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	//设定block数，线程数；保证总线程数大于矢量的长度，这样每个矢量中的元素都会被计算到
	//每个线程计算矢量中的一个元素
	//这里设定线程与block数是一个常用的方法：
	//先设定threadsPerBlock为某个值，然后计算blocksPerGrid，公式为：
	//blocksPerGrid = （要计算的矢量长度+threadsPerBlock-1）/threadsPerBlock
	//这样可以保证总共可调用线程数>=要计算的矢量长度
  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	//数据拷贝回来
  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	//检查gpu运算结果，如果计算的偏差超过了1e-5则输出提示
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

	//释放gpu,cpu内存空间
  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
