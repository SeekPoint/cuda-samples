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
// https://blog.csdn.net/zcy0xy/article/details/84335367 CUDA samples系列 0.1 asyncAPI
// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions


/*
接下来，说道每个thread是怎么解析这些参数的，首先括号里的参数(d_a, value)是传给每个进程的，然后在核函数内：

先求出进程编号： int idx = blockIdx.x * blockDim.x + threadIdx.x;  blockDim.x 是一个block里面有多好个进程。

然后进程就可以知道分给我的半亩三分地是哪里了，默默地去干我的活就完事了：

g_data[idx] = g_data[idx] + inc_value;

记住，核函数基本都是传递的内存首地址，到时候直接根据首地址+偏移就可以得到我（某个线程）被分配到的半亩三分地了。
*/
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

//根据 int main(int argc, char *argv[])以及c++的知识,可以知道argc是指的输入参数的个数,
//如果你不输入的话,argc=1,argv="编译得到的你的可执行文件路径";
int main(int argc, char *argv[]) {
  int devID;
  cudaDeviceProp deviceProps;

  printf("[%s] - Starting...\n", argv[0]);

  // This will pick the best possible CUDA capable device  
  //返回最合适的cuda设备；在main函数的开头调用，返回deviceID
  // 如果你不输入编号,则它会找到最大Gflops/s的显卡,也就是浮点数运算速度最快的;
  devID = findCudaDevice(argc, (const char **)argv);

  //cudaGetDeviceProperties(&deviceProps, devID) 顾名思义,根据显卡的ID,得到这块显卡的性质;
  // checkCudaErrors() 很多cuda自带的函数是有状态返回值的,
  // 如果执行错误的话,就返回错误的编号,这个checkCudaErrors()专门用来根据错误的编号显示错误信息,如果没有错误,就通过了,否则中断在这里;
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

  /*两个计时函数
继续我们的代码解析之前,记住2个计时的函数,一个是cpu计时函数,
这个函数在sdkStartTimer(&timer) 以及 sdkStopTimer(&timer) 之间的程序就是总时间,
而这两个函数会在什么时候执行呢? 答案是在主程序运行到这里的时候, 也就是cpu拿到主程序的控制权的时候.

这所以说这么一句废话是因为这段代码并不是像我们以前的c++程序一样,上一句执行完了才进入下一句,.
根据你的设定,你可以让程序像传统的c++一样,等执行完了<GPU代码运行命令>,才会执行下一句sdkStopTimer(&timer)结束计时,这就是"同步执行";
但是你也可以让显卡执行<GPU代码运行命令>,与此同时你的cpu直接执行下面的sdkStopTimer(&timer),这就是"异步执行".
以上的黑体字并不十分准确,但是在此我们先这样理解"同步"与"异步".

    */

  // 声明
  // create cuda event handles  
  cudaEvent_t start, stop;

  // 创建
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU (all to stream 0)
  checkCudaErrors(cudaProfilerStart());
  sdkStartTimer(&timer); // cpu开始计时

  //<GPU代码运行命令>


  /*
  关于stream, 可以暂时这么理解,一个stream就相当于一个独立的main函数的代码,我们运行程序demo,可以打开终端,输入程序名./demo,
  回车,那么多打开几个终端就可以多运行几个程序. 而cuda语言允许我们在一份代码中执行好几个这样独立的主程序,一个stream就是一个main函数整体,
  你可以看到cuda中有隶属于不同stream的代码,你只要记住他们的本质是不同main函数,互相独立,所以cuda并不是我们看到的那样,上一句完毕了才执行下一句.

    这里的计时函数,需要添加stream的标号,因为他是隶属于不同stream的计时程序,只有指定的stream执行到这里了他才会记一下时间,
    其他的程序走到这里他根本不搭理你,就算是天王老子(比如cpu主程序)也不行.

    待会你会看到这2个计时程序位于代码中同样的位置,然而得到的时间却大不相同,原因很简单,因为他们根本就是在为2个独立的程序计时而已.

    有图！！！ https://blog.csdn.net/zcy0xy/article/details/84335367

    可以看出这两个计时函数是在对不同的代码进行计时，因而得到不同的时间结果也就是理所当然的了。
    */


    //当gpu的这个stream执行到这里时,标记一下这个时间点
    //这里的0指的是stream的编号,0号stream
  cudaEventRecord(start, 0);  // Gpu开始计时

  // 销毁
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);

  /*
  四、核函数的调用
        在 CUDA 中，要执行一个核函数，使用以下的语法：
        函数名称<<<block 数目, thread 数目, shared memory 大小， stream标号>>>(参数...);
        block是很多个thread的集合，顾名思义block：块，也就是进程块；
        thread是进程；
        <<<m, n, 0, 0>>>里的第一个参数是总共准备调用m个block，第二个是每个block里有n个进程，所以总共就是m*n个进程。
        第三个参数，共享内存大小，先设置为0。
        第四个参数，就是指定哪个stream了，指明了这个函数隶属于哪个stream。
  */
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0); // Gpu结束计时

  sdkStopTimer(&timer); // cpu结束计时
  checkCudaErrors(cudaProfilerStop());

  // have CPU do some work while waiting for stage 1 to finish
  unsigned long int counter = 0;

  //循环结束标记是检查到stop标记， 只有stream0走到stop才会检查到
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


/*
五、总结
调用gpu函数一般按照五步走：

开辟一块内存空间A（cudaMalloc或者cudaMallocHost，第二个函数开辟的空间可以在cpu和gpu同时访问到，第一个只能由gpu访问，但是第一个要快很多）
把需要运算的数据拷贝到A
执行运算
把运算结果拷贝回到cpu
卸磨杀驴，过河拆桥，释放开辟的空间A（cudaFreeHost(A)，cudaFree(A)）
下面是输出：

[/root/cuda-workspace/asyncAPI/Release/asyncAPI] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

CUDA device [GeForce GTX 1080 Ti]
time spent executing by the GPU: 11.05
time spent by CPU in CUDA calls: 0.03
CPU executed 49566 iterations while waiting for GPU to finish
两个时间不一样，这个说过了。

最后输出的一样，是那个while的输出，知道cpu查询到stream0走到了<stop标记点>，才会退出循环。

这就是第一个例程，比我想象的要难，涉及了不好本应该是中后期的知识点，看来nvidia官方给的这个samples并不是循序渐进的难度。

这是第一篇，希望自己能写下去，肛到底。

（11.23后注：这并不是第一个例程，只是他是按照字母排序的，所以这个asyncAPI是第一个，下面的我先挑挑，先写简单的例程。）


————————————————
版权声明：本文为CSDN博主「zcy0xy」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zcy0xy/article/details/84335367
*/