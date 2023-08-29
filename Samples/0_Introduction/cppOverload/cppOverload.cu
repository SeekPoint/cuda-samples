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

#define THREAD_N 256 //ÿ��block���趨256���߳�
#define N 1024	//�ܹ�����1024����
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))//���庯�������Ǳ�׼�ļ���block�����ĺ���

 //��������һ��cudaϵ�е���ƪ�Ľ��⣺
 /*�趨block�����߳�������֤���߳�������ʸ���ĳ��ȣ�����ÿ��ʸ���е�Ԫ�ض��ᱻ���㵽
 ÿ���̼߳���ʸ���е�һ��Ԫ��
 �����趨�߳���block����һ�����õķ�����
 ���趨threadsPerBlockΪĳ��ֵ��Ȼ�����blocksPerGrid����ʽΪ��
 blocksPerGrid = ��Ҫ�����ʸ������+threadsPerBlock-1��/threadsPerBlock
 �������Ա�֤�ܹ��ɵ����߳���>=Ҫ�����ʸ������ */


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

//����cpu��麯�����������gpu�˺������������Ƿ��㹻׼ȷ
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

    //��ȡ���õ�GPU�豸��Ϣ
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

    // �����ڴ�ռ�
  // Allocate device memory
    checkCudaErrors(cudaMalloc(&dInput, sizeof(int) * N * 2));
    checkCudaErrors(cudaMalloc(&dOutput, sizeof(int) * N));

    // Allocate host memory
    checkCudaErrors(cudaMallocHost(&hInput, sizeof(int) * N * 2));
    checkCudaErrors(cudaMallocHost(&hOutput, sizeof(int) * N));

    // �趨���ݵ���ֵ1-2048
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

    // 3������ָ��
    void (*func1)(const int*, int*, int);
    void (*func2)(const int2*, int*, int);
    void (*func3)(const int*, const int*, int*, int);
    struct cudaFuncAttributes attr;//��¼�������ԵĽṹ��

    // ���ص�һ������
    func1 = simple_kernel;//����func�Ĳ����������һ��simple_kernel�Ǻϣ����ｫ����֮��
    memset(&attr, 0, sizeof(attr));
    //CacheConfig�趨Ϊ����ʹ�ù����ڴ�
    checkCudaErrors(cudaFuncSetCacheConfig(*func1, cudaFuncCachePreferShared));
    //��ȡ����������
    checkCudaErrors(cudaFuncGetAttributes(&attr, *func1));
    OUTPUT_ATTR(attr);//���������Ϣ
    (*func1) << <DIV_UP(N, THREAD_N), THREAD_N >> > (dInput, dOutput, a);
    checkCudaErrors(cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    funcResult = check_func1(hInput, hOutput, a);//cpu���һ��gpu������Ƿ�׼ȷ
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
��ݴ�������˺˺������صķ������Ƚ���һЩ��Ҫ�Ļ���֪ʶ��

int��int2����
int���ĸ��ֽڣ�32λ��
int2������2��int�����Է�Ϊ2��int��

int2 position��
position.x = 1;
position.y = 3;

ǿ������ת��
reinterpret_castΪǿ������ת��������ֻ�ǰ�ָ�����µķ�ʽ���н�����������������Ķ�����

int *hInput  = NULL;
cudaMallocHost(&hInput , sizeof(int)*N*2);
check_func2(reinterpret_cast<int2 *>(hInput), hOutput, a)��

��������ֻ�ǰ�hInput��int2���н�����ԭ��*��hInput+1����ǰǰ��һ��int��������ǰǰ��2��int����δ�������մ���check_func2()�ĵ�һ������������ int2*�����ˡ�

����ָ��
����ָ����÷�Ϊ��

int func(int x); /* ����һ���������ǵö��� */
int (*f) (int x); /* ����һ������ָ�� */
f = func; /* ��func�������׵�ַ����ָ��f */
(*f) ������������(*f)ȡ��ַ���������������Ȼ��ֱ�ӵ����������
��������������������������������




��������
д������ͬ�����������ǲ�ͬ���������ĺ��������ˡ�
Դ����������ô����simple_kernel���������Ǻ˺�������GPU�����еĺ�������

__global__ void simple_kernel(const int* pIn, int* pOut, int a)��
__global__ void simple_kernel(const int2* pIn, int* pOut, int a)��
__global__ void simple_kernel(const int* pIn1, const int* pIn2, int* pOut, int a)��

��֮��ϣ�������3������ָ�룺

void (*func1)(const int*, int*, int);
void (*func2)(const int2*, int*, int);
void (*func3)(const int*, const int*, int*, int);

�趨�˺�������
�õ���һ���������趨�˺�������ʱ�������ʹ��ʲô�ڴ�

cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared);

�ٷ��ĵ����������������


��ȡ�������ԣ�

memset(&attr, 0, sizeof(attr));
checkCudaErrors(cudaFuncGetAttributes(&attr, *func2));
OUTPUT_ATTR(attr);

�����ʾ���Եĺ������£�

#define OUTPUT_ATTR(attr)  \
    printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);   \
    printf("Constant Size: %d\n", (int)attr.constSizeBytes);                 \
    printf("Local Size:    %d\n", (int)attr.localSizeBytes);                 \
    printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock);          \
    printf("Number of Registers: %d\n", attr.numRegs);                       \
    printf("PTX Version: %d\n", attr.ptxVersion);                            \
    printf("Binary Version: %d\n", attr.binaryVersion);                      \

ShareMemory
֮ǰ��ÿÿ˵����������GPU�ڴ�ռ䣬��ָ����ȫ���ڴ棻
���ڽ����¹����ڴ�ShareMemory�����Ŀռ�ǳ�С��ÿ��block��һ�鼸ʮkB�Ŀռ䣬�����ٶȷǳ��죻��������cudaMemcpy��ֱ�Ӹ������ݹ�ȥ�����������ڼ���ʱ��Ϊһ������̨���ڵġ��������ص��ǣ����Ա�ͬһ��block���������̷��ʵ���

����GPU��Ӳ���߼�ͼ�����е�SM����һ��block��shared memory�ǳ�С���ǳ��죬����ͬһ��SM������е�sp�����Է��ʵ���
������������˺�����

__global__ void simple_kernel(const int* pIn, int* pOut, int a)
{
    __shared__ int sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = sData[threadIdx.x] * a + tid;;
}
\
����ʹ�ù����ڴ�ķ�����ֱ���ں˺����ﶨ��һ����СΪTHREAD_N��int�Ŀռ䣬THREAD_N���趨��һ��block�ж��ٸ��̣߳�Դ������Ϊ#define THREAD_N 256 ��

*/