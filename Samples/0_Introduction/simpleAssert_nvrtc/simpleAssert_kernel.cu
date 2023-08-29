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

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
//! Tests assert function.
//! Thread whose id > N will print assertion failed error message.
////////////////////////////////////////////////////////////////////////////////
/*
* https://blog.csdn.net/q583956932/article/details/78764301  CUDAѧϰ�ʼ�(3) NVRTC�����
* 
��չ��simpleAssert_nvrtc
������simpleAssertʵ��ͬ�����ܵ�һ�ݴ��롣

����������ǣ�ͨ��nvtc����⣬������cuda�˺�����ͬʱ������������Ĵ���ȫ������һ��cpp�ļ���ȥ��

����Ϊʲô��ô�������Բο����

https://blog.csdn.net/q583956932/article/details/78764301

�򵥸����¾��ǣ�����ô���Ļ����͵���cuda c/c++������������cpp���룬���Ƿǳ����ġ������Ļ���������ϵͳ�Դ���c/c++����������cpp����nvrtc����cu�ļ�������ܿ졣

�����¸о���������Դ����ע�ͺ���ϸ��û��Ҫ��ϸ˵�ˡ�����Ŀǰ��ʱһ��Ҳ�����õ����nvrtc���ֿ�����cpp��cuda��
https://blog.csdn.net/zcy0xy/article/details/84424182
*/
extern "C" __global__ void testKernel(int N) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  assert(gtid < N);
}
