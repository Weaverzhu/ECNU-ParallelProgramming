### 骚想法 brain storming

~~$b = b^T$ 然后每行对应点乘~~ 读入不是连续读入，结果慢了4倍

~~怎么样使用 share memory 最赚？需要搞清楚 blockIdx threadIdx 的运作方式~~ 用了和没用一个样子

~~使用异步内存转移 `cudaMemcpyAsync`~~ 可能架构太老了，本地能跑但是oj会炸

`cudaOccupancyMaxPotentialBlockSize` 获取最佳的 `blocksize`

`maxthreadsPerBlock` 获取设备每个block 最多显卡

把浮点数拆开来当整数部分，快了一倍。。。。（老师明确说整数行不通的，看起来数据假了

查看 occupancy，查到古董卡 tesla c2075 的计算能力，https://developer.nvidia.com/cuda-gpus

Determining Registers Per Thread and Shared Memory Per Thread Block
To determine the number of registers used per thread in your kernel, simply compile the kernel code using the option               --ptxas-options=-v to nvcc.  This will output information about register, local memory, shared memory, and constant memory usage for each kernel in the .cu file.  However, if your kernel declares any external shared memory that is allocated dynamically, you will need to add the (statically allocated) shared memory reported by ptxas to the amount you dynamically allocate at run time to get the correct shared memory usage.  An example of the verbose ptxas output is as follows:


Linux-Ubuntu open-mpi运行错误:https://www.jianshu.com/p/fe9649f9596e cma-permission-denied