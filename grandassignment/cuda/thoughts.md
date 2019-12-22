### 骚想法 brain storming

$b = b^T$ 然后每行对应点乘

怎么样使用 share memory 最赚？需要搞清楚 blockIdx threadIdx 的运作方式

使用异步内存转移 `cudaMemcpyAsync`

`cudaOccupancyMaxPotentialBlockSize` 获取最佳的 `blocksize`