#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <algorithm>


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

typedef double ld;
typedef long long LL;

using namespace std;

// ============= config =============

const int N = 5200 * 5200;

// ==================================

ld *genrandomarray(int n, ld l = 0, ld r = 100000) {
    static default_random_engine dre(time(0));
    uniform_int_distribution<int> u((int)round(l * 100), (int)round(r * 100));
    ld *res = (ld*)malloc(n * sizeof(ld));
    for (int i=0; i<n; ++i)
        res[i] = 1.0 * u(dre) / 100;
    return res;
}

inline void handleCudaError(cudaError_t err, string name = "fuck") {
    if (err != cudaSuccess) {
        cerr << name << endl;
        cerr << cudaGetErrorString(err) << endl;
        exit(0);
    }
}


ld *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

__global__ void vectorAdd(ld *d_a, ld *d_b, ld *d_c, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) d_c[index] = d_a[index] + d_b[index];
}

int main() {
    h_a = genrandomarray(N);
    h_b = genrandomarray(N);
    // h_c = genrandomarray(N);
    int size = sizeof(ld) * N;
    h_c = (ld*)malloc(size);

    for (int i=0; i<N; ++i) {
        h_c[i] = h_a[i] + h_b[i];
    }

    handleCudaError(cudaMalloc(&d_a, size));
    handleCudaError(cudaMalloc(&d_b, size));
    handleCudaError(cudaMalloc(&d_c, size));
    handleCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    handleCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int block_size = 2048;
    int grids = (N + block_size - 1) / block_size;
    handleCudaError(cudaGetLastError(), "check before kernel");
    vectorAdd<<<grids, block_size>>>(d_a, d_b, d_c, N);
    handleCudaError(cudaGetLastError(), "after kernel");


    // bool suc = true;
    for (int i=0; i<N; ++i) {
        ld relative_error = fabs(1 - d_c[i] / h_c[i]);
        if (relative_error > 1e-6) {
            puts("WA");
            return 0;
        }
    }
    puts("AC");
    return 0;
}