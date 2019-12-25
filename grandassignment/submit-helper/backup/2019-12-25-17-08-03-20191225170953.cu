#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <cstdlib>
// #include <windows.h>
// #include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

typedef double ld;
typedef long long LL;

// const int max_share_size = 512, chunk_size = 1 << 16;
const int chunk_size = 1<<16;


namespace io_impl
{
inline bool maybe_digit(char c)
{
    return c >= '0' && c <= '9';
}

struct io_s
{
private:
    FILE *fin;
    FILE *fout;

    bool negative;
    bool ok;
    char ch;

    inline char next_char()
    {
        static char buf[100000], *p1 = buf, *p2 = buf;
        return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 100000, fin), p1 == p2) ? EOF : *p1++;
    }

public:
    void init(FILE *_in, FILE *_out)
    {
        fin = _in;
        fout = _out;
        ch = next_char();
        ok = true;
    }

    template <typename T>
    bool run(T &_v)
    {
        _v = 0;
        while (!maybe_digit(ch) && ch != EOF)
            ch = next_char();
        if (ch == EOF)
            return ok = false;
        do
        {
            _v = (_v << 1) + (_v << 3) + ch - '0';
        } while (maybe_digit(ch = next_char()));
        return true;
    }

    template <typename T>
    bool rd(T &_v)
    {
        negative = false;
        _v = 0;
        while (!maybe_digit(ch) && ch != EOF)
        {
            negative = ch == '-';
            ch = next_char();
        }
        if (ch == EOF)
            return ok = false;
        do
        {
            _v = (_v * 10) + (ch - '0');
        } while (maybe_digit(ch = next_char()));
        static double _map[] = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
        if (ch == '.')
        {
            int tp = 0;
            while (maybe_digit(ch = next_char()))
            {
                _v = (_v * 10) + (ch - '0');
                ++tp;
            }
            _v *= _map[tp];
        }
        if (negative)
            _v = -_v;
        return true;
    }
    
};

} // namespace io_impl

using namespace io_impl;

io_s iokb;

namespace output {
    const int OutputBufferSize = 1 << 20;

    char buffer[OutputBufferSize];
    char *s = buffer;
    inline void flush() {
        fwrite(buffer, 1, s-buffer, stdout);
        s = buffer;
        fflush(stdout);
    }
    inline void print(const char ch) {
        // putchar(ch); return;
        if (s-buffer>OutputBufferSize-2) flush();
        *s++ = ch;
    }
    inline void print(char *str) {
        while (*str!=0) print(char(*str++));
    }
    inline void print(int x) {
        // printf("%d", x); return;
        char buf[25] = {0}, *p = buf;
        // if (x<0) print('-'), x=-x;
        // if (x == 0) print('0');
        while (x) *(++p) = x%10, x/=10;
        while (p != buf) print(char(*(p--)+'0'));
    }

    inline void print(LL x) {
        // printf("%d", x); return;
        char buf[25] = {0}, *p = buf;
        
        if (x == 0) print('0');
        while (x) *(++p) = x%10, x/=10;
        while (p != buf) print(char(*(p--)+'0'));
    }

    inline void print(ld v) {
        // printf("%.2f", x);
        // static int stk[70], tp;
        // tp = 0;
        if (fabs(v) < 0.005)
        {
            print('0');
            return;
        }
        else
        {
            LL x = (LL)floor(v * 100 + 0.5);
            if (x<0) print('-'), x=-x;
            // cerr << "x=" << x << endl; exit(0);
            print((LL)(x / 100));
            print('.');
            print((char)(x / 10 % 10 + '0'));
            print((char)(x % 10 + '0'));
        }
    }
}



struct ios {
    
    inline ios & operator >> (int &x){
        iokb.run(x);
        return *this;
    }

   inline ios &operator>>(ld &x)
    {
        iokb.rd(x);
        return *this;
    }
} io;

inline void handleCudaError(cudaError_t err, string name = "fuck") {
    if (err != cudaSuccess) {
        cerr << name << endl;
        cerr << cudaGetErrorString(err) << endl;
        exit(0);
    }
}

ld *d_a, *d_b, *d_c, *h_a, *h_b, *h_c;
int an, am, bn, bm;
int n, m;

void copyMatrix(ld *&src,  ld *&dst, int n, int m) {
    int size = sizeof(ld) * n * m;
    
    handleCudaError(cudaMalloc(&dst, size), "cudaMalloc in copyMatrix");
    handleCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), "memcpy in copyMatrix");
}

void copyMatrixAsync(ld *&src, ld *&dst, int n, int m, cudaStream_t &stream) {
    int size = sizeof(ld) * n * m;
    handleCudaError(cudaMalloc(&dst, size), "cudaMalloc in copyMatrix");
    handleCudaError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream), "memcpyasync in copyMatrix");
}

template<typename T>
__global__ void matrixMult(T *d_a, T *d_b, T *d_c, int an, int bm, int am) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / bm, j = index % bm;
    if (i >= an || j >= bm) return;
    register ld sum = 0;
    int basea = i * am;
   
    for (int k=0; k<am; ++k)
        sum += d_a[basea + k] * d_b[k * bm + j];

    d_c[i * bm + j] = sum;
    // int index = threadIdx.x;
    // if (index < an * bm)
    //     d_c[index] = 1; 
}

void outputMatrix(ld *a, int n, int m) {
    // output::print(n); output::print(',');
    // output::print(m); output::print('\n');
    for (int i=0; i<n; ++i) {
        int base = i * m;
        output::print(a[base]);
        for (int j=1; j<m; ++j) {
            output::print(',');
            output::print(a[base + j]);
        }
        output::print('\n');
    }
}


void outputinterval(ld *c, int l, int r) {
    // printf("%p %d %d, %d %d\n", c, l, r, n, m);
    // printf("%.2lf\n", c[1]);
    // exit(0);
    if (l == 0) {
        // output::print('\n');
        output::print(c[l++]);
    }
    for (register int i=l; i<r; ++i) {
        if (i % m == 0) output::print('\n');
        else output::print(',');
        output::print(c[i]);
    }
    // output::print('\n');
    // output::flush();
    // exit(0);
}
void outputMatrixAsync(ld *&a, ld *&d_a, int n, int m) {



    int st = 0, ed = n * m;
    // printf("st=%d ed=%d, a=%p\n", st, ed, a); 
    cudaStream_t stream[2];
    int mask = 0;
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    int size;
    
    for (; st<ed; st+=size, mask^=1) {
        size = min(chunk_size, ed - st);
        // printf("st=%d st+size=%d, mask=%d\n", st, st+size, mask);
        // handleCudaError(cudaMemcpy(a + st, d_a + st, size * sizeof(ld), cudaMemcpyDeviceToHost));
        handleCudaError(cudaMemcpyAsync(a + st, d_a + st, size * sizeof(ld), cudaMemcpyDeviceToHost, stream[mask]));
        // exit(0);
        if (st - chunk_size >= 0) {
            printf("%d %d\n",st-chunk_size, st);
            handleCudaError(cudaStreamSynchronize(stream[mask^1]));
            outputinterval(a, st-chunk_size, st);
        }
    }
    st -= size;
    // sleep(1000);
    handleCudaError(cudaStreamSynchronize(stream[0]));
    handleCudaError(cudaStreamSynchronize(stream[1]));
    
    outputinterval(a, st, ed);
    output::print('\n');
}

int main()
{
    // #ifndef Weaverzhu
    // freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    iokb.init(fopen("input.txt", "r"), fopen("output.txt", "w"));


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cerr << prop.name << endl;

    cudaStream_t mainstream;
    cudaStreamCreate(&mainstream);

    // #endif
    io >> an >> am; 
    // h_a = (ld*)malloc(sizeof(ld) * an * am);
    handleCudaError(cudaHostAlloc(&h_a, sizeof(ld) * an * am, cudaHostAllocDefault));
    for (int i=0; i<an; ++i)
    for (int j=0; j<am; ++j)
        io >> h_a[i*am + j];
    // copyMatrix(d_a, h_a, an, am);
    copyMatrixAsync(h_a, d_a, an, am, mainstream);


    io >> bn >> bm; 
    // h_b = (ld*)malloc(sizeof(ld) * bn * bm);
    handleCudaError(cudaHostAlloc(&h_b, sizeof(ld) * bn * bm, cudaHostAllocDefault));
    for (int i=0; i<bn; ++i)
    for (int j=0; j<bm; ++j)
        io >> h_b[i*bm + j];

    // copyMatrix(h_a, d_a, an, am);
    // copyMatrix(h_b, d_b, bn, bm);
    copyMatrixAsync(h_b, d_b, bn, bm, mainstream);
    n = an;
    m = bm;
    int block_size = prop.maxThreadsPerBlock, grids = (n * m + block_size - 1) / block_size;
    
    handleCudaError(cudaMalloc(&d_c, sizeof(ld) * n * m), "allocate for h_c");

    matrixMult<<<grids, block_size, 0, mainstream>>>(d_a, d_b, d_c, an, bm, am);
    // h_c = (ld*)malloc(sizeof(ld) * n * m);
    handleCudaError(cudaHostAlloc(&h_c, sizeof(ld) * n * m,cudaHostAllocDefault));
    // int size = sizeof(ld) * n * m;
    // cerr << "before outputmatrixasync" << endl;

    int size = sizeof(ld) * n * m;
    handleCudaError(cudaStreamSynchronize(mainstream));
    // cudaStream_t stream;
    // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // cudaStreamCreate(&stream);
    // handleCudaError(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream));
    // handleCudaError(cudaStreamSynchronize(stream));
    // outputinterval(h_c, 0, n * m);

    handleCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "memcpy back");
    // printf("h_c=%p\n", h_c);
    outputMatrix(h_c, n, m);
    // outputMatrixAsync(h_c, d_c, n, m);
    output::flush();
    
    return 0;
}



