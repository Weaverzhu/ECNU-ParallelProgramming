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
        if (x<0) print('-'), x=-x;
        if (x == 0) print('0');
        while (x) *(++p) = x%10, x/=10;
        while (p != buf) print(char(*(p--)+'0'));
    }

    inline void print(ld x) {
        // printf("%.2f", x);
        static char buf[100];
        sprintf(buf, "%.2f", x);
        print(buf);
    }
}


struct ios {
    static const int IN_LEN=1<<18|1;
    char buf[IN_LEN],*s,*t; 
    inline char read(){
        return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t?-1:*s++;
    }
    inline bool isEOF() {   
        return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t;
    }
    inline ios & operator >> (int &x){
        static char c11,boo;
        for(c11=read(),boo=0;!isdigit(c11);c11=read()){
            if(c11==-1)return *this;
            boo|=c11=='-';
        }
        for(x=0;isdigit(c11);c11=read())x=x*10+(c11^'0');
        boo&&(x=-x);
        return *this;
    }

    inline ios & operator >> (LL &x){
        static char c11,boo;
        for(c11=read(),boo=0;!isdigit(c11);c11=read()){
            if(c11==-1)return *this;
            boo|=c11=='-';
        }
        for(x=0;isdigit(c11);c11=read())x=x*10+(c11^'0');
        boo&&(x=-x);
        return *this;
    }

    inline ios &operator >> (char *s) {
        int len = 0;
        char ch;
        for (ch=read(); ch=='\n' || ch == ' '; ch=read());
        if (ch == -1) {
            s[len] = 0;
            return *this;
        }
        for (; ch!='\n' && ch != ' ' && ch != -1;ch=read())
            s[len++] = ch;
        s[len] = 0;
        return *this;
    }

   inline ios &operator>>(ld &x)
    {

        char ch;
        bool neg = false, dec = false;
        double now = 0.1;
        for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

        if (ch == '-') neg = true;
        else if (ch == '.') { x = 0; dec = true; }
        else if (ch != -1) x = ch-'0';
        else return *this;
        if (!dec) {
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x = x * 10 + ch-'0';
            }
        }

        if (ch == '.')
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x += now * (ch - '0'); now *= 0.1;
            }
        if (neg) x = -x;
        
        return *this;
    }

    inline ios &operator>>(long double &x)
    {

        char ch;
        bool neg = false, dec = false;
        double now = 0.1;
        for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

        if (ch == '-') neg = true;
        else if (ch == '.') { x = 0; dec = true; }
        else if (ch != -1) x = ch-'0';
        else return *this;
        if (!dec) {
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x = x * 10 + ch-'0';
            }
        }

        if (ch == '.')
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x += now * (ch - '0'); now *= 0.1;
            }
        if (neg) x = -x;
        
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
        output::print('\n');
        output::print(c[l++]);
    }
    for (register int i=l; i<r; ++i) {
        if (i % m == 0) output::print('\n');
        else output::print(',');
    }
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
    // handleCudaError(cudaStreamSynchronize(stream[1]));
    
    outputinterval(a, st, ed);
    output::print('\n');
}

int main()
{
    // #ifndef Weaverzhu
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cerr << prop.name << endl;

    // cudaStream_t s_a, s_b;
    // cudaStreamCreate(&s_a);
    // cudaStreamCreate(&s_b);
    // cudaStreamCreateWithFlags(&s_a, cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&s_b, cudaStreamNonBlocking);

    // #endif
    io >> an >> am; h_a = (ld*)malloc(sizeof(ld) * an * am);
    for (int i=0; i<an; ++i)
    for (int j=0; j<am; ++j)
        io >> h_a[i*am + j];
    // copyMatrix(d_a, h_a, an, am);
    // copyMatrixAsync(h_a, d_a, an, am, s_a);


    io >> bn >> bm; h_b = (ld*)malloc(sizeof(ld) * bn * bm);
    for (int i=0; i<bn; ++i)
    for (int j=0; j<bm; ++j)
        io >> h_b[i*bm + j];

    copyMatrix(h_a, d_a, an, am);
    copyMatrix(h_b, d_b, bn, bm);
    // copyMatrixAsync(h_b, d_b, bn, bm, s_b);
    n = an;
    m = bm;
    int block_size = prop.maxThreadsPerBlock, grids = (n * m + block_size - 1) / block_size;


    // cudaStreamSynchronize(s_a);
    // cudaStreamSynchronize(s_b);
    
    handleCudaError(cudaMalloc(&d_c, sizeof(ld) * n * m), "allocate for h_c");

    matrixMult<<<grids, block_size>>>(d_a, d_b, d_c, an, bm, am);
    h_c = (ld*)malloc(sizeof(ld) * n * m);
    // int size = sizeof(ld) * n * m;
    // cerr << "before outputmatrixasync" << endl;

    int size = sizeof(ld) * n * m;
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



