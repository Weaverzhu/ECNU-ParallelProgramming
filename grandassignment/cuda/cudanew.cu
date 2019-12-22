#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

// #include <omp.h>

using namespace std;

typedef double ld;
typedef long long LL;

namespace output {
    const int OutputBufferSize = 1e6+5;

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
        cerr << cudaGetErrorName(err) << endl;
        // cudapeek
        exit(0);
    }
}

ld *d_a, *d_b, *d_c, *h_a, *h_b, *h_c;
int an, am, bn, bm;
int n, m;

// void copyMatrix(ld **pdst, ld **psrc, int n, int m) {
//     ld *dst = *pdst, *src = *psrc;
//     int size = sizeof(ld) * n * m;
//     handleCudaError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), "memcpy in copyMatrix");
// }


template<typename T>
__global__ void matrixMult(T *d_a, T *d_b, T *d_c, int an, int bm, int am) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / bm, j = index % bm;
    if (i >= an || j >= bm) return;
    ld sum = 0;
    if (i < an && j < bm) {
        for (int k=0; k<am; ++k)
            sum += d_a[i * am + k] * d_b[k * bm + j];
    }
    if (i * bm + j < an * bm)
        d_c[i * bm + j] = sum;
}

void outputMatrix(ld *a, int n, int m) {
    // output::print(n); output::print(',');
    // output::print(m); output::print('\n');
    for (int i=0; i<n; ++i) {
        int base = i * m;
        output::print(a[base]);
        // printf("%.2f", a[base]);
        for (int j=1; j<m; ++j) {
            output::print(',');
            output::print(a[base + j]);
            // printf(",%.2f", a[base+j]);
        }
        output::print('\n');
        // putchar('\n');
    }
}

int main()
{
    // #ifndef Weaverzhu
    // ld st = omp_get_wtime();
    freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cerr << prop.name << endl;

    // scanf("%d,%d", &an, &am);
    io >> an >> am;
    // printf("%d %d\n", an, am); exit(0);
    h_a = (ld*)malloc(sizeof(ld) * an * am);
    for (int i=0; i<an; ++i) {
        int base = i * am;
        // scanf("%lf", &h_a[base]);
        io >> h_a[base];
        for (int j=1; j<am; ++j)
            io >> h_a[base+j];
            // scanf(",%lf", &h_a[base+j]);
    }
    
    io >> bn >> bm;
    // scanf("%d,%d", &bn, &bm);
    // printf("%d %d\n", bn, bm); exit(0);
    h_b = (ld*)malloc(sizeof(ld) * bn * bm);
    for (int i=0; i<bn; ++i) {
        int base = i * bm;
        io >> h_b[base];
        // scanf("%lf", &h_b[base]);
        for (int j=1; j<bm; ++j)
            io >> h_b[base+j];
            // scanf(",%lf", &h_b[base+j]);
    }

    n = an; m = bm;
    int size = sizeof(ld) * n * m,
    sizea = sizeof(ld) * an * am,
    sizeb = sizeof(ld) * bn * bm;

    handleCudaError(cudaMalloc(&d_a, sizea), "alloc d_a");
    handleCudaError(cudaMalloc(&d_b, sizeb), "alloc d_b");
    handleCudaError(cudaMalloc(&d_c, size), "alloc d_c");
    handleCudaError(cudaMemcpy(d_a, h_a, sizea, cudaMemcpyHostToDevice), "memcpy h_a to d_a");
    handleCudaError(cudaMemcpy(d_b, h_b, sizeb, cudaMemcpyHostToDevice), "memcpy h_b to d_b");



    handleCudaError(cudaMemcpy(h_a, d_a, sizeof(ld) * an * am, cudaMemcpyDeviceToHost), "test memcpy back");
    // outputMatrix(h_a, an, am);

    
    
    // int block_size = prop.maxThreadsPerBlock,
    int block_size = 256,
     grids = (an * bm + block_size-1) / block_size;
    // ld cst = omp_get_wtime();
    handleCudaError(cudaGetLastError(), "check before running error");
    matrixMult<<<grids, block_size>>>(d_a, d_b, d_c, an, bm, am);
    handleCudaError(cudaGetLastError(), "check after kernel");
    // ld ced = omp_get_wtime();
    h_c = (ld*)malloc(size);
    

    handleCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "memcpy back");
    
    outputMatrix(h_c, n, m);
    output::flush();
    
    // cerr << "Time used: " << omp_get_wtime() - st << endl;
    // cerr << "calc Time used: " << ced - cst << endl;
    return 0;
}



