#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

using namespace std;

typedef double ld;
typedef long long LL;

const int max_shared_size = 128;

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

__global__ void matrixMult2(ld *d_a, ld *d_b, ld *d_c, int an, int bn, int am) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int i = index / bn, j = index % bn;
    if (i >= an || j >= bn) return;
    ld sum = 0;
    for (int k=0; k<am; ++k)
        sum += d_a[i*am + k] * d_b[j*am+k];
    d_c[i * bn + j] = sum;
}

void outputMatrix(ld *a, int n, int m) {
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

void hostMatrixInput(ld **a, int &n, int &m, bool transpose = false) {
    io >> n >> m;
    *a = (ld*)malloc(sizeof(ld) * n * m);
   
    if (!transpose)
        for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) {
            io >> (*a)[i * m + j];
        }
    else {
        for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) {
            io >> (*a)[j * n + i];
        }
        swap(n, m);
    }
}



int main()
{
    // #ifndef Weaverzhu
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cerr << prop.name << endl;
    hostMatrixInput(&h_a, an, am);
    hostMatrixInput(&h_b, bn, bm, true);


    n = an;
    m = bn;
    int block_size = prop.maxThreadsPerBlock, grids = (n * m + block_size - 1) / block_size;
    copyMatrix(h_a, d_a, an, am);
    copyMatrix(h_b, d_b, bn, bm);
    handleCudaError(cudaMalloc(&d_c, sizeof(ld) * n * m), "allocate for h_c");
    // cerr << an << ' ' << am << ' ' << bn << ' ' << bm << ' ' << endl;
    matrixMult2<<<grids, block_size>>>(d_a, d_b, d_c, an, bn, am);
    h_c = (ld*)malloc(sizeof(ld) * n * m);
    int size = sizeof(ld) * n * m;


    handleCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "memcpy back");
    
    outputMatrix(h_c, n, m);
    output::flush();
    
    return 0;
}



