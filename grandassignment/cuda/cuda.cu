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

using namespace std;

typedef double ld;
typedef long long LL;

// namespace output {
//     const int OutputBufferSize = 1e6+5;

//     char buffer[OutputBufferSize];
//     char *s = buffer;
//     inline void flush() {
//         fwrite(buffer, 1, s-buffer, stdout);
//         s = buffer;
//         fflush(stdout);
//     }
//     inline void print(const char ch) {
//         // putchar(ch); return;
//         if (s-buffer>OutputBufferSize-2) flush();
//         *s++ = ch;
//     }
//     inline void print(char *str) {
//         while (*str!=0) print(char(*str++));
//     }
//     inline void print(int x) {
//         // printf("%d", x); return;
//         char buf[25] = {0}, *p = buf;
//         if (x<0) print('-'), x=-x;
//         if (x == 0) print('0');
//         while (x) *(++p) = x%10, x/=10;
//         while (p != buf) print(char(*(p--)+'0'));
//     }

//     inline void print(ld x) {
//         // printf("%.2f", x);
//         static char buf[100];
//         sprintf(buf, "%.2f", x);
//         print(buf);
//     }
// }


// struct ios {
//     static const int IN_LEN=1<<18|1;
//     char buf[IN_LEN],*s,*t; 
//     inline char read(){
//         return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t?-1:*s++;
//     }
//     inline bool isEOF() {   
//         return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t;
//     }
//     inline ios & operator >> (int &x){
//         static char c11,boo;
//         for(c11=read(),boo=0;!isdigit(c11);c11=read()){
//             if(c11==-1)return *this;
//             boo|=c11=='-';
//         }
//         for(x=0;isdigit(c11);c11=read())x=x*10+(c11^'0');
//         boo&&(x=-x);
//         return *this;
//     }

//     inline ios & operator >> (LL &x){
//         static char c11,boo;
//         for(c11=read(),boo=0;!isdigit(c11);c11=read()){
//             if(c11==-1)return *this;
//             boo|=c11=='-';
//         }
//         for(x=0;isdigit(c11);c11=read())x=x*10+(c11^'0');
//         boo&&(x=-x);
//         return *this;
//     }

//     inline ios &operator >> (char *s) {
//         int len = 0;
//         char ch;
//         for (ch=read(); ch=='\n' || ch == ' '; ch=read());
//         if (ch == -1) {
//             s[len] = 0;
//             return *this;
//         }
//         for (; ch!='\n' && ch != ' ' && ch != -1;ch=read())
//             s[len++] = ch;
//         s[len] = 0;
//         return *this;
//     }

//    inline ios &operator>>(ld &x)
//     {

//         char ch;
//         bool neg = false, dec = false;
//         double now = 0.1;
//         for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

//         if (ch == '-') neg = true;
//         else if (ch == '.') { x = 0; dec = true; }
//         else if (ch != -1) x = ch-'0';
//         else return *this;
//         if (!dec) {
//             for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
//                 x = x * 10 + ch-'0';
//             }
//         }

//         if (ch == '.')
//             for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
//                 x += now * (ch - '0'); now *= 0.1;
//             }
//         if (neg) x = -x;
        
//         return *this;
//     }

//     inline ios &operator>>(long double &x)
//     {

//         char ch;
//         bool neg = false, dec = false;
//         double now = 0.1;
//         for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

//         if (ch == '-') neg = true;
//         else if (ch == '.') { x = 0; dec = true; }
//         else if (ch != -1) x = ch-'0';
//         else return *this;
//         if (!dec) {
//             for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
//                 x = x * 10 + ch-'0';
//             }
//         }

//         if (ch == '.')
//             for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
//                 x += now * (ch - '0'); now *= 0.1;
//             }
//         if (neg) x = -x;
        
//         return *this;
//     }
// } io;

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

    // handleCudaError(cudaMemcpy(src, dst, size, cudaMemcpyDeviceToHost), "check in copyMatrix");
    // cerr << "end in copyMatrix" << endl;
}

// ld *copyMatrixBack(const ld *src, int n, int m) {
//     ld *res;
//     int size = sizeof(ld) * n * m;
//     res = (ld*)malloc(size);
//     cerr << "in copyMatrixBack: size=" << size << endl;
//     handleCudaError(cudaMemcpy(res, src, size, cudaMemcpyDeviceToHost), "memcpy in copyMatrixBack");
//     // memcpy(res.a, ptr, size);)
//     return res;
// }

template<typename T>
__global__ void matrixMult(T *d_a, T *d_b, T *d_c, int an, int bm, int am) {
    int i = blockDim.x * blockIdx.x + threadIdx.x,
    j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= an || j >= bm) return;
    ld sum = 0;
    if (i < an && j < bm) {
        for (int k=0; k<am; ++k)
            sum += d_a[i * am + k] * d_b[k * bm + j];
    }
    if (i * bm + j < an * bm)
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
        // output::print(a[base]);
        printf("%.2f", a[base]);
        for (int j=1; j<m; ++j) {
            // output::print(',');
            // output::print(a[base + j]);
            printf(",%.2f", a[base+j]);
        }
        // output::print('\n');
        putchar('\n');
    }
}

int main()
{
    // #ifndef Weaverzhu
    freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
    // #endif
    // io >> an >> am;
    scanf("%d,%d", &an, &am);
    // printf("%d %d\n", an, am); exit(0);
    h_a = (ld*)malloc(sizeof(ld) * an * am);
    for (int i=0; i<an; ++i) {
        int base = i * am;
        scanf("%lf", &h_a[base]);
        for (int j=1; j<am; ++j)
            scanf(",%lf", &h_a[base+j]);
    }
    

    scanf("%d,%d", &bn, &bm);
    // printf("%d %d\n", bn, bm); exit(0);
    h_b = (ld*)malloc(sizeof(ld) * bn * bm);
    for (int i=0; i<bn; ++i) {
        int base = i * bm;
        scanf("%lf", &h_b[base]);
        for (int j=1; j<bm; ++j)
            scanf(",%lf", &h_b[base+j]);
    }
    // B.readtrans();

    // outputMatrix(h_a, an, am);
    // outputMatrix(h_b, bn, bm);
    // exit(0);

    int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid((an + threads.x - 1) / threads.x, (bm + threads.y - 1) / threads.y);
    n = an;
    m = bm;

    // fprintf(stderr, "grid= %d,%d,%d threads= %d,%d,%d\n", grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);

    // read into main memory
    copyMatrix(h_a, d_a, an, am);
    copyMatrix(h_b, d_b, bn, bm);
    handleCudaError(cudaMalloc(&d_c, sizeof(ld) * n * m), "allocate for h_c");

    // puts("entering danger");
    matrixMult<<<threads, grid>>>(d_a, d_b, d_c, an, bm, am);
    // if (cudaGetLastError() != cudaSuccess) {
    //     cerr << "failed in matrixMult" << endl;
    //     exit(0);
    // } else cerr << "looks good in matrixMult" << endl;
    // puts("FUCK");
    // ld *c = copyMatrixBack(d_c, n, m);
    h_c = (ld*)malloc(sizeof(ld) * n * m);
    int size = sizeof(ld) * n * m;


    handleCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "memcpy back");
    
    outputMatrix(h_c, n, m);
    // output::flush();
    
    return 0;
}



