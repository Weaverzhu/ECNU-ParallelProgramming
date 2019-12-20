#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long long ll;
typedef double ld;
// using namespace std;

// fast io ========================

inline void flush();
inline void print(const char ch);
inline void printstr(char *str);
inline void printint(int x);
inline void printld(ld x);
inline void readint(int *x);
inline void readld(ld *x);

// ================================

#define isdigit _isdigit

inline int isdigit(char ch) { return ch >= '0' && ch <= '9';}

int main(int argc, char **argv)
{
    // MPI_Init(&argc, &argv);
    // int world_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // int world_size;
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // MPI_Finalize();
    return 0;
}


char buffer[1000000+5];
char *s = buffer;
inline void flush() {
    fwrite(buffer, 1, s-buffer, stdout);
    s = buffer;
    fflush(stdout);
}
inline void print(const char ch) {
    // putchar(ch); return;
    if (s-buffer>1000000+5-2) flush();
    *s++ = ch;
}
inline void printstr(char *str) {
    while (*str!=0) print((char)(*str++));
}
inline void printint(int x) {
    // printf("%d", x); return;
    char buf[25] = {0}, *p = buf;
    if (x<0) print('-'), x=-x;
    if (x == 0) print('0');
    while (x) *(++p) = x%10, x/=10;
    while (p != buf) print((char)(*(p--)+'0'));
}

inline void printld(ld x) {
    // printf("%.2f", x);
    char buf[100];
    sprintf(buf, "%.2f", x);
    printstr(buf);
}


#define IN_LEN 1<<18|1
char buf[IN_LEN],*s,*t; 
inline char read(){
    return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t?-1:*s++;
}
inline int isEOF() {   
    return (s==t)&&(t=(s=buf)+fread(buf,1,IN_LEN,stdin)),s==t;
}
inline void readint (int *x){
    char c11,boo;
    for(c11=read(),boo=0;!isdigit(c11);c11=read()){
        if(c11==-1) return;
        boo|=c11=='-';
    }
    for(x=0;isdigit(c11);c11=read()) *x=(*x)*10+(c11^'0');
    boo&&(*x=-*x);
    return;
}
inline void readld (ld *x)
{

    char ch;
    int neg = 0, dec = 0;
    double now = 0.1;
    for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

    if (ch == '-') neg = 1;
    else if (ch == '.') { *x = 0; dec = 1; }
    else if (ch != -1) *x = ch-'0';
    else return;
    if (!dec) {
        for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
            *x = (*x) * 10 + ch-'0';
        }
    }

    if (ch == '.')
        for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
            *x += now * (ch - '0'); now *= 0.1;
        }
    if (neg) *x = -*x;
    
    return;
}
