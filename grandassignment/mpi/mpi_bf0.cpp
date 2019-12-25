#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include <mpi.h>

using namespace std;

typedef double ld;
typedef long long LL;

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
        if (x<0) print('-'), x=-x;
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
            // cerr << "x=" << x << endl;
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
// ============================== fastio ======================

// const int R_CHUNKSIZE = 

const int N = 10000;

int world_size, world_rank;

ld *a, *b, *c;

int an, am, bn, bm, n, m;

int workload_size, size996;

MPI_Request requests[3][N];

void input() {
    if (world_rank == 0) {
        iokb.init(fopen("input.txt", "r"), fopen("output.txt", "w"));
        freopen("output.txt", "w", stdout);
    }
}


void quit() {
    MPI_Finalize(); exit(0);
}

void readBcast(ld *&a, int &n, int &m, int id = 0) {
    if (world_rank == 0)
        io >> n >> m;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("proc %d, %d %d\n", world_rank, n, m);

    a = (ld*)malloc(sizeof(ld) * n * m);
    
    int base = 0;
    for (int i=0; i<n; ++i, base += m) {
        if (world_size == 0)
            for (int j=0; j<m; ++j) {
                io >> a[base + j];
            }
        // MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Ibcast(a + base, m, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[id][i]);
        // MPI_Bcast(a + base, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(a, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    quit();
}

void awaitAll() {
    for (int i=0; i<an; ++i)
        MPI_Wait(&requests[0][i], MPI_STATUS_IGNORE);
    for (int i=0; i<bn; ++i)
        MPI_Wait(&requests[1][i], MPI_STATUS_IGNORE);
}

pair<int, int> calcworkload(int rank) {
    int st = min(rank, size996) * (workload_size + 1) + max(0, rank-size996) * (workload_size),
    ed = st + workload_size + (rank < size996);
    fprintf(stderr, "stderr: workload_size=%d, rank=%d, st=%d, ed=%d\n", workload_size, rank, st, ed);
    return make_pair(st, ed);
}

void compute(int rank) {
    auto tmp = calcworkload(rank);
    int st = tmp.first, ed = tmp.second;

    for (int i=st; i<ed; ++i) {
        for (int j=0; j<bm; ++j) {
            register double sum = 0;
            for (int k=0; k<am; ++k) {
                sum += a[i*am + k] * b[j*bm + k];
            }
            c[i*bm+j] = sum;
            fprintf(stderr, "rank=%d, i=%d, j=%d, sum=%.2lf, a[0][0]=%.2lf\n", rank, i, j, sum, a[0]);
        }
    }

    // if (rank > 0) MPI_Isend(c + st * bm, (ed-st) * bm, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[3][rank + world_size]);
    int size = (ed - st) * bm;
    fprintf(stderr, "stderr: proc %d, st = %d, ed = %d, bm = %d, size=%d\n", rank, st, ed, bm, size);
    if (rank > 0 && size > 0) MPI_Send(c + st * bm, size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // ===========================
    
    input();
    // quit();
    readBcast(a, an, am, 0);
    readBcast(b, bn, bm, 1);
    workload_size = (an) / world_size;
    size996 = an % world_size;

    // printf("%d %d\n", workload_size, size996);
    
    awaitAll();

    n = an;
    m = bm;
    c = (ld*)malloc(sizeof(ld) * n * m);

    compute(world_rank);
    // quit();
    if (world_rank == 0) {
        fprintf(stderr, "stderr: %d %d %d %d\n", an, am, bn, bm);
        for (int i=1; i<world_size; ++i) {
            auto tmp = calcworkload(i);
            int st = tmp.first, ed = tmp.second, size = (ed-st) * bm;

            // fprintf(stderr, "stderr: st=%d, ed=%d, size=%d\n", st, ed, size);
            // quit();
            // fprintf(stderr, "stderr: %d %d %d %d\n", an, am, bn, bm);
            if (size > 0) {
                MPI_Irecv(c + st * bm, size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[3][i]);            
            }
        }
        fprintf(stderr, "stderr: %d %d %d %d\n", an, am, bn, bm);
        // quit();
        for (int i=0; i<an; ++i) {
            int index;
            if (workload_size > 0) {
                index = i % workload_size? -1 : i / workload_size;
            } else index = i;
            // fprintf(stderr, "stderr: %d %d %d %d\n", an, am, bn, bm);
            // fprintf(stderr, "stderr: i=%d, index=%d, bm=%d\n", i, index, bm);
            if (~index && index > 0) MPI_Wait(&requests[3][index], MPI_STATUS_IGNORE);
            int base = i * bm;
            output::print(c[base]);
            for (int j=1; j<bm; ++j) {
                output::print(',');
                output::print(c[base + j]);
            }
            output::print('\n');
            // quit();
        }
        output::flush();
    }
    // ===========================
    MPI_Finalize();
    return 0;
}