#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const int N = 2e6+5;

namespace serial {
    bool prime[N];

    void go() {
        int i, j;
        for (i=2; i<=N; ++i)
            prime[i] = true;
        for (i=2; i<=sqrt(N); ++i)
            if (prime[i]) {
                for (j=i+i; j<N; j=j+i) prime[j] = false;
            }
    }
    bool check(bool p1[], bool p2[]) {
        for (int i=0; i<N; ++i) {
            if (p1[i] != p2[i]) {
                return false;
            }
        }
        return true;
    }
}

namespace parallel {
    bool prime[N];

    void go() {
        int i, j;
        #pragma omp parallel for
        for (i=2; i<=N; ++i)
            prime[i] = true;
        int ed = sqrt(N);
        
        for (i=2; i<=ed; ++i)
            if (prime[i])
            {
                #pragma omp parallel for
                for (j=i*i; j<N; j+=i)
                    prime[j] = false;
            }
    }

    void show() {
        for (int i=2; i<=20; ++i) {
            if (prime[i]) printf("%d\n", i);
        }
    }
}

int main(int argc, char const *argv[])
{
    serial::go();
    parallel::go();
    bool res = serial::check(serial::prime, parallel::prime);
    parallel::show();
    if (res) puts("AC");
    else puts("WA");


    return 0;
}
