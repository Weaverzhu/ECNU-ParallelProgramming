#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define N 50

int cmp(const void *a, const void *b) {
    int *pa = (int*)a, *pb = (int*)b;
    return *pa - *pb;
}

void output(int a[], int n, int rank) {
    qsort(a, n, sizeof(a[0]), cmp);
    printf("Thread %d: Iterations ", rank);
    for (int i=0, j; i<n; i=j+1) {
        j = i;
        while (j+1<n && a[j+1] == a[j] + 1) ++j;
        printf("%d--%d ", i,j);
    }
    puts("");
}

void doit(double *sum, int n, int *pi) 
{
    int rank = omp_get_thread_num();
    int tmp[N], tcnt = 0;
    for (*pi=0; *pi<=n; ++*pi) {
        *sum += *pi;
        tmp[tcnt++] = n;
    }
    output(tmp, tcnt, rank);
}

int main(int argc, char const *argv[])
{
    double sum = 0;
    assert(argc == 5);
    int thread_count = atoi(argv[1]), i, n = atoi(argv[2]), method = atoi(argv[3]), chunksize = atoi(argv[4]);
    switch (method)
    {
    case 0:
        # pragma omp parallel for num_threads(thread_count) \
        reduction(+:sum) schedule(static, chunksize)
        for (i=0; i<=n; ++i) {
            sum += i;
            int rank = omp_get_thread_num();
            printf("Thread %d, Iteration %d\n", rank, i);
        }
        break;
    case 1:
        # pragma omp parallel for num_threads(thread_count) \
        reduction(+:sum) schedule(dynamic, chunksize)
        for (i=0; i<=n; ++i) {
            sum += i;
            int rank = omp_get_thread_num();
            printf("Thread %d, Iteration %d\n", rank, i);
        }

        break;
    case 2:
        # pragma omp parallel for num_threads(thread_count) \
        reduction(+:sum) schedule(guided, chunksize)
        for (i=0; i<=n; ++i) {
            sum += i;
            int rank = omp_get_thread_num();
            printf("Thread %d, Iteration %d\n", rank, i);
        }
        break;
    
    default:
        break;
    }
    
    
    return 0;
}
