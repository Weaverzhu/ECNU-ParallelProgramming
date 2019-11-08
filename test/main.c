#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

#define N 50000

int chunksize, thread_count;

void Count_sort(int a[], const int n) {
    int i, j, count;
    clock_t st = clock();
    int *temp = malloc(n * sizeof(int));
    for (i=0; i<n; ++i) {
        count = 0;
        for (j=0; j<n; ++j)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        temp[count] = a[i];
    }
    clock_t ed = clock();
    memcpy(a, temp, n * sizeof(int));
    if (n < 10) {
        for (int i=0; i<n; ++i) printf("%d ", a[i]);
        puts("");
    } else {
        for (int i=0; i<n-1; ++i) assert(a[i] <= a[i+1]);
        printf("Time used in original program %.0lfms\n", 1.0*(ed-st) / CLOCKS_PER_SEC * 1000);
    }
    
    free(temp);
}
void Count_sort_parallel(int a[], const int n, int thread_count) {
    int i, j, count;
    int *temp = malloc(n * sizeof(int));
    double st = omp_get_wtime(), ed;
    for (i=0; i<n; ++i) {
        count = 0;
        # pragma omp parallel for num_threads(thread_count) \
            reduction(+: count)
        for (j=0; j<n; ++j)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count ++;
        temp[count] = a[i];
    }
    ed = omp_get_wtime();
    
    for (int i=0; i<n-1; ++i) assert(temp[i] <= temp[i+1]);
    printf("Time used in parallel program %.5lfms, with %d threads\n", ed - st, thread_count);
    
    free(temp);
}

void doit(int thread_count) {
    int sum = 0;
    const int n = 1000000000;
    int i;
    double st = omp_get_wtime();
    # pragma omp parallel for num_threads(thread_count) \
        reduction(+: sum)
    for (i=1; i<=n; ++i) {
        sum += i;
        // printf("In %d of %d, add %d\n", omp_get_thread_num(), omp_get_num_threads(), i);
    }
    printf("%d threads, Time used %.5lfms\n", thread_count, (omp_get_wtime() - st));
    printf("%d\n", sum);
}

int t[N];

int main(int argc, char const *argv[])
{
    int n =  atoi(argv[1]);
    for (int i=0; i<n; ++i) t[i] = rand();   
    for (int i=1; i<=8; ++i)
        Count_sort_parallel(t, n, i);

    return 0;
}

