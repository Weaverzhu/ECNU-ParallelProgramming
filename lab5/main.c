#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

#define N 5000

int chunksize, thread_count;

void Count_sort(int a[], int n) {
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
void Count_sort_parallel(int a[], int n) {
    int i, j, count;
    int *temp = malloc(n * sizeof(int));
    clock_t st = clock();
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
    clock_t ed = clock();
    memcpy(a, temp, n * sizeof(int));
    if (n < 10) {
        for (int i=0; i<n; ++i) printf("%d ", a[i]);
        puts("");
    } else {
        for (int i=0; i<n-1; ++i) assert(a[i] <= a[i+1]);
        printf("Time used in parallel program %.0lfms\n", 1.0*(ed-st) / CLOCKS_PER_SEC * 1000);
    }
    free(temp);
}

int main(int argc, char const *argv[])
{
    if (argc >= 2) chunksize = atoi(argv[1]);
    chunksize = 100;

    thread_count = 4;

    int a[N];

    srand(time(0));
    int n = N;
    for (int i=0; i<n; ++i) {
        a[i] = rand() % 20 + 1;
    }
    int t1[N], t2[N];
    memcpy(t1, a, sizeof(a));
    memcpy(t2, a, sizeof(a));

    Count_sort(t1, n);
    Count_sort_parallel(t2, n);

    return 0;
}


