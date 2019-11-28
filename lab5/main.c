#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int main(int argc, char const *argv[])
{
    int thread_count = 0;
    int n = 200;
    for (thread_count=1; thread_count<=8; ++thread_count) {
        puts("-----------------------------------");
        printf("thread_count = %d\n", thread_count);
        printf("using static schedule:\n");
        double sum = 0;
        int i, flag = 0;
        #pragma omp parallel num_threads(thread_count)
        {
            int *a = malloc(sizeof(int) * (n)), sz = 0;
            #pragma omp for schedule(static, 2)
            for (i=1; i<=n; ++i) {
                a[sz++] = i;
                sum += i;
            }
            int rank = omp_get_thread_num();
            while (flag != rank);
            #pragma omp critical
            {
                ++flag;
                printf("Thread %d:", rank);
                for (int i=0; i<sz; ++i)
                    printf(" %d", a[i]);
                printf("\n");
            }
        }
    }

    for (thread_count=1; thread_count<=8; ++thread_count) {
        puts("-----------------------------------");
        printf("thread_count = %d\n", thread_count);
        printf("using dynamic schedule:\n");
        double sum = 0;
        int i, flag = 0;
        #pragma omp parallel num_threads(thread_count)
        {
            int *a = malloc(sizeof(int) * (n)), sz = 0;
            #pragma omp for schedule(dynamic, 2)
            for (i=1; i<=n; ++i) {
                a[sz++] = i;
                sum += i;
            }
            int rank = omp_get_thread_num();
            while (flag != rank);
            #pragma omp critical
            {
                ++flag;
                for (int j=0; j<10000; ++j) {
                    int x = j;
                    x = x + 1;
                }
                printf("Thread %d:", rank);
                for (int i=0; i<sz; ++i)
                    printf(" %d", a[i]);
                printf("\n");
            }
        }
    }

    for (thread_count=1; thread_count<=8; ++thread_count) {
        puts("-----------------------------------");
        printf("thread_count = %d\n", thread_count);
        printf("using guided schedule:\n");
        double sum = 0;
        int i, flag = 0;
        #pragma omp parallel num_threads(thread_count)
        {
            int *a = malloc(sizeof(int) * (n)), sz = 0;
            #pragma omp for schedule(guided, 1)
            for (i=1; i<=n; ++i) {
                a[sz++] = i;
                sum += i;
            }
            int rank = omp_get_thread_num();
            while (flag != rank);
            #pragma omp critical
            {
                ++flag;
                printf("Thread %d:", rank);
                for (int i=0; i<sz; ++i)
                    printf(" %d", a[i]);
                printf("\n");
            }
        }
    }


    return 0;
}
