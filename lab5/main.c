#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define max(a, b) (a<b? a:b)
#define min(a, b) (a>b? a:b)

double f(double x);
void busywaiting();
void mutex(int n, int thread_count, double a, double b);
void semaphore(int n, int thread_count, double a, double b);

double global_sum;
int flag, sem;
int n;
double a, b, h;
int thread_count;
int remain, num;

int main(int argc, char const *argv[])
{
    a = 1; b = 10;
    busywaiting(2);
    
    return 0;
}

double f(double x) { return x;}

void *bw(void *rank) {
    long my_rank = (long)rank;
    int l = min(my_rank, remain) * (num+1) + max(0, my_rank-remain) * num;
    int r = l + num - (my_rank >= remain);
    double local_sum = 0;
    for (int i=l; i<=r; ++i) {
        local_sum += f(h*(i+1)+a);
    }
    while(flag != my_rank);
    global_sum += local_sum;
    printf("%.5lf %.5lf %ld\n", global_sum, local_sum, my_rank);
    flag++;
    return NULL;
}

void busywaiting(int tc) {
    thread_count = tc;
    long thread;
    flag = 0;
    pthread_t *thread_handles = malloc(thread_count * sizeof(pthread_t));
    global_sum = (f(a) + f(b)) / 2;
    h = (b-a) / n;
    num = (n-1) / thread_count;
    remain = (n-1) % thread_count;
    
    for (thread=0; thread<thread_count; ++thread)
        pthread_create(&thread_handles[thread], NULL, bw, (void*)thread);
    
    for (thread=0; thread<thread_count; ++thread)
        pthread_join(thread_handles[thread], NULL);

    global_sum *= h;
    printf("Globalsum = %.5lf with %d\n", global_sum, tc);
    free(thread_handles);
}
