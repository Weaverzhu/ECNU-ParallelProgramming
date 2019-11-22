#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

#define max(a, b) (a<b? a:b)
#define min(a, b) (a>b? a:b)

double f(double x);
void busywaiting(int tc, int n);
void mutex(int tc, int n);
void semaphore(int tc, int n);

double global_sum;
int flag;
int n;
double a, b, h;
int thread_count;
int remain, num;

pthread_mutex_t mut;
sem_t sem;

int main(int argc, char const *argv[])
{
    a = 1; b = 10; n = 1000000;

    for (int tc=1; tc<=10; ++tc) {
        for (n = 100000000; n<=100000000; n*=10) {
            puts("-----------");
            busywaiting(tc, n);
            mutex(tc, n);
            semaphore(tc, n);
            puts("-----------");
        }
    }
    

    
    return 0;
}

double f(double x) { return x * x;}

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
    flag++;
    return NULL;
}

void busywaiting(int tc, int n) {
    struct timespec st, ed;
    double timediff;
    clock_gettime(CLOCK_MONOTONIC, &st);

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
    clock_gettime(CLOCK_MONOTONIC, &ed);
    timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
    printf("Busy waiting: Globalsum = %.5lf with %d, n = %d, time used: %.5lfs\n", global_sum, tc, n, timediff);
    free(thread_handles);
}

void *m(void *rank) {
    long my_rank = (long)rank;
    int l = min(my_rank, remain) * (num+1) + max(0, my_rank-remain) * num;
    int r = l + num - (my_rank >= remain);
    double local_sum = 0;
    for (int i=l; i<=r; ++i) {
        local_sum += f(h*(i+1)+a);
    }
    pthread_mutex_lock(&mut);
    global_sum += local_sum;
    pthread_mutex_unlock(&mut);
    return NULL;
}

void mutex(int tc, int n) {
    struct timespec st, ed;
    double timediff;
    clock_gettime(CLOCK_MONOTONIC, &st);

    thread_count = tc;
    long thread;
    flag = 0;
    pthread_t *thread_handles = malloc(thread_count * sizeof(pthread_t));
    global_sum = (f(a) + f(b)) / 2;
    h = (b-a) / n;
    num = (n-1) / thread_count;
    remain = (n-1) % thread_count;

    pthread_mutex_init(mut, NULL);

    for (thread=0; thread<thread_count; ++thread)
        pthread_create(&thread_handles[thread], NULL, bw, (void*)thread);
    
    for (thread=0; thread<thread_count; ++thread)
        pthread_join(thread_handles[thread], NULL);

    global_sum *= h;
    clock_gettime(CLOCK_MONOTONIC, &ed);
    timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
    printf("Busy waiting: Globalsum = %.5lf with %d, n = %d, time used: %.5lfs\n", global_sum, tc, n, timediff);
    free(thread_handles);
}

void *s(void *rank) {
    long my_rank = (long)rank;
    int l = min(my_rank, remain) * (num+1) + max(0, my_rank-remain) * num;
    int r = l + num - (my_rank >= remain);
    double local_sum = 0;
    for (int i=l; i<=r; ++i) {
        local_sum += f(h*(i+1)+a);
    }
    sem_wait(&sem);
    global_sum += local_sum;
    sem_post(&sem);
    return NULL;
}

void semaphore(int tc, int n) {
    struct timespec st, ed;
    double timediff;
    clock_gettime(CLOCK_MONOTONIC, &st);

    thread_count = tc;
    long thread;
    flag = 0;
    pthread_t *thread_handles = malloc(thread_count * sizeof(pthread_t));
    global_sum = (f(a) + f(b)) / 2;
    h = (b-a) / n;
    num = (n-1) / thread_count;
    remain = (n-1) % thread_count;

    sem_init(&sem, 0, 1);
    pthread_mutex_init(mut, NULL);

    for (thread=0; thread<thread_count; ++thread)
        pthread_create(&thread_handles[thread], NULL, bw, (void*)thread);
    
    for (thread=0; thread<thread_count; ++thread)
        pthread_join(thread_handles[thread], NULL);

    global_sum *= h;
    clock_gettime(CLOCK_MONOTONIC, &ed);
    timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
    printf("Busy waiting: Globalsum = %.5lf with %d, n = %d, time used: %.5lfs\n", global_sum, tc, n, timediff);
    free(thread_handles);
}


