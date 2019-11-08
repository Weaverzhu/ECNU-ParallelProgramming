#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
void Hello(void) {
    int rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("Hello from %d of %d\n", rank, thread_count);
}

int main(int argc, char const *argv[])
{
    
    int thread_count = strtol(argv[1], NULL, 10);
    #pragma omp parallel num_threads(thread_count)
    Hello();

    puts("FUCK");
    return 0;
}
