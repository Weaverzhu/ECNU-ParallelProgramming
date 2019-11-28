#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_PROC 4

double randd(double k) {
    return k * rand() / RAND_MAX;
}

int main(int argc, char **argv)
{

    // -------------------------------------
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // -------------------------------------

    for (int num_try = 500000; num_try <= 50000000; num_try*=10) {
        struct timespec st, ed;
        double timediff;
        clock_gettime(CLOCK_MONOTONIC, &st);
        int hit = 0;
        int cx = 1, cy = 1;
        for (int i=0; i<num_try; ++i) {
            double x = randd(2), y = randd(2), dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy < 1) {
                ++hit;
            }
        }

        int global_hit = 0, global_try = world_size * num_try;
        MPI_Reduce(&hit, &global_hit, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &ed);
        timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
        if (world_rank == 0) 
            printf("num_try = %d, In total: %.2lf, Time used=%lf\n",num_try, 4.0 * global_hit / global_try, timediff);
    }

    

    // -------------------------------------
    MPI_Finalize();

    return 0;
}
