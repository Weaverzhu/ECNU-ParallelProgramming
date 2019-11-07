#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    int num_try = atoi(argv[1]), hit = 0;
    int cx = 1, cy = 1;
    for (int i=0; i<num_try; ++i) {
        double x = randd(2), y = randd(2), dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy < 1) {
            ++hit;
        }
    }

    int global_hit = 0, global_try = world_size * num_try;
    MPI_Reduce(&hit, &global_hit, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) 
        printf("In total: %.2lf\n", 4.0 * global_hit / global_try);

    // -------------------------------------
    MPI_Finalize();

    return 0;
}
