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

    double sx, sy,  cx = 1, cy = 1;
    int seed = atoi(argv[1]); srand(seed);
    int num_try = atoi(argv[2]) / 4;
    switch (world_rank)
    {
    case 0:
        sx = sy = 0;
        break;
    case 1:
        sx = sy = 1;
        break;
    case 2:
        sx = 0; sy = 1;
        break;
    case 3:
        sx = 1; sy = 0;
        break;
    
    default:
        break;
    }

    int hit = 0, global_hit = 0;
    for (int i=0; i<num_try; ++i) {
        double x = sx + randd(1), y = sy + randd(1), dx = x - cx, dy = y - cy;
        if (dx*dx + dy*dy <= 1) ++hit;
    }
    MPI_Reduce(&hit, &global_hit, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("In proc %d, Pi = %.4lf\n", world_rank, 4.0 * global_hit / (num_try * world_size));
    }

    // -------------------------------------
    MPI_Finalize();

    return 0;
}
