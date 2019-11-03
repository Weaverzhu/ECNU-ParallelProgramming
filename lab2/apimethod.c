#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (1<<20)

#define PING_PONG_LIMIT 5



int main(int argc, char const *argv[])
{

    // -------------------------------------
    MPI_Init(&argc, argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // -------------------------------------

    int num_per_proc = atoi(argv[1]), num_total_proc = atoi(argv[2]);
    int *arr = malloc(sizeof(int) * num_per_proc), local_sum = 0, global_sum;
    for (int i=0; i<num_per_proc; ++i) {
        arr[i] = num_per_proc * world_rank + i;
        local_sum += arr[i];
    }
    printf("Local sum for process %d, %d\n", world_rank, local_sum);

    for (int i=1; i<num_total_proc; i<<=1) {
        for (int j=i; j<num_total_proc; j+=i) {
            if (world_rank == j) {
                int target = j - i;
                MPI_Send(&global_sum, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            } else if (world_rank == j-i) {
                int target = i, rec;
                MPI_Recv(&rec, 1, MPI_INT, target, 0, MPI_INT, MPI_STATUS_IGNORE);
                global_sum += rec;
            }
        }
    }
    
    if (world_rank == 0) {
        printf("Total sum in process 0: %d\n", world_rank);
    }
    // -------------------------------------
    MPI_Finalize();

    return 0;
}
