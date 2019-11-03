#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (1<<20)

#define PING_PONG_LIMIT 5

char buf[N];

void handlemsg(int tag, int num, int world_rank, int count) {
    if (world_rank == 0) {
        if (tag) {
            memset(buf, 'a', num);
            MPI_Send(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD);

            printf("Process %d send at %d\n", world_rank, count);
        } else {
            MPI_Recv(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            printf("Process %d recv at %d\n", world_rank, count);
        }
    } else {
        if (!tag) {
            memset(buf, 'a', num);
            MPI_Send(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD);

            printf("Process %d send at %d\n", world_rank, count);
        } else {
            MPI_Recv(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            printf("Process %d recv at %d\n", world_rank, count);
        }
    }
}

int main(int argc, char const *argv[])
{
    MPI_Init(&argc, argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int pingpongcnt = atoi(argv[1]) << 1, targetrank = world_rank ^ 1, datalen = atoi(argv[2]);

    // -------------------------------------

    int i;
    for (i=0; i<pingpongcnt; ++i) {
        // printf("%d\n", i);
        if (i & 1) { // 0 to 1
            handlemsg(0, datalen, world_rank, i);
        } else { // 1 to 0
            handlemsg(1, datalen, world_rank, i);
        }
    }

    // -------------------------------------
    MPI_Finalize();

    return 0;
}
