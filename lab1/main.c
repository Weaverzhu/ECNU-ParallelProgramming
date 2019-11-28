#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (1<<20)

int outputflag;
char buf[N];

void handlemsg(int tag, int num, int world_rank, int count) {
    // printf("%d\n", world_rank);
    int dest = world_rank ^ 1;
    printf("%d\n", dest);
    if (world_rank == 0) {
        if (tag) {
            memset(buf, 'a', num);
            MPI_Send(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD);

            if (outputflag) printf("Process %d send at %d\n", world_rank, count);
        } else {
            MPI_Recv(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (outputflag) printf("Process %d recv at %d\n", world_rank, count);
        }
    } else {
        if (!tag) {
            memset(buf, 'a', num);
            MPI_Send(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD);

            if (outputflag) printf("Process %d send at %d\n", world_rank, count);
        } else {
            MPI_Recv(buf, num, MPI_CHAR, world_rank ^ 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (outputflag) printf("Process %d recv at %d\n", world_rank, count);
        }
    }
}

void go(int pingpongcnt, int datalen, int world_rank) {
    for (int i=0; i<pingpongcnt << 1; ++i) {
        // printf("%d\n", i);
        handlemsg((i&1)^1, datalen, world_rank, i);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int targetrank = world_rank ^ 1, datalen = atoi(argv[1]);
    // -------------------------------------

    for (int pingpongcnt = 1; pingpongcnt <= 100000; pingpongcnt *= 10) {
         struct timespec st, ed;
        double timediff;
        clock_gettime(CLOCK_MONOTONIC, &st);
        go(pingpongcnt, datalen, world_rank);
        clock_gettime(CLOCK_MONOTONIC, &ed);
        timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
        printf("Pingpong %d, datalen=%d, Time used: %.5lf\n", pingpongcnt, datalen, timediff);
    }

    int pingpongcnt = 100;
    for (datalen=1; datalen<=16384; datalen<<=1) {
         struct timespec st, ed;
        double timediff;
        clock_gettime(CLOCK_MONOTONIC, &st);
        go(pingpongcnt, datalen, world_rank);
        clock_gettime(CLOCK_MONOTONIC, &ed);
        timediff = ed.tv_sec - st.tv_sec + (ed.tv_nsec - st.tv_nsec) / 1000000000.0;
        printf("Pingpong %d, datalen=%d, Time used: %.5lf\n", pingpongcnt, datalen, timediff);
    }

    // -------------------------------------
    MPI_Finalize();

    return 0;
}
