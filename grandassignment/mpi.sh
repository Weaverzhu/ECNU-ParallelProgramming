set -o errexit
mpic++ $1 -O3 -o mpi.out
time mpiexec -n 4 mpi.out