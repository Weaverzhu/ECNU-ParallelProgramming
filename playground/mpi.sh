mpicxx $1 -o mpi.out -Wall -std=c++0x 
echo ================
time mpiexec mpi.out -n 4 mpi