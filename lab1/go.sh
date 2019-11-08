set -o errexit
make main
echo running 1000 pingpong with 16 byte
time mpiexec -n 2 ./bin/main 10 16
echo runing 1000000 pingpong with 16 byte
time mpiexec -n 2 ./bin/main 1000000 16
echo running 1000 pingpong with 4096 byte
time mpiexec -n 2 ./bin/main 1000 4096
echo running 1000 pingpong with 16384 byte
time mpiexec -n 2 ./bin/main 1000 16384