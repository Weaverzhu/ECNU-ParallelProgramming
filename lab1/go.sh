set -o errexit
make main
echo running with 16 byte
time mpiexec -n 2 ./bin/main 16
