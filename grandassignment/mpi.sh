set -o errexit
mpic++ $1 -O3
time ./a.out