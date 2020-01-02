g++ -Wall -fopenmp -o omp.out $1
echo ================
time ./omp.out < input.txt