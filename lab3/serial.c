#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

double randd(double k) {
    return k * rand() / RAND_MAX;
}


int main(int argc, char const *argv[])
{
    int seed = atoi(argv[1]);
    int num_try = atoi(argv[2]);
    srand(seed);

    double cx = 1, cy = 1;

    int hit = 0;

    for (int i=0; i<num_try; ++i) {
        double x = randd(2.0), y = randd(2.0);
        double dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy <= 1) {
            ++hit;   
        }
    }

    printf("Outcome in serial program: %.4lf\n", 4.0 * hit / num_try);

    return 0;
}
