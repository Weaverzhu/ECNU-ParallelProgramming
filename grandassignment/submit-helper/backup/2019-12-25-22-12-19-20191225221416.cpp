#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
const int N = 5000;

typedef long long LL;

LL **a, **b, **c;


int main(int argc, char const *argv[])
{
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);

	int an, am, bn, bm;
	scanf("%d,%d", &an, &am);
	a = (LL**)malloc(sizeof(LL*) * an);
	register double x;
	for (int i=0; i<an; ++i) {
		a[i] = (LL*)malloc(sizeof(LL) * am);
		scanf("%lf", &x); a[i][0] = round(x * 1000);
		for (int j=1; j<am; ++j) {
			scanf(",%lf", &x);
			a[i][j] = round(x * 1000);
		}
	}

	scanf("%d,%d", &bn, &bm);
	b = (LL**)malloc(sizeof(LL*) * bn);
	for (int i=0; i<bn; ++i) {
		b[i] = (LL*)malloc(sizeof(LL) * bm);
		scanf("%lf", &x); b[i][0] = round(x * 1000);
		for (int j=1; j<bm; ++j) {
			scanf(",%lf", &x); b[i][j] = round(x * 1000);
		}
	}
	assert(am == bn);
	c = (LL**)malloc(sizeof(LL) * an);
	for (int i=0; i<an; ++i) {
		c[i] = (LL*)malloc(sizeof(LL) * bm);
		for (int j=0; j<bm; ++j) {
			c[i][j] = 0;
			// for (int k=0; k<am; ++k) c[i][j] += a[i][k] * b[k][j];
			for (int k=am-1; k>=0; --k) c[i][j] += a[i][k] * b[k][j];
		}
	}
	

	for (int i=0; i<an; ++i) {
		
		for (int j=0; j<bm; ++j) {
			// int remain = c[i][j] % 1000000 / 10000;
			double remain = 1.0 * c[i][j] / 1000000;
			// if (c[i][j] % 1000000 / 1000 % 10 >= 5) ++remain;
			// printf("%lld.", c[i][j] / 1000000);
			// if (remain < 10) putchar('0');
			// printf("%lld%c", remain, j==bm-1? '\n':',');
			printf("%.2lf%c", remain, j==bm-1? '\n':',');
		}
			// printf("%.2lf%c", c[i][j], j==bm-1? '\n':',');
	}
    return 0;
}

