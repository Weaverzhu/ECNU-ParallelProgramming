#include <bits/stdc++.h>
using namespace std;
using LL = long long;

typedef int T;


struct Matrix {
	vector< vector<T> > a;
	Matrix(int n, int m) {
		a.resize(n);
		for (int i=0; i<n; ++i)
			a[i].resize(m);
	}

	Matrix operator * (const Matrix &other) const {
		assert(a[0].size() == other.a.size());
		Matrix res(a.size(), other.a[0].size());
		int n = a.size(), m = other.a[0].size(), kk = a[0].size();
		for (int i=0; i<n; ++i) {
			for (int j=0; j<m; ++j) {
				res.a[i][j] = 0;
				for (int k=0; k<kk; ++k) {
					res.a[i][j] += a[i][k] * other.a[k][j];
				}
			}
		}
		return res;
	}

	void padwithzero(int n) {
		int on = a.size(), om = a[0].size();
		a.resize(n);
		for (int i=0; i<on; ++i) {
			a[i].resize(n);
			for (int j=om; j<n; ++j)
				a[i][j] = 0;
		}
		for (int i=on; i<n; ++i) {
			a[i].resize(n);
			fill(a[i].begin(), a[i].end(), 0);
		}
	}

	pair<int, int> size() const {
		return make_pair(a.size(), a[0].size());
	}

	void output() const {
		int n = a.size(), m = a[0].size();
		for (int i=0; i<n; ++i) {
			for (int j=0; j<m; ++j) {
				printf("%d%c", a[i][j], j==m? '\n':' ');
			}
			putchar('\n');
		}
	}
};

namespace Strassen {
	
const int N = 256;

const int threshold = 32;

int a[N][N], b[N][N], c[N][N], nn;

void dump(const Matrix &x, int a[N][N]) {
	auto p = x.size();
	memset(a, 0, sizeof a);
	for (int i=0; i<p.first; ++i)
	for (int j=0; j<p.second; ++j) {
		a[i][j] = x.a[i][j];
	}
}

void doit(int n) {

}

Matrix strassen(Matrix aa, Matrix bb) {
	pair<int, int> s1 = aa.size(), s2 = bb.size();
	int N = max(s1.first, s1.second);
	N = max(N, s2.first); N = max(N, s2.second);
	nn = 1;
	while (nn < N) nn <<= 1;
	aa.padwithzero(nn);
	bb.padwithzero(nn);
	dump(aa, a);
	dump(bb, b);
}

}



int main(int argc, char const *argv[])
{
	int n;
	scanf("%d", &n);
	Matrix a(n, n), b(n, n);

	for (int i=0; i<n; ++i)
	for (int j=0; j<n; ++j)
		scanf("%d", &a.a[i][j]);
    
	for (int i=0; i<n; ++i)
	for (int j=0; j<n; ++j)
		scanf("%d", &b.a[i][j]);
	
	Matrix c = a * b; c.output();

	// a.padwithzero(8); b.padwithzero(8);
	// c = a * b;
	// c.output();
	return 0;
}

