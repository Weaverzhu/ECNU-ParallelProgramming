#include <iostream>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>

using namespace std;

typedef pair<int, int> P;
typedef double ld;

template<typename T>
class Matrix {
private:
    P st, ed;
    T **a;
public:
    int n, m;
    Matrix () {
        a = NULL;
        n = m = 0;
        st = ed = {0, 0};
    }

    Matrix (int n, int m) {
        a = (T**)malloc(sizeof(T*) * n);
        printf("Allocate %p\n", a);
        st = {0, 0};
        ed = {n, m};
        this->n = n;
        this->m = m;
        for (int i=0; i<n; ++i)
            a[i] = (T*)malloc(sizeof(T) * m);
    }

    inline T &at(int x, int y) {
        assert(st.first+x < ed.first);
        assert(st.second+y<ed.second);
        return a[st.first+x][st.second+y];
    }

    ~Matrix() {
        // printf("Deallocate %p\n", a);
        // free(a);
    }

    void read() {
        for (int i=0; i<n; ++i) {
            scanf("%lf", &a[i + st.first][0 + st.second]);
            for (int j=1; j<=m; ++j) {
                scanf("%lf", &a[i + st.first][j + st.second]);
            }
        }
    }

    void dump(Matrix &other, P sht = {0, 0}) const {
        for (int i=st.first; i<ed.first; ++i) {
            int size = sizeof(T) * (ed.second - st.second);
            // memcpy((void*)&other.a[other.st.first + sht.first][other.st.second + sht.second], (void*)a[st.first+i][st.second], size);
            memcpy((void*)&other.at(sht.first, sht.second), (void*)&this->at(i, 0), size);
        }
    }

    Matrix subMatrix(P st1, P ed1) const {
        assert(st.first + ed1.first < ed.first);
        assert(st.second + ed1.second < ed.second);
        assert(st1.first <= ed1.first);
        assert(st1.second <= ed1.second);

        Matrix res; res.a = this->a;
        res.st = make_pair(st.first + st1.first, st.second + st1.second);
        res.ed = make_pair(st.first + ed1.first, st.second + ed1.second);
        res.n = ed1.first - st1.first;
        res.m = ed1.second - st1.second;
    }

    Matrix operator + (const Matrix &other) const {
        assert(n == other.n);
        assert(m == other.m);
        Matrix res(n, m);
        for (int i=0; i<n; ++i) {
            for (int j=0; j<m; ++j) {
                res.at(i, j) = at(i, j) + other.at(i, j);
            }
        }
    }

    Matrix operator * (Matrix &other) {
        assert(m == other.n);
        Matrix res(n, other.m);
        for (int i=0; i<n; ++i) {
            for (int j=0; j<other.m; ++j) {
                res.at(i, j) = 0;
                for (int k=0; k<m; ++k) {
                    res.at(i, j) += at(i, k) * other.at(k, j);
                }
            }
        }
        return res;
    }

    inline void output() const {
        for (int i=0; i<n; ++i) {
            printf("%.2f", this->at(i, 0));
            for (int j=1; j<m; ++j) {
                printf(" %.2f", at(i, j));
            }
            putchar('\n');
        }
    }

    void deallocate() {
        for (int i=0; i<n; ++i)
            free(a[i]);
        free(a);
    }
};

template<typename T>
class StrassenMatrix : public Matrix<T> {
private:
    bool checkParity() const {
        return n % 2 == 0 && m % 2 == 0;
    }

public:
    StrassenMatrix operator * (StrassenMatrix &other) {
        assert(m == other.n);
        assert(checkParity() && other.checkParity());
        StrassenMatrix<T> res(n, other.m);

        
    }
};


void foo(const Matrix<ld> &a) {
    Matrix<ld> b = a;
    b.output();
}

int main(int argc, char const *argv[])
{   
    int n, m;
    scanf("%d%d", &n, &m);
    Matrix<ld> a(n, m);
    a.at(1, 0) = 1;
    a.at(0, 1) = 2;
    printf("%d %d\n", a.n, a.m);
    a.output();

    Matrix<ld> b(2, 2);
    b.at(0,0) = 2;
    b.at(1,1) = 2;
    Matrix<ld> c = a * b;
    foo(a);

    b.deallocate();
    a.deallocate();
    c.deallocate();

    return 0;
}
