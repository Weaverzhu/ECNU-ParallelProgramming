#include <bits/stdc++.h>
using namespace std;
using LL = long long;

const int N = 5000;
double **a, **b, **c;


namespace input
{
inline bool scan_d(int &num)
{
    char in;
    bool IsN = false;
    in = getchar();
    if (in == EOF)
        return false;
    while (in != '-' && (in < '0' || in > '9'))
        in = getchar();
    if (in == '-')
    {
        IsN = true;
        num = 0;
    }
    else
        num = in - '0';
    while (in = getchar(), in >= '0' && in <= '9')
    {
        num *= 10, num += in - '0';
    }
    if (IsN)
        num = -num;
    return true;
}
inline bool scan_lf(double &num)
{
    char in;
    double Dec = 0.1;
    bool IsN = false, IsD = false;
    in = getchar();
    if (in == EOF)
        return false;
    while (in != '-' && in != '.' && (in < '0' || in > '9'))
        in = getchar();
    if (in == '-')
    {
        IsN = true;
        num = 0;
    }
    else if (in == '.')
    {
        IsD = true;
        num = 0;
    }
    else
        num = in - '0';
    if (!IsD)
    {
        while (in = getchar(), in >= '0' && in <= '9')
        {
            num *= 10;
            num += in - '0';
        }
    }
    if (in != '.')
    {
        if (IsN)
            num = -num;
        return true;
    }
    else
    {
        while (in = getchar(), in >= '0' && in <= '9')
        {
            num += Dec * (in - '0');
            Dec *= 0.1;
        }
    }
    if (IsN)
        num = -num;
    return true;
}
} // namespace input

struct ios
{
    static const int IN_LEN = 1 << 18 | 1;
    char buf[IN_LEN], *s, *t;
    inline char read()
    {
        return (s == t) && (t = (s = buf) + fread(buf, 1, IN_LEN, stdin)), s == t ? -1 : *s++;
    }
    inline bool isEOF()
    {
        return (s == t) && (t = (s = buf) + fread(buf, 1, IN_LEN, stdin)), s == t;
    }
    inline ios &operator>>(int &x)
    {
        // input::scan_d(x);
        // return *this;


        static char c11, boo;
        for (c11 = read(), boo = 0; !isdigit(c11); c11 = read())
        {
            if (c11 == -1)
                return *this;
            boo |= c11 == '-';
        }
        for (x = 0; isdigit(c11); c11 = read())
            x = x * 10 + (c11 ^ '0');
        boo && (x = -x);
        return *this;
    }

    inline ios &operator>>(LL &x)
    {
        static char c11, boo;
        for (c11 = read(), boo = 0; !isdigit(c11); c11 = read())
        {
            if (c11 == -1)
                return *this;
            boo |= c11 == '-';
        }
        for (x = 0; isdigit(c11); c11 = read())
            x = x * 10 + (c11 ^ '0');
        boo && (x = -x);
        return *this;
    }

    inline ios &operator>>(char *s)
    {
        int len = 0;
        char ch;
        for (ch = read(); ch == '\n' || ch == ' '; ch = read())
            ;
        if (ch == -1)
        {
            s[len] = 0;
            return *this;
        }
        for (; ch != '\n' && ch != ' ' && ch != -1; ch = read())
            s[len++] = ch;
        s[len] = 0;
        return *this;
    }

    inline ios &operator>>(double &x)
    {

        char ch;
        bool neg = false, dec = false;
        double now = 0.1;
        for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

        if (ch == '-') neg = true;
        else if (ch == '.') { x = 0; dec = true; }
        else if (ch != -1) x = ch-'0';
        else return *this;
        if (!dec) {
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x = x * 10 + ch-'0';
            }
        }

        if (ch == '.')
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x += now * (ch - '0'); now *= 0.1;
            }
        if (neg) x = -x;
        
        return *this;
    }

    inline ios &operator>>(long double &x)
    {

        char ch;
        bool neg = false, dec = false;
        double now = 0.1;
        for (ch=read(); !isdigit(ch) && (ch!='.' && ch!='-') && ch!=-1; ch=read());

        if (ch == '-') neg = true;
        else if (ch == '.') { x = 0; dec = true; }
        else if (ch != -1) x = ch-'0';
        else return *this;
        if (!dec) {
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x = x * 10 + ch-'0';
            }
        }

        if (ch == '.')
            for (ch=read(); isdigit(ch) && ch!=-1; ch=read()) {
                x += now * (ch - '0'); now *= 0.1;
            }
        if (neg) x = -x;
        
        return *this;
    }
} io;

namespace output
{
const int OutputBufferSize = 1e6 + 5;

char buffer[OutputBufferSize];
char *s = buffer;
inline void flush()
{
    fwrite(buffer, 1, s - buffer, stdout);
    s = buffer;
    fflush(stdout);
}
inline void print(const char ch)
{
    if (s - buffer > OutputBufferSize - 2)
        flush();
    *s++ = ch;
}
inline void print(char *str)
{
    while (*str != 0)
        print(char(*str++));
}
inline void print(int x)
{
    char buf[25] = {0}, *p = buf;
    if (x < 0)
        print('-'), x = -x;
    if (x == 0)
        print('0');
    while (x)
        *(++p) = x % 10, x /= 10;
    while (p != buf)
        print(char(*(p--) + '0'));
}

inline void print(double x, char ch = ' ')
{
    static char buf[45];
    sprintf(buf, "%.2f%c", x, ch);
    print(buf);
}
} // namespace output


int main(int argc, char const *argv[])
{

    int an, am, bn, bm;
    // scanf("%d,%d", &an, &am);
    io >> an >> am;
    // cerr << an << ' ' << am << endl;
    a = (double **)malloc(sizeof(double *) * an);
    for (int i = 0; i < an; ++i)
    {
        a[i] = (double *)malloc(sizeof(double) * am);
        // scanf("%lf", &a[i][0]);
        io >> a[i][0];
        for (int j = 1; j < am; ++j)
        {
            // scanf(",%lf", &a[i][j]);
            // io.read();
            io >> a[i][j];
        }
    }

    // scanf("%d,%d", &bn, &bm);
    io >> bn >> bm;
    // cerr << bn << ' ' << bm << endl;
    b = (double **)malloc(sizeof(double *) * bn);
    for (int i = 0; i < bn; ++i)
    {
        b[i] = (double *)malloc(sizeof(double) * bm);
        // scanf("%lf", &b[i][0]);
        io >> b[i][0];
        for (int j = 1; j < bm; ++j)
        {
            // scanf(",%lf", &b[i][j]);
            // io.read();
            io >> b[i][j];
        }
    }
    assert(am == bn);
    c = (double **)malloc(sizeof(double) * an);
    for (int i = 0; i < an; ++i)
    {
        c[i] = (double *)malloc(sizeof(double) * bm);
        for (int j = 0; j < bm; ++j)
        {
            c[i][j] = 0;
            for (int k = 0; k < am; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
    }

    for (int i = 0; i < an; ++i)
    {

        for (int j = 0; j < bm; ++j)
        {
            printf("%.2lf%c", c[i][j], j == bm - 1 ? '\n' : ',');
            // output::print(c[i][j], j==bm? '\n':' ');
        }
    }
    // output::flush();
    return 0;
}
