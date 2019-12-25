from random import randint, random
import os, math

import time

# ============= config =================
outputFile = ".\\input.txt"
cudasource = ".\\cuda\\cudakbio.cu"
brutalforcesource = ".\\baoli\\main.cpp"

cudarun = ".\\cuda.bat"
brutalforcerun = ".\\bf.bat"

msize = [30, 41]
ele_range = [-1222220,1220220]

# gendata = True;
gendata = True

def randomfloat(l, r):
    return random() * (r-l) + l


def randomMatrixStr(n, m, L=0, R=1000000):
    print("log: random Matrix prepare started")
    bucket = []
    bucket.append("{},{}\n".format(n, m))
    for i in range(n):
        if (i % 500 == 0):
            print("log: processing {}th row".format(i))
        bucket.append("{:.3f}".format(randomfloat(L,R)))
        for j in range(1,m):
            if randint(0, 1) == 1:
                bucket.append(",{:.3f}".format(randomfloat(L,R)))
            else:
                bucket.append(",{:.3f}".format(randomfloat(L,R)))
        

        bucket.append("\n")
    res = "".join(bucket)
    print("log: random Matrix prepare completed")
    return res

class Runner:
    def __init__(self, sourcepath, name, runner):
        self.sourcepath = sourcepath
        self.name = name
        self.runner = runner
        self.outputfile = ".\\judger\\" + self.name + ".txt"

    def go(self):
        print("log: start to run {}".format(self.runner))
        starttime = time.time()
        ret = os.system("{} {}".format(self.runner, self.sourcepath))
        endtime = time.time()
        if ret != 0:
            print("err: running {} failed".format(self.name))
        
        ret = os.system("xcopy /y .\\output.txt {}".format(self.outputfile))
        
        if ret != 0:
            print("err: copying the output of {} failed".format(self.name))
        print("log: running {} completed, time used: {}".format(self.runner, endtime - starttime))


    def diff(self, otherRunner):
        os.system("fc {} {}".format(self.outputfile, otherRunner.outputfile))
        pass

class matrixcmp:
    def __init__(self, mstr = ""):
        self.mstr = mstr
        self.rows = self.mstr.split('\n')
        self.a = [ [] for i in self.rows]
        self.n = self.m = 0
        self.n = self.rows.__len__() - 1
        idx = 0
        for row in self.rows:
            if (row.__len__() == 0):
                continue
            tmps = row.split(',')
            if (self.m == 0):
                self.m = tmps.__len__()
            elif self.m != tmps.__len__():
                print("FUCK")
                exit()
            for floatstr in tmps:
                self.a[idx].append(float(floatstr))
            # print(self.a[idx].__len__())
            idx = idx + 1
    def diff(self, other, eps = 1e-9):
        if self.n != other.n or self.m != other.m:
            print("log: diff matrix size, n={}, m={}, other.n={}, other.m={}", self.n, self.m, other.n, other.m)
            return True
        for i in range(self.n):
            for j in range(self.m):
                # print(i, j)
                f1 = self.a[i][j]
                f2 = other.a[i][j]
                
                err_val = math.fabs((f1 - f2) / max([f2, 1]))
                if err_val > eps:
                    print("log: diff val: i={} j={} f1={} f2={}".format(i, j, f1, f2))
                    return True
        print("log: no diff err!, Accepted")
        return False

        pass
        
# n = randint(msize[0], msize[1])
# m = randint(msize[0], msize[1])
# k = randint(msize[0], msize[1])

n = 1
m = 5000
k = 1

cuda = Runner(cudasource, "cuda", cudarun)
brutalforce = Runner(brutalforcesource, "bf", brutalforcerun)

if gendata:
    f = open(outputFile, "w")
    f.write(randomMatrixStr(n, m, ele_range[0], ele_range[1]))
    f.write(randomMatrixStr(m, k, ele_range[0], ele_range[1]))
    f.close()

cuda.go()

if gendata:
    brutalforce.go()

    # cuda.diff(brutalforce)
    f1 = open(cuda.outputfile, "r")
    f2 = open(brutalforce.outputfile, "r")

    cudaAns = matrixcmp(f1.read())
    bfAns = matrixcmp(f2.read())

    res = cudaAns.diff(bfAns)

    f1.close()
    f2.close()

