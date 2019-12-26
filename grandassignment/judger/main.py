from random import randint, random
import os, math

import time

# ============= config =================
outputFile = ".\\input.txt"
inputFileLinux = "./input.txt"

cudarun = ".\\cuda.bat"
brutalforcerun = ".\\bf.bat"
mpirun = "./mpi.sh"
brutallinux = "./bf.sh"

msize = [123, 1000]
ele_range = [1,1000000]
# gendata = True
gendata = False

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
    tmpinputfile = [
        ".\\input.txt",
        "./input.txt"
    ]

    tmpoutputfile = [
        ".\\output.txt",
        "./output.txt"
    ]


    def __init__(self, sourcepath, name, runner, isInLinux = 0):
        self.sourcepath = sourcepath
        self.name = name
        self.runner = runner
        prefix = ".\\judger\\"
        self.isLinux = isInLinux
        if isInLinux == 1:
            prefix = "./judger/"
        self.outputfile = prefix + self.name + ".txt"

    def go(self):
        print("log: start to run {}".format(self.runner))
        starttime = time.time()
        ret = os.system("{} {}".format(self.runner, self.sourcepath))
        endtime = time.time()
        if ret != 0:
            print("err: running {} failed".format(self.name))
        
        ret = os.system("xcopy /y {} {}".format(Runner.tmpoutputfile[self.isLinux], self.outputfile))
        
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
                    print("log: diff val: i={} j={} f1={} f2={}, err_val={}".format(i, j, f1, f2, err_val))
                    return True
        print("log: no diff err!, Accepted")
        return False

        pass
        
n = randint(msize[0], msize[1])
m = randint(msize[0], msize[1])
k = randint(msize[0], msize[1])
# n = 1
# m = 5000
# k = 1
# n = 1000
# m = 10
# k = 1000
# n = 33
# m = 1
# k = 33

cuda = Runner(".\\cuda\\tilewithoutcheat.cu", "cuda", cudarun)
brutalforce = Runner(".\\baoli\\main.cpp", "bf", brutalforcerun)
brutalforcelinux = Runner("./baoli/main.cpp", "main", brutallinux)
cudakbio = Runner(".\\cuda\\asyncio3.cu", "cudakbio", cudarun)
mpi = Runner("./mpi/mpi.cpp", "mpi", mpirun)

def runtest(gendata = True, main = cuda, std = brutalforce, ele_range = [0, 1000000], isInLinux = False):
    if gendata:
        inputfile = outputFile
        if (isInLinux):
            inputfile = inputFileLinux


        f = open(inputfile, "w")
        f.write(randomMatrixStr(n, m, ele_range[0], ele_range[1]))
        f.write(randomMatrixStr(m, k, ele_range[0], ele_range[1]))
        f.close()
    
    main.go()
    
    if gendata:
        
        std.go()
        f1 = open(main.outputfile, "r")
        f2 = open(std.outputfile, "r")

        mainAns = matrixcmp(f1.read())
        bfAns = matrixcmp(f2.read())
        res = mainAns.diff(bfAns)
        f1.close()
        f2.close()

def cudatest():
    runtest(gendata, cuda, brutalforce, ele_range)

def mpitest():
    runtest(gendata, mpi, brutalforcelinux, ele_range, True)

cudatest();
# mpitest();
# test1 = Runner('.\\baoli\\main2.cpp', 'main2', brutalforcerun)
# test2 = Runner('.\\baoli\\main.cpp', 'main', brutalforcerun)
# runtest(True, test1, test2, [0,1000])