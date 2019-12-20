from random import randint, random
import os, math

# ============= config =================
outputFile = ".\\input.txt"
cudasource = ".\\cuda\\cuda.cu"
brutalforcesource = ".\\baoli\\main2.cpp"

cudarun = ".\\cuda.bat"
brutalforcerun = ".\\bf.bat"

msize = [100, 500]
ele_range = [0,10000]


gendata = False
# gendata = True
# ======================================

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
                bucket.append(",{:.3f}".format(randomfloat(0, 1)))
        j = 1

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
        ret = os.system("{} {}".format(self.runner, self.sourcepath))
        if ret != 0:
            print("err: running {} failed".format(self.name))
        ret = os.system("xcopy /y .\\output.txt {}".format(self.outputfile))
        if ret != 0:
            print("err: copying the output of {} failed".format(self.name))
        print("log: running {} completed".format(self.runner))

    def diff(self, otherRunner):
        os.system("fc {} {}".format(self.outputfile, otherRunner.outputfile))
        pass


n = randint(msize[0], msize[1])
m = randint(msize[0], msize[1])
k = randint(msize[0], msize[1])

# n = 1
# m = 7000
# k = 1

cuda = Runner(cudasource, "cuda", cudarun)
brutalforce = Runner(brutalforcesource, "bf", brutalforcerun)

if gendata:
    f = open(outputFile, "w")
    f.write(randomMatrixStr(n, m, ele_range[0], ele_range[1]))
    f.write(randomMatrixStr(m, k, ele_range[0], ele_range[1]))
    f.close()

cuda.go()
# brutalforce.go()

# cuda.diff(brutalforce)
# f1 = open(cuda.outputfile, "r")
# f2 = open(brutalforce.outputfile, "r")

# ans1 = float(f1.read())
# ans2 = float(f2.read())

# print(ans1, ans2)
# print(math.fabs(1 - ans1 / ans2))