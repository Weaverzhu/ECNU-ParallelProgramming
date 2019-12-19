from random import randint, random
import os


# ============= config =================
outputFile = ".\\input.txt"
cudasource = ".\\cuda\\cuda.cu"
brutalforcesource = ".\\baoli\\main2.cpp"

cudarun = ".\\cuda.bat"
brutalforcerun = ".\\bf.bat"

msize = [1000, 1100]
ele_range = [0, 500]
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
            bucket.append(",{:.3f}".format(randomfloat(L,R)))
            # res = res + ",{:.3f}".format(randomfloat(L,R))
        # res = res + "\n"
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

cuda = Runner(cudasource, "cuda", cudarun)
brutalforce = Runner(brutalforcesource, "bf", brutalforcerun)

f = open(outputFile, "w")
f.write(randomMatrixStr(n, m, ele_range[0], ele_range[1]))
f.write(randomMatrixStr(m, k, ele_range[0], ele_range[1]))
f.close()

# os.system("{} {}".format(cudarun, cudasource))
# os.system("xcopy ./output.txt ./judger/cuda.txt")

# os.system("{} {}".format(brutalforcerun, brutalforcesource))
# os.system("xcopy ./output.txt ./judger/bf.txt")

cuda.go()
# brutalforce.go()

# cuda.diff(brutalforce)