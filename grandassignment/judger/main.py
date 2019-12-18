from random import randint, random
import os


# ============= config =================
outputFile = ".\\input.txt"
cudasource = ".\\cuda\\cuda.cu"
brutalforcesource = ".\\baoli\\main2.cpp"

cudarun = ".\\cuda.bat"
brutalforcerun = ".\\bf.bat"
# ======================================

def randomfloat(l, r):
    return random() * (r-l) + l


def randomMatrixStr(n, m, L=0, R=1000000):
    res = ""
    res = res + "{},{}\n".format(n, m)
    for i in range(n):
        res = res + "{:.3f}".format(randomfloat(L,R))
        for j in range(1,m):
            res = res + ",{:.3f}".format(randomfloat(L,R))
        res = res + "\n"

    return res

class Runner:
    def __init__(self, sourcepath, name, runner):
        self.sourcepath = sourcepath
        self.name = name
        self.runner = runner

    def go(self):
        print("log: start to run {}".format(self.runner))
        ret = os.system("{} {}".format(self.runner, self.sourcepath))
        if ret != 0:
            print("err: running {} failed".format(self.name))
        ret = os.system("xcopy /y .\\output.txt {}".format(".\\judger\\" + self.name + ".txt"))
        if ret != 0:
            print("err: copying the output of {} failed".format(self.name))
        print("log: running {} completed".format(self.runner))

    def diff(self, otherRunner):
        pass


n = randint(2, 2)
m = randint(2, 2)
k = randint(2, 2)

cuda = Runner(cudasource, "cuda", cudarun)
brutalforce = Runner(brutalforcesource, "bf", brutalforcerun)

f = open(outputFile, "w")
f.write(randomMatrixStr(n, m))
f.write(randomMatrixStr(m, k))
f.close()

# os.system("{} {}".format(cudarun, cudasource))
# os.system("xcopy ./output.txt ./judger/cuda.txt")

# os.system("{} {}".format(brutalforcerun, brutalforcesource))
# os.system("xcopy ./output.txt ./judger/bf.txt")

cuda.go()
brutalforce.go()