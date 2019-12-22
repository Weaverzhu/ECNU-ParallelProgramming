from sys import argv
import sys, os
# ================ CONFIG ====================

nameListFile = "namelist.txt"
DBG = True

# ============================================

def loadPastNames():
    f = open(nameListFile, "r")
    text = f.read()
    res = text.split('\n')
    return res

def savePastNames(namelist):
    bak = sys.stdout
    sys.stdout = open(nameListFile, "w")
    for s in namelist:
        if (s.__len__() > 0):
            print(s)
        # pass
    sys.stdout = bak
    
def tryNewName(name, namelist):
    n = 100000
    for i in range(n):
        newname = "{}{}".format(name, i)
        # print("i={}, newname={}".format(i, newname))
        if (not namelist.count(newname)):
            return newname

namelist = loadPastNames()

if argv.__len__() != 2:
    print("Usage: python main.py {path}")

p = argv[1].split('\\')
# ass = argv[2]
filename = p[-1].split('.')[0]
ass = p[-1].split('.')[1]
ass = "." + ass
newname = tryNewName(filename, namelist)
print("newname={}".format(newname))
newname = newname
namelist.append(newname)
savePastNames(namelist)

# os.system("del /q port")
os.system("type {} > {}".format(argv[1], "./port/" + newname + ass))

