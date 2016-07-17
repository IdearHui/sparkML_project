# coding=utf-8
import numpy as np
fin = open("../covtype.data", "r")
fout = open("rfData.data", "w")
data = []
for line in fin.readlines()[:20000]:
    text = line.split(",")
    string = text[-1][:-1]
    data.append([int(string)])
    for i in range(len(text)-1):
        data[-1].append(int(text[i]))
fin.close()
data = np.array(data)
for i in range(len(data)):
    fout.write(str(int(data[i][0])))
    for j in range(1, len(data[i])):
        fout.write(" "+str(j)+":"+str(data[i][j]))
    fout.write("\n")
fout.close()
