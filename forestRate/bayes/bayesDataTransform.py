# coding=utf-8
import numpy as np
fin = open("../covtype.data", "r")
fout = open("bayesData.data", "w")
data = []
for line in fin.readlines():
    text = line.split(",")
    string = text[-1][:-1]
    data.append([int(string)])
    for i in range(len(text)-1):
        data[-1].append(float(text[i]))
fin.close()
data = np.array(data)
for i in range(1, 11):
    data[:, i] = (data[:, i]-data[:, i].min())/(data[:, i].max()-data[:, i].min()+1)*3
for i in range(len(data)):
    fout.write(str(int(data[i][0]))+",")
    for j in range(1, len(data[i])):
        fout.write(str(int(data[i][j]))+(j == len(data[i])-1 and "\n" or " "))
fout.close()
