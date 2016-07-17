import numpy as np
fin = open("../covtype.data", "r")
sampleData = []
label = []
for line in fin.readlines()[:50000]:
    column = line.split(",")
    label.append(int(column[-1]))
    sampleData.append([])
    for i in range(len(column)-1):
        sampleData[-1].append(float(column[i]))
sampleData = np.array(sampleData)
label = np.array(label)
fin.close()
for i in range(10):
    sampleData[:, i] = (sampleData[:, i]-sampleData[:, i].min())/(sampleData[:, i].max()-sampleData[:, i].min())
fout = open("knnData_50000.data", "w")
for i in range(len(sampleData)):
    fout.write(str(i)+" "+str(label[i]))
    for j in range(len(sampleData[i])):
        fout.write(" "+str(sampleData[i][j]))
    fout.write("\n")
fout.close()
