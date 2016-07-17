from __future__ import print_function
from pyspark import SparkContext
import numpy as np
import Queue
import time


class Element:
    def __init__(self, id, value):
        self.id = id
        self.value = value

    def __lt__(self, element):
        return self.value > element.value


def knn(sampleData, label, data, k):
    difference=np.zeros(len(sampleData))
    q=Queue.PriorityQueue()
    for i in range(len(difference)):
        t = sampleData[i]-data
        difference[i] = t.dot(t)**0.5
        e = Element(i, difference[i])
        if q.qsize() < k:
            q.put(e)
            continue
        maxe = q.get()
        q.put(difference[i] < maxe.value and e or maxe)
    count=np.zeros(max(label)+1)
    while not q.empty():
        count[label[q.get().id]] += 1
    return count.argmax()

sampleData = []
label = []


def mp1(x):
    column = x.split(" ")
    label.append(int(column[1]))
    d=[]
    for i in range(2, len(column)):
        d.append(float(column[i]))
    d = np.array(d)
    id = int(column[0])
    if id < trainingSize:
        sampleData.append(d)
        return []
    return [(id, d)]


if __name__=="__main__":
    begin=time.time()
    sc = SparkContext(appName="knn")
    lines = sc.textFile("knnData_5000.data", 1)
    trainingSize = int(lines.count() * 0.7)
    result = lines.flatMap(mp1)
    result = result.flatMap(lambda x:knn(sampleData, label, x[1], 6) == label[x[0]] and [(x[0], 1)] or [])
    print(result.count()*1./(lines.count()-trainingSize))
    sc.stop()
    end=time.time()
    print(end-begin)
