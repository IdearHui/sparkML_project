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


def selectTopK(data, k):
    minIdx = 0
    maxIdx = len(data) - 1
    while True:
        i = minIdx
        j = maxIdx
        while i < j:
            while data[minIdx] < data[j]:
                j -= 1
            while i < j and data[minIdx] >= data[i]:
                i += 1
            t = data[i]
            data[i] = data[j]
            data[j] = t
        t = data[i]
        data[i] = data[minIdx]
        data[minIdx] = t
        rank = i - minIdx + 1
        if rank == k:
            return data[i]
        if rank > k:
            maxIdx = i - 1
        else:
            k -= rank
            minIdx = i + 1


def partition(data):
    mid = selectTopK(data.copy(), int((len(data) + 1) / 2))
    leftDataId = []
    rightDataId = []
    for i in range(len(data)):
        if data[i] <= mid:
            leftDataId.append(i)
        else:
            rightDataId.append(i)
    return mid, leftDataId, rightDataId


def getDistance(vertex, point):
    distance = 0
    for i in range(vertex[0].shape[0]):
        if point[i] < vertex[0, i]:
            distance += (point[i] - vertex[0, i]) ** 2
        elif point[i] > vertex[1, i]:
            distance += (point[i] - vertex[1, i]) ** 2
    return distance ** 0.5


class KDNode:
    def __init__(self, data, index, vertex, columnId, k):
        self.data = data
        self.index = index
        self.vertex = vertex
        self.leftChild = None
        self.rightChild = None
        i = columnId
        while True:
            self.mid, leftDataId, rightDataId = partition(data[:, i])
            if len(leftDataId) > 0 and len(rightDataId) > 0:
                self.columnId = i
                break
            i = (i + 1) % data.shape[1]
            if i == columnId:
                return
        if len(leftDataId) > 0 and len(rightDataId) > 0:
            leftVertex = vertex.copy()
            leftVertex[1, columnId] = self.mid
            self.leftChild = KDNode(data[leftDataId], index[leftDataId], leftVertex, (columnId + 1) % data.shape[1], k)
            rightVertex = vertex.copy()
            rightVertex[0, columnId] = self.mid
            self.rightChild = KDNode(data[rightDataId], index[rightDataId], rightVertex, (columnId + 1) % data.shape[1],
                                     k)

    def visit(self, d, q, k, maxDistance):
        if self.leftChild is None or self.data.shape[0] + q.qsize() <= k:
            for i in range(self.data.shape[0]):
                t = self.data[i] - d
                distance = t.dot(t) ** 0.5
                if q.qsize() < k:
                    q.put(Element(self.index[i], distance))
                elif distance < maxDistance:
                    q.get()
                    q.put(Element(self.index[i], distance))
            if q.qsize() == k:
                maxElement = q.get()
                maxDistance = maxElement.value
                q.put(maxElement)
            return maxDistance
        if d[self.columnId] <= self.mid:
            maxDistance = self.leftChild.visit(d, q, k, maxDistance)
            distance = getDistance(self.rightChild.vertex, d)
            if distance < maxDistance:
                maxDistance = self.rightChild.visit(d, q, k, maxDistance)
        else:
            maxDistance = self.rightChild.visit(d, q, k, maxDistance)
            distance = getDistance(self.leftChild.vertex, d)
            if distance < maxDistance:
                maxDistance = self.leftChild.visit(d, q, k, maxDistance)
        return maxDistance


class KDTree:
    def __init__(self, sampleData, k):
        self.sampleData = sampleData
        self.k = k
        vertex = np.zeros(2 * sampleData.shape[1]).reshape(2, sampleData.shape[1])
        for i in range(sampleData.shape[1]):
            vertex[0][i] = sampleData[:, i].min()
            vertex[1][i] = sampleData[:, i].max()
        self.root = KDNode(sampleData, np.arange(sampleData.shape[0]), vertex, 0, k)

    def visit(self, data):
        q = Queue.PriorityQueue()
        self.root.visit(data, q, self.k, 1e20)
        return q


def knn(kdTree, label, data):
    q = kdTree.visit(data)
    count = np.zeros(label.max()+1)
    result=[]
    while not q.empty():
        element=q.get()
        count[label[element.id]] += 1
        result.append((element.id,element.value))
    return count.argmax()


def mp1(x):
    column=x.split(" ")
    id = int(column[0])
    return id>=trainingSize and [(id,sampleData[id])] or []

if __name__=="__main__":
    begin=time.time()
    sc = SparkContext(appName="knn")
    lines = sc.textFile("knnData_50000.data", 1)
    trainingSize = int(lines.count() * 0.7)
    sampleData = []
    label = []
    for line in lines.collect():
        column = line.split(" ")
        label.append(int(column[1]))
        sampleData.append([])
        for i in range(2, len(column)):
            sampleData[-1].append(float(column[i]))
    sampleData = np.array(sampleData)
    label = np.array(label)
    kdTree = KDTree(sampleData[:trainingSize], 6)
    result=lines.flatMap(mp1).flatMap(lambda x:knn(kdTree, label,x[1])==label[x[0]] and [(x[0],1)] or [])
    print(result.count() * 1. / (lines.count() - trainingSize))
    sc.stop()
    end=time.time()
    print(end-begin)