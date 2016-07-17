# 森林植被分类
## 1. 项目思路
问题描述：依据环境、土壤等信息对森林植被类型进行分类预测  
问题解决流程：  
1.	预处理：数据集—>清理与规约—>数据划分  
2.	训练：训练集—>特征选取—>训练—>分类器  
3.	分类：测试集—>特征选取—>分类—>判决  
算法选取：对于分类器，选取了典型的贝叶斯、Lazy Learning和Trees三类算法加以实现
## 2. 运行环境
集群部署：master4G，两个worker各2G  
环境：Spark + Pycharm  
语言：Python  
## 3. 数据集描述
数据源：UCI数据集  
数据量：581012  
字段描述：54个属性字段，一个类别字段  
环境属性10个：数值属性，描述海拔、方位、斜角等环境信息  
野生区域属性4个：二元属性，标定区域类型  
土壤类型属性40个：二元属性，标记土壤类型  
森林植被类型：1~7，标记七种不同类型  
## 4. 算法描述
### 4.1 朴素贝叶斯
核心代码：  
&nbsp;*# 将数据按照60%和40%的比例分为训练集和测试集*  
&nbsp;(trainingData, testData) = data.randomSplit([0.6, 0.4], seed=0)  
&nbsp;*# 训练朴素贝叶斯模型* 
&nbsp;model = NaiveBayes.train(training, 1.0)  
参数说明：  
&nbsp;Addictive Smoothing:拉普拉斯平滑，为解决零概率问题进行平滑所需参数，这里为1.0
### 4.2 决策树
核心代码：  
&nbsp;*# 将数据按照70%和30%的比例分为训练集和测试集*  
&nbsp;(trainingData, testData) = data.randomSplit([0.7, 0.3])  
&nbsp;*# 训练决策树模型*  
&nbsp;model=DecisionTree.trainClassifier(trainingData,numClasses=8,categoricalFeaturesInfo={},impurity='gini', maxDepth=15, maxBins=32)  
参数说明：  
&nbsp;numClasses:分类数，需比实际类别数量大，这里设置为8；  
&nbsp;categoricalFeaturesInfo:特征类别信息，为空，意为所有特征为连续型变量；  
&nbsp;impurity:信息纯度度量，进行分类时可选择熵或基尼，这里设置为基尼；  
&nbsp;maxDepth:决策树最大深度，这里设为15；  
&nbsp;maxBins:特征分裂时的最大划分数量,这里设为32。
### 4.3 K近邻
核心代码：  
&nbsp;def knn(sampleData, label, data, k):  
&nbsp;&nbsp;difference=np.zeros(len(sampleData))  
&nbsp;&nbsp;q=Queue.PriorityQueue()  
&nbsp;&nbsp;for i in range(len(difference)):  
&nbsp;&nbsp;&nbsp;t = sampleData[i]-data  
&nbsp;&nbsp;&nbsp;difference[i] = t.dot(t)**0.5  
&nbsp;&nbsp;&nbsp;e = Element(i, difference[i])  
&nbsp;&nbsp;&nbsp;if q.qsize() < k:  
&nbsp;&nbsp;&nbsp;&nbsp;q.put(e)  
&nbsp;&nbsp;&nbsp;&nbsp;continue  
&nbsp;&nbsp;&nbsp;maxe = q.get()  
&nbsp;&nbsp;&nbsp;q.put(difference[i] < maxe.value and e or maxe)  
&nbsp;&nbsp;count=np.zeros(max(label)+1)  
&nbsp;&nbsp;while not q.empty():  
&nbsp;&nbsp;&nbsp;count[label[q.get().id]] += 1  
&nbsp;return count.argmax()  
参数说明：  
&nbsp;k为6，距离权重均为1，样本间距离使用欧几里得距离
### 4.4 随机森林
核心代码：
&nbsp;*# 将数据按照70%和30%的比例分为训练集和测试集*  
(trainingData, testData) = data.randomSplit([0.7, 0.3])  
&nbsp;*# 训练随机森林模型*  
&nbsp;model=RandomForest.trainClassifier(trainingData,numClasses=8,categoricalFeaturesInfo={},numTrees=20,featureSubsetStrategy="auto",impurity='gini',maxDepth=18,maxBins=32)  
参数说明：  
&nbsp;&nbsp;&nbsp;&nbsp;numClasses:分类数，需比实际类别数量大，这里设置为8；  
&nbsp;&nbsp;&nbsp;&nbsp;categoricalFeaturesInfo:特征类别信息，为空，意为所有特征为连续型变量；  
&nbsp;&nbsp;&nbsp;&nbsp;numTrees:森林中树的数量，这里设为20；  
&nbsp;&nbsp;&nbsp;&nbsp;featureSubsetStrategy:特征子集采样策略，auto表示算法自主选取；  
&nbsp;&nbsp;&nbsp;&nbsp;impurity:信息纯度度量，进行分类时可选择熵或基尼，这里设置为基尼；  
&nbsp;&nbsp;&nbsp;&nbsp;maxDepth:决策树最大深度，这里设为18；  
&nbsp;&nbsp;&nbsp;&nbsp;maxBins:特征分裂时的最大划分数量,这里设为32。  

