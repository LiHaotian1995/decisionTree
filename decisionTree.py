from math import log
import operator
import numpy as np


# calculate the Shannon Entropy, which is Info(D)
def calculateEntropy(dataSet):
    numberEntropy = len(dataSet)
    # print("\n numberEntropy: " + str(numberEntropy))
    labelCount = {}
    for featureVec in dataSet:
        # print("\n featureVec: " + str(featureVec))
        # print("\n featureVec[-1]: " + str(featureVec[-1]))
        currentLabel = featureVec[-1]
        # print("\n labelCount.keys(): \n" + str(labelCount.keys()))
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    # print("\n labelCount: \n" + str(labelCount))

    entropy = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numberEntropy
        entropy -= prob*log(prob,2)

    return entropy


# split the dataSet according to the choose feature(by axis and its value)
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    # the selected feature is not in retDataSet
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    featureNumber = len(dataSet[0]) - 1
    basedEntropy = calculateEntropy(dataSet)  # Info(D)
    bestInfoGain = 0.0
    bestFeature = -1
    # print("\n basedEntropy: " + str(basedEntropy))

    for i in range(featureNumber):
        # get feature of each axis one by one, and save as featureList
        featureList = [feature[i] for feature in dataSet]
        # set: an unordered collection of unique elements(uniqueValues means feature Labels)
        uniqueValues = set(featureList)

        featureEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # print("\n prob: \n" + str(prob))
            featureEntropy += prob*calculateEntropy(subDataSet)   # Info_feature(D)
        infoGain = basedEntropy - featureEntropy   # Gain_feature = Info(D) - Info_feature(D)

        # update the best feature
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityClassName(classNameList):
    classNameCount = {}
    for vote in classNameList:
        if vote not in classNameCount:
            classNameCount[vote] = 0
        classNameCount[vote] += 1
    sortedClassNameCount = sorted(classNameCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassNameCount[0][0]


def createTree(dataSet, labels):
    # classNameList is the label name list(like ['yes', 'yes', 'no', 'no', 'no'])
    classNameList = [data[-1] for data in dataSet]

    # count: return the number of the classNameList[0](here 0 means the location of classNameList, first one)
    # if the first one of classNameList == the length of classNameList
    # (means every value of classNameList is same)
    if classNameList.count(classNameList[0]) == len(classNameList):
        return classNameList[0]
    # if the length of the first one of dataSet(such as [1, 1, 'yes']) == 1
    # (means there is no any other features)
    if len(dataSet[0]) == 1:
        # so we can directly use vote to decide which one is the bestFeature in this time
        return majorityClassName(classNameList)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]

    # decisionTree is a dict, {key: value}, (key is bestFeature, value is {})
    decisionTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])

    featureValues = [data[bestFeature] for data in dataSet]
    uniqueValue = set(featureValues)

    for value in uniqueValue:
        subLabels = labels[:]
        decisionTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)

    return decisionTree


# plot the decisionTree


# classify
def classify(decisionTree, featureLabelsName, testVec):

    # get the name of firstKey's str in decisionTree
    firstKeyStr = decisionTree.keys()[0]
    # print("\n firstKeyStr: " + str(firstKeyStr))

    # get the first firstKey's Value
    firstKeyValue = decisionTree[firstKeyStr]
    # print("\n firstKeyValue: " + str(firstKeyValue))

    # get the index number of featureLabelsName with given featureLabelsName
    featureIndex = featureLabelsName.index(firstKeyStr)
    # print("\n featureIndex: " + str(featureIndex))

    # # print("\n firstKeyValue.keys(): " + str(firstKeyValue.keys()))
    # for key in firstKeyValue.keys():
    #     # firstKeyValue.keys(): 0, 1
    #     if testVec[featureIndex] == key:
    #         # print type(firstKeyValue[key]).__name__
    #         # print firstKeyValue[key]
    #         if type(firstKeyValue[key]).__name__ == 'dict':
    #             classLabels = classify(firstKeyValue[key], featureLabelsName, testVec)
    #         else:
    #             classLabels = firstKeyValue[key]

    key = testVec[featureIndex]
    valueOfKey = firstKeyValue[key]
    if isinstance(valueOfKey, dict):
        classLabels = classify(valueOfKey, featureLabelsName, testVec)
    else:
        classLabels = valueOfKey

    return classLabels



# a small dataSet for test
def test():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no' ],
               [0,1,'no' ],
               [0,1,'no' ]]
    labels = ['no surfacing', 'flipper']

    # print("\ndataSet: \n" + str(dataSet))

    # result = calculateEntropy(dataSet)
    # print("\nresult: \n" + str(result))

    # retDataSet = splitDataSet(dataSet, 0, 1)
    # print("\nretDataSet: \n" + str(retDataSet))

    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print("\nbestFeature: \n" + str(bestFeature))

    decisionTree = createTree(dataSet, labels)
    print("\ndecisionTree: \n" + str(decisionTree))
    return decisionTree


def loadDataSet():
    f = open('/Users/lihaotian/PycharmProjects/decisionTree/lenses.txt', 'r')
    lenses = [data.strip().split('\t') for data in f.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    print("\n lenses: \n" + str(np.array(lenses)))
    return lenses, labels


if __name__ == '__main__':
    # decisionTree = test()
    # labels = ['no surfacing', 'flipper']

    # result = classify(decisionTree, labels, [1,1])
    # print("\n result: " + str(result))

    # test
    lenses, label = loadDataSet()
    # labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # tree = createTree(lenses, label)
    # print tree
    # result = classify(tree, labels, ['pre', 'hyper', 'yes', 'normal'])
    # print("\n result: " + str(result))
    calculateEntropy(lenses)
