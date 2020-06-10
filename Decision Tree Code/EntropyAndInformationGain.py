import importData
import math
import tree
import copy
import sys

totalRowsCount = 0
totalAttributesCount = 0

attributeIndices = []

values = [0,1]

results = []

mainRoot = tree.node(None, None)

def startDecisionTreeForDataSetOne():
	f = open("results.txt", "w+")
	examples, validationAttributes, testAttributes, headings = importData.getDataOfSetOne()
	attributeCount = len(headings)

	global attributeIndices
	for i in range(attributeCount-1):
		attributeIndices.append(i)

	informationGainForAllAttrs = informationGainOverGivenAttributes(examples, attributeIndices)

	highestValueIndex = informationGainForAllAttrs.index(max(informationGainForAllAttrs))

	ID3(examples, attributeIndices, None)

	global mainRoot
	
	if sys.argv[4] == "yes":
		displayTree(mainRoot, mainRoot, 0)
	
	f.write("Data Set 1: "+ "\ntest data result:"+ str(testTreeOnTestDataSet(testAttributes, headings)) + 
		"\nvalidation data set results: "+ str(testTreeOnTestDataSet(validationAttributes, headings)))	
	

def startDecisionTreeForDataSetTwo():
	f = open("results.txt", "a+")
	examples, testAttributes, validationAttributes, headings = importData.getDataOfSetTwo()
	attributeCount = len(headings)
	global attributeIndices
	attributeIndices = []
	for i in range(attributeCount-2):
		attributeIndices.append(i)

	informationGainForAllAttrs = informationGainOverGivenAttributes(examples, attributeIndices)

	highestValueIndex = informationGainForAllAttrs.index(max(informationGainForAllAttrs))

	ID3(examples, attributeIndices, None)

	global mainRoot
	
	if sys.argv[3] == "yes":
		displayTree(mainRoot, mainRoot, 0)

	f.write("\n\nData Set 2: "+ " \ntest data result:"+ str(testTreeOnTestDataSet(testAttributes, headings)) + 
		"\nvalidation data set results: "+ str(testTreeOnTestDataSet(validationAttributes, headings)))

def testTreeOnTestDataSet(testAttributes, headings):
	global mainRoot
	global results
	results = []
	root = mainRoot

	for row in testAttributes:
		test(row, headings, root)

	count = findCorrectValueCount()	
	totalRowsCount = len(results)
	return (count/totalRowsCount)

def test(row, headings, root):
	index = headings.index(root.attributeName)
	value = row[index]	
	
	global results

	successors = root.successors

	for i in successors:
		#print(type(i.value), type(value))
		if(len(i.successors) != 0):
			svalue = i.successors[0].value
			if svalue == 0 or svalue == 1:
				#print("in ")
				if int(row[len(row)-1]) == svalue:
			#		print("found right value")
					results.append(1)
					return
				else:
			#		print("found wrong value")
					results.append(0)
					return

		if int(value) == i.value:
			if(len(i.successors) != 0):
				root = i.successors[0]
				test(row, headings, root)

def findCorrectValueCount():
	global results
	count = 0
	for i in results:
		if i == 1: 
			count = count + 1
	return count

def displayTree(root, parent, numberOfTabs):
	tabs = ""
	for i in range(numberOfTabs):
		tabs = tabs + " | "

	successors = root.successors

	if len(successors) == 0:
		sys.stdout.write(str(root.value))
		#print("leaf : " + str(root.value))
		return

	if root.value != None:
		if root.value != -1:
			#print("Value:",root.value)  parent.attributeName + " : " +
			sys.stdout.write("\n" + tabs + parent.attributeName + " = " + str(root.value) + " : ")

	numberOfTabs = numberOfTabs + 1

	for i in successors:
		displayTree(i, root, numberOfTabs)

def ID3(examples, attributeIndices, parent):
	
	if len(attributeIndices) == 0:
		#print("Attributes empty")
		return 

	if len(examples) == 0:
		#print("no examples ")
		return

	# check for examples if all examples are positive then return single root with label yes
	positiveClassCount, negativeClassCount = getPositiveAndNegativeClassCount(examples)

	if positiveClassCount > 0 and negativeClassCount == 0:
		#print("positive class : ",parent.attributeName, positiveClassCount)
		parent.successors.append(tree.node(1, None))
		return

	if positiveClassCount == 0 and negativeClassCount > 0:
		parent.successors.append(tree.node(0, None))
		return

	else:
		# begin with attribute with highest information gain
		informationGainForAllAttrs = informationGainOverGivenAttributes(examples, attributeIndices)
		if len(informationGainForAllAttrs) == 0:
			return

		highestValueIndex = informationGainForAllAttrs.index(max(informationGainForAllAttrs))
		highestValueIndexInAttributes = attributeIndices[highestValueIndex]
		highInfoGainAttr = importData.getHeadingOfAttribute(highestValueIndexInAttributes)

		# removing the highest gain attribute from the attributes list
		attributeIndices.remove(highestValueIndexInAttributes)

		# create a root with the attribute with highest information gain
		
		root = tree.node(None, None)

		if parent != None:
			root = tree.node(-1, highInfoGainAttr)
			parent.successors.append(root)

		else:
			global mainRoot
			root = tree.node(-1, highInfoGainAttr)
			mainRoot = root
		
		for i in values:
			examplesForGivenValue = getExamplesWithGivenValue(examples, i, highestValueIndexInAttributes)
			nodeWithGivenValue = tree.node(None, None)
			
			if len(examplesForGivenValue) != 0:
				nodeWithGivenValue = tree.node(i, None)
				root.successors.append(nodeWithGivenValue)
				ID3(examplesForGivenValue, attributeIndices, nodeWithGivenValue)


def calculateEntropy(examples):
	# Get first count of positive and negative class values
	# iterate over all rows and find the last column value and count 

	positiveClassCount, negativeClassCount = getPositiveAndNegativeClassCount(examples)
	return getEntropy(len(examples), positiveClassCount, negativeClassCount)


def getPositiveAndNegativeClassCount(examples):
	global totalRowsCount
	global totalAttributesCount

	totalRowsCount = len(examples)
	positiveClassCount = 0
	negativeClassCount = 0

	if(len(examples) != 0):
		totalAttributesCount = len(examples[0])

	for i in range(totalRowsCount):
		if(int(examples[i][totalAttributesCount-1]) == 1):
			positiveClassCount = positiveClassCount + 1

		if(int(examples[i][totalAttributesCount-1]) == 0):
			negativeClassCount = negativeClassCount + 1

	return positiveClassCount, negativeClassCount


def getEntropy(totalRowsCount, positiveClassCount, negativeClassCount):
	if totalRowsCount == 0:
		return 0

	fractionOfPositiveValues = positiveClassCount/totalRowsCount
	fractionOfNegativeValues = negativeClassCount/totalRowsCount
	
	logarithmOffractionOfPositiveValues = 0
	logarithmOffractionOfNegativeValues = 0

	if fractionOfPositiveValues > 0:
		logarithmOffractionOfPositiveValues = math.log(fractionOfPositiveValues, 2)

	else:	
		logarithmOffractionOfPositiveValues = 0

	if fractionOfNegativeValues > 0:
		logarithmOffractionOfNegativeValues = math.log(fractionOfNegativeValues, 2)

	else:
		logarithmOffractionOfNegativeValues = 0


	return (-(fractionOfPositiveValues * logarithmOffractionOfPositiveValues)
	 	-(fractionOfNegativeValues * logarithmOffractionOfNegativeValues))

def calculateInformationGainOfAttributes(examples, attributeIndices):
	fractionOfZeroValues = 0
	fractionOfOneValues = 0

	global totalRowsCount
	global totalAttributesCount

	for attrIndex in range(totalAttributesCount-1):
		totalRowsCount = len(examples)
		# count the number of instances with specific values for attribute i
		attrsWithValueOne = []
		attrsWithValueZero = []	

		for i in range(totalRowsCount):
			if int(examples[i][attrIndex]) == 0:
				attrsWithValueZero.append(examples[i])

			if int(examples[i][attrIndex]) == 1:
				attrsWithValueOne.append(examples[i])

		fractionOfZeroValues = len(attrsWithValueZero)/totalRowsCount
		fractionOfOneValues = len(attrsWithValueOne)/totalRowsCount

		informationGainForAllAttrs.append(calculateEntropy(examples)
		- ((fractionOfZeroValues * calculateEntropy(attrsWithValueZero)) 
		+ (fractionOfOneValues * calculateEntropy(attrsWithValueOne))))
	
	return informationGainForAllAttrs


def informationGainOverGivenAttributes(examples, attributeIndices):
	informationGainForAllAttrs = []
	for i in range(len(attributeIndices)-1):
		informationGainForAllAttrs.append(informationGainOverAnAttribute(examples, i))
	return informationGainForAllAttrs

def informationGainOverAnAttribute(examples, attributeNumber):
	fractionOfZeroValues = 0
	fractionOfOneValues = 0

	totalRowsCount = len(examples)

	if totalRowsCount == 0:
		return 

	attrsWithValueOne = []
	attrsWithValueZero = []	

	for i in range(totalRowsCount):
#		print("i, attributeN:", i, attributeNumber)
		if int(examples[i][attributeNumber]) == 0:
			attrsWithValueZero.append(examples[i])

		if int(examples[i][attributeNumber]) == 1:
			attrsWithValueOne.append(examples[i])

	fractionOfZeroValues = len(attrsWithValueZero)/totalRowsCount
	fractionOfOneValues = len(attrsWithValueOne)/totalRowsCount

	if sys.argv[5] == "information_gain":
		return (calculateEntropy(examples)
		- ((fractionOfZeroValues * calculateEntropy(attrsWithValueZero)) 
		+ (fractionOfOneValues * calculateEntropy(attrsWithValueOne))))

	if sys.argv[5] == "variance_impurity":
		return (calculateVarianceImpurity(examples)
			- ((fractionOfZeroValues * calculateVarianceImpurity(attrsWithValueZero)) 
			+ (fractionOfOneValues * calculateVarianceImpurity(attrsWithValueOne))))


def getExamplesWithGivenValue(examples, value, attributeNumber):
	totalRowsCount = len(examples)

	attrsWithValue = []	

	for i in range(totalRowsCount):
		if int(examples[i][attributeNumber]) == value:
			attrsWithValue.append(examples[i])

	return attrsWithValue


def calculateVarianceImpurity(examples):
	examplesCount = len(examples)
	if examplesCount == 0:
		return 0
	
	positiveClassCount, negativeClassCount = getPositiveAndNegativeClassCount(examples)
	return ((positiveClassCount * negativeClassCount)/(examplesCount * examplesCount))
	

startDecisionTreeForDataSetOne()
startDecisionTreeForDataSetTwo()