# This file just imports all the data from the first dataset and fills in the class object called Attributes
import csv
import sys

attributesOfDataSetOne = []
testAttributesDataSetOne = []
validationAttributesDataSetOne = []

attributesOfDataSetTwo = []
testAttributesDataSetTwo = []
validationAttributesDataSetTwo = []

pathForDatasetOne = "dataset_1/dataset_1/"
pathForDatasetTwo = "data_sets2/"

headingsOfAttributes = []

with open(pathForDatasetOne + sys.argv[1]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
			headingsOfAttributes.extend(row)
			firstRow = True
			continue

		attributesOfDataSetOne.append(row)

# sys.argv[2] is for validation set
with open(pathForDatasetOne + sys.argv[2]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
			#headingsOfAttributes.extend(row)
			firstRow = True
			continue

		validationAttributesDataSetOne.append(row)


with open(pathForDatasetOne + sys.argv[3]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
			#headingsOfAttributes.extend(row)
			firstRow = True
			continue

		testAttributesDataSetOne.append(row)


with open(pathForDatasetTwo + sys.argv[1]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
		#	headingsOfAttributes.extend(row)
			firstRow = True
			continue

		attributesOfDataSetTwo.append(row)


#Validation dataset
with open(pathForDatasetTwo + sys.argv[2]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
		#	headingsOfAttributes.extend(row)
			firstRow = True
			continue

		validationAttributesDataSetTwo.append(row)


with open(pathForDatasetTwo + sys.argv[3]) as csv_file:
	firstRow = False
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if firstRow == False:
		#	headingsOfAttributes.extend(row)
			firstRow = True
			continue

		testAttributesDataSetTwo.append(row)

def getDataOfSetOne():
	return attributesOfDataSetOne, validationAttributesDataSetOne, testAttributesDataSetOne, headingsOfAttributes

def getDataOfSetTwo():
	return attributesOfDataSetTwo, validationAttributesDataSetTwo, testAttributesDataSetTwo, headingsOfAttributes

def getHeadingOfAttribute(attributeNumber):
	return headingsOfAttributes[attributeNumber]