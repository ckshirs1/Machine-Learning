import os
import pandas as pd
import stopWords
import math

seriesInitToZeroForHam = []
columnIndexForHam = []
seriesInitToZeroForSpam = []
columnIndexForSpam = []

dfForHam = pd.DataFrame()
dfForSpam = pd.DataFrame()

docIndexForSpam = 0
docIndexForHam = 0

# find unique words append to the list if not in the list to the first row of the list
def appendUniqueWordsAsColumnsForHam(tokens):
	for token in tokens:
		if token not in dfForHam.columns.values:
			# add column with that column label and init all the elements to zero of the elements in spam or ham 
			dfForHam[token] = pd.Series(seriesInitToZeroForHam, columnIndexForHam)
			dfForHam.at[docIndexForHam, token] = dfForHam.at[docIndexForHam, token] + 1

		else:
			#if token is found
			# update for the current document index the values for the given column token
			dfForHam.at[docIndexForHam, token] = dfForHam.at[docIndexForHam, token] + 1

def appendUniqueWordsAsColumnsForSpam(tokens):
	for token in tokens:
		if token not in dfForSpam.columns.values:
			# add column with that column label and init all the elements to zero of the elements in spam or ham 
			dfForSpam[token] = pd.Series(seriesInitToZeroForSpam, columnIndexForSpam)
			dfForSpam.at[docIndexForSpam, token] = dfForSpam.at[docIndexForSpam, token] + 1

		else:
			#if token is found
			# update for the current document index the values for the given column token
			dfForSpam.at[docIndexForSpam, token] = dfForSpam.at[docIndexForSpam, token] + 1


pathForHam = "./train/train/ham"
pathForSpam = "./train/train/spam"

filesInHam = []
filesInSpam = []
# r=root, d=directories, f = files


def init():
	for i in range(len(filesInHam)):
		seriesInitToZeroForHam.append(0)
		columnIndexForHam.append(i)
		seriesInitToZeroForSpam.append(0)
		columnIndexForSpam.append(i)

#reading from the ham files
for r, d, f in os.walk(pathForHam):
    for file in f:
        if '.txt' in file:
            #frequencyTable.append([])
            filesInHam.append(os.path.join(r, file))


init()
# parsing the content in ham files
for file in filesInHam:	
	fileContent = open(file, "r").read()
	tokens = fileContent.split()
	appendUniqueWordsAsColumnsForHam(tokens)
	print("filename:", file)
	#addFrequencyCount(tokens)
	docIndexForHam = docIndexForHam + 1

print(dfForHam)

for r, d, f in os.walk(pathForSpam):
    for file in f:
        if '.txt' in file:
            filesInSpam.append(os.path.join(r, file))


for file in filesInSpam:
	f = open(file, "r", encoding="latin-1")	
	fileContent = f.read()
	tokens = fileContent.split()
	appendUniqueWordsAsColumnsForSpam(tokens)
	print("filename:", file)
	docIndexForSpam = docIndexForSpam + 1
	
print(dfForSpam)

def getTotalVocabularyCount():
	# it is sum of total spam + total ham words found	
	return len(dfForHam.columns) + len(dfForSpam.columns)

def getTotalNForSpam():
	return calculateNForDF(dfForSpam)

def getTotalNForHam():
	return calculateNForDF(dfForHam)

def calculateNForDF(df):
	tokens = df.columns.values
	docIndexForDF = 0
	print("row count:",len(df.index))
	rowCount = len(df.index)
	totalN = 0
	for token in tokens:
		for i in range(rowCount):
			totalN = totalN + df.at[i, token]

	return totalN

totalVocabulary = getTotalVocabularyCount()

totalNForHam = getTotalNForHam()
totalNForSpam = getTotalNForSpam()

print(totalVocabulary)

print(totalNForSpam)
print(totalNForHam)

# Now to calculate probability for every word given the label

# so say for ham words

probabilitiesForHam = pd.DataFrame(columns = ["token", "probability"])
probabilitiesForSpam = pd.DataFrame(columns = ["token", "probability"])

def calculateProbabilitiesForAllHam():
	global probabilitiesForHam
	tokens = dfForHam.columns
	for token in tokens:
		probabilitiesForHam = probabilitiesForHam.append({"token": token, "probability": 
			calculateProbabilityForToken(token, dfForHam, totalNForHam)}, ignore_index = True)
	print(probabilitiesForHam)
	return probabilitiesForHam

def calculateProbabilitiesForAllSpam():
	global probabilitiesForSpam
	tokens = dfForSpam.columns
	for token in tokens:
		probabilitiesForSpam = probabilitiesForSpam.append({"token": token, "probability": 
			calculateProbabilityForToken(token, dfForSpam, totalNForSpam)}, ignore_index = True)
	print(probabilitiesForSpam)
	return probabilitiesForSpam

def calculateProbabilityForToken(token, df, n):
	nk = 0
	for docIndex in range(len(df.index)):	
		nk = df.at[docIndex, token] + nk

	return ((nk + 1)/(n + totalVocabulary))


def calculateProbabilityOfSpam():
	return len(filesInSpam)/(len(filesInHam) + len(filesInSpam))


def calculateProbabilityOfHam():
	return len(filesInHam)/(len(filesInHam) + len(filesInSpam))



# Test class for the testing the sample on the model


count = 0

probabilitiesForHam = calculateProbabilitiesForAllHam()
probabilitiesForSpam = calculateProbabilitiesForAllSpam()

probabilityOfSpam = calculateProbabilityOfSpam()
probabilityOfHam = calculateProbabilityOfHam()



pathForTestHam = "./test/ham"
pathForTestSpam = "./test/spam"

testFilesInHam = []
testFilesInSpam = []
# r=root, d=directories, f = files

defaultProbabilityForHam = (1/(totalNForHam + totalVocabulary))
defaultProbabilityForSpam = (1/(totalNForSpam + totalVocabulary))

def calculateProbabilityForSpam(tokens):
	#print("Inside the calculate")
	probability = math.log(probabilityOfSpam)
	for token in tokens:
		if (probabilitiesForSpam["token"] == token).any():
			row = probabilitiesForSpam.loc[probabilitiesForSpam["token"] == token]
			#print("token:",token , " probability:", row["probability"])
			probability = probability + math.log(row["probability"])
			#print("calc probability:", probability)
		else:
		#	print("In else : defaultProbabilityForSpam:",defaultProbabilityForSpam)
			probability = probability + math.log(defaultProbabilityForSpam)
	print("\nTotal probability log for spam:",probability)
	print("\n\ntotal prob for spam:",math.exp(probability))
	return probability
	#return math.exp(probability)
	
def calculateProbabilityForHam(tokens):
	#print("Inside the calculate")
	probability = math.log(probabilityOfHam)
	for token in tokens:
		if (probabilitiesForHam["token"] == token).any():
			row = probabilitiesForHam.loc[probabilitiesForHam["token"] == token]
			#print("token:", row["probability"])
			probability = probability + math.log(row["probability"])
			#print("probability:", probability," exp:", math.exp(probability))
		else:
		#	print("In else : defaultProbabilityForHam:",defaultProbabilityForHam)
			probability = probability + math.log(defaultProbabilityForHam)
	print("\nTotal probability log for ham:",probability)
	print("\n\ntotal prob for ham:",math.exp(probability))
	return probability
	#return math.exp(probability)

resultForSpamDocs = []
resultForHamDocs = []

def predictSpamOrHamForHamDocs(fileContent):
	#print("In predict")
	tokens = fileContent.split()
	probabilityForSpam = calculateProbabilityForSpam(tokens)
	probabilityForHam = calculateProbabilityForHam(tokens)
	print("probability of spam:", probabilityForSpam)
	print("probability of ham:", probabilityForHam)
	if probabilityForHam > probabilityForSpam: 
		resultForHamDocs.append(1)
		print("it is ham")		
	else:
		resultForHamDocs.append(0)
		print("it is spam")


def predictSpamOrHamForSpamDocs(fileContent):
	#print("In predict")
	tokens = fileContent.split()
	probabilityForSpam = calculateProbabilityForSpam(tokens)
	probabilityForHam = calculateProbabilityForHam(tokens)
	print("probability of spam:", probabilityForSpam)
	print("probability of ham:", probabilityForHam)
	if probabilityForHam > probabilityForSpam: 
		resultForSpamDocs.append(1)
		print("it is ham")		
	else:
		resultForSpamDocs.append(0)
		print("it is spam")


count = 0
#reading from the ham files
for r, d, f in os.walk(pathForTestHam):
    for file in f:
       	if count < 1000:
	       	if '.txt' in file:
		        #frequencyTable.append([])
		        testFilesInHam.append(os.path.join(r, file))
        count = count + 1

for file in testFilesInHam:	 
	fileContent = open(file, "r").read()
	predictSpamOrHamForHamDocs(fileContent)


#reading from the spam files
for r, d, f in os.walk(pathForTestSpam):
    for file in f:
       	if '.txt' in file:
	        testFilesInSpam.append(os.path.join(r, file))
	    
for file in testFilesInSpam:	 
	fileContent = open(file, "r", encoding="latin-1").read()
	predictSpamOrHamForSpamDocs(fileContent)

def getAccuracyForHamDocs():
	count = 0 # count for spam
	for i in resultForHamDocs:
		if i == 1:
			count = count + 1
	return (count)/(len(resultForHamDocs))	

def getAccuracyForSpamDocs():
	count = 0 # count for spam
	for i in resultForSpamDocs:
		if i == 0:
			count = count + 1
	return (count)/(len(resultForSpamDocs))	


print(getAccuracyForHamDocs())
print(getAccuracyForSpamDocs())

print(resultForHamDocs)
print(resultForSpamDocs)

stringForHam = "\n\nAccuracy with test dataset for Ham with stop words included:" + str(getAccuracyForHamDocs())

stringForSpam = "\nAccuracy with test dataset for Spam with stop words included:" + str(getAccuracyForSpamDocs())

f = open("./results.txt", "a")
f.write(stringForHam)
f.write(stringForSpam)
