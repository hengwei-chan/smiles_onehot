import numpy as np
import pandas

from datetime import datetime
#np.set_printoptions(threshold=sys.maxsize)

FILENAME = "tox21.csv"
DEBUG = True
VERBOSE = True

start = datetime.now()
#Obtains the dictionary of Symbols used in Smiles across the dataset, as well as setting them inside a dictionary as keys to be used for index lookup
def getSymbols(toxData):

    #get appropriate column
    smilesArray = toxData["smiles"]
    
    
    symbols = []
    #turn array into a single long string
    smilesString = ''.join(smilesArray)

    #for each character
    for index, char in enumerate(smilesString[:-1]):
        #skip c
        if char.islower() and char!='c':
            continue
        #if the next character is lowercase, not a c, this and next are alphanumerical and current one is uppercase
        if smilesString[index+1].islower() and smilesString[index+1]!='c' and smilesString[index+1].isalpha() and char.isalpha() and char.isupper():
            #create a combined symbol
            candidateSymbol = char + smilesString[index+1] 
        else:
            #in all other cases add the character directly
            candidateSymbol=char
        
        #if the symbol isn't in the list add it
        if candidateSymbol not in symbols:
            symbols.append(candidateSymbol)

    #add the last character if it's not lowercase (it was already added as partr of the previous) and not in symbols
    if smilesString[-1].isupper() and smilesString[-1] not in symbols:
        symbols.append(candidateSymbol)
    
    #generate indexes
    indexes = np.arange(len(symbols))
    #build dictionary and return it
    symbolsDict = dict(zip(symbols, indexes))
    return symbolsDict

#fills a tensor
def fillTensor(smile, inToxData):
    #get symbols from within Dataset
    containedSymbols = getSymbols(inToxData)
    #set the width of the tensor as the longest smile
    #not perfect, but good enough - counts all characters instead of symbols, but guarantees enough spaces
    inputTensorWidth = len(max(inToxData["smiles"], key=len))
    #set the height as the number of Symbols
    #perfect but not good enough
    inputtensorHeight = len(containedSymbols)
    
    #generate empty one hot matrix

    zerosTensor = np.zeros((inputtensorHeight, inputTensorWidth))
    #print("Zero Tensor Shape: ", zerosTensor.shape)
    smileArray = []

    #same as the getSymbols()
    for index, char in enumerate(smile[:-1]):
        if char.islower() and char!='c':
            continue
        if smile[index+1].islower() and smile[index+1]!='c' and smile[index+1].isalpha() and char.isalpha() and char.isupper():
            candidateSymbol = char + smile[index+1] 
        else:
            candidateSymbol=char
            
        smileArray.append(candidateSymbol)
    #include last
    if smile[-1].isupper() or not smile[-1].isalpha() or smile[-1]=="c":
        smileArray.append(smile[-1])
    
    
    
    #index moves through the columns on each iteration
    #sets to 1 on any the index from the lookup symbol dictionary of the current column
    index=0
    for i in smileArray:
        
        zerosTensor[containedSymbols[i]][index]=1
        #print(i, "(", index, containedSymbols[i], ")", zerosTensor[containedSymbols[i]][index])
        index+=1
    #print(zerosTensor.sum())
    return zerosTensor


#Load data
if VERBOSE:
    print("Reading data from FILENAME ", FILENAME, ".")

inToxData = pandas.read_csv(FILENAME)

if VERBOSE:
    print("Data Loaded.")
    print("Enable DEBUG to show symbol list.")
#gets the list of symbols from input smiles

#print each dictionary entry in newline
if DEBUG:
    containedSymbols = getSymbols(inToxData)
    print("List of symbols and indices:")
    print("{" + "\n".join("{!r}: {!r},".format(v, k) for k, v in containedSymbols.items()) + "}")

if DEBUG:
    print("Test for Tensor Fill, should be 9: ", int(fillTensor("CcNaBr#(S)c", inToxData).sum().sum()))


#iterate over rows
dataset = []

if VERBOSE:
    print("Generating dataset, enable DEBUG to see each iteration output")
for index, row in inToxData.iterrows():
    #print(row["smiles"])
    datapoint = []
    #iterate over columns, skip last which stores the smile
    for column in row.drop("smiles"):
        datapoint.append(column)

    #for smiles, transform into array, and append  
    datapoint.append(fillTensor(row["smiles"], inToxData))
    


    if DEBUG:
        print("DATAPOINT:: Index, Datapoint, smile sum: ", index, datapoint, sum(sum(datapoint[-1])))

        print("DATAPOINT:: LEN: ", len(datapoint))
    #append to final output tensor, to be used as input for training    
    dataset.append(datapoint)
    if DEBUG:
        print("DATASET:: Index, Datapoint, smile sum: ", index, dataset[-1], sum(sum(dataset[-1][-1])))

        print("DATASET:: LEN: ", len(dataset[-1]))
if VERBOSE:
    print("Converting dataset to NUMPY array...")
dataset = np.array(dataset)

if VERBOSE:
    print("Conversion Done.")
    print("Saving .NPY ...")
#save as NPY and NPZ
np.save(FILENAME.split('.')[0]+".npy", dataset)
if VERBOSE:
    print(".NPY saved.")
    print("Saving .NPZ ...")
np.savez(FILENAME.split('.')[0]+".npz", dataset)
if VERBOSE:
    print(".NPZ saved.")
    print("Dataset converted in ", str(datetime.now()-start), "s.")
