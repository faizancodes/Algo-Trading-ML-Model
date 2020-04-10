import pandas as pd
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model, metrics
from matplotlib import pyplot as plt
import math

rowData = []
counter = 0


def clean(stng):
    
    output = ''

    for letter in stng:
        if letter != '[' and letter != ']' and letter != "'" and letter != ' ':
            output += letter
        
    return output


def addClassificationFeatures():
    
    with open(fileName +  datasetName, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    
    for x in range(len(data)):
    
        if x >= 2:
            #print(data[x - 1][1 : 33])
            rowData.append([data[x][:-2], data[x - 1][1 : 34], data[x][-2:]])
        elif x != 0:
            rowData.append([data[x][:-2], [0 for x in range(33)], data[x][-2:]])



    MyFile = open(fileName + saveAs, 'w')

    header = 'Date, High, Low, Open, Close, Volume, Adj Close, % Change, MA5, MA10, MA15, MA20, MA25, MA30, MA50, MA100, MA200, EMA5, EMA10, EMA15, EMA20, EMA25, EMA30, EMA50, EMA100, EMA200, Upper Band, Lower Band, Sto %K, Sto %D, CCI, ROC, Force Index, RSI(14), Prev High, Prev Low, Prev Open, Prev Close, Prev Volume, Prev Adj Close, Prev % Change, Prev MA5, Prev MA10, Prev MA15, Prev MA20, Prev MA25, Prev MA30, Prev MA50, Prev MA100, Prev MA200, Prev EMA5, Prev EMA10, Prev EMA15, Prev EMA20, Prev EMA25, Prev EMA30, Prev EMA50, Prev EMA100, Prev EMA200, Prev Upper Band, Prev Lower Band, Prev Sto %K, Prev Sto %D, Prev CCI, Prev ROC, Prev Force Index, Prev RSI(14), Next Day Up, Next Day Down' + '\n'

    MyFile.write(header)

    for element in rowData:
        MyFile.write(clean(str(element)))
        MyFile.write('\n')

    MyFile.close()
    print('\n\nSaved as: ' + fileName + saveAs)


def addRegressionFeatures():
    
    with open(fileName +  datasetName, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    
    for x in range(len(data)):
    
        if x >= 2 and x != len(data) - 1:
            rowData.append([data[x][:-2], data[x - 1][1 : 34], data[x + 1][6], data[x][-2:]])
        elif x != 0 and x != len(data) - 1:
            rowData.append([data[x][:-2], [0 for x in range(33)], data[x + 1][6], data[x][-2:]])


    MyFile = open(fileName + saveAs, 'w')

    header = 'Date, High, Low, Open, Close, Volume, Adj Close, % Change, MA5, MA10, MA15, MA20, MA25, MA30, MA50, MA100, MA200, EMA5, EMA10, EMA15, EMA20, EMA25, EMA30, EMA50, EMA100, EMA200, Upper Band, Lower Band, Sto %K, Sto %D, CCI, ROC, Force Index, RSI(14), Prev High, Prev Low, Prev Open, Prev Close, Prev Volume, Prev Adj Close, Prev % Change, Prev MA5, Prev MA10, Prev MA15, Prev MA20, Prev MA25, Prev MA30, Prev MA50, Prev MA100, Prev MA200, Prev EMA5, Prev EMA10, Prev EMA15, Prev EMA20, Prev EMA25, Prev EMA30, Prev EMA50, Prev EMA100, Prev EMA200, Prev Upper Band, Prev Lower Band, Prev Sto %K, Prev Sto %D, Prev CCI, Prev ROC, Prev Force Index, Prev RSI(14), Next Day Adj Close, Next Day Up, Next Day Down' + '\n'

    MyFile.write(header)

    for element in rowData:
        MyFile.write(clean(str(element)))
        MyFile.write('\n')

    MyFile.close()
    print('\n\nSaved as: ' + fileName + saveAs)


def testModel():

    features = pd.read_csv(fileName + saveAs)
    #print(features.head(5))
    print(features.describe())

    #features = pd.get_dummies(features)

    # Labels are the values we want to predict
    labels = np.array(features[' Next Day Up'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop(' Next Day Up', axis = 1)
    features = features.drop(' Next Day Down', axis = 1)
    features = features.drop('Date', axis = 1)

    #print(features)

    # Saving feature names for later use
    feature_list = list(features.columns)
    print(feature_list)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
    #rf = GaussianNB()
    
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    
    posError = 0
    posCounter = 0

    negError = 0
    negCounter = 0
    correctCounter = 0
    incorrectCounter = 0
 
    for x in range(len(test_labels)):
        
        if test_labels[x] == 1:
            posError += abs(test_labels[x] - predictions[x])
            posCounter += 1

        if test_labels[x] == predictions[x]:
            correctCounter += 1
        else:
            incorrectCounter += 1
        
        if test_labels[x] == 0:
            negError += abs(test_labels[x] - predictions[x])
            negCounter += 1

        #print(predictions[x], test_labels[x])

    print('\n\nMean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
    
    print('\nSensitivity:', str(posError / posCounter))
    print('Specificity:', str(negError / negCounter))
    
    print('\nCorrect Classifications:', correctCounter, '\nIncorrect Classifications:', incorrectCounter, '\nTotal Ratio:', correctCounter / (correctCounter + incorrectCounter))


def testLinearRegression():

    trainSet = 'AAPLMLDataset1990-2010TrainSet.csv'
    testSet = 'AAPLMLDataset2015-2020TestSet.csv'

    df = pd.read_csv(fileName + trainSet)

    features = list(df.columns)
    print(features)

    X = df[[' High', ' Low', ' Open', ' Close', ' Volume', ' Adj Close', ' % Change', ' MA5', ' MA10', ' MA15', ' MA20', ' MA25', ' MA30', ' MA50', ' MA100', ' MA200', ' EMA5', ' EMA10', ' EMA15', ' EMA20', ' EMA25', ' EMA30', ' EMA50', ' EMA100', ' EMA200', ' Upper Band', ' Lower Band', ' Sto %K', ' Sto %D', ' CCI', ' ROC', ' Force Index', ' RSI(14)', ' Prev High', ' Prev Low', ' Prev Open', ' Prev Close', ' Prev Volume', ' Prev Adj Close', ' Prev % Change', ' Prev MA5', ' Prev MA10', ' Prev MA15', ' Prev MA20', ' Prev MA25', ' Prev MA30', ' Prev MA50', ' Prev MA100', ' Prev MA200', ' Prev EMA5', ' Prev EMA10', ' Prev EMA15', ' Prev EMA20', ' Prev EMA25', ' Prev EMA30', ' Prev EMA50', ' Prev EMA100', ' Prev EMA200', ' Prev Upper Band', ' Prev Lower Band', ' Prev Sto %K', ' Prev Sto %D', ' Prev CCI', ' Prev ROC', ' Prev Force Index', ' Prev RSI(14)']] 
    Y = df[' Next Day Adj Close']   

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    #print('Intercept: \n', regr.intercept_)
    #print('Coefficients: \n', regr.coef_)

    #Predict Team Wins
    testData = pd.read_csv(fileName + testSet)

    predictions = regr.predict(testData[[' High', ' Low', ' Open', ' Close', ' Volume', ' Adj Close', ' % Change', ' MA5', ' MA10', ' MA15', ' MA20', ' MA25', ' MA30', ' MA50', ' MA100', ' MA200', ' EMA5', ' EMA10', ' EMA15', ' EMA20', ' EMA25', ' EMA30', ' EMA50', ' EMA100', ' EMA200', ' Upper Band', ' Lower Band', ' Sto %K', ' Sto %D', ' CCI', ' ROC', ' Force Index', ' RSI(14)', ' Prev High', ' Prev Low', ' Prev Open', ' Prev Close', ' Prev Volume', ' Prev Adj Close', ' Prev % Change', ' Prev MA5', ' Prev MA10', ' Prev MA15', ' Prev MA20', ' Prev MA25', ' Prev MA30', ' Prev MA50', ' Prev MA100', ' Prev MA200', ' Prev EMA5', ' Prev EMA10', ' Prev EMA15', ' Prev EMA20', ' Prev EMA25', ' Prev EMA30', ' Prev EMA50', ' Prev EMA100', ' Prev EMA200', ' Prev Upper Band', ' Prev Lower Band', ' Prev Sto %K', ' Prev Sto %D', ' Prev CCI', ' Prev ROC', ' Prev Force Index', ' Prev RSI(14)']])

    plt.plot(testData[' Next Day Adj Close'], label = 'Actual Price')
    plt.plot(predictions, label = 'Predicted Price')
    plt.title('AAPL Stock Price Predictions (2015 - 2020)')
    plt.legend() 
    plt.show()
    
    testData = testData.values.tolist()

    errSum = 0

    print("\n\nDate, Price, Predicted Price")

    for x in range(len(testData)):
        print(testData[x][0], testData[x][6], math.ceil(predictions[x]), 'Error: (' + str(abs(testData[x][6] - math.ceil(predictions[x]))) + ')\n') 
        errSum += abs(testData[x][6] - math.ceil(predictions[x]))

    print('Avg Error', errSum / len(testData))
  


fileName = 'C:\\Users\\faiza\\OneDrive\\Desktop\\Datasets\\'

datasetName = 'AAPLAlgoMLDataset1990-2010.csv'
saveAs = 'AAPLMLDataset1990-2010TrainSet.csv'


#testModel()

#addRegressionFeatures()

testLinearRegression()

#1:33