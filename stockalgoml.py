from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix 

import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import csv

data = []
featuresData = []
dateFeaturesData = []
classifiedData = []
rsi = []
dates = []

sym = ''


def progress(count, total, status=''):
    
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def loadData(symbol):

    global data
    global rsi
    global sym
    global dates
    global start
    global end

    sym = web.DataReader(symbol, 'yahoo', start, end)
    
    #print(sym.tail())
    print('Loading daily ' + stock + ' data from ' + str(startMonth) + '-' + str(startDay) + '-' + str(startYear) + " -> " + str(endMonth) + '-' + str(endDay) + '-' + str(endYear))

    data = sym.values.tolist()

    rawRSI = computeRSI(sym['Adj Close'], 14)
    rsi = rawRSI.values.tolist()


def getMovingAverage(start, days):

    global data

    summ = 0

    if start - days < 0:
        return 0

    for x in range(start - days, start):

        try:
            summ += float(data[x][5])
        except:
            print()

    return summ / days
       

def computeRSI(data, time_window):
    
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    
    return rsi


def convertToPct(adjClose, movingAvg):

    if movingAvg == '-' or movingAvg == 0 or math.isnan(movingAvg):
        return 0
    
    return str((adjClose - movingAvg) / movingAvg * 100) + '%'


def stoK(close, low, high, n): 

    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK


def stoD(close, low, high, n):
    
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    
    return STOD


def CCI(data, n): 
    
    TP = (data['High'] + data['Low'] + data['Adj Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(n).mean()) / (0.015 * TP.rolling(n).std()), name = 'CCI') 
    
    return CCI


def ROC(data, n):
    
    N = data['Adj Close'].diff(n)
    D = data['Adj Close'].shift(n)
    ROC = pd.Series((N/D) * 100, name='Rate of Change')

    return ROC


def forceIndex(data, n): 
    
    fIndex = pd.Series(data['Adj Close'].diff(n) * data['Volume'], name = 'ForceIndex') 
    return fIndex / 100000


def cleanNan(num):
    
    if math.isnan(num):
        return 0
    
    return num


def addFeatures():

    global data
    global featuresData
    global rsi
    global sym

    for x in range(len(data)):
        
        adjClose = data[x][5]

        fiveDayMA = convertToPct(adjClose, getMovingAverage(x, 5))
        tenDayMA = convertToPct(adjClose, getMovingAverage(x, 10))
        fifteenDayMA = convertToPct(adjClose, getMovingAverage(x, 15))
        twentyDayMA = convertToPct(adjClose, getMovingAverage(x, 20))
        twenty5DayMA = convertToPct(adjClose, getMovingAverage(x, 25))
        thirtyDayMA = convertToPct(adjClose, getMovingAverage(x, 30))
        fiftyDayMA = convertToPct(adjClose, getMovingAverage(x, 50))
        hundredDayMA = convertToPct(adjClose, getMovingAverage(x, 100))
        twoHundredDayMA = convertToPct(adjClose, getMovingAverage(x, 200))

        ema5 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=5).mean()[x])
        ema10 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=10).mean()[x])
        ema15 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=15).mean()[x])
        ema20 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=20).mean()[x])
        ema25 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=25).mean()[x])
        ema30 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=30).mean()[x])
        ema50 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=50).mean()[x])
        ema100 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=100).mean()[x])
        ema200 = convertToPct(adjClose, pd.Series.ewm(sym['Adj Close'], span=200).mean()[x])

        std = sym['Adj Close'].rolling(window=20).std()
        upperBand = '-'
        lowerBand = '-'
        
        cci = cleanNan(CCI(sym, 14)[x])
        roc = cleanNan(ROC(sym, 9)[x])

        stok = cleanNan(stoK(sym['Adj Close'], sym['Low'], sym['High'], 14)[x])
        stod = cleanNan(stoD(sym['Adj Close'], sym['Low'], sym['High'], 14)[x])
        fIndex = cleanNan(forceIndex(sym, 8)[x])

        try:
            upperBand = convertToPct(adjClose, getMovingAverage(x, 20) + (std[x] * 2))
            lowerBand = convertToPct(adjClose, getMovingAverage(x, 20) - (std[x] * 2))

            if float(upperBand[:-1]) > 100 or float(upperBand[:-1]) < -100:
                upperBand = 0
            if float(lowerBand[:-1]) > 100 or float(lowerBand[:-1]) < -100:
                lowerBand = 0
        except:
            upperBand = 0
            lowerBand = 0

        rsiVal = '-'
        pctChange = '-'
        
        if x > 0:
            pctChange = str((adjClose - data[x - 1][5]) / data[x - 1][5] * 100) + '%'

        if x > 0:
            rsiVal = cleanNan(rsi[x - 1])

        progress(x, len(data), status='Adding Features...')

        featuresData.append([data[x][0], data[x][1], data[x][2], data[x][3], data[x][4], data[x][5], pctChange, 
        fiveDayMA, tenDayMA, fifteenDayMA, twentyDayMA, twenty5DayMA, thirtyDayMA, fiftyDayMA, hundredDayMA, twoHundredDayMA, 
        ema5, ema10, ema15, ema20, ema25, ema30, ema50, ema100, ema200, upperBand, lowerBand, stok, stod, cci, roc, fIndex, rsiVal])


def convertable(counter, days):
    return counter > days


def classifyData(fileName):
    
    global classifiedData
    global startYear
    global endYear

    counter = 0

    with open(fileName) as csvfile:
        
        readCSV = csv.reader(csvfile, delimiter=',')
        
        rawData = pd.read_csv(fileName)
        data = rawData.values.tolist()

        nextDayUp = 0
        nextDayDown = 0

        for row in readCSV:
            
            nextDayChange = ''

            try:
                nextDayChange = data[counter][7]

                if '-' in nextDayChange:
                    nextDayUp = 0
                    nextDayDown = 1
                else:
                    nextDayUp = 1
                    nextDayDown = 0

            except:
                print('')
            
            counter += 1

            if counter == 1:
                classifiedData.append([row, 'Next Day Up', 'Next Day Down'])
            else:
                classifiedData.append([row, nextDayUp, nextDayDown])


            progress(counter, len(data), status='Classifing Data...')


def addDates(symbol, dataset):

    global dateFeaturesData
    global featuresData
    global sym
    global dates

    df = web.DataReader(symbol, 'yahoo', start, end)
    path_out = 'C:\\Users\\faiza\\OneDrive\\Desktop\\pyTest\\'
    df.to_csv(path_out + '$' + symbol + '.txt')
    classFile = open("C:\\Users\\faiza\\OneDrive\\Desktop\\pyTest\\$" + symbol + '.txt', "r")

    for line in classFile:

        if 'Date' not in line:
            dates.append(line[0 : line.find(',')])

    for x in range(len(featuresData)):
        dateFeaturesData.append([dates[x], featuresData[x]])


def clean(stng):
    
    output = ''

    for letter in stng:
        if letter != '[' and letter != ']' and letter != "'" and letter != ' ':
            output += letter
        
    return output
         

def exportData(fileName, dataset, datasetType):

    #MyFile = open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + stock + 'PriceLabels2012.csv','w')
    MyFile = open(fileName, 'w')

    if datasetType == 'Features':
        header = 'Date, High, Low, Open, Close, Volume, Adj Close, % Change, MA5, MA10, MA15, MA20, MA25, MA30, MA50, MA100, MA200, EMA5, EMA10, EMA15, EMA20, EMA25, EMA30, EMA50, EMA100, EMA200, Upper Band, Lower Band, Sto %K, Sto %D, CCI, ROC, Force Index, RSI(14)' + '\n'

        MyFile.write(header)

    for element in dataset:
        MyFile.write(clean(str(element)))
        MyFile.write('\n')

    MyFile.close()
    print('\n\nSaved as: ' + fileName)


def convertDataset(fileName, symbol):
    
    data = []
    mlData = []

    print('\nConverting dataset from ' + fileName)

    with open(fileName) as csvfile:

        readCSV = csv.reader(csvfile, delimiter=',')
        
        for row in readCSV:
            data.append(row)

    counter = 0

    for row in data:
        
        pcts = []
        inds = []
        rating = ''

        for x in range(7, 28):
            
            if counter == 0:
                pcts.append(row[x])
                
            else:
                num = row[x]
                strNum = str(num[:-1])
                updatedNum = ''

                try:
                    updatedNum = float(strNum) / 100
                except:
                    updatedNum = 0

                pcts.append(updatedNum)

            
        for x in range(28, 34):
            num = row[x]

            if num == '-':
                inds.append(0)
            else:
                inds.append(num)

        
        if counter == 0:
            rating = 'Next Day Up, Next Day Down'

        counter += 1

        mlData.append([row[0 : 7], pcts, inds, row[34], row[35]])


    MyFile = open(fileName,'w')

    for row in mlData:
        MyFile.write(clean(str(row)))
        MyFile.write('\n')

    MyFile.close()
    print('\nSaved as ' + fileName)


def trainModel(fileName, symbol):

    print('\nTraining model...')

    features = pd.read_csv(fileName)
    #print(features.head(5))
    print(features.describe())

    #features = pd.get_dummies(features)

    # Labels are the values we want to predict
    labels = np.array(features['NextDayUp'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('NextDayUp', axis = 1)
    features = features.drop('NextDayDown', axis = 1)
    features = features.drop('Date', axis = 1)

    '''
    features = features.drop('High', axis = 1)
    features = features.drop('Low', axis = 1)
    features = features.drop('Open', axis = 1)
    features = features.drop('Close', axis = 1)
    features = features.drop('AdjClose', axis = 1)
    '''
    
    print(features.head(5))

    # Saving feature names for later use
    feature_list = list(features.columns)
    print(feature_list)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20, random_state = 42)

    print('\nTraining Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with decision trees
    #rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
    #rf = GaussianNB()
    rf = DecisionTreeClassifier()
    
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




startYear = 1990
endYear = 2020

startMonth = 1
startDay = 1

endMonth = 3
endDay = 30
 
start = datetime.datetime(startYear, startMonth, startDay)
end = datetime.datetime(endYear, endMonth, endDay)


stock = 'AAPL'

fileName = 'C:\\Users\\faiza\\OneDrive\\Desktop\\' + stock + 'AlgoMLDataset' + str(startYear) + '-' + str(endYear) + '.csv'


loadData(stock)
addFeatures()

addDates(stock, featuresData)
exportData(fileName, dateFeaturesData, 'Features')

classifyData(fileName)
exportData(fileName, classifiedData, 'Classified')

convertDataset(fileName, stock)


trainModel(fileName, stock)