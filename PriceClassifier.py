from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

data = []
featuresData = []
dateFeaturesData = []
classifiedData = []
rsi = []
dates = []

sym = ''

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2020, 2, 27)


def loadData(symbol):

    global data
    global rsi
    global sym
    global dates
    global start
    global end

    sym = web.DataReader(symbol, 'yahoo', start, end)
    
    print(sym.tail())
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

        featuresData.append([data[x][0], data[x][1], data[x][2], data[x][3], data[x][4], data[x][5], pctChange, 
        fiveDayMA, tenDayMA, fifteenDayMA, twentyDayMA, twenty5DayMA, thirtyDayMA, fiftyDayMA, hundredDayMA, twoHundredDayMA, 
        ema5, ema10, ema15, ema20, ema25, ema30, ema50, ema100, ema200, upperBand, lowerBand, stok, stod, cci, roc, fIndex, rsiVal])


def convertable(counter, days):
    return counter > days


def classifyData():
    
    global classifiedData

    counter = 0
    ratingCounter = 0
    classified = []
    

    with open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + stock + 'PriceLabels.csv') as csvfile:
        
        readCSV = csv.reader(csvfile, delimiter=',')
        
        rating = 'Hold'
        prevRating = 'Hold'

        for row in readCSV:
            
            counter += 1
            adjClose = ''
            rsiVal = ''
            changed = False

            try:
                adjClose = float(row[6])
            except:
                print(row)

            try:
                rsiVal = float(row[-1]) 
            except:
                print(row)


            #if convertable(counter, 20) == True and (rsiVal <= 35 or getMovingAverage(counter, 8) > getMovingAverage(counter, 12)):
            if convertable(counter, 200) == True and (adjClose > getMovingAverage(counter, 200) and getMovingAverage(counter, 5) > getMovingAverage(counter, 8)):    
            
                if rating == 'Hold' and prevRating != 'Buy':
                    rating = 'Buy'
                    prevRating = 'Buy'
                    changed = True

            #if convertable(counter, 20) == True and (rsiVal >= 70 or getMovingAverage(counter, 12) > getMovingAverage(counter, 8)):
            if convertable(counter, 200) == True and (getMovingAverage(counter, 8) > getMovingAverage(counter, 5)):    

                if rating == 'Hold' and prevRating == 'Buy':
                    rating = 'Sell'
                    prevRating = 'Hold'
                    changed = True


            if not changed:
                rating = 'Hold'

            if counter == 1:
                classifiedData.append([row, 'Rating'])
            else:
                classifiedData.append([row, rating])


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
         

def clean2(stng):

    output = ''

    for letter in stng:
        if letter != ',':
            output += letter
        
    return output


def exportData(dataset, datasetType):

    MyFile = open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + stock + 'PriceLabels.csv','w')

    if datasetType == 'Features':
        header = 'Date, High, Low, Open, Close, Volume, Adj Close, % Change, MA5, MA10, MA15, MA20, MA25, MA30, MA50, MA100, MA200, EMA5, EMA10, EMA15, EMA20, EMA25, EMA30, EMA50, EMA100, EMA200, Upper Band, Lower Band, Sto %K, Sto %D, CCI, ROC, Force Index, RSI(14)' + '\n'

        MyFile.write(header)

    for element in dataset:
        MyFile.write(clean(str(element)))
        MyFile.write('\n')

    MyFile.close()


def convertDataset(symbol):
    
    data = []
    mlData = []

    with open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + symbol + 'PriceLabels.csv') as csvfile:

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
            rating = 'Buy,Hold,Sell'

        counter += 1

        if row[34] == 'Buy':
            rating = '1,0,0'
        if row[34] == 'Hold':
            rating = '0,1,0'
        if row[34] == 'Sell':
            rating = '0,0,1'

        mlData.append([row[0 : 6], pcts, inds, rating])


    MyFile = open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + symbol + 'MLDataset.csv','w')

    for row in mlData:
        MyFile.write(clean(str(row)))
        MyFile.write('\n')

    MyFile.close()


def checkPerformance(symbol):

    buys = []
    sells = []

    totalProfit = 0
    winningTrades = 0
    losingTrades = 0

    avgWin = 0
    avgLoss = 0
    avgTrade = 0

    with open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + symbol + 'PriceLabels.csv') as csvfile:
    
        readCSV = csv.reader(csvfile, delimiter=',')
        
        for row in readCSV:
            date = row[0]
            adjClose = row[6]
            rating = row[-1]

            if rating == 'Buy':
                buys.append([date, adjClose])

            if rating == 'Sell':
                sells.append([date, adjClose])


    with open('C:\\Users\\faiza\\OneDrive\\Desktop\\' + symbol + 'AlgoStratPerformance.csv', 'w') as f:

        f.write('Buy Date, Sell Date, Buy Price, Sell Price, % Change' + '\n')

        for x in range(len(buys)):
            
            try:
                buyPrice = float(buys[x][1])
                sellPrice = float(sells[x][1])

                pctChange = str((sellPrice - buyPrice) / buyPrice * 100) + '%'
                totalProfit += (sellPrice - buyPrice)
                avgTrade += float(pctChange[:-1])

                if float(pctChange[:-1]) < 0: 
                    losingTrades += 1
                    avgLoss += float(pctChange[:-1])
                else:
                    winningTrades += 1
                    avgWin += float(pctChange[:-1])
                
                f.write(buys[x][0] + ',' + sells[x][0] + ',' + buys[x][1] + ',' + sells[x][1] + ',' + pctChange + '\n')

            except:
                f.write('\n')


        firstBuy = float(buys[0][1])

        ratio = ''
        avgLossPct = ''

        if losingTrades == 0:
            ratio = 100
            avgLoss = 0
            avgLossPct = '0'
        else:
            ratio = winningTrades / losingTrades 
            avgLossPct = avgLoss / losingTrades 

        f.write('\n\nTotal Profit:,' + str(totalProfit))
        f.write('\nGain %:,' + str((totalProfit / firstBuy) * 100) + '%')
        f.write('\nWinning Trades:,' + str(winningTrades))
        f.write('\nLosing Trades:,' + str(losingTrades))
        f.write('\nRatio:,' + str(ratio))
        f.write('\nAvg Win%:,' + str(avgWin / winningTrades) + '%')
        f.write('\nAvg Loss%:,' + str(avgLossPct) + '%')
        f.write('\nAvg Trade%:,' + str(avgTrade / len(buys)) + '%')


def trainModel(symbol):

    features = pd.read_csv('C:\\Users\\faiza\\OneDrive\\Desktop\\SPYAlgoDataset.csv')
    #print(features.head(5))
    #print(features.describe())

    #features = pd.get_dummies(features)

    # Labels are the values we want to predict
    labels = np.array(features['Buy'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features = features.drop('Buy', axis = 1)
    features = features.drop('Sell', axis = 1)
    features = features.drop('Hold', axis = 1)
    features = features.drop('Date', axis = 1)

    #features = features.drop('Rating_Sell', axis = 1)
    #features = features.drop('Rating_Hold', axis = 1)

    print(features)

    # Saving feature names for later use
    feature_list = list(features.columns)
    #print(feature_list)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.50, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)


    print('\n\nMean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))


'''
    newFeatures = pd.read_csv('C:\\Users\\faiza\\OneDrive\\Desktop\\AAPLPriceLabels.csv')
    
    newFeatures = pd.get_dummies(newFeatures)
    print('\n\n\nNew Features:', newFeatures)

    newLabels = np.array(newFeatures['Rating_Buy'])

    newFeatures = newFeatures.drop('Rating_Buy', axis = 1)
    newFeatures = newFeatures.drop('Rating_Hold', axis = 1)
    newFeatures = newFeatures.drop('Rating_Sell', axis = 1)

    feature_list2 = list(newFeatures.columns)

    newFeatures = np.array(newFeatures)

    train_features2, test_features2, train_labels2, test_labels2 = train_test_split(newFeatures, newLabels, test_size = 0.5, random_state = 42)
    
    print('\n\n\nTest Features:', test_features2)

    rf.fit(train_features2, train_labels2)

    predictions2 = rf.predict(test_features2)

    print('\n\n\nPredictions:', predictions2)
'''



stock = 'AAPL'

'''
loadData(stock)
addFeatures()

addDates(stock, featuresData)
exportData(dateFeaturesData, 'Features')

classifyData()
exportData(classifiedData, 'Classified')
'''

#checkPerformance(stock)

trainModel(stock)

#https://pythonprogramminglanguage.com/machine-learning-classifier/
#https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code

