import datetime
import pandas as pd
import pandas_datareader.data as web
import csv

data = []
featuresData = []
dateFeaturesData = []
classifiedData = []
rsi = []
dates = []

sym = ''

start = datetime.datetime(2016, 2, 25)
end = datetime.datetime(2020, 2, 25)

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
        return '-'

    for x in range(start - days, start):

        try:
            summ += float(data[x][5])
        except:
            print(data[x])

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


def addFeatures():

    global data
    global featuresData
    global rsi

    for x in range(len(data)):
        
        fiveDayMA = getMovingAverage(x, 5)
        tenDayMA = getMovingAverage(x, 10)
        fifteenDayMA = getMovingAverage(x, 15)
        twentyDayMA = getMovingAverage(x, 20)
        fiftyDayMA = getMovingAverage(x, 50)
        hundredDayMA = getMovingAverage(x, 100)
        twoHundredDayMA = getMovingAverage(x, 200)

        rsiVal = 'nan'

        if x > 0:
            rsiVal = rsi[x - 1]

        featuresData.append([data[x][0], data[x][1], data[x][2], data[x][3], data[x][4], data[x][5], fiveDayMA, tenDayMA, fifteenDayMA, twentyDayMA, fiftyDayMA, hundredDayMA, twoHundredDayMA, rsiVal])



'''
def classifyData():
    
    global counter
    global classifiedData

    with open('C:\\Users\\faiza\\OneDrive\\Desktop\\SPY Prices.csv') as csvfile:
        
        readCSV = csv.reader(csvfile, delimiter=',')
        
        for row in readCSV:
            
            counter += 1
            
            if counter >= 21:
                
                try:
                    adjClose = float(row[5])
                except:
                    print(row)

                if adjClose > getMovingAverage(counter, 20) and adjClose > getMovingAverage(counter, 9):
                    row[7] = 'Buy'

                elif adjClose < getMovingAverage(counter, 20) and adjClose < getMovingAverage(counter, 9):
                    row[7] = 'Sell'
            
            classifiedData.append(row)

'''

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

        adjClose = line

        for y in range(6):
            adjClose = adjClose[adjClose.find(',') + 1 : ]

        if 'Date' not in line:
            dates.append(line[0 : line.find(',')])

    for x in range(len(featuresData)):
        dateFeaturesData.append([dates[x], featuresData[x]])


def clean(stng):
    
    output = ''

    for letter in stng:
        if letter != '[' and letter != ']' and letter != "'":
            output += letter
        
    return output
         

def exportData(dataset):

    MyFile = open('C:\\Users\\faiza\\OneDrive\\Desktop\\PriceLabels.csv','w')

    header = 'Date, High, Low, Open, Close, Volume, Adj Close, MA5, MA10, MA15, MA20, MA50, MA100, MA200, RSI(14)' + '\n'
    MyFile.write(header)

    for element in dataset:
        
        MyFile.write(clean(str(element)))
        MyFile.write('\n')

    MyFile.close()


stock = 'SPY'

loadData(stock)  
addFeatures()

addDates(stock, featuresData)
exportData(dateFeaturesData)

#classifyData()
#print(classifiedData)
