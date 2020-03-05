import datetime
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import csv
import math

'''
data = []
featuresData = []
dateFeaturesData = []
classifiedData = []
rsi = []
dates = []

sym = ''

start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2020, 2, 28)


def loadData(symbol):

    global data
    global rsi
    global sym
    global dates
    global start
    global end

    sym = web.DataReader(symbol, 'yahoo', start, end)
    
    #print(sym.tail())
    data = sym.values.tolist()



loadData('SPY')

emas = pd.Series.ewm(sym['Adj Close'], span=10).mean()[0]

ma = sym['Adj Close'].rolling(window=20).mean()

std = sym['Adj Close'].rolling(window=20).std()

lowerBand = ma[100] - (std[100] * 2)


def stoK(close, low, high, n): 

    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK


def stoD(close, low, high, n):
    
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    
    return STOD


def CCI(data, ndays): 
    
    TP = (data['High'] + data['Low'] + data['Adj Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
    name = 'CCI') 
    
    return CCI


def ROC(data,n):
    N = data['Adj Close'].diff(n)
    D = data['Adj Close'].shift(n)
    ROC = pd.Series((N/D) * 100, name='Rate of Change')

    return ROC


def ForceIndex(data, n): 
    
    fIndex = pd.Series(data['Adj Close'].diff(n) * data['Volume'], name = 'ForceIndex') 
    return fIndex / 100000




k = stoK(sym['Adj Close'], sym['Low'], sym['High'], 14)[0]
print(k)

print()

d = stoD(sym['Adj Close'], sym['Low'], sym['High'], 14)
print(d[len(d) - 1])


roc = ROC(sym, 9)

f = ForceIndex(sym, 8)[0]

if math.isnan(f):
        print('-')
    
'''    

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


data = []
mlData = []

with open('C:\\Users\\faiza\\OneDrive\\Desktop\\SPYHandPriceLabels.csv') as csvfile:

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


MyFile = open('C:\\Users\\faiza\\OneDrive\\Desktop\\TestAlgoDataset.csv','w')

for row in mlData:
    MyFile.write(clean(str(row)))
    MyFile.write('\n')

MyFile.close()


'''
for x in range(len(data)):
    
    if x > 0:
    
        cell = clean(str(data[x]))
        modCell = cell[0 : cell.rfind(',')]

        cellAdj = modCell

        p1 = modCell[0 : modCell.find('%') - 6]
        p2 = ''
        p3 = modCell[modCell.rfind('%') + 2 : ]
        
        if x == 1:
            p1 = modCell[0 : modCell.find('-') - 1]
            
        for x in range(21):

            pct = clean2(cellAdj[cellAdj.find('%') - 5 : cellAdj.find('%')])

            if pct == '--' or '.' not in pct:
                break
            else:
                cellAdj = cellAdj[cellAdj.find('%') + 1 : ]
                p2 += str(float(pct) / 100) + ','
        

        row = p1 + p2 + p3

        print(row + '\n')

        MyFile.write(row)
        
        if 'Buy' in cell:
            MyFile.write(',1,0,0')
        if 'Hold' in cell:
            MyFile.write(',0,1,0')
        if 'Sell' in cell:
            MyFile.write(',0,0,1')

        MyFile.write('\n')


MyFile.close()
'''

#print(data)

