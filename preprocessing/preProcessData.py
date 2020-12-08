from datetime import datetime, timedelta
import sys, os


# inputCSVPath = "c:/Users/Nishant/Documents/CSE575_SML/csv/part2.csv"

inputCSVPath = "./yellow_tripdata_2009-01_point.csv"

# outputDirpath = "c:/Users/Nishant/Documents/CSE575_SML/csv/results/part2_2"

outputDirpath = "./results2"

startDateTime = "2009-01-01 00:00:00"
endDateTime = "2009-02-01 00:00:00"
minX = -74.50
maxX = -73.70
minY = 40.50
maxY = 40.90
numRows = 75
numColumns = 75
minuteWindow = 30
isMinuteWindow = False
hourWindow = 1
isHourWindow = True
possibleMinuteWindows = {5, 10, 15, 20, 30}
possibleHourWindows = {1, 2, 3, 4, 6, 8, 12}

def validate():
    if ((isMinuteWindow and isHourWindow) or ((not isMinuteWindow) and (not isHourWindow))):
        raise Exception('Minute and Hour time windows can not be set together!') 

    if (isMinuteWindow and minuteWindow not in possibleMinuteWindows):
        raise Exception('Unsupported minute window! supported values are: '+ str(possibleMinuteWindows)) 

    if (isHourWindow and hourWindow not in possibleHourWindows):
        raise Exception('Unsupported hour window! supported values are: '+ str(possibleHourWindows))

def roundDateTime(dt) :
    dt = dt.split(" ")
    date = dt[0].replace('-', "")
    time = dt[1].split(':')
    
    if (isHourWindow):
        time[1] = '00'
        hour = hourWindow * (int(time[0]) / hourWindow)
        if hour < 10:
            time[0] = '0' + str(hour)
        else:
            time[0] = str(hour)
    elif (isMinuteWindow):
        minute = minuteWindow * (int(time[1]) / minuteWindow)
        if minute < 10:
            time[1] = '0' + str(minute)
        else:
            time[1] = str(minute)
    
    time[2] = '00'
    dt = date+time[0]+time[1]+time[2]
    
    return dt


def initializeResults():
    minDT = datetime.strptime(startDateTime, "%Y-%m-%d %H:%M:%S")
    maxDT = datetime.strptime(endDateTime, "%Y-%m-%d %H:%M:%S")

    startResults = {}
    endResults = {}
    while (minDT < maxDT):
        timStr = str(minDT)
        timStr = timStr.replace(" ", "").replace(":", "").replace("-","")
        startResults[timStr] = [[0 for i in range(numColumns)] for j in range(numRows)]
        endResults[timStr] = [[0 for i in range(numColumns)] for j in range(numColumns)]
        if isHourWindow:
            minDT = minDT + timedelta(hours= hourWindow)
        elif isMinuteWindow:
            minDT = minDT + timedelta(minutes= minuteWindow)
    
    return startResults, endResults


def initializeGridMap():
    stepX = (maxX - minX)/numColumns
    stepY = (maxY - minY)/numRows

    gridMap = [[[0 for i in range(4)] for j in range(numColumns)] for k in range(numRows)]

    startX = minX
    startY = maxY

    for i in range(numRows):
        for j in range(numColumns):
            gridMap[i][j] = [startX, startX + stepX, startY, startY - stepY]
            startX = startX + stepX
        startX = minX
        startY = startY - stepY
    
    return gridMap


def addToResult(x, y, dt, gridMap, results):
    j = 0
    k = numColumns - 1
    while (j < numRows and k >=0):
        x1 = gridMap[j][k][0]
        x2 = gridMap[j][k][1]
        y1 = gridMap[j][k][2]
        y2 = gridMap[j][k][3]
        if (x >= x1 and x < x2 and y <= y1 and y > y2):
            results[dt][j][k] += 1
            
            return True
        elif x < x1:
            k -= 1
        elif y <= y2:
            j += 1
        else:
            break
    
    return False


def writeOutputToFile(startResults, endResults):
    startOutputDir = outputDirpath + '/' + 'start'
    endOutputDir = outputDirpath + '/' + 'end'
    if not os.path.exists(startOutputDir):
        os.makedirs(startOutputDir)
    if not os.path.exists(endOutputDir):
        os.makedirs(endOutputDir)

    for outputFileName in startResults:
        f1 = open(startOutputDir+ "/" + outputFileName + ".txt", "w")
        f2 = open(endOutputDir+ "/" + outputFileName + ".txt", "w")
        for j in range(numRows):
            for k in range(numColumns):
                if (k == numColumns-1):
                    f1.write(str(startResults[outputFileName][j][k]))
                    f2.write(str(endResults[outputFileName][j][k]))
                else:
                    f1.write(str(startResults[outputFileName][j][k])+",")
                    f2.write(str(endResults[outputFileName][j][k])+",")
            f1.write("\n")
            f2.write("\n")
        f1.close()
        f2.close()


def main():
    startTime = datetime.now()
    validStartCount = 0
    validEndCount = 0
    totalPoints = 0

    validate()

    startResults, endResults = initializeResults()
    gridMap = initializeGridMap()

    fr = open(inputCSVPath, "r")
    
    for line in fr:
        totalPoints += 1
        
        col = line.strip().split(';')
        dt = col[1]
        startLocation = col[5].replace("(","").replace(")","").split(",")
        endLocation = col[8].replace("(","").replace(")","").split(",")

        dt = roundDateTime(dt)

        x = float(startLocation[0])
        y = float(startLocation[1])

        if addToResult(x, y, dt, gridMap, startResults):
            validStartCount +=1
        
        x = float(endLocation[0])
        y = float(endLocation[1])

        if addToResult(x, y, dt, gridMap, endResults):
            validEndCount += 1

    writeOutputToFile(startResults, endResults)

    print ("total valid start points: ", validStartCount)
    print ("total valid end points: ", validEndCount)
    print ("total points:", totalPoints)
    timeElapsed = datetime.now() - startTime
    print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))


if __name__ == "__main__":
    main()
