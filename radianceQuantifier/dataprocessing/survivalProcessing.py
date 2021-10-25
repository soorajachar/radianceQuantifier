#! /usr/bin/env python3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,pickle,sys,shutil
from sklearn.preprocessing import MinMaxScaler
sns.set_context('talk')

def addPercentageToSurvivalCountDf(countDf):
    timeLabels = [int(x[1:]) for x in countDf.index.get_level_values('Time')]
    countDf = countDf.assign(Day=timeLabels).set_index('Day', append=True)
    percentDf = countDf.copy()
    initialTime = sorted(countDf.index.unique('Day').tolist())[0]
    initialCountDf = countDf.xs([initialTime],level=['Day'])
    percentList,tupleList = [],[]
    for day in percentDf.index.unique('Day'):
        dayDf = percentDf.xs([day],level=['Day'])
        initialConditions = [list(dayDf.iloc[x,:].name)[:-1] for x in range(dayDf.shape[0])]
        for row in range(initialCountDf.shape[0]):
            name = list(initialCountDf.iloc[row,:].name)[:-1]
            time = list(dayDf.index.unique('Time'))[0]
            if name in initialConditions:
                count = dayDf.xs(name,level=initialCountDf.index.names[:-1]).values[0,0]
                percent = count / initialCountDf.xs(name,level=dayDf.index.names[:-1]).values[0]
                percent = percent[0]
            else:
                count = 0
                percent = 0
            tupleList.append(name+[time,day])
            percentList.append([count,percent*100])
    newMI = pd.MultiIndex.from_tuples(tupleList,names=countDf.index.names)
    survivalDf = pd.DataFrame(percentList,index=newMI,columns=['Survival Count','Survival Percentage'])
    newList,newTuples = [],[]
    offset = 0.001
    allDays = sorted(survivalDf.index.unique('Day').tolist())
    for row in range(survivalDf.shape[0]):
        name = list(survivalDf.iloc[row,:].name)
        oldDay = name[-1]
        if oldDay != allDays[-1]:
            newDay = oldDay+offset
            newName = name[:-1]+[newDay]
            newTuples.append(newName)
            newVals = survivalDf.xs(name[:-2]+[allDays[allDays.index(oldDay)+1]],level=list(survivalDf.index.names)[:-2]+['Day'])
            newList.append(newVals.values)
    newMatrix = np.vstack(newList)
    newMI = pd.MultiIndex.from_tuples(newTuples,names=survivalDf.index.names)
    newDf = pd.DataFrame(newMatrix,index=newMI,columns=survivalDf.columns)
    survivalDf = pd.concat([survivalDf,newDf],keys=['Original','Dummy'],names=['PointType'])
    return survivalDf

def createSurvivalDf(radianceStatisticDf,otherGroupingLevels,outputName,saveDf=False):
    #Create count df (grouped by levels selected)
    allGroupingLevels = [x for x in radianceStatisticDf.index.names if x not in ['Sample']+otherGroupingLevels]
    countDf = radianceStatisticDf.groupby(allGroupingLevels).count().iloc[:,0].to_frame('Count')
    #Create survival df
    survivalDf = addPercentageToSurvivalCountDf(countDf)
    if saveDf:
        subsetDf = countDf.copy()
        subsetDf.index.names = [x if x != 'Time' else 'Day' for x in subsetDf.index.names]
        subsetDf['Time'] = [int(x[1:]) for x in subsetDf.index.get_level_values('Day')]
        allGroupIndices = [x for x in subsetDf.index.names if x not in ['Day','Time']]
        subsetDf = subsetDf.reset_index().set_index(allGroupIndices+['Day','Time']).sort_values(by='Time')
        subsetDf.to_pickle('outputData/survivalStatisticPickleFile-'+outputName+'.pkl')
        subsetDf.to_excel('outputData/survivalStatisticExcelFile-'+outputName+'.xlsx')
    return survivalDf

def createSurvivalPlot(plottingDf,allPlottingKwargs,outputName):
    plottingTranslationDict = {'Marker':'style','Color':'hue','Column':'col','Row':'row','Size':'size'}
    plottingKwargs = {plottingTranslationDict[k]:v for k,v in zip(list(allPlottingKwargs.keys()),list(allPlottingKwargs.values())) if k != 'None'}
    g = sns.relplot(data=plottingDf,x='Day',y='Survival Percentage',kind='line',**plottingKwargs)
    g.fig.savefig('plots/survivalCurve-'+outputName+'.png')
