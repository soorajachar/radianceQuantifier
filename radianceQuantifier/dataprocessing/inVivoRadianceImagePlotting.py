#General purpose data wrangling/visualization packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import os,pickle,sys,shutil
from sklearn.preprocessing import MinMaxScaler
sns.set_context('talk')

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

# #Miscellaneous
import warnings
warnings.filterwarnings("ignore")

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    #mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    mycmap._lut[0,-1] = 0
    return mycmap

def selectMatrices(pMatrix,groups='all',days='all',samples='all'):
    if groups != 'all':
        groupValString = ','.join(groups)
    else:
        groupValString = 'all'
    groupTitle = '-'.join(['Group',groupValString])
    if days != 'all':
        dayValString = ','.join(days)
    else:
        dayValString = 'all'
    dayTitle = '-'.join(['Day',dayValString])
    selectionTitle = '_'.join([groupTitle,dayTitle])

    selectionDict = {}
    print(pMatrix.files)
    allDays,allGroups,allSamples = [x.split('-')[0] for x in pMatrix.files],[x.split('-')[1] for x in pMatrix.files],[x.split('-')[2] for x in pMatrix.files]
    if days == 'all':
        days = pd.unique(allDays).tolist()
    if groups == 'all':
        groups = pd.unique(allGroups).tolist()
    if samples == 'all':
        samples = pd.unique(allSamples).tolist()
    if type(days) != list:
        days = [days]
    if type(groups) != list:
        groups = [groups]
    if type(samples) != list:
        samples = [samples]
    selectionKeysList = []
    for day in days:
        for group in groups:
            for sample in samples:
                fullKey = '-'.join([day,group,sample])
                if fullKey in pMatrix.files:
                    selectionDict[fullKey] = pMatrix[fullKey]
                    selectionKeysList.append([day,group,sample])
                    
    selectionKeyMI = pd.MultiIndex.from_tuples(selectionKeysList,names=['Day','Group','Sample'])
    selectionKeyDf = pd.DataFrame(list(selectionDict.keys()),index=selectionKeyMI,columns=['Key'])    
    
    return selectionDict,selectionKeyDf,selectionTitle

def adaptMatricesForPlotting(radianceMatrix,brightfieldMatrix,trueMin):

    brightfieldAndBackgroundMatrix = np.add((radianceMatrix == -1).astype(int),(brightfieldMatrix == 0).astype(int))
    brightfieldAndBackgroundMatrix= (brightfieldAndBackgroundMatrix != 0).astype(int) 
    additionMatrix = np.multiply(brightfieldAndBackgroundMatrix,trueMin-1)
    plottingMatrix = np.multiply(radianceMatrix,1-brightfieldAndBackgroundMatrix) + additionMatrix            
    
    return plottingMatrix

def returnTailCropIndex(axes,cmap,cbar_ax,pMatrixDict,minScaleDict,selectionKeysDf,row,col,r,c,rowVal,colVal,twoDaxes=True,groupRenamingDict={},marginTitles=True,numericDays=False,fontDict={}):
    trueVals,trueLevels,trueAxisIndices = [],[],[]
    for val,level,index in zip([rowVal,colVal],[row,col],[r,c]):
        if val != '':
            trueVals.append(val)
            trueLevels.append(level)
            trueAxisIndices.append(index)
    sampleKey = selectionKeysDf.xs(trueVals,level=trueLevels,drop_level=False).values[0,0]
    
    if len(groupRenamingDict) != 0:
        for i,level in enumerate(trueLevels):
            if level == 'Group' and trueVals[i] in groupRenamingDict.keys():
                trueVals[i] = groupRenamingDict[trueVals[i]]
    if numericDays:
        for i,level in enumerate(trueLevels):
            if level == 'Day':
                trueVals[i] = trueVals[i][1:]
    
    sampleMatrix = pMatrixDict[sampleKey]
    trueMin = minScaleDict[sampleKey][0]
    trueMax = minScaleDict[sampleKey][1]
    
    radianceMatrix,brightfieldMatrix,b = sampleMatrix[:,:,0],sampleMatrix[:,:,1],sampleMatrix[:,:,2]
    plottingDf = pd.DataFrame(brightfieldMatrix)
    plottingDf.index.name = 'Row'
    plottingDf.columns.name = 'Column'
    #plottingDf.loc[:,:] = MinMaxScaler().fit_transform(plottingDf.values)
    columnBrightfield = plottingDf.sum(axis=1).to_frame('Value')
    maxIndex = np.argmax(columnBrightfield)
    maxVal = np.max(columnBrightfield).values[0]
    limit = 0.1*maxVal
    for i in range(maxIndex,columnBrightfield.shape[0]):
        if columnBrightfield.iloc[i,0] < limit:
            tailCropIndex = i
            break
#     g = sns.relplot(data=plottingDf.sum(axis=1).to_frame('Value'),x='Row',y="Value",kind='line')
#     g.axes.flat[0].axvline(color='k',linestyle='--',x=tailCropIndex)
#     g.axes.flat[0].axhline(color='r',linestyle='--',y=limit)
    return tailCropIndex

def plotSingleMouseImage(axes,cmap,cbar_ax,pMatrixDict,minScaleDict,selectionKeysDf,row,col,r,c,rowVal,colVal,groupRecoloringDict={},tailCrop=-1,twoDaxes=True,groupRenamingDict={},marginTitles=True,numericDays=False,fontDict={}):
    trueVals,trueLevels,trueAxisIndices = [],[],[]
    for val,level,index in zip([rowVal,colVal],[row,col],[r,c]):
        if val != '':
            trueVals.append(val)
            trueLevels.append(level)
            trueAxisIndices.append(index)
    sampleKey = selectionKeysDf.xs(trueVals,level=trueLevels,drop_level=False).values[0,0]
    
    if len(groupRenamingDict) != 0:
        for i,level in enumerate(trueLevels):
            if level == 'Group' and trueVals[i] in groupRenamingDict.keys():
                trueVals[i] = groupRenamingDict[trueVals[i]]
    if numericDays:
        for i,level in enumerate(trueLevels):
            if level == 'Day':
                trueVals[i] = trueVals[i][1:]
    
    sampleMatrix = pMatrixDict[sampleKey]
    trueMin = minScaleDict[sampleKey][0]
    trueMax = minScaleDict[sampleKey][1]
    
    radianceMatrix,brightfieldMatrix,b = sampleMatrix[:,:,0],sampleMatrix[:,:,1],sampleMatrix[:,:,2]
    plottingDf = pd.DataFrame(brightfieldMatrix)
    if tailCrop != -1:
        radianceMatrix = radianceMatrix[:tailCrop,:]
        brightfieldMatrix = brightfieldMatrix[:tailCrop,:]
    plottingMatrix = adaptMatricesForPlotting(radianceMatrix,brightfieldMatrix,trueMin)
    sampleDf = pd.DataFrame(plottingMatrix)
    sampleDf.iloc[-1,-1] = trueMax
    if twoDaxes:
        g = sns.heatmap(sampleDf,cmap=transparent_cmap(cmap),cbar = r == 0,ax=axes[r,c],norm=matplotlib.colors.LogNorm(),vmin=np.nanmin(sampleDf.values),vmax=trueMax,cbar_ax=cbar_ax,cbar_kws={'label':'Radiance (p/sec/cm$^2$/sr)'})
        if not marginTitles:
            axes[r,c].set_title('-'.join(trueVals),**fontDict)
        else:
            if r == 0:
                axes[r,c].set_title(trueVals[1],{**fontDict,**{'color':groupRecoloringDict[colVal]}})
            if c == 0:
                axes[r,c].text(-0.15,0.5,trueVals[0],verticalalignment='center',horizontalalignment='center',transform=axes[r,c].transAxes,**fontDict)
        axes[r,c].imshow(b,zorder=0, cmap='gray')
    else:
        g = sns.heatmap(sampleDf,cmap=transparent_cmap(cmap),cbar = trueAxisIndices[0] == 0,ax=axes[trueAxisIndices[0]],norm=matplotlib.colors.LogNorm(),vmin=np.nanmin(sampleDf.values),vmax=trueMax,cbar_ax=cbar_ax,cbar_kws={'label':'Radiance (p/sec/cm$^2$/sr)'})
        #TODO: Make faceting work with only one row or column as well
        if not marginTitles:
            axes[trueAxisIndices[0]].set_title(trueVals[1])
        else:
            axes[trueAxisIndices[0]].set_title(trueVals[1])
        axes[trueAxisIndices[0]].imshow(b,zorder=0, cmap='gray')

#Pad all 3 levels differently (-1, 0, minVal), then stack
def stackMouseImages(matrixList,orientation='h',unifiedPaddingShape = [],staggeredPadding=False):
    if len(unifiedPaddingShape) == 0:
        maxLength = max([x.shape[0] for x in matrixList])
        maxWidth = max([x.shape[1] for x in matrixList])
    else:
        maxLength = unifiedPaddingShape[0]
        maxWidth = unifiedPaddingShape[1]
    newMatrixList = []
    if orientation == 'h':
        for matrix in matrixList:
            dstackList = []
            for i,paddingConstant in enumerate([-1,0,np.min(matrix[:,:,2])]):
                newMatrix = np.multiply(np.ones([maxLength,matrix.shape[1]]),paddingConstant)
                newMatrix[:matrix.shape[0],:matrix.shape[1]] = matrix[:,:,i]
                dstackList.append(newMatrix)
            newMatrixList.append(np.dstack(dstackList))
        paddedMatrix = np.hstack(newMatrixList)
    else:
        for matrix in matrixList:
            dstackList = []
            for i,paddingConstant in enumerate([-1,0,np.min(matrix[:,:,2])]):
                newMatrix = np.multiply(np.ones([matrix.shape[0],maxWidth]),paddingConstant)
                newMatrix[:matrix.shape[0],:matrix.shape[1]] = matrix[:,:,i]
                dstackList.append(newMatrix)
            newMatrixList.append(np.dstack(dstackList))
        paddedMatrix = np.vstack(newMatrixList)
    if len(unifiedPaddingShape) != 0:
        dstackList = []
        matrixWidthSum = sum([x.shape[1] for x in matrixList])
        paddingDifference = maxWidth-matrixWidthSum
        if not staggeredPadding or paddingDifference < 5:
            for i,paddingConstant in enumerate([-1,0,np.min(paddedMatrix[:,:,2])]):
                newMatrix = np.multiply(np.ones([maxLength,maxWidth]),paddingConstant)
                newMatrix[:paddedMatrix.shape[0],:paddedMatrix.shape[1]] = paddedMatrix[:,:,i]
                dstackList.append(newMatrix)
        else:
            padPerMouse = int(paddingDifference/5)
            for i,paddingConstant in enumerate([-1,0,np.min(paddedMatrix[:,:,2])]):
                newMatrix = np.multiply(np.ones([maxLength,maxWidth]),paddingConstant)
                for j,nMatrix in enumerate(newMatrixList):
                    padStart = sum([x.shape[1]+padPerMouse for x in matrixList[:j]])
                    newMatrix[:paddedMatrix.shape[0],padStart:padStart+nMatrix.shape[1]] = nMatrix[:,:,i]
                dstackList.append(newMatrix)
        paddedMatrix = np.dstack(dstackList)
    
    return paddedMatrix

def concatenateImage(pMatrixDict,minScaleDict,selectionKeysDf,kwargDict,kwargValsDict,unifiedPaddingShape=[]):
    
    matrixList,minList,maxList = [],[],[]
    selectionKeysDf = pd.concat([selectionKeysDf],keys=['dummy'],names=['dummy'])
    #TODO: ADD DUMMY IMAGES (BLACK) TO INNERROW/INNERCOL PARAMETERS
    if kwargDict['innerRow'] != '':
        if kwargDict['innerCol'] != '':
            for val in kwargValsDict['innerRow']:
                if val in selectionKeysDf.index.unique(kwargDict['innerRow']):
                    matrixList2 = []
                    for val2 in kwargValsDict['innerCol']:
                        if val2 in selectionKeysDf.index.unique(kwargDict['innerCol']):
                            sampleKey = selectionKeysDf.xs([val,val2],level=[kwargDict['innerRow'],kwargDict['innerCol']],drop_level=False).values[0,0]
                            sampleMatrix,trueMin,trueMax = pMatrixDict[sampleKey],minScaleDict[sampleKey][0],minScaleDict[sampleKey][1]
                            matrixList2.append(sampleMatrix)
                            minList.append(trueMin)
                            maxList.append(trueMax)
                matrixList.append(stackMouseImages(matrixList2,orientation='h',unifiedPaddingShape = unifiedPaddingShape))
            fullMatrix = stackMouseImages(matrixList,orientation='v',unifiedPaddingShape = unifiedPaddingShape)
        else:
            for val in kwargValsDict['innerRow']:
                if val in selectionKeysDf.index.unique(kwargDict['innerRow']):
                    sampleKey = selectionKeysDf.xs([val],level=[kwargDict['innerRow']],drop_level=False).values[0,0]
                    sampleMatrix,trueMin,trueMax = pMatrixDict[sampleKey],minScaleDict[sampleKey][0],minScaleDict[sampleKey][1]
                else:
                    for tempIndex,tempVal in enumerate(kwargValsDict['innerRow']):
                        if tempVal in selectionKeysDf.index.unique(kwargDict['innerRow']):
                            break
                    sampleKey = selectionKeysDf.xs([selectionKeysDf.index.unique(kwargDict['innerRow']).tolist()[tempIndex]],level=[kwargDict['innerRow']],drop_level=False).values[0,0]
                    tempMatrix,trueMin,trueMax = pMatrixDict[sampleKey],minScaleDict[sampleKey][0],minScaleDict[sampleKey][1]
                    sampleMatrix = np.dstack([np.ones(tempMatrix.shape[:2])*-1,np.zeros(tempMatrix.shape[:2]),np.ones(tempMatrix.shape[:2])*np.min(tempMatrix[:,:,2])])

                matrixList.append(sampleMatrix)
                minList.append(trueMin)
                maxList.append(trueMax)
            fullMatrix = stackMouseImages(matrixList,orientation='v',unifiedPaddingShape = unifiedPaddingShape)
    else:
        if kwargDict['innerCol'] != '':
            for val2 in kwargValsDict['innerCol']:
                if val2 in selectionKeysDf.index.unique(kwargDict['innerCol']):
                    sampleKey = selectionKeysDf.xs([val2],level=[kwargDict['innerCol']],drop_level=False).values[0,0]
                    sampleMatrix,trueMin,trueMax = pMatrixDict[sampleKey],minScaleDict[sampleKey][0],minScaleDict[sampleKey][1]
                else:
                    for tempVal in kwargValsDict['innerCol']:
                        if tempVal in selectionKeysDf.index.unique(kwargDict['innerCol']):
                            tempIndex = selectionKeysDf.index.unique(kwargDict['innerCol']).tolist().index(tempVal)
                            break
                    sampleKey = selectionKeysDf.xs([selectionKeysDf.index.unique(kwargDict['innerCol']).tolist()[tempIndex]],level=[kwargDict['innerCol']],drop_level=False).values[0,0]
                    tempMatrix,trueMin,trueMax = pMatrixDict[sampleKey],minScaleDict[sampleKey][0],minScaleDict[sampleKey][1]
                    sampleMatrix = np.dstack([np.ones(tempMatrix.shape[:2])*-1,np.zeros(tempMatrix.shape[:2]),np.ones(tempMatrix.shape[:2])*np.min(tempMatrix[:,:,2])])
                plottingDf = pd.DataFrame(pMatrixDict[sampleKey][:,:,1])
                plottingDf.index.name = 'Row'
                plottingDf.columns.name = 'Column'
                columnBrightfield = plottingDf.sum(axis=0).to_frame('Value')
                boundary = 110
                if plottingDf.shape[1] > boundary:
                    leftMax = np.max(columnBrightfield.iloc[:boundary].values)#.values[0]
                    leftMaxInd = np.argmax(columnBrightfield.iloc[:boundary].values)
                    rightMax = np.max(columnBrightfield.iloc[boundary:].values)#.values[0]
                    rightMaxInd = boundary+np.argmax(columnBrightfield.iloc[boundary:].values)
                    if rightMax >= 0.5*leftMax:
                        trueBoundary = leftMaxInd+np.argmin(columnBrightfield.iloc[leftMaxInd:rightMaxInd].values)
                        sampleMatrix = sampleMatrix[:,:trueBoundary,:]
                        g = sns.relplot(data=columnBrightfield,x='Column',y="Value",kind='line')
                        g.axes.flat[0].set_title(sampleKey)
                        g.axes.flat[0].axvline(color='k',linestyle='--',x=trueBoundary)
                matrixList.append(sampleMatrix) 
                minList.append(trueMin)
                maxList.append(trueMax)
            fullMatrix = stackMouseImages(matrixList,orientation='h',unifiedPaddingShape = unifiedPaddingShape,staggeredPadding=True)
        else:
            fullMatrix = list(pMatrixDict.values())[0]
            minScale = list(minScaleDict.values())[0]
    
    return fullMatrix,[min(minList),max(maxList)]

def plotMouseImages(pMatrixDict,minScaleDict,selectionKeysDf,titleRenamingDict={'row':'Day','col':'Group'},groupRecoloringDict={},tailCrop=False,row='',col='',innerRow='',innerCol='',row_order=[],col_order=[],innerRowOrder=[],innerColOrder=[],cmap='magma',groupRenamingDict={},marginTitles=True,numericDays=True,useConstantImageSize=True,colorbarScale=2,font='Helvetica',fontsize=40,image_dir='',save_image=False,imageTitle='',fontScale=1,maxTextLength=20):

    fontDict = {}
    for param,paramVal in zip(['fontname','fontsize'],[font,fontsize]):
        if paramVal != '':
            fontDict[param] = paramVal

    kwargDict = {'row':row,'col':col,'innerRow':innerRow,'innerCol':innerCol}
    kwargLenDict,kwargValsDict = {},{}
    for kwarg in kwargDict:
        if kwargDict[kwarg] != '' and len(selectionKeysDf.index.unique(kwargDict[kwarg])) != 1:
            kwargLenDict[kwarg] = len(selectionKeysDf.index.unique(kwargDict[kwarg]))
            #Make sure samples/days are sorted from lowest to highest
            if kwargDict[kwarg] == 'Sample':
                kwargValsDict[kwarg] = sorted(selectionKeysDf.index.unique(kwargDict[kwarg]).tolist())
            else:
                kwargValsDict[kwarg] = selectionKeysDf.index.unique(kwargDict[kwarg]).tolist()
        else:
            kwargLenDict[kwarg] = 1
            kwargValsDict[kwarg] = ['']
            kwargDict[kwarg] = ''

    if kwargLenDict['row'] == 1 or kwargLenDict['col'] == 1:
        twoDaxes = False
    else:
        twoDaxes = True

    if kwargDict['innerRow'] != '' or kwargDict['innerCol'] != '':
        wspace = 0.01
        hspace=0.01
        nonNestedVarLocations = [x for x in range(len(selectionKeysDf.index.names)) if list(selectionKeysDf.index.names)[x] in [kwargDict['row'],kwargDict['col']]]
        nestedVarLocations = [x for x in range(len(selectionKeysDf.index.names)) if list(selectionKeysDf.index.names)[x] in [kwargDict['innerRow'],kwargDict['innerCol']]]
        uniqueNestedVars,uniqueNonNestedVars = {},{}
        for rowIndex in range(selectionKeysDf.shape[0]):
            name = list(selectionKeysDf.iloc[rowIndex,:].name)
            nonNestedVars = '-'.join([name[x] for x in range(len(name)) if x in nonNestedVarLocations])
            nestedVars = '-'.join([name[x] for x in range(len(name)) if x in nestedVarLocations])
            if nonNestedVars not in list(uniqueNonNestedVars.keys()):
                uniqueNonNestedVars[nonNestedVars] = rowIndex
        newSelectionKeysDf = selectionKeysDf.iloc[list(uniqueNonNestedVars.values()),:].droplevel(nestedVarLocations)
        newSelectionKeysDf.iloc[:,0] = list(uniqueNonNestedVars.keys())

        newPmatrixDict,newMinScaleDict = {},{}
        for rowIndex in range(newSelectionKeysDf.shape[0]):
            levelValues = list(newSelectionKeysDf.iloc[rowIndex,:].name)
            levels = list(newSelectionKeysDf.index.names)
            keysToCombine = selectionKeysDf.xs(levelValues,level=levels)
            newKey = newSelectionKeysDf.iloc[rowIndex,0]
            concatenatedImage,minScale = concatenateImage(pMatrixDict,minScaleDict,keysToCombine,kwargDict,kwargValsDict)
            newPmatrixDict[newKey] = concatenatedImage
            newMinScaleDict[newKey] = minScale
        if useConstantImageSize:
            unifiedPadLength = max([newPmatrixDict[x].shape[0] for x in newPmatrixDict])
            unifiedPadWidth = max([newPmatrixDict[x].shape[1] for x in newPmatrixDict])
            unifiedPaddingShape = [unifiedPadLength,unifiedPadWidth]
            for rowIndex in range(newSelectionKeysDf.shape[0]):
                levelValues = list(newSelectionKeysDf.iloc[rowIndex,:].name)
                levels = list(newSelectionKeysDf.index.names)
                keysToCombine = selectionKeysDf.xs(levelValues,level=levels)
                newKey = newSelectionKeysDf.iloc[rowIndex,0]
                concatenatedImage,minScale = concatenateImage(pMatrixDict,minScaleDict,keysToCombine,kwargDict,kwargValsDict,unifiedPaddingShape=unifiedPaddingShape)
                newPmatrixDict[newKey] = concatenatedImage

        selectionKeysDf = newSelectionKeysDf.copy()
        pMatrixDict = newPmatrixDict.copy()
        minScaleDict = newMinScaleDict.copy()
    else:
        wspace = None
        hspace=None
     
    plottedParameterIndices = [list(selectionKeysDf.index.names).index(x) for x in selectionKeysDf.index.names if x in [kwargDict['row'],kwargDict['col']]]
    plottedParameterTuples = []
    for indexTuple in selectionKeysDf.values[:,0].tolist():
        tempList = []
        indexTupleList = indexTuple.split('-')
        for index in plottedParameterIndices:
            tempList.append(indexTupleList[index])
        plottedParameterTuples.append(set(tempList))
    
    if tailCrop:
        tailCropScalingFactor = 0.6
    else:
        tailCropScalingFactor = 1
    
    fig, axes = plt.subplots(kwargLenDict['row'],kwargLenDict['col'],figsize=(2.5*kwargLenDict['col']*kwargLenDict['innerCol']*0.5,tailCropScalingFactor*4.55*kwargLenDict['row']*kwargLenDict['innerRow']))
    fig.subplots_adjust(right=0.8)
    r = fig.canvas.get_renderer()

    if not marginTitles:
        titleList = [x for x in [kwargDict['row'],kwargDict['col']] if x != '']
        fig.suptitle('-'.join(titleList),**fontDict)
        fig.subplots_adjust(top=1-0.25/kwargLenDict['row'],wspace=wspace)
    else:
        fig.subplots_adjust(wspace=wspace,hspace=hspace)
                            
    #Fitted hyperbolic saddle that describes relationship between font size, text length, and text width in pixels
    widthBounds = 0.9
    bbox = axes[0,0].get_window_extent(renderer=r)
    axWidth, axHeight = bbox.width, bbox.height
    textWidth = widthBounds*axWidth
    textSize = (textWidth-0.025496721637466635)/(maxTextLength*0.8620902)
    fontDict['fontsize'] = min(40,int(textSize*fontScale))

    fontDict2 = fontDict.copy()
    fontDict2['fontweight'] = 'bold'
    if 'fontsize' in fontDict2:
        fontDict2['fontsize'] = fontDict2['fontsize']*1.2
    else:
        fontDict2['fontsize'] = 12
    
    levelTitles = []
    if kwargDict['row'] != '':
        if kwargLenDict['col'] != 1:
            axbox1 = axes[0,0].get_position().extents
        else:
            axbox1 = axes[0].get_position().extents
        rightXPos = axbox1[0]
        a1 = plt.figtext(rightXPos,0.5,titleRenamingDict['row']+'\n\n', rotation = 90,horizontalalignment='right',verticalalignment='center',**fontDict2)
        levelTitles.append(a1)
        #fig.suptitle(kwargDict['row']+'\n',x=rightXPos,y=0.5, rotation = 90,horizontalalignment='right',verticalalignment='center',**fontDict2)
    if kwargDict['col'] != '':
        if kwargLenDict['col'] % 2 == 0:
            if kwargLenDict['row'] != 1:
                axbox1 = axes[0,int(kwargLenDict['col']/2) -1].get_position().extents
                axbox2 = axes[0,int(kwargLenDict['col']/2)].get_position().extents
            else:
                axbox1 = axes[int(kwargLenDict['col']/2) -1].get_position().extents
                axbox2 = axes[int(kwargLenDict['col']/2)].get_position().extents
            middleXPos = 0.5*(0.5*(axbox1[0] + axbox1[2]) + 0.5*(axbox2[0] + axbox2[2]))
        else:
            if kwargLenDict['row'] != 1:
                axbox1 = axes[0,int(kwargLenDict['col']/2)].get_position().extents
            else:
                axbox1 = axes[int(kwargLenDict['col']/2)].get_position().extents
            middleXPos = 0.5*(axbox1[0] + axbox1[2])
        bottomYPos = axbox1[3]
        a2 = plt.figtext(middleXPos,bottomYPos,titleRenamingDict['col']+'\n',horizontalalignment='center',verticalalignment='bottom',**fontDict2)
        levelTitles.append(a2)

    barWidth = colorbarScale*0.02*(2/len(kwargValsDict['col']))
    barHeight = (1/tailCropScalingFactor)*colorbarScale*0.8*(1/len(kwargValsDict['row']))
    cbar_ax = fig.add_axes([0.86-0.005*len(kwargValsDict['col']), 0.5-(0.1+barHeight/2)+0.1, barWidth, barHeight])
    
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cmap = sns.color_palette(cmap, as_cmap=True)
    #Grab tailcropping indices
    if tailCrop:
        tailCropIndices = []
        for r,rowVal in enumerate(kwargValsDict['row']):
            for c,colVal in enumerate(kwargValsDict['col']):
                if not twoDaxes:
                    if kwargLenDict['row'] == 1:
                        plottedParameterList = [colVal]
                        axes[c].axis('off')
                    else:
                        plottedParameterList = [rowVal]
                        axes[r].axis('off')
                else:
                    plottedParameterList = [rowVal,colVal]
                    axes[r,c].axis('off')
                if set(plottedParameterList) in plottedParameterTuples:
                    tailCropIndex = returnTailCropIndex(axes,cmap,cbar_ax,pMatrixDict,minScaleDict,selectionKeysDf,kwargDict['row'],kwargDict['col'],r,c,rowVal,colVal,twoDaxes=twoDaxes,groupRenamingDict=groupRenamingDict,marginTitles=marginTitles,numericDays=numericDays,fontDict=fontDict)
                    tailCropIndices.append(tailCropIndex)
        minTailCrop = max(245,min(tailCropIndices))
    else:
        minTailCrop = -1
    
    if 'col' in list(kwargValsDict.keys()) and len(col_order) != 0:
        kwargValsDict['col'] = col_order 
    for r,rowVal in enumerate(kwargValsDict['row']):
        for c,colVal in enumerate(kwargValsDict['col']):
            if not twoDaxes:
                if kwargLenDict['row'] == 1:
                    plottedParameterList = [colVal]
                    axes[c].axis('off')
                else:
                    plottedParameterList = [rowVal]
                    axes[r].axis('off')
            else:
                plottedParameterList = [rowVal,colVal]
                axes[r,c].axis('off')

            if set(plottedParameterList) in plottedParameterTuples:
                plotSingleMouseImage(axes,cmap,cbar_ax,pMatrixDict,minScaleDict,selectionKeysDf,kwargDict['row'],kwargDict['col'],r,c,rowVal,colVal,groupRecoloringDict=groupRecoloringDict,tailCrop=minTailCrop,twoDaxes=twoDaxes,groupRenamingDict=groupRenamingDict,marginTitles=marginTitles,numericDays=numericDays,fontDict=fontDict)
            else:
                if marginTitles:
                    trueVals,trueLevels,trueAxisIndices = [],[],[]
                    for val,level,index in zip([rowVal,colVal],[row,col],[r,c]):
                        if val != '':
                            trueVals.append(val)
                            trueLevels.append(level)
                            trueAxisIndices.append(index)
                    if len(groupRenamingDict) != 0:
                        for i,level in enumerate(trueLevels):
                            if level == 'Group' and trueVals[i] in groupRenamingDict.keys():
                                trueVals[i] = groupRenamingDict[trueVals[i]]
                    if numericDays:
                        for i,level in enumerate(trueLevels):
                            if level == 'Day':
                                trueVals[i] = trueVals[i][1:]
                    if r == 0:
                        axes[r,c].set_title(trueVals[1],**fontDict)
                    if c == 0:
                        axes[r,c].text(-0.15,0.5,trueVals[0],verticalalignment='center',horizontalalignment='center',transform=axes[r,c].transAxes,**fontDict)
                        #axes[r,c].text(0,0.5,trueVals[0],verticalalignment='center',horizontalalignment='right',transform=axes[r,c].transAxes,**fontDict)
    cbar_ax.set_frame_on(True)
    cbar_ax.tick_params(which='both',width=colorbarScale*1.5)
    if fontsize != '':
        cbar_ax.yaxis.label.set_fontsize(fontsize)
        for l in cbar_ax.yaxis.get_ticklabels():
            l.set_fontsize(fontsize)
    if font != '':
        cbar_ax.yaxis.label.set_family(font)
        for l in cbar_ax.yaxis.get_ticklabels():
            l.set_family(font)
    if save_image:
        paramTitleList = []
        for param in kwargDict:
            if kwargDict[param] != '':
                paramTitleList.append('-'.join([param,kwargDict[param]]))
        paramTitle = '_'.join(paramTitleList)
        experimentName = os.getcwd().split(dirSep)[-1]
        imageTitle = '_'.join(['mouseImage',experimentName,imageTitle,paramTitle])
        fig.savefig('plots/'+imageTitle+'.png',bbox_extra_artists=(cbar_ax,*levelTitles),bbox_inches='tight')
