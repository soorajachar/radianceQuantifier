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

#Image processing packages
from matplotlib import image as mplImage
import pytesseract
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    dirSep = '\\'
else:
    dirSep = '/'
from PIL import Image
from scipy import ndimage
import cv2

#Clustering
import hdbscan

#Miscellaneous
from itertools import tee
from tqdm.auto import trange
from scipy.signal import argrelmin,find_peaks,savgol_filter
import warnings
import shutil

warnings.filterwarnings("ignore")

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_blocks(values,cutoff):
    result = []
    for i,val in enumerate(values):
        values2 = [values[x] for x in range(len(values)) if x != i]
        group = [val]
        for val2 in values2:
            if abs(val2 - val) < cutoff:
                group.append(val2)
        result.append(group)
    return result

def np_array_to_hex(array):
    array = np.asarray(array, dtype='uint32')
    array = ((array[:, :, 0]<<16) + (array[:, :, 1]<<8) + array[:, :, 2])
    return array

def img_array_to_single_val(image, color_codes):
    image = image.dot(np.array([65536, 256, 1], dtype='int32'))
    result = np.ndarray(shape=image.shape, dtype=int)
    result[:,:] = -1
    for rgb, idx in color_codes.items():
        rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        result[image==rgb] = idx
    return result

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    #mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    mycmap._lut[0,-1] = 0
    return mycmap

def returnColorScale(colorBar):    
    rgbaList = np.dsplit(colorBar,colorBar.shape[2])
    colorDfList = []
    for d in rgbaList:
        colorDf = pd.DataFrame(d[:,:,0],index=list(range(colorBar.shape[0])),columns=list(range(colorBar.shape[1])))
        colorDf.index.name = 'Row'
        colorDf.columns.name = 'Column'
        colorDfList.append(colorDf)
    colorDf = pd.concat(colorDfList,keys=['R','G','B','A'],names=['Color'])
    colorScale = colorDf.loc[['R','G','B']].iloc[:,int(colorDf.shape[1]/2)].unstack('Color').values.tolist()[::-1] + [[1,0,0]]

    rgbColorScale = []
    for elem in colorScale:
        elem = [int(255*x) for x in elem]
        rgbColorScale.append(elem)
    
    trueRGBColorScale = []
    offset = 1
    for r in range(rgbColorScale[0][0]+offset,-1,-1):
        trueRGBColorScale.append([r,0,255])
    for g in range(1,256):
        trueRGBColorScale.append([0,g,255])
    for b in range(255,-1,-1):
        trueRGBColorScale.append([0,255,b])
    for r in range(1,256):
        trueRGBColorScale.append([r,255,0])
    for g in range(255,-1,-1):
        trueRGBColorScale.append([255,g,0])

    return trueRGBColorScale

def returnColorScaleSpan(legend,colorScale,colorBarScale):
    croppedLegend = np.multiply(legend[60:105,:75],255)
    pIm = Image.fromarray(np.uint8(croppedLegend))
    fullString = pytesseract.image_to_string(pIm)
    splitStrings = fullString.split('\n')
    for splitString in splitStrings:
        if 'Min' in splitString or 'Max' in splitString:
          #Account for occasinoal pytesseract failures
          if ' = ' in splitString:
            splitChar = ' = '
          else:
            if ' =' in splitString:
              splitChar = ' ='
            elif '= ' in splitString:
              splitChar = '= '
            else:
              if '=' in splitString:
                splitChar = '='
              else:
                print('PyTessearct error; equal sign not found. Split string is:'+splitString)
          if 'Min' in splitString:
              number = splitString.split(splitChar)[1]
              minScalingFactor = 10**int(number[-1])
              scaleStart = float(number[:4])*minScalingFactor
          elif 'Max' in splitString:
              number = splitString.split(splitChar)[1]
              maxScalingFactor = 10**int(number[-1])
              scaleEnd = float(number[:4])*maxScalingFactor
            
    linearScale = np.linspace(scaleStart,scaleEnd,num=len(colorScale))
    return linearScale,scaleStart

def returnLuminescentImageComponents(luminescent,visualize=False):
    occupancyCutoff = 300
    headerCutoff = 50

    rgbSample = luminescent[:,:,:]
    #Separate colorbar from mouse samples
    bwMatrix = np.zeros(luminescent.shape)[:,:,:]
    for row in range(luminescent.shape[0]):
        for col in range(luminescent.shape[1]):
            pixel = luminescent[row,col].tolist()[:3]
            if pixel != [1,1,1]:
                bwMatrix[row,col] = 1

    #Split horizontally
    componentSeparationDfHorizontal = pd.DataFrame(bwMatrix[:,:,0].sum(axis=1),index=list(range(bwMatrix.shape[0])),columns=['Sample Pixel Number'])
    componentSeparationDfHorizontal.index.name = 'Row'
    rowBreakpoints = []
    for row in range(1,componentSeparationDfHorizontal.shape[0]-1):
        pixelNum = componentSeparationDfHorizontal.iloc[row,0]
        if (componentSeparationDfHorizontal.iloc[row-1,0] < occupancyCutoff and componentSeparationDfHorizontal.iloc[row,0] > occupancyCutoff) or (componentSeparationDfHorizontal.iloc[row,0] > occupancyCutoff and componentSeparationDfHorizontal.iloc[row+1,0] < occupancyCutoff):
            rowBreakpoints.append(row)

    #Split vertically
    componentSeparationDf = pd.DataFrame(bwMatrix[:,:,0].sum(axis=0),index=list(range(bwMatrix.shape[1])),columns=['Sample Pixel Number'])
    componentSeparationDf.index.name = 'Column'
    columnBreakpoints = []
    for row in range(1,componentSeparationDf.shape[0]-1):
        pixelNum = componentSeparationDf.iloc[row,0]
        if (componentSeparationDf.iloc[row-1,0] < occupancyCutoff and componentSeparationDf.iloc[row,0] > occupancyCutoff) or (componentSeparationDf.iloc[row,0] > occupancyCutoff and componentSeparationDf.iloc[row+1,0] < occupancyCutoff):
            columnBreakpoints.append(row)
    
    #Split off colorbar
    rgbScale = luminescent[:,columnBreakpoints[2]:columnBreakpoints[3]+1,:]
    colorBarRowIndices,colorBarColumnIndices = [],[]
    for i in range(rgbScale.shape[0]): # for every pixel:
        for j in range(rgbScale.shape[1]):
            pixel = rgbScale[i,j].tolist()[:3]
            if len(np.unique(pixel)) == 1 and np.unique(pixel) in [0,1]:
                pass
            else:
                colorBarRowIndices.append(i)
                colorBarColumnIndices.append(j)
    
    if rowBreakpoints[0] > headerCutoff:
        interval = 596
    else:
        interval = 706
    if rowBreakpoints[-1] - rowBreakpoints[-2] != interval:
      #print(rowBreakpoints)
      rowBreakpoints[-1] = rowBreakpoints[-2]+interval
    if columnBreakpoints[1] - columnBreakpoints[0] != interval:
      #print(columnBreakpoints)
      columnBreakpoints[1] = columnBreakpoints[0]+interval
    colorBar = rgbScale[min(colorBarRowIndices):max(colorBarRowIndices)+1,min(colorBarColumnIndices):max(colorBarColumnIndices)+1,:]
    miceSamples = luminescent[rowBreakpoints[-2]:rowBreakpoints[-1],columnBreakpoints[0]:columnBreakpoints[1],:]
    legend = luminescent[max(colorBarRowIndices):,columnBreakpoints[2]:]
    colorBarScale = luminescent[min(colorBarRowIndices):max(colorBarRowIndices)+1,columnBreakpoints[2]+max(colorBarColumnIndices)+1:,:]
    
    if visualize:
        sns.relplot(data=componentSeparationDfHorizontal,x='Row',y='Sample Pixel Number',kind='line',aspect=luminescent.shape[1]/luminescent.shape[0])
        sns.relplot(data=componentSeparationDf,x='Column',y='Sample Pixel Number',kind='line',aspect=luminescent.shape[1]/luminescent.shape[0])
    
    return miceSamples,colorBar,legend,colorBarScale

def findBrightfieldCutoff(brightfield,visualize=False):

    brightfieldDf = pd.DataFrame(brightfield)
    brightfieldDf.index.name = 'Row'
    brightfieldDf.columns.name = 'Column'
    plottingDf = brightfieldDf.stack().to_frame('Intensity')

    visualBrightfieldMatrix = MinMaxScaler(feature_range=(11000,65000)).fit_transform(np.clip(plottingDf.values,a_min=0,a_max=5500))

    hist,bins = np.histogram(plottingDf.values,bins='auto', density=True)
    hist = savgol_filter(hist,7,2)
    maxHist = np.argmax(hist)
    elbow = maxHist+1000
    if visualize:
        g = sns.displot(data=plottingDf,x='Intensity',element='poly')
        g.axes.flat[0].axvline(x=elbow,linestyle=':',color='k')
        blackpointDf = plottingDf.copy()
        blackpointDf.loc[:,:] = visualBrightfieldMatrix
        blackpointDf2 = plottingDf.copy()
        g = sns.displot(data=pd.concat([blackpointDf,blackpointDf2],keys=['Yes','No'],names=['Scaled']),hue='Scaled',kind='hist',x='Intensity',element='poly',fill=False)
        g.set(yscale='log')

    return elbow,visualBrightfieldMatrix.reshape(brightfield.shape)

def rescaleBrightfieldImage(brightfield,brightfield2,luminescentSamples,visualize=False):
    
    rescaledBrightfieldMatrix = cv2.resize(brightfield, dsize=luminescentSamples.shape[:2])
    
    if np.mean(rescaledBrightfieldMatrix) > 10000:
        brightfieldCutoff = 90
        visualBrightfieldMatrix = cv2.resize(brightfield2, dsize=luminescentSamples.shape[:2])
        mouseBrightfieldMatrix = (visualBrightfieldMatrix[:,:,0] > brightfieldCutoff).astype(int)
    else:        
        brightfieldCutoff,visualBrightfieldMatrix = findBrightfieldCutoff(rescaledBrightfieldMatrix,visualize=visualize)
        mouseBrightfieldMatrix = (rescaledBrightfieldMatrix > brightfieldCutoff).astype(int)
    
    if visualize:
        fig = plt.figure()
        sns.heatmap(mouseBrightfieldMatrix,cmap='Greys')
    
    return mouseBrightfieldMatrix,visualBrightfieldMatrix

def horizontallySeparateMice(brightfieldSamples,visualize=False):
    brightfieldDf = pd.DataFrame(brightfieldSamples,index=list(range(brightfieldSamples.shape[0])),columns=list(range(brightfieldSamples.shape[1])))
    brightfieldDf.index.name = 'Row'
    brightfieldDf.columns.name = 'Column'
    brightfieldDf = brightfieldDf.sum(axis=1).to_frame('Count')
    brightfieldDf.loc[:,:] = MinMaxScaler().fit_transform(brightfieldDf.values)

    maxPointRangeCutoff = 0.2
    rangeBreakpoints = []
    for row in range(brightfieldDf.shape[0]-1,-1,-1):
        if brightfieldDf.iloc[row,0] > maxPointRangeCutoff:
            rangeBreakpoints.append(row)
            break        
    for row in range(rangeBreakpoints[0]-1,-1,-1):
        if brightfieldDf.iloc[row,0] < maxPointRangeCutoff:
            rangeBreakpoints.append(row)
            break        
    
    maxPoint = np.argmax(brightfieldDf.iloc[rangeBreakpoints[1]:rangeBreakpoints[0],0]) + rangeBreakpoints[1]
    breakpoints = [0,brightfieldDf.shape[0]-1]

    cutOff = maxPointRangeCutoff/2
    for row in range(maxPoint,-1,-1):
        if brightfieldDf.iloc[row,0] < cutOff:
            breakpoints[0] = row
            break        
    for row in range(maxPoint,brightfieldDf.shape[0]):
        if brightfieldDf.iloc[row,0] < 0.01:
            breakpoints[1] = row
            break        

    if visualize:
        g = sns.relplot(data=brightfieldDf,x='Row',y='Count',kind='line')
        g.axes.flat[0].axhline(y=maxPointRangeCutoff,linestyle=':',color='k')
        g.axes.flat[0].axhline(y=cutOff,linestyle=':',color='r')
        for breakpoint in breakpoints:
            g.axes.flat[0].axvline(x=breakpoint,linestyle=':',color='k')
        g.axes.flat[0].plot(maxPoint, brightfieldDf.iloc[maxPoint,0],color='k',marker='o')

    return breakpoints

def verticallySeparateMice(mouseBrightfieldMatrix,breakpoints,visualize=False):
    
    croppedBrightfieldMatrix = mouseBrightfieldMatrix[breakpoints[0]:breakpoints[1]+1,:]
    verticalMouseSeparationDf = pd.DataFrame(croppedBrightfieldMatrix,index=list(range(croppedBrightfieldMatrix.shape[0])),columns=list(range(croppedBrightfieldMatrix.shape[1])))
    verticalMouseSeparationDf.index.name = 'Row'
    verticalMouseSeparationDf.columns.name = 'Column'
    verticalMouseSeparationDf = verticalMouseSeparationDf.sum(axis=0).to_frame('Count').rolling(window=25).mean().fillna(value=0)
    verticalMouseSeparationDf.loc[:,:] = MinMaxScaler().fit_transform(verticalMouseSeparationDf.values)

    data = savgol_filter(verticalMouseSeparationDf.values.flatten(),31,2)
    data = abs(data)
    mins, _ = find_peaks(-1*data)
    #Further selection; mice have to be a certain distance apart
    trueMins = []
    minMouseWidth = 90
    minGroups = get_blocks(mins,minMouseWidth)
    for group in minGroups:
        if len(group) == 1:
            minIndex = group[0]
        else:
            dataVals = [data[x] for x in group]
            #minIndex = sorted(group)[-1]
            minIndex = group[dataVals.index(min(dataVals))]
        trueMins.append(minIndex)
    trueMins.append(verticalMouseSeparationDf.shape[0]-1)
    trueMins = np.unique(trueMins).tolist()
    #Even further selection; each interval must have a max of at least 15% the overall max
    splitMice = []
    maxCutoff = 0.2
    keptIntervals = []
    trueMins2 = []
    for i,trueMin in enumerate(trueMins[:-1]):
        maxI = 10
        if trueMin+maxI > verticalMouseSeparationDf.shape[0]-1:
            maxI = verticalMouseSeparationDf.shape[0]-trueMin-1
        if data[trueMin+maxI] >= data[trueMin]:
            trueMins2.append(trueMin)
    if data[trueMins[-1]-10] >= data[trueMins[-1]]:
        trueMins2.append(trueMins[-1])
    trueMins = trueMins2
    trueCutoff = min(data) + maxCutoff * (max(data)-min(data))
    for interval in pairwise(trueMins):
        if max(data[interval[0]:interval[1]]) > trueCutoff and abs(interval[1] - interval[0]) >= minMouseWidth:
            keptIntervals.append(interval)

    peaks,finalKeptIntervals = [],[]
    secondaryCutoff = 0.1
    for interval in keptIntervals:
        startpoint = interval[0]
        endpoint = interval[1]
        intervalValues = verticalMouseSeparationDf.query("Column > @startpoint and Column < @endpoint").reset_index()
        peakPoint = np.argmax(intervalValues['Count'].values)
        peak = intervalValues['Column'][peakPoint]
        peaks.append(peak)
        #Even further selection; crop sides of interval at cutoff point; starting from max point
        leftEndpoint = interval[0]
        rightEndpoint = interval[1]
        for val in range(peak,rightEndpoint):
            if intervalValues.query("Column == @val")['Count'].values[0] < secondaryCutoff:
                rightEndpoint = val
                break
        for val in range(peak,leftEndpoint,-1):
            if intervalValues.query("Column == @val")['Count'].values[0] < secondaryCutoff:
                leftEndpoint = val
                break
        finalKeptIntervals.append([leftEndpoint,rightEndpoint])
    
    if visualize:
        verticalMouseSeparationDf.loc[:,:] = data.reshape(-1,1)
        g = sns.relplot(data=verticalMouseSeparationDf,x='Column',y='Count',kind='line')
        g.axes.flat[0].axhline(y=trueCutoff,linestyle=':',color='k')
        g.axes.flat[0].axhline(y=secondaryCutoff,linestyle=':',color='r')
        for interval in finalKeptIntervals:
            g.axes.flat[0].axvline(x=interval[0],linestyle=':',color='k')
            g.axes.flat[0].axvline(x=interval[1],linestyle=':',color='k')
        
    return finalKeptIntervals,peaks

def fullySeparateMice(luminescentSamples,brightfieldSamples,originalBrightfieldSamples,verticalBreakpoints,horizontalBreakpoints,visualize=False):
    
    miceSamples = np.dstack([luminescentSamples,brightfieldSamples])
    splitMice,splitBrightfields = [],[]
    posthocCutoff = 10
    for interval in verticalBreakpoints:
        splitMouse = miceSamples[horizontalBreakpoints[0]:horizontalBreakpoints[1],max(interval[0]-posthocCutoff,0):interval[1]-posthocCutoff,:]
        if len(originalBrightfieldSamples.shape) > 2:
            splitBrightfield = originalBrightfieldSamples[horizontalBreakpoints[0]:horizontalBreakpoints[1],max(interval[0]-posthocCutoff,0):interval[1]-posthocCutoff,:]
        else:
            splitBrightfield = originalBrightfieldSamples[horizontalBreakpoints[0]:horizontalBreakpoints[1],max(interval[0]-posthocCutoff,0):interval[1]-posthocCutoff]            
        splitMice.append(splitMouse)
        splitBrightfields.append(splitBrightfield)
    if visualize:
        fig, axes = plt.subplots(2,len(splitMice),figsize=(2.5*len(splitMice),5))
        for i,sample in enumerate(splitMice):
            if len(splitMice) > 1:
                axes[0,i].set_title(str(i+1))
                axes[0,i].imshow(sample[:,:,:3])
                axes[1,i].imshow(sample[:,:,3],cmap='Greys')
            else:
                axes[0].set_title(str(i+1))
                axes[0].imshow(sample[:,:,:3])
                axes[1].imshow(sample[:,:,3],cmap='Greys')
            if i == 0:
                if len(splitMice) > 1:
                    for ax in [axes[0,i],axes[1,i]]:
                        ax.set_xlabel('')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)                
                    axes[0,i].set_ylabel('Luminescent')
                    axes[1,i].set_ylabel('Brightfield')
                else:
                    for ax in [axes[0],axes[1]]:
                        ax.set_xlabel('')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)                
                    axes[0].set_ylabel('Luminescent')
                    axes[1].set_ylabel('Brightfield')
            else:
                if len(splitMice) > 1:
                    axes[0,i].axis('off')
                    axes[1,i].axis('off')
                else:
                    axes[0].axis('off')
                    axes[1].axis('off')

        fig.tight_layout()

    return splitMice,splitBrightfields

def returnRadianceMetrics(imageTitle,samples,splitBrightfields,colorScale,linearScale,trueMin,sampleNames=[],save_pixel_df=False,visualize=False):            
    outputDir = 'outputData/'
    statisticList = []
    if len(sampleNames) == 0:
        sampleNames = list(map(str,list(range(1,len(samples)+1))))

    radianceColorDict,pixelIntensityColorDict = {},{}
    for i,color in enumerate(colorScale):
        radianceColorDict[tuple(color)] = linearScale[i]
        pixelIntensityColorDict[tuple(color)] = i
    pixelDfList,pixelMatrixList,pixelMatrixNameList, = [],[],[]
    
    for sn,rgbSample in enumerate(samples):
        #Convert to RGB
        trueRGBSample = np.multiply(255,rgbSample[:,:,:3]).astype(int)
        #Create pixelwise radiance dataframe
        if visualize or save_pixel_df:
          intensityMatrix = img_array_to_single_val(trueRGBSample, pixelIntensityColorDict)
          radianceMatrix = img_array_to_single_val(trueRGBSample, radianceColorDict)
          brightfieldMatrix = rgbSample[:,:,3]
          if visualize:
              pixelIntensityDf = pd.DataFrame(intensityMatrix)
              pixelIntensityDf.index.name = 'Row'
              pixelIntensityDf.columns.name = 'Column'
              pixelRadianceDf = pd.DataFrame(radianceMatrix)
              pixelRadianceDf.index.name = 'Row'
              pixelRadianceDf.columns.name = 'Column'
              pixelBrightfieldDf = pd.DataFrame()
              pixelBrightfieldDf.index.name = 'Row'
              pixelBrightfieldDf.columns.name = 'Column'
              pixelRadianceAndBrightfieldDf = pd.concat([pixelIntensityDf.stack(),pixelRadianceDf.stack(),pixelBrightfieldDf.stack()],axis=1,keys=['Intensity','Radiance','Brightfield'])
              pixelDfList.append(pixelRadianceAndBrightfieldDf)
          else:
           imageMatrixPath = outputDir+'imageMatrices/'
           np.save(imageMatrixPath+'-'.join([imageTitle,sampleNames[sn]]),np.dstack([radianceMatrix,brightfieldMatrix,splitBrightfields[sn]]))
           with open(imageMatrixPath+'-'.join([imageTitle,sampleNames[sn]])+'.pkl','wb') as f:
               pickle.dump([trueMin,linearScale[-1]],f)

#             pixelMatrixList.append(np.dstack([radianceMatrix,brightfieldMatrix,splitBrightfields[sn]]))
#             pixelMatrixNameList.append('-'.join([imageTitle,sampleNames[sn]]))

        #Remove non mouse pixels
        brightfieldMask = rgbSample[:,:,3] == 1
        trueRGBSample = trueRGBSample[brightfieldMask]
        
        #Convert pixel intensity arrays to radiance arrays
        trueRGBSample = trueRGBSample[:,np.newaxis,:]
        scoringMatrix = img_array_to_single_val(trueRGBSample, pixelIntensityColorDict)
        radianceMatrix = img_array_to_single_val(trueRGBSample, radianceColorDict)

        #Selects non color scale pixels
        colorScaleBoolean = (radianceMatrix == -1).astype(int)
        additionMatrix = np.multiply(colorScaleBoolean,0) + colorScaleBoolean
                
        #Set mouse pixels (white/greyscale) to minimum detectable colorbar value
        radianceMatrix = np.add(radianceMatrix,additionMatrix)
        
        #Image statistics
        avgPixelIntensity = np.sum(scoringMatrix+1)/(scoringMatrix.shape[0])
        numTumorPixels = radianceMatrix[radianceMatrix != 0].shape[0]
        
        #Radiance statistics
        avgRadiance = np.sum(radianceMatrix)/(trueRGBSample.shape[0])
        totalRadiance = np.sum(radianceMatrix)
        
        #Tumor spatial area statistics
        tumorAreaCovered = numTumorPixels/trueRGBSample.shape[0]
        
        statisticList.append([avgPixelIntensity,avgRadiance,totalRadiance,tumorAreaCovered])

    if visualize:
      pixelDf = pd.concat(pixelDfList,keys=sampleNames,names=['Sample'])
      if 'processedImages' not in os.listdir(outputDir):
        os.mkdir(outputDir+'processedImages')
    else:
      pixelDf = []
    statisticMatrix = np.matrix(statisticList)
    statisticDf = pd.DataFrame(statisticMatrix,index=sampleNames,columns=['Average Pixel Intensity','Average Radiance','Total Radiance','Tumor Fraction'])
    statisticDf.index.name = 'Sample'

    if visualize:
      intensityDf = pixelDf['Radiance'].unstack('Column')
      brightfieldDf = pixelDf['Brightfield'].unstack('Column')
      initialDf = np.add((intensityDf.values == -1).astype(int),(brightfieldDf.values == 0).astype(int))
      initialDf = (initialDf != 0).astype(int) 
      additionDf = np.multiply(initialDf,trueMin-1)
      intensityDf.iloc[:,:] = np.multiply(intensityDf.values,1-initialDf) + additionDf
      numSamples = len(intensityDf.index.unique('Sample'))
      fig, axes = plt.subplots(1,numSamples,figsize=(2.5*numSamples,5))
      fig.subplots_adjust(right=0.8)
      cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
      for i,sample in enumerate(intensityDf.index.unique('Sample')):
        b = splitBrightfields[i]
        sampleDf = intensityDf.query("Sample == @sample").dropna(axis=1) 
        cmap = sns.color_palette("magma", as_cmap=True)
        sampleDf.iloc[-1,-1] = np.nanmax(linearScale[-1])
        if len(intensityDf.index.unique('Sample')) > 1:
            g = sns.heatmap(sampleDf,cmap=transparent_cmap(cmap),cbar= i == 0,ax=axes[i],norm=matplotlib.colors.LogNorm(),vmin=np.nanmin(intensityDf.values),vmax=linearScale[-1],cbar_ax=cbar_ax,cbar_kws={'label':'Radiance\n(p/sec/cm$^2$/sr)'})
            axes[i].set_title('Sample '+str(i+1))
            axes[i].imshow(b,zorder=0, cmap='gray')
            axes[i].axis('off')
        else:
            g = sns.heatmap(sampleDf,cmap=transparent_cmap(cmap),cbar= i == 0,ax=axes,norm=matplotlib.colors.LogNorm(),vmin=np.nanmin(intensityDf.values),vmax=linearScale[-1],cbar_ax=cbar_ax,cbar_kws={'label':'Radiance\n(p/sec/cm$^2$/sr)'})
            axes.set_title('Sample '+str(i+1))
            axes.imshow(b,zorder=0, cmap='gray')
            axes.axis('off')
            
            
      fig.suptitle(imageTitle)
      fig.subplots_adjust(top=0.85)
      fig.savefig(outputDir+'processedImages'+imageTitle+'.png',bbox_inches='tight')
      plt.close()
    
    return statisticDf

def processGroupedMouseImage(imageTitle,luminescent,brightfield,brightfield2,visualize=False,save_pixel_df=False,sampleNames = [],save_images=False):
    
    #Preprocess images
    luminescentSamples,colorBar,legend,colorBarScale = returnLuminescentImageComponents(luminescent,visualize=visualize)
    
    colorScale = returnColorScale(colorBar)
    linearScale,trueMin = returnColorScaleSpan(legend,colorScale,colorBarScale)
    brightfieldSamples,originalBrightfieldSamples = rescaleBrightfieldImage(brightfield,brightfield2,luminescentSamples,visualize=visualize)

    #Crop images
    horizontalBreakpoints = horizontallySeparateMice(brightfieldSamples,visualize=visualize)
    verticalBreakpoints,peaks = verticallySeparateMice(brightfieldSamples,horizontalBreakpoints,visualize=visualize)    
    splitMice,splitBrightfields = fullySeparateMice(luminescentSamples,brightfieldSamples,originalBrightfieldSamples,verticalBreakpoints,horizontalBreakpoints,visualize=visualize)
    
    #Create radiance statistic dataframe and pixel-wise radiance dataframe
    radianceStatisticDf = returnRadianceMetrics(imageTitle,splitMice,splitBrightfields,colorScale,linearScale,trueMin,sampleNames = sampleNames,save_pixel_df=save_pixel_df,visualize=save_images)
    return radianceStatisticDf,peaks

def addTrueIndexToDataframe(radianceStatisticDf,sampleNameFile):
    matrixList,tupleList = [],[]
    for row in range(sampleNameFile.shape[0]):
        sampleName = list(sampleNameFile['Group'])[row]
        time = list(sampleNameFile['Day'])[row]
        sampleStatistics = radianceStatisticDf.xs([sampleName,time],level=['Group','Day'])
        matrixList.append(sampleStatistics.values)
        for sample in sampleStatistics.index.unique('Sample'):
          tupleList.append(sampleNameFile.iloc[row,:].values.tolist()+[sample])

    fullMatrix = np.vstack(matrixList)
    multiIndex = pd.MultiIndex.from_tuples(tupleList,names=list(sampleNameFile.columns)+['Sample']).droplevel('Group')
    completeDf = pd.DataFrame(fullMatrix,index=multiIndex,columns=radianceStatisticDf.columns)
    if 'SampleNames' in completeDf.index.names:
        completeDf = completeDf.droplevel('SampleNames')
    return completeDf

def addTrueIndexToPixelDataframe(radiancePixelDf,sampleNameFile):
    matrixList,tupleList = [],[]
    for row in range(sampleNameFile.shape[0]):
        sampleName = list(sampleNameFile['Group'])[row]
        time = list(sampleNameFile['Day'])[row]
        sampleStatistics = radiancePixelDf.xs([sampleName,time],level=['Group','Day'])
        matrixList.append(sampleStatistics.values)
        for sample in sampleStatistics.index.unique('Sample'):
          sampleDf = sampleStatistics.query("Sample == @sample").reset_index()
          for i in range(sampleDf.shape[0]):
            tupleList.append(sampleNameFile.iloc[row,:].values.tolist()+[sample,sampleDf['Row'][i],sampleDf['Column'][i]])

    fullMatrix = np.vstack(matrixList)
    multiIndex = pd.MultiIndex.from_tuples(tupleList,names=list(sampleNameFile.columns)+['Sample','Row','Column']).droplevel('Group')
    completeDf = pd.DataFrame(fullMatrix,index=multiIndex,columns=radiancePixelDf.columns)
    if 'SampleNames' in completeDf.index.names:
        completeDf = completeDf.droplevel('SampleNames')
    return completeDf

def luminescentBrightfieldMatchCheck(sampleNameFile,save_pixel_df=False):
  
  inputDir = 'inputData/'
  outputDir = 'outputData/'
  days = [x for x in pd.unique(sampleNameFile['Day'])]

  unmatchedGroups = []
  if save_pixel_df:
    if 'imageMatrices' not in os.listdir(outputDir):
      os.mkdir(outputDir+'imageMatrices')    
    
  for i in range(len(days)):
      day = days[i]
      tempDf = sampleNameFile.query("Day == @day")
      groups = list(pd.unique([x for x in pd.unique(tempDf['Group'])]))
      luminescentImages = [x.split('.')[0][0] for x in os.listdir(inputDir+'luminescent/'+day+'/') if '.DS' not in x]
      brightfieldImages = [x.split('.')[0][0] for x in os.listdir(inputDir+'brightfield/'+day+'/') if '.DS' not in x]
      for j in range(len(groups)):
          group = groups[j]
          if group not in luminescentImages or group not in brightfieldImages:
            if group not in luminescentImages:
              unmatchedGroups.append('luminescent/'+day+'/'+group)
            else:
              unmatchedGroups.append('brightfield/'+day+'/'+group)
  return unmatchedGroups

def amendSampleNames(fullDf,allPeaks,sampleNameFile,fullSplitGroupDict,save_pixel_df=False):
    base_dir = 'outputData/'

    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(len(allPeaks)*0.1))
    cluster_labels = clusterer.fit_predict(np.array(allPeaks).reshape(-1,1))
    cluster_labels = [str(x+1) for x in cluster_labels]  
    tempDf = pd.DataFrame({'Max':allPeaks,'Cluster':cluster_labels})
    sortedClusterList = tempDf.groupby(['Cluster']).mean().sort_values(by='Max').index.get_level_values('Cluster').tolist()
    
    topX = 5
    topXclusters = tempDf.groupby(['Cluster']).count().sort_values(by=['Max'],ascending=False).index.get_level_values('Cluster').tolist()[:topX]    
    allClusterMeansDf = tempDf.groupby(['Cluster']).mean()
    clusterMeans = sorted([allClusterMeansDf.loc[x].values[0] for x in topXclusters])
    partitions = [0] + [(clusterMeans[x]+clusterMeans[x+1])/2 for x in range(len(clusterMeans)-1)] + [np.max(tempDf['Max'])]
    new_cluster_labels = []
    for row in range(tempDf.shape[0]):
        val = tempDf['Max'][row]
        for i in range(len(partitions)-1):
            if val > partitions[i] and val <= partitions[i+1]:
                new_cluster_labels.append(str(i+1))
                break
    
    #new_cluster_labels = [str(sortedClusterList.index(x)+1) for x in cluster_labels]
    
    newFullDf = fullDf.copy()
    newFullDf.index.names = ['Position' if x == 'Sample' else x for x in fullDf.index.names]
    newFullDf = newFullDf.assign(Sample=new_cluster_labels).set_index('Sample', append=True)
    
    fullDfList = []
    splitGroups = [',,'.join(x.split(',,')[:2]) for x in fullSplitGroupDict]
    matrixRenamingDict = {}
    for currentDay in newFullDf.index.unique('Day'):
        dayDf = newFullDf.query("Day == @currentDay")
        for currentGroup in dayDf.index.unique('Group'):
            indexingKey = ',,'.join([currentDay,currentGroup])
            if indexingKey in splitGroups:
                keys = []
                for index,elem in enumerate(splitGroups):
                    if elem == indexingKey:
                        keys.append(list(fullSplitGroupDict.keys())[index])
                if type(keys) == str:
                    keys = [keys]
                subsetDfList = []
                splitPositionDict = {}
                for key in keys:
                    splitPositionDict[key.split(',,')[2]] = 0
                for key in keys:
                    splitKey = key.split(',,')
                    day,group,samples = splitKey[0],splitKey[1],splitKey[2].split(',')
                    renamingPositions = fullSplitGroupDict[key]
                    subsetDf = newFullDf.query("Day == @day and Group == @group and Position == @samples").query("Sample == @renamingPositions")
                    for position,renamedPosition in zip(subsetDf.index.unique('Position'),renamingPositions):
                        splitPositionDict[splitKey[2]]+=1
                        trueIndex = newFullDf.query("Day == @day and Group == @group and Position == @samples").index.unique('Sample').tolist().index(renamedPosition)
                        matrixRenamingDict['-'.join([currentDay,currentGroup,renamedPosition])] = '-'.join([currentDay,currentGroup+','.join(renamingPositions),str(trueIndex+1)])
                    subsetDfList.append(subsetDf)
                renamedEntry = pd.concat(subsetDfList).sort_values(by=['Sample'])
            else:
                renamedEntry = newFullDf.query("Day == @currentDay and Group == @currentGroup")
                for position,renamedPosition in zip(renamedEntry.index.unique('Position'),renamedEntry.index.unique('Sample')):
                    matrixRenamingDict['-'.join([currentDay,currentGroup,renamedPosition])] = '-'.join([currentDay,currentGroup,position])
            if 'SampleNames' in sampleNameFile.columns:
                sampleNameVal = sampleNameFile[(sampleNameFile["Day"] == currentDay) & (sampleNameFile["Group"] == currentGroup)]['SampleNames'].values[0]
                if not pd.isna(sampleNameVal) and sampleNameVal != '':
                    renamingDict = {}
                    for oldSampleName,newSampleName in zip(renamedEntry.index.unique('Sample').tolist(),sampleNameVal.split(',')):
                        renamingDict[oldSampleName] = newSampleName
                        oldSampleKey = matrixRenamingDict.pop('-'.join([currentDay,currentGroup,oldSampleName]))
                        matrixRenamingDict['-'.join([currentDay,currentGroup,newSampleName])] = oldSampleKey
                    renamedEntry = renamedEntry.rename(renamingDict,level='Sample')
            renamedEntry = renamedEntry.sort_values(by=['Sample']).droplevel('Position')
            fullDfList.append(renamedEntry)
    
    if save_pixel_df:
        savezDict,minScaleDict = {},{}
        for savezKey in matrixRenamingDict:
            oldFileName = matrixRenamingDict[savezKey]
            savezDict[savezKey] = np.load(base_dir+'imageMatrices/'+oldFileName+'.npy')            
            minScaleDict[savezKey] = pickle.load(open(base_dir+'imageMatrices/'+oldFileName+'.pkl','rb'))
        
        #Save concatenated files
        experimentName = os.getcwd().split(dirSep)[-1]
        np.savez_compressed(base_dir+experimentName+'-pixel',**savezDict)
        with open(base_dir+experimentName+'-minScale.pkl','wb') as f:
            pickle.dump(minScaleDict,f)
        #Delete temporary directory
        shutil.rmtree(base_dir+'imageMatrices/')
        
    fullDf = pd.concat(fullDfList)    
    return fullDf

def checkSplitGroups(day,group,allPeaks,sampleNames=[],visualize=False,save_df=True,save_pixel_df=False,save_images=False):
    base_dir = 'inputData/'
    luminescentImages = sorted([x.split('.')[0] for x in os.listdir(base_dir+'luminescent/'+day+'/') if len(x.split('.')[0]) > 1 and group in x.split('.')[0]])
    brightfieldImages = sorted([x.split('.')[0] for x in os.listdir(base_dir+'brightfield/'+day+'/') if len(x.split('.')[0]) > 1 and group in x.split('.')[0]])
    splitGroupDict = {}
    if len(luminescentImages) > 1:
        if set(luminescentImages) == set(brightfieldImages):
            groupDfList,groupPixelDfList,groupPeaksList = [],[],[]
            splitIndex = 0
            for splitLuminescentFileName in luminescentImages:
                #Read in luminescent image
                if splitLuminescentFileName+'.png' in os.listdir(base_dir+'luminescent/'+day):
                  fileName = base_dir+'luminescent/'+day+'/'+splitLuminescentFileName+'.png'
                else:
                  fileName = base_dir+'luminescent/'+day+'/'+splitLuminescentFileName+'.PNG'
                luminescent = mplImage.imread(fileName)[:,:,:3]

                #Read in brightfield image
                if splitLuminescentFileName+'.tif' in os.listdir(base_dir+'brightfield/'+day):
                  fileName = base_dir+'brightfield/'+day+'/'+splitLuminescentFileName+'.tif'
                else:
                  fileName = base_dir+'brightfield/'+day+'/'+splitLuminescentFileName+'.TIF'
                brightfield = mplImage.imread(fileName)
                brightfield2 = cv2.imread(fileName)        
                positionsToKeep = splitLuminescentFileName[1:].split(',')
                groupDf,groupPeaks = processGroupedMouseImage(day+'-'+splitLuminescentFileName,luminescent,brightfield,brightfield2,sampleNames=sampleNames,visualize=visualize,save_images=save_images,save_pixel_df=save_pixel_df)
                if splitIndex != 0:
                    originalSamples = groupDf.index.unique('Sample').tolist()
                    newSamples = [str(int(x)+splitIndex) for x in originalSamples]
                    renamingDict = {}
                    for og,new in zip(originalSamples,newSamples):
                        renamingDict[og] = new
                    groupDf = groupDf.rename(renamingDict,level='Sample')
                
                splitIndex+=len(groupDf.index.unique('Sample').tolist())
                splitGroupDict[day+',,'+group+',,'+','.join(groupDf.index.unique('Sample').tolist())] = positionsToKeep
                groupDfList.append(groupDf)
                groupPeaksList+=groupPeaks
            groupDf = pd.concat(groupDfList)
            return groupDf,groupPeaksList,splitGroupDict           
        else:
            print('These images are not shared:\n')
            for missingImage in list(set(luminescentImages) ^ set(brightfieldImages)):
              print(day+'-'+missingImage)
            print('\nExiting...')
            sys.exit(0)
    else:
        #Read in luminescent image
        if group+'.png' in os.listdir(base_dir+'luminescent/'+day):
          fileName = base_dir+'luminescent/'+day+'/'+group+'.png'
        else:
          fileName = base_dir+'luminescent/'+day+'/'+group+'.PNG'
        luminescent = mplImage.imread(fileName)[:,:,:3]

        #Read in brightfield image
        if group+'.tif' in os.listdir(base_dir+'brightfield/'+day):
          fileName = base_dir+'brightfield/'+day+'/'+group+'.tif'
        else:
          fileName = base_dir+'brightfield/'+day+'/'+group+'.TIF'
        brightfield = mplImage.imread(fileName)
        brightfield2 = cv2.imread(fileName)        
        groupDf,groupPeaks = processGroupedMouseImage(day+'-'+group,luminescent,brightfield,brightfield2,sampleNames=sampleNames,visualize=visualize,save_images=save_images,save_pixel_df=save_pixel_df)
        return groupDf,groupPeaks,splitGroupDict

def moveRawImages(sampleNameFile,pathToRawImages):
    fileExtensionDict = {'brightfield':'.TIF','luminescent':'.PNG'}
    dayRenamingDict = {}
    for imageType in ['brightfield','luminescent']:
        if imageType not in os.listdir('inputData'):
            os.mkdir('inputData/'+imageType)
        for day in list(pd.unique(sampleNameFile['Day'])):
            newDay = 'D'+''.join([i for i in day.split() if i.isdigit()])
            dayRenamingDict[day] = newDay
            if day in os.listdir(pathToRawImages):
                if newDay not in os.listdir('inputData/'+imageType):
                    os.mkdir('inputData/'+imageType+'/'+newDay)
                for group in list(pd.unique(sampleNameFile['Group'])):
                    if group in os.listdir(pathToRawImages+'/'+day):
                        initialPath = pathToRawImages+'/'+day+'/'+group+'/'
                        if imageType == 'brightfield':
                            initialName = 'photograph.TIF'
                        else:
                            for fileName in os.listdir(pathToRawImages+'/'+day+'/'+group):
                                if '.png' in fileName or '.PNG' in fileName:
                                    initialName = fileName
                                    break
                        finalPath =  'inputData/'+imageType+'/'+newDay+'/'
                        finalName = group+fileExtensionDict[imageType]
                        shutil.copyfile(initialPath+initialName,finalPath+finalName)
    
    dayIndex = list(sampleNameFile.columns).index('Day')
    for i in range(sampleNameFile.shape[0]):
        oldDay = sampleNameFile.iloc[i,dayIndex]
        sampleNameFile.iloc[i,dayIndex] = dayRenamingDict[oldDay]
    return sampleNameFile

def fullInVivoImageProcessingPipeline(sampleNameFile,visualize=False,save_df=True,save_pixel_df=False,save_images=False,pathToRawImages=''):
  outputDir = 'outputData/'
  if pathToRawImages != '':
      sampleNameFile = moveRawImages(sampleNameFile,pathToRawImages)
  unmatchedGroups = luminescentBrightfieldMatchCheck(sampleNameFile,save_pixel_df=save_pixel_df)
  if len(unmatchedGroups) == 0:
    days = [x for x in pd.unique(sampleNameFile['Day'])]

    dayDfList,dayPixelDfList,dayPeaksList = [],[],[]
    fullSplitGroupDict = {}
    for i in trange(len(days), desc='Processing Days:'):
        day = days[i]
        groupDfList,groupPixelDfList = [],[]
        tempDf = sampleNameFile.query("Day == @day")
        groups = list(pd.unique([x for x in pd.unique(tempDf['Group'])]))
        #print(day)
        for j in trange(len(groups), desc='Processing Groups:',leave=False):
            group = groups[j]
            sampleNames = []
            
            groupDf,groupPeaks,splitGroupDict = checkSplitGroups(day,group,dayPeaksList,sampleNames=sampleNames,visualize=visualize,save_images=save_images,save_pixel_df=save_pixel_df)
            groupDfList.append(groupDf)
            dayPeaksList+=groupPeaks
            fullSplitGroupDict = {**fullSplitGroupDict,**splitGroupDict}
            #print(group)
        dayDf = pd.concat(groupDfList,keys=groups,names=['Group'])
        dayDfList.append(dayDf)

    experimentName = os.getcwd().split(dirSep)[-1]
    outputFileName = 'radianceStatisticPickleFile-'+experimentName
    if 'SampleNames' not in tempDf.columns:
        sampleNamesColumn = []
    else:
        sampleNamesColumn = sampleNameFile['SampleNames'].tolist()
    
    fullDf = pd.concat(dayDfList,keys=days,names=['Day'])
    fullDf = amendSampleNames(fullDf,dayPeaksList,sampleNameFile,fullSplitGroupDict,save_pixel_df=save_pixel_df)
    radianceStatisticDf = addTrueIndexToDataframe(fullDf,sampleNameFile)
    radianceStatisticDf['Time'] = [int(x[1:]) for x in radianceStatisticDf.index.get_level_values('Day').tolist()]
    radianceStatisticDf = radianceStatisticDf.set_index(['Time'],append=True)
    
    #Ensure order is "all group indices",day,time,sample
    allGroupIndices = [x for x in radianceStatisticDf.index.names if x not in ['Day','Time','Sample']]
    radianceStatisticDf = radianceStatisticDf.reset_index().set_index(allGroupIndices+['Day','Time','Sample'])

    if save_df:
      radianceStatisticDf.to_pickle(outputDir+outputFileName+'.pkl')
      radianceStatisticDf.to_excel(outputDir+outputFileName+'.xlsx')    
    return radianceStatisticDf
  else:
    print('These images are missing:\n')
    for missingImage in unmatchedGroups:
      print(missingImage)
    print('\nExiting...')
    return []

