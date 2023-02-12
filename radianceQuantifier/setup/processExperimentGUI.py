#! /usr/bin/env python3
import pickle, os, json, math, subprocess, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from radianceQuantifier.dataprocessing.inVivoRadianceProcessing import fullInVivoImageProcessingPipeline
from radianceQuantifier.dataprocessing.survivalProcessing import createSurvivalDf

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

class ProcessExperimentWindow(tk.Frame):
    def __init__(self, master, backPage,selectedExperiment, sampleNameFile,pathToRawImages):
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10)
        
        l1 = tk.Label(mainWindow,text='Set color bar limits:')
        v = tk.StringVar(value='manual')
        rb1a = tk.Radiobutton(mainWindow, text="Automatically (pytesseract)",padx = 20, variable=v, value='auto')
        rb1b = tk.Radiobutton(mainWindow,text="Manually",padx = 20, variable=v, value='manual')
        l1.grid(row=0,column=0)
        rb1a.grid(row=1,column=0,sticky=tk.W)
        rb1b.grid(row=2,column=0,sticky=tk.W)
        tk.Label(mainWindow,text='Min:').grid(row=2,column=1,sticky=tk.W)
        minEntry = tk.Entry(mainWindow,width=5)
        minEntry.grid(row=2,column=2,sticky=tk.W)
        tk.Label(mainWindow,text='Max:').grid(row=2,column=3,sticky=tk.W)
        maxEntry = tk.Entry(mainWindow,width=5)
        maxEntry.grid(row=2,column=4,sticky=tk.W)

        def collectInput():
            if v.get() == 'auto':
                cbar_lim = []
            else:
                cbar_lim = [float(minEntry.get()),float(maxEntry.get())]
            
            #Radiance df
            try:
                radianceStatisticDf = fullInVivoImageProcessingPipeline(sampleNameFile,save_pixel_df=True,pathToRawImages=pathToRawImages,cbar_lim=cbar_lim)
            except UnboundLocalError:
                tk.messagebox.showinfo(title='Error', message='Automatic colorbar reading failed. Please manually enter color bar range.')

            print(radianceStatisticDf)
            
            #Survival df
            #Legacy formatting
            subsetDf = radianceStatisticDf.copy()
            subsetDf = subsetDf.droplevel('Time')
            subsetDf.index.names = [x if x != 'Day' else 'Time' for x in subsetDf.index.names]
            #Create ungrouped survival dataframe
            survivalDf = createSurvivalDf(subsetDf,[],selectedExperiment,saveDf=True)
            
            tk.messagebox.showinfo(title='Success', message='Experiment processing complete!')
            master.switch_frame(backPage, selectedExperiment)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='OK', command=(lambda : collectInput())).grid(row=5, column=0)
        tk.Button(buttonWindow, text='Back', command=(lambda : master.switch_frame(backPage, selectedExperiment))).grid(row=5, column=1)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=5, column=2)
