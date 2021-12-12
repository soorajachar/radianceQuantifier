#! /usr/bin/env python3
import pickle, os, json, math, subprocess, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

class ExperimentSetupStartPage(tk.Frame):

    def __init__(self, master, fName, bPage):
        global backPage
        global folderName
        folderName = fName
        backPage = bPage
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=(tk.TOP), padx=10, pady=10)
        tk.Label(mainWindow, text='Add groups, days, and labels (Input):').grid(row=2, column=0, sticky=(tk.W), pady=(10,0))
        modeVar = tk.StringVar(value='template')
        rb1 = tk.Radiobutton(mainWindow, text='Manually', padx=20, variable=modeVar, value='manual')
        rb2 = tk.Radiobutton(mainWindow, text='From template', padx=20, variable=modeVar, value='template')
        rb1.grid(row=3, column=0, sticky=(tk.W))
        rb2.grid(row=4, column=0, sticky=(tk.W))
        self.template = []
        self.rawimagepath = ''

        def load_rawImages():
            self.rawimagepath = fd.askdirectory()
            if self.rawimagepath[-1] != '/':
                self.rawimagepath+='/'
            rawImageLabel['text'] = self.rawimagepath
            rawImageLabel['font'] = 'Helvetica 14 bold'
        
        def load_template():
            self.filename = fd.askopenfilename(filetypes=[('Excel Files', '*.xlsx'), ('CSV Files', '*.csv')])
            if self.filename != '':
                if '.csv' in self.filename:
                    self.template = pd.read_csv(self.filename)
                else:
                    self.template = pd.read_excel(self.filename)
                templateLabel['text'] = self.filename.split(dirSep)[(-1)]
                templateLabel['font'] = 'Helvetica 14 bold'

        templateFrame = tk.Frame(mainWindow)
        templateFrame.grid(row=4, column=1, columnspan=2, sticky=(tk.W))
        templateSelectionButton = tk.Button(templateFrame, text='Browse:', command=(lambda : load_template()))
        templateSelectionButton.grid(row=0, column=0, sticky=(tk.W))
        templateLabel = tk.Label(templateFrame, text='')
        templateLabel.grid(row=0, column=1, sticky=(tk.W))
        
        modeVar2 = tk.StringVar(value='automatic')
        tk.Label(mainWindow, text='Format raw IVIS output:').grid(row=5, column=0, sticky=(tk.W), pady=(10,0))
        rb1b = tk.Radiobutton(mainWindow, text='Manually', padx=20, variable=modeVar2, value='manual')
        rb2b = tk.Radiobutton(mainWindow, text='Automatically (select path to IVIS output)', padx=20, variable=modeVar2, value='automatic')
        rb1b.grid(row=6, column=0, sticky=(tk.W))
        rb2b.grid(row=7, column=0, sticky=(tk.W))
        rawImageFrame = tk.Frame(mainWindow)
        rawImageFrame.grid(row=7, column=1, columnspan=2, sticky=(tk.W))
        rawImageSelectionButton = tk.Button(rawImageFrame, text='Browse:', command=(lambda : load_rawImages()))
        rawImageSelectionButton.grid(row=0, column=0, sticky=(tk.W))
        rawImageLabel = tk.Label(rawImageFrame, text='')
        rawImageLabel.grid(row=0, column=1, sticky=(tk.W))

        def collectInput():
            method = modeVar.get()
            method2 = modeVar2.get()
            if method == 'manual':
                tk.messagebox.showinfo(title='Not supported', message='Option not currently supported. Please select a template file and try again.')
            else:
                if method2 == 'manual' and 'brightfield' not in os.listdir('inputData') and 'luminescent' not in os.listdir('inputData'):
                    for imageType in ('luminescent', 'brightfield'):
                        if imageType not in os.listdir('inputData/'):
                            os.mkdir('inputData/' + imageType)
                        for day in list(pd.unique(self.template['Day'])):
                            if day not in os.listdir('inputData/' + imageType):
                                os.mkdir('inputData/' + imageType + '/' + day)
                if 'templatePathDict.pkl' in os.listdir(master.homedirectory + 'misc'):
                    templatePathDict = pickle.load(open(master.homedirectory + 'misc/templatePathDict.pkl', 'rb'))
                else:
                    templatePathDict = {}
                projectName = os.getcwd().split(dirSep)[(-2)]
                experimentName = os.getcwd().split(dirSep)[(-1)]
                templatePathDict[projectName + '/' + experimentName] = self.filename
                with open(master.homedirectory + 'misc/templatePathDict.pkl', 'wb') as (f):
                    pickle.dump(templatePathDict, f)
                
                if 'rawImagePathDict.pkl' in os.listdir(master.homedirectory + 'misc'):
                    rawImagePathDict = pickle.load(open(master.homedirectory + 'misc/rawImagePathDict.pkl', 'rb'))
                else:
                    rawImagePathDict = {}
                rawImagePathDict[projectName + '/' + experimentName] = self.rawimagepath 
                with open(master.homedirectory + 'misc/rawImagePathDict.pkl', 'wb') as (f):
                    pickle.dump(rawImagePathDict, f)
                tk.messagebox.showinfo(title='Success', message='Experiment setup complete!')
                master.switch_frame(backPage, folderName)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='OK', command=(lambda : collectInput())).grid(row=5, column=0)
        tk.Button(buttonWindow, text='Back', command=(lambda : master.switch_frame(backPage, folderName))).grid(row=5, column=1)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=5, column=2)
