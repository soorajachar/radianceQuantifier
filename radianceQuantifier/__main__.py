#! /usr/bin/env python3
import pickle,os,subprocess,sys
import tkinter as tk
import tkinter.ttk
import pandas as pd
from PIL import Image,ImageTk
from importlib_metadata import version
from radianceQuantifier.dataprocessing.inVivoRadianceProcessing import fullInVivoImageProcessingPipeline
from radianceQuantifier.dataprocessing.survivalProcessing import createSurvivalDf
from radianceQuantifier.dataprocessing.miscFunctions import setMaxWidth
from radianceQuantifier.setup.experimentCreationGUI import NewExperimentWindow,NewProjectWindow,RemoveProjectWindow
from radianceQuantifier.setup.experimentSetupGUI import ExperimentSetupStartPage
from radianceQuantifier.plotting.plottingGUI import PlotExperimentWindow 
import radianceQuantifier

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

#Root class; handles frame switching in gui
class MainApp(tk.Tk):
    def __init__(self):
        self.root = tk.Tk.__init__(self)
        self.title('radianceQuantifier '+version('radianceQuantifier'))
        self._frame = None
        self.homedirectory = dirSep.join(os.path.abspath(radianceQuantifier.__file__).split(dirSep)[:-1])
        if self.homedirectory[-1] != '/':
            self.homedirectory+='/'
        print('radianceQuantifier location: '+self.homedirectory)
        self.switch_frame(RadianceQuantifierHomePage)

    def switch_frame(self, frame_class,*args):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self,*args)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

#Top level actions for experiments
class RadianceQuantifierHomePage(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10)
        
        load = Image.open(master.homedirectory+"radianceQuantifierLogo.png")
        width, height = load.size
        SCALEFACTOR = 0.3
        load = load.resize((int(SCALEFACTOR*width), int(SCALEFACTOR*height)), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(mainWindow, image=render)
        img.image = render
        img.grid(row=0,column=0,columnspan=3)
        v = tk.StringVar(value='slt')
        rb1c = tk.Button(mainWindow, text="Associate new experiment with project",padx = 20, command=lambda:master.switch_frame(NewExperimentWindow,RadianceQuantifierHomePage))
        rb1c.grid(row=3,column=0,sticky=tk.EW,columnspan=3,pady=1)
        rb1a = tk.Button(mainWindow, text="Create new project",padx = 20, command=lambda:master.switch_frame(NewProjectWindow,RadianceQuantifierHomePage))
        rb1b = tk.Button(mainWindow, text="Remove project",padx = 20, command=lambda:master.switch_frame(RemoveProjectWindow,RadianceQuantifierHomePage))
        rb1a.grid(row=1,column=0,sticky=tk.EW,columnspan=3,pady=(20,1))
        rb1b.grid(row=2,column=0,sticky=tk.EW,columnspan=3,pady=1)
        
        expSelectionFrame = tk.Frame(mainWindow,borderwidth=1,relief=tk.SOLID)
        expSelectionFrame.grid(row=4,column=0,sticky=tk.EW,columnspan=3,pady=(10,0))
        rb1d = tk.Button(expSelectionFrame,text="Work on existing experiment ",padx = 20, command=lambda: collectInput())
        
        def getUpdateData(event):
            projectName = self.projectMenu.get()
            pathName = self.pathDict[projectName]
            experiments = []
            for experimentName in os.listdir(pathName+projectName):
                if '.DS' not in experimentName:
                    experiments.append(experimentName)
            experiments = sorted(experiments)[::-1]
            self.experimentMenu['values'] = experiments
            if len(experiments) == 1:
                self.experimentMenu.set(self.experimentMenu['values'][0])
            self.experimentMenu['width'] = len(max(experiments,key=len))
        
        if 'misc' not in os.listdir(master.homedirectory):
            subprocess.run(['mkdir',master.homedirectory+'misc'])
        if 'pathDict.pkl' not in os.listdir(master.homedirectory+'misc'):
            self.pathDict = {}
        else:
            self.pathDict = pickle.load(open(master.homedirectory+'misc/pathDict.pkl','rb'))
        projects = list(self.pathDict.keys())
        self.projectMenu = tkinter.ttk.Combobox(expSelectionFrame,values = projects)
        if len(self.pathDict) > 0:
            self.projectMenu['width'] = len(max(projects,key=len))
        tk.Label(expSelectionFrame,text='Project name: ').pack()
        self.projectMenu.pack()
        self.projectMenu.bind('<<ComboboxSelected>>', getUpdateData)

        self.experimentMenu = tkinter.ttk.Combobox(expSelectionFrame)
        tk.Label(expSelectionFrame,text='Experiment name: ').pack()
        self.experimentMenu.pack()
        rb1d.pack(expand=True,fill=tk.BOTH,pady=(5,0))

        def collectInput():
            projectName = self.projectMenu.get()
            pathName = self.pathDict[projectName]
            selectedExperiment = self.experimentMenu.get()
            os.chdir(pathName+projectName+'/'+selectedExperiment)
            master.switch_frame(ExperimentActionWindow,selectedExperiment)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,padx=10,pady=(50,10))
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(side=tk.LEFT)

class ExperimentActionWindow(tk.Frame):
    def __init__(self,master,selectedExperiment):
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10)
        
        l1 = tk.Label(mainWindow,text='Choose an action:')
        v = tk.StringVar(value='plt')
        rb1a = tk.Radiobutton(mainWindow, text="Setup experiment",padx = 20, variable=v, value='se')
        rb1b = tk.Radiobutton(mainWindow,text="Process experiment",padx = 20, variable=v, value='pd')
        rb1c = tk.Radiobutton(mainWindow,text="Plot experiment",padx = 20, variable=v, value='plt')
        l1.grid(row=0,column=0)
        rb1a.grid(row=1,column=0,sticky=tk.W)
        rb1b.grid(row=2,column=0,sticky=tk.W)
        rb1c.grid(row=3,column=0,sticky=tk.W)
        
        def collectInput():
            action = v.get()
            if action == 'se':
                master.switch_frame(ExperimentSetupStartPage,selectedExperiment,ExperimentActionWindow)
            elif action == 'pd':
                answer = tkinter.messagebox.askokcancel(title='Confirmation',message='Do you want to process all image data?',icon=tkinter.messagebox.WARNING)
                if answer:
                    #Get input df
                    if 'templatePathDict.pkl' in os.listdir(master.homedirectory + 'misc'):
                        templatePathDict = pickle.load(open(master.homedirectory + 'misc/templatePathDict.pkl', 'rb'))
                    else:
                        templatePathDict = {}
                    #Get raw images
                    if 'rawImagePathDict.pkl' in os.listdir(master.homedirectory + 'misc'):
                        rawImagePathDict = pickle.load(open(master.homedirectory + 'misc/rawImagePathDict.pkl', 'rb'))
                    else:
                        rawImagePathDict = {}
                    projectName = os.getcwd().split(dirSep)[(-2)]
                    experimentName = os.getcwd().split(dirSep)[(-1)]
                    templatePath = templatePathDict[projectName + '/' + experimentName]
                    rawImagePath = rawImagePathDict[projectName + '/' + experimentName]
                    if '.csv' in templatePath:
                        sampleNameFile = pd.read_csv(templatePath)
                    else:
                        sampleNameFile = pd.read_excel(templatePath)

                    #Radiance df
                    radianceStatisticDf = fullInVivoImageProcessingPipeline(sampleNameFile,save_pixel_df=True,pathToRawImages=rawImagePath)
                    print(radianceStatisticDf)
                    
                    #Survival df
                    #Legacy formatting
                    subsetDf = radianceStatisticDf.copy()
                    subsetDf = subsetDf.droplevel('Time')
                    subsetDf.index.names = [x if x != 'Day' else 'Time' for x in subsetDf.index.names]
                    #Create ungrouped survival dataframe
                    survivalDf = createSurvivalDf(subsetDf,[],selectedExperiment,saveDf=True)
                    
                    tk.messagebox.showinfo(title='Success', message='Experiment processing complete!')
            elif action == 'plt':
                master.switch_frame(PlotExperimentWindow,selectedExperiment,ExperimentActionWindow)

        def backCommand():
            os.chdir(master.homedirectory)
            master.switch_frame(RadianceQuantifierHomePage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,padx=10,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInput()).pack(side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: backCommand()).pack(side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(side=tk.LEFT)

if __name__== "__main__":
    app = MainApp()
    app.mainloop()
