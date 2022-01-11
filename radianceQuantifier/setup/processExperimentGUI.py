#! /usr/bin/env python3
import pickle, os, json, math, subprocess, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

class ProcessExperimentWindow(tk.Frame):

    def __init__(self, master, backPage, sampleNameFile,pathToRawImages):

        def collectInput():
                master.switch_frame(backPage, sampleNameFile,pathToRawImages)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='OK', command=(lambda : collectInput())).grid(row=5, column=0)
        tk.Button(buttonWindow, text='Back', command=(lambda : master.switch_frame(backPage, folderName))).grid(row=5, column=1)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=5, column=2)
