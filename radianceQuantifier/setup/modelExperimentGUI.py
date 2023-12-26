#! /usr/bin/env python3
import pickle, os, json, math, subprocess, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from radianceQuantifier.dataprocessing.inVivoRadianceProcessing import loadPickle 
from radianceQuantifier.dataprocessing.modelingFunctions import identify_phases, fit_data, get_rates, make_bayesian_plots

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

class ModelExperimentWindow(tk.Frame):

    def __init__(self, master, fName, bPage):
        global backPage
        global folderName
        folderName = fName
        backPage = bPage
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=(tk.TOP), padx=10, pady=10)

        def on_region_selected(*args):
            ''' Update the selected region from the drop down.'''
            global selected_region_str
            selected_region_str = selected_region.get()
            print(f'Selected region: {selected_region_str.split("_")[-1]}')

        # select data to model
        choose_region_label = tk.Label(mainWindow, text='Select Region to Model:')
        choose_region_label.pack(pady=10)
        regions2choose = [item for item in os.listdir('outputData/ROI Radiance Calculation') if item.startswith('left')]
        selected_region = tk.StringVar() 
        selected_region.set(regions2choose[0]) # Initialize with the first item in the list
        region_selection_button = tk.OptionMenu(mainWindow, selected_region, *regions2choose) # dropdown menu button
        region_selection_button.pack(pady=10)

        def collectInput():
            data_dir = 'outputData/ROI Radiance Calculation'

            selected_region.trace("w", on_region_selected) # update to be region chosen from dropdown
            selected_region_str = selected_region.get() # convert selected region from tk.StringVar to actual string
            print(f'Modeling {selected_region_str.split("_")[-1]} region:')
            
            '''
            Identify Growth, Decay, and Relapse phases.
            '''
            print('Identifying Growth, Decay, and Relapse phases.')
            plot_data = loadPickle(f'{data_dir}/{selected_region_str}/{os.getcwd().split(dirSep)[-1]}_df_from_images_{selected_region_str}.pkl')
            mice_notfit_df = identify_phases(plot_data) # perform the phase identification
            mice_notfit_df.to_pickle(f'{data_dir}/{selected_region_str}/{os.getcwd().split(dirSep)[-1]}_notfit_phaseslabeled_{selected_region_str}.pkl')  # save the dataframe that contains phases
            print('DataFrame with Growth, Decay, and Relapse phases saved.')
            
            '''
            Fit phases to differential equations and calculates rates.
            '''
            print('Fitting Data.')
            phases_df = mice_notfit_df.copy() # reload in the merged data with Mouse IDs
            
            #### Look at inital distributions of rates before Bayesian Priors ####
            alphas=[0,0,0,0,0] # all zero so no Bayesian Priors yet
            mice_fit_df_noBayesian = fit_data(data=phases_df,alphas=alphas)  # no Bayesian Priors
            # save the data
            if not os.path.exists(f'{data_dir}/{selected_region_str}/BayesianPriors'): os.makedirs(f'{data_dir}/{selected_region_str}/BayesianPriors') # make dir if doesn't exist
            mice_fit_df_noBayesian.to_pickle(f'{data_dir}/{selected_region_str}/BayesianPriors/fit2model_all_alphas_{alphas[0]}_{alphas[1]}_{alphas[2]}_{alphas[3]}_{alphas[4]}_{selected_region_str}.pkl')
    
            # initial mean and stdev of pop INCLUDING outliers -- before Bayesian priors
            growth_rates, decay_rates, relapse_rates = get_rates(mice_fit_df_noBayesian,include_outliers=True) # extract the rates from the df
            print('### Initial Distributions ###')
            print(f'Growth: mean {np.mean(growth_rates)}, std {np.std(growth_rates)}')
            print(f'Decay: mean {np.mean(decay_rates)}, std {np.std(decay_rates)}')
            print(f'Relapse: mean {np.mean(relapse_rates)}, std {np.std(relapse_rates)}')

            # make plots
            plot_dir = f'plots/Before Bayesian Priors/{selected_region_str}'
            make_bayesian_plots(mice_fit_df_noBayesian,growth_rates,decay_rates,relapse_rates,plot_dir,bayesian_key='Before')
    



        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='OK', command=(lambda : collectInput())).grid(row=5, column=0)
        tk.Button(buttonWindow, text='Back', command=(lambda : master.switch_frame(backPage, folderName))).grid(row=5, column=1)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=5, column=2)