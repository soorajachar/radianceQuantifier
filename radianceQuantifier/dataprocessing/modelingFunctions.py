#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import pickle5 as pickle
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from lmfit import Model, Minimizer, Parameters, fit_report, minimize
if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'


def find_mice2ignore(df):
  '''
  Function to find mice that have only one timepoint.
  
  Input: df -- df of data for all mice to look through
  Output: mice2ignore -- list of MouseID of mice to ignore
  '''

  mice_list = df.reset_index().MouseID.unique()

  mice2ignore = []
  for mouse in mice_list:
    mouse_df = df.query('MouseID == @mouse')

    # only one timepoint
    yexp = mouse_df['Average Radiance'].to_numpy() # experimental values of tumor
    if (len(yexp) <= 1):
      mice2ignore.append(mouse)

  return mice2ignore


def identify_phases(data):
  '''
  Function to identify Growth, Decay, and Relapse phases for each mouse.
  
  Input -- dataframe with Mouse IDs
  Output -- dataframe with phases for each mouse
  '''

  # identify the different phases for each mouse

  index_names = ['Date','ExperimentName','Researcher','CAR_Binding','CAR_Costimulatory','Tumor','TumorCellNumber','TCellNumber','bloodDonorID','Perturbation','Group','Phase','Day','Time','Sample','MouseID','ImageID']

  # only look at data from start of CAR-T administration (Time >= 0)
  data = data.query('Time>=0')

 # list of all mice
  mice_list = data.reset_index().MouseID.unique()

  # these mice have only one time point
  mice2ignore = find_mice2ignore(data)
  print(f'Skipping mice with only one data point: {mice2ignore}')

  # label phases for each mouse
  tg_list = []
  mice_df_list = []
  mice_badfit = []
  for mouse in tqdm(mice_list,disable=False):
    if mouse not in mice2ignore: # skip these mice
      mouse_df = data.query('MouseID == @mouse') # df for one mouse at a time
      num_of_timepoints = mouse_df.shape[0] # total number of timepoints of data (non-zero)
      mouse_df['Phase'] = np.nan #-9999 # set up default values to be overwritten

      # phase flags
      stop_phase1=False # growth
      stop_phase2=False # decay
      stop_phase3=False # relapse
      stop_phase4=False # decay2
      stop_phase5=False # relapse2

      # check if tumor first decreases
      if mouse_df['Average Radiance'][1]-mouse_df['Average Radiance'][0] < 0:
        # skip the initial growth and go to decay
        # 2. Decay
        for j in range(2,num_of_timepoints+1): # range starts at 2 because need at least two timepoints
          df4model = mouse_df.reset_index('Time').iloc[0:j] # data upto given timepoint to use for model
          df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
          yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
          # keep adding points until stops decreasing (stop if starts increasing by 3x)
          if yexp[-1]/yexp[-2] > 3: # stop
            stop_phase2 = True
            num4model = len(yexp)-1
            # import pdb; pdb.set_trace()
            break
        
        # add info to df
        if (stop_phase2==False): # used up all remaining timepoints in decay region
          mouse_df = mouse_df.reset_index()
          mouse_df.loc[0:, 'Phase'] = 'Decay'
          mouse_df = mouse_df.set_index(index_names)
        if (stop_phase2==True): # still have more time points for next phase
          mouse_df = mouse_df.reset_index()
          mouse_df.loc[0:num4model-1, 'Phase'] = 'Decay' # fill timepoints from start to end of phase 1 as Decay
          mouse_df = mouse_df.set_index(index_names)
          
          # 3. Relapse
          for k in range(j,num_of_timepoints):
            df4model = mouse_df.reset_index('Time').iloc[j-1:k+1]
            df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
            yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
            # keep adding points until starts decreasing (more than factor of 10 less than prior value)
            if yexp[-1]/yexp[-2] < 0.1: # stop
              stop_phase3 = True
              num4model = len(yexp)-1
              # import pdb; pdb.set_trace()
              break
          
          # add info to df
          if (stop_phase3==False): # used of all remaining timepoints in relapse region
            mouse_df = mouse_df.reset_index()
            mouse_df.loc[j-1:, 'Phase'] = 'Relapse' # remaining timepoints are Relapse
            mouse_df = mouse_df.set_index(index_names)
          if (stop_phase3==True): # still have more time points for next phase
            mouse_df = mouse_df.reset_index()
            mouse_df.loc[j-1:k-1, 'Phase'] = 'Relapse' # fill timepoints from end of phase 2 to end of phase 3 as Relapse
            mouse_df = mouse_df.set_index(index_names)

          
      else: # start with initial growth
        # 1. Growth
        for i in range(2,num_of_timepoints+1): # range starts at 2 because need at least two timepoints
          df4model = mouse_df.reset_index('Time').iloc[0:i] # data upto given timepoint to use for model
          df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
          yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
          # keep adding points until starts decreasing (more than 75% less than prior value)
          if yexp[-1]/yexp[-2] < 0.75: # stop
            stop_phase1 = True
            num4model = len(yexp)-1
            break

        # add info to df
        if (stop_phase1==False): # used all timepoints in initial growth region
          mouse_df = mouse_df.reset_index()        
          mouse_df.loc[0:, 'Phase'] = 'Growth'
          mouse_df = mouse_df.set_index(index_names)
        if (stop_phase1==True): # still have more time points for next phase
          mouse_df = mouse_df.reset_index()
          # fill timepoints from first time to idx of last timepoint used in this phase (subtract 1 bc index starts at 0)
          mouse_df.loc[0:num4model-1, 'Phase'] = 'Growth'
          mouse_df = mouse_df.set_index(index_names)

          # 2. Decay
          for j in range(i,num_of_timepoints):
            df4model = mouse_df.reset_index('Time').iloc[i-1:j+1]
            df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
            yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
            # keep adding points until stops decreasing (stop if starts increasing by 3x)
            if yexp[-1]/yexp[-2] > 3: # stop
              stop_phase2 = True
              num4model = len(yexp)-1
              break
          
          # add info to df
          if (stop_phase2==False): # used up all remaining timepoints in decay region
            mouse_df = mouse_df.reset_index()
            mouse_df.loc[i-1:, 'Phase'] = 'Decay'
            mouse_df = mouse_df.set_index(index_names)
          if (stop_phase2==True): # still have more time points for next phase
            mouse_df = mouse_df.reset_index()
            mouse_df.loc[i-1:j-1, 'Phase'] = 'Decay' # fill timepoints from end of phase 1 to end of phase 2 as Decay
            mouse_df = mouse_df.set_index(index_names)
        
            # 3. Relapse
            for k in range(j,num_of_timepoints):
              df4model = mouse_df.reset_index('Time').iloc[j-1:k+1]
              df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
              yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
              # keep adding points until starts decreasing (more than factor of 10 less than prior value)
              if yexp[-1]/yexp[-2] < 0.1: # stop
                stop_phase3 = True
                num4model = len(yexp)-1
                break

            # add info to df
            if (stop_phase3==False): # used up all remaining timepoints in relapse region
              mouse_df = mouse_df.reset_index()
              mouse_df.loc[j:, 'Phase'] = 'Relapse' # remaining timepoints are Relapse
              mouse_df = mouse_df.set_index(index_names)
            if (stop_phase3==True): # still have more time points for next phase
              mouse_df = mouse_df.reset_index()
              mouse_df.loc[j:k-1, 'Phase'] = 'Relapse' # fill timepoints from end of phase 2 to end of phase 3 as Relapse
              mouse_df = mouse_df.set_index(index_names)

      ## now see if other phases ##
      if (stop_phase3==True):
          # 4. Decay2
          for m in range(k,num_of_timepoints):
              df4model = mouse_df.reset_index('Time').iloc[k-1:m+1]
              df4model = df4model.reset_index().set_index(index_names) # reset the multi-index
              yexp = df4model['Average Radiance'].to_numpy() # experimental values of tumor
              # keep adding points until stops decreasing (stop if starts increasing by 3x)
              if yexp[-1]/yexp[-2] > 3: # stop
                stop_phase4 = True
                num4model = len(yexp)-1
                break

          # add info to df
          if (stop_phase4==False): # used up all remaining timepoints in decay2 region
              mouse_df = mouse_df.reset_index()
              mouse_df.loc[k:, 'Phase'] = 'Decay2' # remaining timepoints are Decay2
              mouse_df = mouse_df.set_index(index_names)
          if (stop_phase4==True): # still have more time points for next phase
              mouse_df = mouse_df.reset_index()
              mouse_df.loc[k:m-1, 'Phase'] = 'Decay2' # fill timepoints from end of phase 3 to end of phase 4 as Decay2

              # 5. Relapse2
              mouse_df.loc[m:, 'Phase'] = 'Relapse2' # remaining timepoints are Relapse2                
              mouse_df = mouse_df.set_index(index_names)
              

      # # change phase label to indicate no CAR mice
      # mouse_df = mouse_df.reset_index()
      # if (mouse_df.CAR_Binding.unique() == 'Mock') or (mouse_df.CAR_Binding.unique() == 'None'):
      #   mouse_df['Phase'] = mouse_df['Phase'].astype(str) + ' (no CAR)'
      # mouse_df = mouse_df.set_index(index_names)

      mice_df_list.append(mouse_df) # keep list of df to concatenate later

  # combine all the data together
  mice_notfit_df = pd.concat(mice_df_list)

  return mice_notfit_df

def get_params(df, phase='Growth'):
  '''
  Function to find the parameters for tumor growth function from dataframe.

  INPUT:
  df -- dataframe that contains the average radiance values over time

  OUTPUT:
  Ti -- initial tumor radiance at time ti
  C0 -- tumor carrying capacity
  t -- array (series) of time points when have experimental data
  ti -- first time point when have experimental data
  '''

  # get mean of tumor average radiance values at each time point
  mean_radiance = df.mean(level='Time').sort_index()['Average Radiance'].to_numpy()

  # initial tumor radiance
  Ti = mean_radiance[0]

  # carrying capacity is max value of tumor radiance
  # C0 = np.max(mean_radiance) # max value is carrying capacity

  # if last two values within factor of 10 of eachother -- at carrying capacity
  if np.abs(np.log10(mean_radiance[-1]) - np.log10(mean_radiance[-2])) < 1:
    C0 = (mean_radiance[-1] + mean_radiance[-2]) / 2 # use average of last two vals as C0
  elif np.max(mean_radiance) > 1e7:
    C0 = np.max(mean_radiance) # use max if greater than 1e7
  else:
    C0 = 1e7 # hardcoded based on values for all mice -- all are about 1e7 with little variability

  # if C0<1e7: # not yet at carrying capacity
  #   C0= 1e7 

  # time where have data
  t = df.reset_index().Time

  # first time point for initial condition
  ti = np.sort(df.reset_index().Time.unique())[0]

  if (phase=='Growth'):
    return Ti, C0, t, ti
  if (phase=='Decay'):
    R = mean_radiance[-1]
    return Ti, R, t, ti


def tumor_growth(params, t):
  '''
  Model logistic growth.

  params - Parameters dictionary
  t      - array of timepoints to model
  '''
  growth = params['growth']
  carrying_capacity = params['carrying_capacity']
  initial_time = params['initial_time']
  initial_tumor = params['initial_tumor']

  A = (carrying_capacity-initial_tumor)/initial_tumor * np.exp(growth*initial_time)

  model = carrying_capacity / (1 + A*np.exp(-growth*t))
  
  return model


def objective_growth(params, t, data, alpha, phase):
  '''
  Objective function to be minimized.
  (This is used for growth and relapse phases)
  '''
  # Bayesian Priors
  if phase=='growth':
    k_bayes = 0.83
    sigma = 0.42
  if phase=='relapse':
    k_bayes = 0.60
    sigma = 0.32
  return np.sum((np.log10(data) - np.log10(tumor_growth(params, t)))**2) + alpha*np.sum(((params['growth']-k_bayes)**2)/(sigma**2))


def tumor_killing(params, t):
  '''
  Model exponential decay.

  params - Parameters dictionary
  t      - array of timepoints to model
  '''
  decay = params['decay']
  baseline = params['baseline']
  initial_time = params['initial_time']
  initial_tumor = params['initial_tumor']

  model = initial_tumor * np.exp(-decay*(t-initial_time)) + baseline
  
  return model


def objective_killing(params, t, data, alpha):
  '''
  Objective function to be minimized.
  '''
  # Bayesian Priors
  k_bayes = 4.15
  sigma = 5.43
  return np.sum((np.log10(data) - np.log10(tumor_killing(params, t)))**2) + alpha*np.sum(((params['decay']-k_bayes)**2)/(sigma**2))


def fit_data(data,alphas=[0.01,0,0,0,0]):
  '''
  Function to fit all phases of tumor behavior.

  Inputs:
  data -- dataframe with radiance data to fit (has info for each phase from "part1" code)
  alphas -- (array) weights for Bayesian prior penalty for each phase [growth, decay, relapse, decay2, relapse2]

  Output:
  mice_fit_df -- dataframe with raw data with rate and phase info
  '''
  index_names = list(data.reset_index().columns)[:-4]
  mice = data.reset_index().MouseID.unique()
  mice_df_list = []
  for mouse in tqdm(mice):
    # df for one mouse at a time
    mouse_df = data.query('MouseID == @mouse')

    # set up default values to be overwritten after fitting
    rate_list=[np.nan,np.nan,np.nan,np.nan,np.nan]
    start_list=[-np.inf,np.inf,np.inf,np.inf,np.inf] # dummy vals for start of each phase
    mouse_df['Rate'] = np.nan
    mouse_df['R2'] = np.nan
    mouse_df = mouse_df.reset_index()

    # filter by phase
    phase1_df = mouse_df[mouse_df['Phase']=='Growth']
    phase2_df = mouse_df[mouse_df['Phase']=='Decay']
    phase3_df = mouse_df[mouse_df['Phase']=='Relapse']
    phase4_df = mouse_df[mouse_df['Phase']=='Decay2']
    phase5_df = mouse_df[mouse_df['Phase']=='Relapse2']

    # 1. fit initial growth phase
    if len(phase1_df) >= 2:
      # create set of Parameters
      Ti, C0, t, ti = get_params(phase1_df.reset_index().set_index(index_names), phase='Growth')
      params = Parameters()
      params.add('growth', value=1, min=0.0, vary=True) # , max=2.0
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = phase1_df['Average Radiance'].to_numpy() 

      # fit the data
      fit = minimize(objective_growth, params, args=(t, yexp, alphas[0], 'growth'))

      # get optimal parameter values
      tg = fit.params['growth'].value
      phase1_df['Rate'] = tg
      rate_list[0] = tg
      start_list[0] = list(t)[0] # start of growth phase
      params.add('growth', value=tg, vary=False)

      # predicted values for each timepoint
      ypred = tumor_growth(params, t).values
      phase1_df['Average Radiance Prediction'] = ypred

      # calculate r-squared
      if np.isnan(ypred).any(): print(f'{mouse} ADAM\n{ypred}\n{params}\n{t}')
      r2 = r2_score(np.log10(yexp),np.log10(tumor_growth(params, t).values))
      phase1_df['R2'] = np.round(r2,3)


    # 2. fit decay phase
    if len(phase2_df) >= 1:
      if len(phase1_df) > 0: # need to use last value of the first phase as input into the model for decay phase
        dummy_df = mouse_df.iloc[list(phase1_df.index)[-1]:list(phase2_df.index)[-1]+1]
        extra=True
      else: # if don't have values for first phase then it initially starts decreasing
        dummy_df = phase2_df
        extra=False

      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)
      Ti, R, t, ti = get_params(dummy_df, phase='Decay')
      params = Parameters()
      params.add('decay', value=1, min=0, vary=True)
      params.add('baseline', value=R, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy() 

      # fit the data
      fit = minimize(objective_killing, params, args=(t, yexp, alphas[1]))

      # get optimal parameter values
      tk = fit.params['decay'].value
      phase2_df['Rate'] = tk
      rate_list[1] = tk
      start_list[1] = list(t)[0] # start of decay phase
      params.add('decay', value=tk, vary=False)

      # predicted values for each timepoint
      if extra:
        ypred = tumor_killing(params, t[1:]).values
        phase2_df['Average Radiance Prediction'] = ypred
      else:
        ypred = tumor_killing(params, t).values
        phase2_df['Average Radiance Prediction'] = ypred

      # calculate r-squared
      if np.isnan(ypred).any(): print(f'{mouse} ADAM\n{ypred}\n{params}\n{t}')
      try:
        r2 = r2_score(np.log10(yexp),np.log10(tumor_killing(params, t).values))
      except:
        print(yexp)
        print(ypred)
        print(tumor_killing(params, t).values)
      phase2_df['R2'] = np.round(r2,3)


    # 3. fit relapse phase
    if len(phase3_df) >= 1:
      dummy_df = mouse_df.iloc[list(phase2_df.index)[-1]:list(phase3_df.index)[-1]+1] # need to use last value of the second phase as input into the model for relapse phase
      # dummy_df = remove_outliers(dummy_df) # remove outliers
      # phase3_df = remove_outliers(phase3_df)
      # create set of Parameters
      Ti, C0, t, ti = get_params(dummy_df.reset_index().set_index(index_names), phase='Growth')
      params = Parameters()
      params.add('growth', value=1, min=0.0, max=5.0, vary=True)
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy()
      
      # fit the data
      fit = minimize(objective_growth, params, args=(t, yexp, alphas[2], 'relapse'))

      # get optimal parameter values
      tg = fit.params['growth'].value
      phase3_df['Rate'] = tg
      rate_list[2] = tg
      start_list[2] = list(t)[0] # start of relapse phase
      params.add('growth', value=tg, vary=False)

      # predicted values for each timepoint
      ypred = tumor_growth(params, t[1:]).values
      phase3_df['Average Radiance Prediction'] = ypred

      # calculate r-squared
      if np.isnan(ypred).any(): print(f'{mouse} ADAM\n{ypred}\n{params}\n{t}')
      r2 = r2_score(np.log10(yexp),np.log10(tumor_growth(params, t).values))
      phase3_df['R2'] = np.round(r2,3)


    # 4. fit second decay phase
    if len(phase4_df) >= 1:
      # need to use last value of the first phase as input into the model for decay phase
      dummy_df = mouse_df.iloc[list(phase3_df.index)[-1]:list(phase4_df.index)[-1]+1]

      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)
      Ti, R, t, ti = get_params(dummy_df, phase='Decay')
      params = Parameters()
      params.add('decay', value=1, min=0, vary=True)
      params.add('baseline', value=R, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy() 

      # fit the data
      fit = minimize(objective_killing, params, args=(t, yexp, alphas[3]))

      # get optimal parameter values
      tk = fit.params['decay'].value
      phase4_df['Rate'] = tk
      rate_list[3] = tk
      start_list[3] = list(t)[0] # start of decay phase
      params.add('decay', value=tk, vary=False)

      # predicted values for each timepoint
      ypred = tumor_killing(params, t[1:]).values
      phase4_df['Average Radiance Prediction'] = ypred

      # calculate r-squared
      if np.isnan(ypred).any(): print(f'{mouse} ADAM\n{ypred}\n{params}\n{t}')
      r2 = r2_score(np.log10(yexp),np.log10(tumor_killing(params, t).values))
      phase4_df['R2'] = np.round(r2,3)

        
    # 5. fit second relapse phase
    if len(phase5_df) >= 1:
      dummy_df = mouse_df.iloc[list(phase4_df.index)[-1]:list(phase5_df.index)[-1]+1] # need to use last value of the second phase as input into the model for relapse phase
      # create set of Parameters
      Ti, C0, t, ti = get_params(dummy_df.reset_index().set_index(index_names), phase='Growth')
      params = Parameters()
      params.add('growth', value=1, min=0.0, max=5.0, vary=True)
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy()
      
      # fit the data
      fit = minimize(objective_growth, params, args=(t, yexp, alphas[4], 'relapse'))

      # get optimal parameter values
      tg = fit.params['growth'].value
      phase5_df['Rate'] = tg
      rate_list[4] = tg
      start_list[4] = list(t)[0] # start of relapse phase
      params.add('growth', value=tg, vary=False)

      # predicted values for each timepoint
      ypred = tumor_growth(params, t[1:]).values
      phase5_df['Average Radiance Prediction'] = ypred

      # calculate r-squared
      if np.isnan(ypred).any(): print(f'{mouse} ADAM\n{ypred}\n{params}\n{t}')
      r2 = r2_score(np.log10(yexp),np.log10(tumor_growth(params, t).values))
      phase5_df['R2'] = np.round(r2,3)        
        
    # combine all three phases together
    mouse_df = pd.concat([phase1_df, phase2_df, phase3_df, phase4_df, phase5_df])
    mouse_df['GrowthRate'] = rate_list[0]
    mouse_df['DecayRate'] = rate_list[1]
    mouse_df['RelapseRate'] = rate_list[2]
    mouse_df['Decay2Rate'] = rate_list[3]
    mouse_df['Relapse2Rate'] = rate_list[4]
    mouse_df['StartGrowth'] = start_list[0]
    mouse_df['StartDecay'] = start_list[1]
    mouse_df['StartRelapse'] = start_list[2]
    mouse_df['StartDecay2'] = start_list[3]
    mouse_df['StartRelapse2'] = start_list[4]
    mice_df_list.append(mouse_df) # keep list of df to concatenate later

  # combine all the data together
  mice_fit_df = pd.concat(mice_df_list)
  index_names2 = ['Date','ExperimentName','Researcher','CAR_Binding','CAR_Costimulatory','Tumor','TumorCellNumber','TCellNumber','bloodDonorID','Perturbation','GrowthRate','DecayRate','RelapseRate','Decay2Rate','Relapse2Rate','StartGrowth','StartDecay','StartRelapse','StartDecay2','StartRelapse2','Phase','Rate','R2','Day','Time','Sample','MouseID','Group','ImageID']
  mice_fit_df = mice_fit_df.set_index(index_names2)

  return mice_fit_df



def get_rates(mice_fit_df,include_outliers=False):
  '''
  Get rates for each phase to use in Bayesian Priors analyses.
  Input: dataframe with data for each mice
  Output: list of rates for growth, decay, relapse phases
  '''
  print('Extracting rates.')
  growth_rates = []
  decay_rates = []
  relapse_rates = []
  for mouse in tqdm(mice_fit_df.reset_index().MouseID.unique()):
    growth_rate = mice_fit_df.query('MouseID == @mouse').reset_index().GrowthRate.unique()
    decay_rate = mice_fit_df.query('MouseID == @mouse').reset_index().DecayRate.unique()
    relapse_rate = mice_fit_df.query('MouseID == @mouse').reset_index().RelapseRate.unique()

    # growth
    if include_outliers == False:
      if (~np.isnan(growth_rate)) and (growth_rate < 3): # don't use outliers to find mean/std of population
        growth_rates.append(growth_rate[0])
    else:
      if (~np.isnan(growth_rate)):
        growth_rates.append(growth_rate[0])

    # decay
    if include_outliers == False:
      if (~np.isnan(decay_rate)) and (decay_rate < 50): # don't use outliers to find mean/std of population
        decay_rates.append(decay_rate[0])
    else:
      if (~np.isnan(decay_rate)):
        decay_rates.append(decay_rate[0])


    # relapse
    if include_outliers == False:
      if (~np.isnan(relapse_rate)) and (relapse_rate < 3): # don't use outliers to find mean/std of population
        relapse_rates.append(relapse_rate[0])
    else:
      if (~np.isnan(relapse_rate)):
        relapse_rates.append(relapse_rate[0])
 
  return growth_rates, decay_rates, relapse_rates


def add_rates_to_df_old(data):
  '''
  Function to add information on rates (p, p-d, p-d', p', d, d', d2, p2') to dataframe.
  '''
  exps = data.reset_index().ExperimentName.unique()
  mice_df_list = []
  for i,exp in enumerate(exps):
    exp_df = data.query('ExperimentName == @exp')
    p = np.mean(exp_df.query('CAR_Binding == "Mock" or CAR_Binding == "None"').reset_index().GrowthRate.unique()) # p --> growth rate without CAR

    if ('Mock' in exp_df.reset_index().CAR_Binding.unique()) or ('None' in exp_df.reset_index().CAR_Binding.unique()):
      for mouse in exp_df.reset_index().MouseID.unique():
        mouse_df = exp_df.query('MouseID == @mouse')
        p_d = mouse_df.reset_index().GrowthRate.unique()[0] # p minus d --> growth rate with CAR
        d = p - p_d # d --> inital CAR killing rate
        p_dprime = mouse_df.reset_index().DecayRate.unique()[0] # p minus d' --> decay rate
        dprime = p - p_dprime # dprime --> second CAR killing rate
        pprime = mouse_df.reset_index().RelapseRate.unique()[0] # p prime --> relapse rate
        d2 = mouse_df.reset_index().Decay2Rate.unique()[0] # decady rate during second decay phase (after relapse)
        pprime2 = mouse_df.reset_index().Relapse2Rate.unique()[0] # relapse rate during second relapse phase (after second decay)

        mouse_df['p'] = p
        mouse_df['p-d'] = p_d
        mouse_df['p-d\''] = p_dprime
        mouse_df['p\''] = pprime
        mouse_df['d'] = d
        mouse_df['d\''] = dprime
        mouse_df['d2'] = d2
        mouse_df['p2\''] = pprime2
        mice_df_list.append(mouse_df)

    else:
      for mouse in exp_df.reset_index().MouseID.unique():
        mouse_df = exp_df.query('MouseID == @mouse')
        p_d = mouse_df.reset_index().GrowthRate.unique()[0] # p minus d --> growth rate with CAR
        # d = p - p_d # d --> inital CAR killing rate
        p_dprime = mouse_df.reset_index().DecayRate.unique()[0] # p minus d' --> decay rate
        # dprime = p - p_dprime # dprime --> second CAR killing rate
        pprime = mouse_df.reset_index().RelapseRate.unique()[0] # p prime --> relapse rate
        d2 = mouse_df.reset_index().Decay2Rate.unique()[0] # decay rate during second decay phase (after relapse)
        pprime2 = mouse_df.reset_index().Relapse2Rate.unique()[0] # relapse rate during second relapse phase (after second decay)

        mouse_df['p'] = np.nan#-9999
        mouse_df['p-d'] = p_d
        mouse_df['p-d\''] = p_dprime
        mouse_df['p\''] = pprime
        mouse_df['d'] = np.nan#-9999
        mouse_df['d\''] = np.nan#-9999
        mouse_df['d2'] = d2
        mouse_df['p2\''] = pprime2

        mice_df_list.append(mouse_df)

  df_all_rates = pd.concat(mice_df_list)
  index_names2 = ['Date','ExperimentName','Researcher','CAR_Binding','CAR_Costimulatory','Tumor','TumorCellNumber','TCellNumber','bloodDonorID','Perturbation','GrowthRate','DecayRate','RelapseRate','Decay2Rate','Relapse2Rate','p','p-d','p-d\'','p\'','d','d\'','d2','p2\'','StartGrowth','StartDecay','StartRelapse','StartDecay2','StartRelapse2','Phase','Rate','R2','Day','Time','Sample','MouseID']
  df_all_rates = df_all_rates.reset_index().set_index(index_names2)

  # add category label to all data
  df_all_rates['Categories'] = df_all_rates.reset_index().groupby('MouseID')['Phase'].transform(lambda x: ' + '.join(x.unique())).values

  ## correct categories for cases when have data before CAR-T infusion ##
  # correcting "Growth + Decay"
  mice = df_all_rates.query('Time<0 and StartDecay==0 and Categories=="Growth + Decay"').reset_index().MouseID.unique()
  condition = list(df_all_rates.reset_index()['MouseID'].isin(mice))
  df_all_rates.loc[condition, ['Categories']] = 'Decay'

  # correcting "Growth + Decay + Relapse"
  mice = df_all_rates.query('Time<0 and StartDecay==0 and Categories=="Growth + Decay + Relapse"').reset_index().MouseID.unique()
  condition = list(df_all_rates.reset_index()['MouseID'].isin(mice))
  df_all_rates.loc[condition, ['Categories']] = 'Decay + Relapse'

  return df_all_rates

def find_carrying_capacity(radiance_vals):
    '''
    Function to find the carrying capacity from the radiance values for one mouse.
    '''
    if (len(radiance_vals)>1) and (np.abs(np.log10(radiance_vals[-1]) - np.log10(radiance_vals[-2])) < 1):
        C0 = (radiance_vals[-1] + radiance_vals[-2]) / 2 # use average of last two vals as C0
    elif np.max(radiance_vals) > 1e7:
        C0 = np.max(radiance_vals) # use max if greater than 1e7
    else:
        C0 = 1e7 # hardcoded based on values for all mice -- all are about 1e7 with little variability
    
    return C0

def find_tumor_baseline(radiance_vals):
    '''
    Function to find the tumor baseline value from the radiance values for one mouse.
    '''
    # if (len(radiance_vals)>1) and (np.abs(np.log10(radiance_vals[-1]) - np.log10(radiance_vals[-2])) < 1):
    #     T_baseline = (radiance_vals[-1] + radiance_vals[-2]) / 2 # use average of last two vals as baseline
    # else:
    T_baseline = radiance_vals[-1] # use value at last point
    
    return T_baseline


def add_rates_to_df(df):
  '''
  Function to add the carrying capacity, tumor baseline, and initial tumor to the dataframe.
  '''

  df_list = []
  for mouse in tqdm(df.reset_index().MouseID.unique()):
      mouse_df = df.query('MouseID==@mouse')
      phases = mouse_df.reset_index().Phase.unique()
      
      # get carrying capacity from growth phase
      if 'Growth' in phases:
          radiance_vals = mouse_df.query('Phase=="Growth"')['Average Radiance'].values
          C0 = find_carrying_capacity(radiance_vals)
          mouse_df['Tinf_Growth'] = C0
      
      # get carrying capacity from relapse phase
      if 'Relapse' in phases:
          radiance_vals = mouse_df.query('Phase=="Relapse"')['Average Radiance'].values
          C0 = find_carrying_capacity(radiance_vals)
          mouse_df['Tinf_Relapse'] = C0
          
      # get tumor baseline from decay phase
      if 'Decay' in phases:
          radiance_vals = mouse_df.query('Phase=="Decay"')['Average Radiance'].values
          T_baseline = find_tumor_baseline(radiance_vals)
          mouse_df['Tb'] = T_baseline
      
      # add initial tumor
      mouse_df['T0'] = mouse_df['Average Radiance'].values[0]
      
      # add category label to all data
      mouse_df['Categories'] = ' + '.join([item for item in phases if 'Decay2' not in item and 'Relapse2' not in item])

      # merge
      df_list.append(mouse_df)

  new_df_without_p = pd.concat(df_list)

  # now need to add back the initial growth of the tumor for the no CAR condition
  exps = new_df_without_p.reset_index().ExperimentName.unique()
  exp_df_list = []
  for i,exp in enumerate(exps):
    exp_df = new_df_without_p.query('ExperimentName == @exp')
    p = np.mean(exp_df.query('CAR_Binding == "Mock" or CAR_Binding == "None"').reset_index().GrowthRate.unique()) # p --> growth rate without CAR
    exp_df['ControlGrowthRate'] = p
    exp_df_list.append(exp_df)

  new_df = pd.concat(exp_df_list)
  
  return new_df


def generate_final_params_df(df):
  '''
  Returns df with one row per mouse. Columns are the 9 parameters for the model. (Two carrying capacities: one for growth and one for relapse)
  '''
  new_df = add_rates_to_df(df)
  final_params_df = new_df.reset_index().drop(['Day','Time','Phase','Rate','R2','Sample','ImageID','Average Radiance','Total Radiance','Average Radiance Prediction','StartGrowth','StartDecay2','StartRelapse2','Decay2Rate','Relapse2Rate'],axis=1).drop_duplicates()
  final_params_df = final_params_df.set_index(['Date','ExperimentName','Researcher','CAR_Binding','CAR_Costimulatory','Tumor','TumorCellNumber','TCellNumber','bloodDonorID','Perturbation','Categories','Group','MouseID']).sort_values(by='MouseID')
  return final_params_df

def count_mice_by_phase_type(data):
  '''
  Count number of mice in five different tumor behavor groups.

  cat1 -- mice with phases 1. Growth
  cat2 -- mice with phases 2. Growth + Decay
  cat3 -- mice with phases 3. Growth + Decay + Relapse
  cat4 -- mice with phases 6. Decay
  cat5 -- mice with phases 7. Decay + Relapse

  Output:
  cat_df -- dataframe with number of mice in each phase
  cat_dict -- dictionary with MouseIDs in each phase
  '''

  cat1 = data.query('Categories=="Growth"').reset_index().MouseID.unique()
  cat2 = data.query('Categories=="Growth + Decay"').reset_index().MouseID.unique()
  cat3 = data.query('Categories=="Growth + Decay + Relapse"').reset_index().MouseID.unique()
  cat4 = data.query('Categories=="Decay"').reset_index().MouseID.unique()
  cat5 = data.query('Categories=="Decay + Relapse"').reset_index().MouseID.unique()
 

  # set up df
  cat_df = pd.DataFrame(data=[len(cat1),len(cat2),len(cat3),len(cat4),len(cat5)],
                        index=['Growth',
                               'Growth + Decay',
                               'Growth + Decay + Relapse',
                               'Decay',
                               'Decay + Relapse'],
                        columns=['Number'])
  
  cat_df['Percent'] = np.round((cat_df['Number']/len(data.reset_index().MouseID.unique()))*100,2)


  # set up dictionary
  cat_dict = {'Growth':cat1,
             'Growth + Decay':cat2,
             'Growth + Decay + Relapse':cat3,
             'Decay':cat4,
             'Decay + Relapse':cat5}


  return cat_df, cat_dict
