import os
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm
from functools import partialmethod
from lmfit import Parameters

from radianceQuantifier.dataprocessing.modelingFunctions import tumor_growth, tumor_killing, get_params
from radianceQuantifier.dataprocessing.miscFunctions import render_mpl_table, loadPickle, add_category_labels

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

def make_bayesian_plots(data, growth_rates, decay_rates, relapse_rates, plot_dir, bayesian_key):
  '''
  Generate plots that show distributions of growth, decay, and relapse rates before/after Bayesian Priors correction applied.

  Input:
  data -- dataframe after fitting that contains rate information (before/after Bayesian Priors performed)
  growth_rates -- list of growth rates as returned by get_rates()
  decay_rates -- list of decay rates as returned by get_rates()
  relapse_rate -- list of relapse rates as returned by get_rates()
  plot_dir -- full directory to folder where plots will be saved
  bayesian_key -- ("Before" or "After") -- flag to signal if data before or after Bayesian Priors correction

  Output: saves plots to plot_dir
  '''
  print('Generating Figures')
  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  # KDE Plot
  fig = plt.figure(figsize=(8, 6))
  g = sns.kdeplot(np.log10(growth_rates),label='Growth',cut=0,color='green')
  g = sns.kdeplot(np.log10(decay_rates),label='Decay',cut=0,color='blue')
  g = sns.kdeplot(np.log10(relapse_rates),label='Relapse',cut=0,color='red')
  plt.text(1.4,0.2,f'N={len(data.reset_index().MouseID.unique())} mice',fontsize='x-small')
  plt.xlim([-3,3])
  plt.xlabel('$log_{10}$(Rate)')
  plt.title(f'All Data - {bayesian_key} Bayesian Priors')
  plt.legend(loc='upper left')
  plt.savefig(f'{plot_dir}/KDE_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')
  
  ## GROWTH PHASE PLOTS ##
  print('Generating Plots for Growth Phase')
  # Swarm #
  fig = plt.figure(figsize=(8, 6))
  g = sns.stripplot(x=growth_rates,label='Growth',color='green',legend=False)
  greenrect=mpatches.Rectangle((3,-0.15),200,0.3, fill=False,color="green",linestyle='--',linewidth=2)
  plt.gca().add_patch(greenrect)
  plt.text(3.2,-0.17,'outliers to correct with Bayesian Priors',fontsize='xx-small')
  plt.text(50,0.47,f'N={len(growth_rates)} mice',fontsize='x-small')
  g.set_xscale('log')
  g.set_xlim([0.04,400])
  g.set_xlabel('Tumor Growth Rate ($k_{G}$)',color='green')
  g.set_title(f'{bayesian_key} Bayesian Priors - Growth Rate')
  plt.savefig(f'{plot_dir}/growth_swarm_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  # Rate vs Number of Data Points #
  fig = plt.figure(figsize=(8, 6))
  counter=0
  for mouse in data.reset_index().MouseID.unique():
    mouse_df = data.query('MouseID == @mouse').reset_index()
    phase1_df = mouse_df[mouse_df['Phase']=='Growth']
    num_datapts = phase1_df.shape[0]
    growth_rate = phase1_df.reset_index().GrowthRate.unique()
    if num_datapts > 0 and growth_rate > 3: 
      counter+=1
      plt.scatter(growth_rate,num_datapts,color='r',s=3)
    elif num_datapts > 0:
      counter+=1
      plt.scatter(growth_rate,num_datapts,color='k',s=3)

  greenrect=mpatches.Rectangle((3,1.8),200,2.4, fill=False,color="green",linestyle='--',linewidth=2)
  plt.gca().add_patch(greenrect)
  plt.text(3,4.32,'outliers to correct with Bayesian Priors',fontsize='xx-small')
  plt.text(50,1,f'N={counter} mice',fontsize='x-small')
  plt.xscale('log')
  plt.xlim([0.04,400])
  plt.xlabel('Tumor Growth Rate ($k_{G}$)',color='green')
  plt.ylabel('Number or Data Points')
  plt.title(f'{bayesian_key} Bayesian Priors - Growth Rate')
  plt.ylim([0.8,11])
  plt.savefig(f'{plot_dir}/growth_rates_vs_num_data_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  # R-Squared vs Number of Data Points #
  fig = plt.figure(figsize=(8, 6))
  fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(16, 6),sharey=False,sharex=False)

  counter_outliers=0
  counter_normal=0
  for mouse in data.reset_index().MouseID.unique():
    mouse_df = data.query('MouseID == @mouse').reset_index()
    phase1_df = mouse_df[mouse_df['Phase']=='Growth']
    num_datapts = phase1_df.shape[0]
    growth_rate = phase1_df.reset_index().GrowthRate.unique()
    r2 = phase1_df.reset_index().R2.unique()
    if num_datapts > 0 and growth_rate > 3:
      counter_outliers+=1
      axs[0].scatter(r2,num_datapts,color='r',s=3,alpha=0.5)
    elif num_datapts > 0:
      counter_normal+=1
      axs[1].scatter(r2,num_datapts,color='k',s=3,alpha=0.5)

  for ax in axs.flatten():
    ax.set_xlabel('R Squared Value')
    ax.set_ylabel('Number of Data Points')
    ax.set_ylim([0.8,11])
    ax.set_xlim([0.4,1.05])
    ax.set_xticks([0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_xticklabels([0.4,0.5,0.6,0.7,0.8,0.9,1])

  axs[0].set_title(f'{bayesian_key} Bayesian Priors - Growth Rate - Outliers',fontsize='small')
  axs[1].set_title(f'{bayesian_key} Bayesian Priors - Growth Rate - Normal',fontsize='small')
  axs[0].text(0.9,1,f'N={counter_outliers} mice',fontsize='x-small')
  axs[1].text(0.9,1,f'N={counter_normal} mice',fontsize='x-small')
  plt.savefig(f'{plot_dir}/growth_rsquared_vs_num_data_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  ## DECAY PHASE PLOTS ##
  print('Generating Plots for Decay Phase')
  # Swarm #
  fig = plt.figure(figsize=(8, 6))
  g = sns.stripplot(x=decay_rates,label='Decay',color='blue',legend=False)
  bluerect=mpatches.Rectangle((50,-0.15),200,0.3, fill=False,color="blue",linestyle='--',linewidth=2)
  plt.gca().add_patch(bluerect)
  plt.text(50,-0.17,'outliers to remove',fontsize='xx-small')
  plt.text(50,0.47,f'N={len(decay_rates)} mice',fontsize='x-small')
  g.set_xscale('log')
  g.set_xlabel('Tumor Decay Rate ($k_{D}$)',color='blue')
  g.set_title(f'{bayesian_key} Bayesian Priors - Decay Rate')
  plt.savefig(f'{plot_dir}/decay_swarm_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  # Rate vs Number of Data Points #
  fig = plt.figure(figsize=(8, 6))
  counter=0
  for mouse in data.reset_index().MouseID.unique():
    mouse_df = data.query('MouseID == @mouse').reset_index()
    phase2_df = mouse_df[ mouse_df['Phase'].str.contains('Decay')]
    num_datapts = phase2_df.shape[0]
    initial_rate = phase2_df.reset_index().GrowthRate.unique()
    decay_rate = phase2_df.reset_index().DecayRate.unique()
    if initial_rate != np.nan: # then tumor was growing before decay so need to add one datapoint
      num_datapts += 1
    if num_datapts > 0 and decay_rate > 50: 
      counter+=1
      plt.scatter(decay_rate,num_datapts,color='r',s=3)
    elif num_datapts > 0:
      counter+=1
      plt.scatter(decay_rate,num_datapts,color='k',s=3)

  bluerect=mpatches.Rectangle((50,1.6),200,4.7, fill=False,color="blue",linestyle='--',linewidth=2)
  plt.gca().add_patch(bluerect)
  plt.text(50,6.8,'outliers to remove',fontsize='xx-small')
  plt.yticks([2,4,6,8,10,12,14,16,18])
  plt.text(0.08,0.75,f'N={counter} mice',fontsize='x-small')
  plt.xscale('log')
  plt.xlim([0.04,400])
  plt.xlabel('Tumor Decay Rate ($k_{D}$)',color='blue')
  plt.ylabel('Number of Data Points')
  plt.title(f'{bayesian_key} Bayesian Priors - Decay Rate')
  plt.ylim([0.5,19])
  plt.savefig(f'{plot_dir}/decay_rates_vs_num_data_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  ## RELAPSE PHASE PLOTS ##
  print('Generating Plots for Relapse Phase')
  # Swarm #
  fig = plt.figure(figsize=(8, 6))
  g = sns.stripplot(x=relapse_rates,label='Relapse',color='red',legend=False)
  redrect=mpatches.Rectangle((3,-0.15),200,0.3, fill=False,color="red",linestyle='--',linewidth=2)
  plt.gca().add_patch(redrect)
  plt.text(3.2,-0.17,'outliers to correct with Bayesian Priors',fontsize='xx-small')
  plt.text(50,0.47,f'N={len(relapse_rates)} mice',fontsize='x-small')
  g.set_xscale('log')
  g.set_xlabel('Tumor Relapse Rate ($k_{R}$)',color='red')
  g.set_title(f'{bayesian_key} Bayesian Priors - Relapse Rate')
  g.set_xlim([0.04,400])
  plt.savefig(f'{plot_dir}/relapse_swarm_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  # Rate vs Number of Data Points #
  fig = plt.figure(figsize=(8, 6))
  counter=0
  for mouse in data.reset_index().MouseID.unique():
    mouse_df = data.query('MouseID == @mouse').reset_index()
    phase3_df = mouse_df[ mouse_df['Phase'].str.contains('Relapse')]
    num_datapts = phase3_df.shape[0] 
    relapse_rate = phase3_df.reset_index().RelapseRate.unique()
    if relapse_rate != np.nan: # need to include last point from decay phase
      num_datapts += 1
    if num_datapts > 0 and relapse_rate > 3: 
      counter+=1
      plt.scatter(relapse_rate,num_datapts,color='r',s=3)
    elif num_datapts > 0:
      counter+=1
      plt.scatter(relapse_rate,num_datapts,color='k',s=3)

  redrect=mpatches.Rectangle((3,1.8),200,9.5, fill=False,color="red",linestyle='--',linewidth=2)
  plt.gca().add_patch(redrect)
  plt.text(3,12,'outliers to correct with Bayesian Priors',fontsize='xx-small')
  plt.text(20,0.7,f'N={counter} mice',fontsize='x-small')
  plt.yticks(np.arange(2,26,2))
  plt.xscale('log')
  plt.xlim([0.04,400])
  plt.xlabel('Tumor Relapse Rate ($k_{R}$)',color='red')
  plt.ylabel('Number of Data Points')
  plt.title(f'{bayesian_key} Bayesian Priors - Relapse Rate')
  plt.ylim([0.5,25])
  plt.savefig(f'{plot_dir}/relapse_rates_vs_num_data_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  print(f'Figures saved in {plot_dir}')

def plot_slanted_image(matrix, i, b, m, plot_dir):
  '''
  Function to plot image with OLS regression line.
  
  Inputs:
  matrix -- raw image data matrix
  i -- image ID
  b -- intercept from OLS regression
  m -- slope from OLS regression
  plot_dir -- full directory to folder where plots will be saved
  
  Output: saves plots to plot_dir
  '''
  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  # convert slope to angle in degrees
  angle = np.abs(np.degrees(np.arctan(m)))
  
  # make plot
  plt.figure()
  plt.imshow(cv2.rotate(matrix[:,:,2],cv2.ROTATE_90_CLOCKWISE),cmap='Greys_r',origin='lower')
  plt.axline(xy1=(0, (matrix[:,:,2].shape[1] - b)), slope=-m) # image was originally flipped
  plt.title(f'ImageID: {i} (Angle = {np.round(angle,2)})')
  plt.savefig(f'{plot_dir}/{os.getcwd().split(dirSep)[-1]}-slanted_image_{i}.pdf',format='pdf',bbox_inches='tight')

def slanted_images_summary_plot(angle_df, thresh, plot_dir):
  '''
  KDE and rug plot of planted images

  Inputs:
  angle_df -- dataframe with angles of slant for each image
  thresh -- angle threshold to classify images as slanted
  plot_dir -- full directory to folder where plots will be saved
  
  Output: saves plots to plot_dir
  '''
  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  plt.figure()
  sns.kdeplot(angle_df['Angle'],cut=0,c='k')
  sns.rugplot(angle_df['Angle'],c='r')
  plt.axvline(x=thresh,linestyle='--',c='grey') # mice with angle > thresh
  plt.xlim([0,20])
  plt.xlabel('Angle of Slant')
  plt.text(2,0.3,s=f'{angle_df.shape[0]-angle_df[angle_df>thresh].count()[0]}\nimages',fontsize='x-small');
  plt.text(10,0.3,s=f'{angle_df[angle_df>thresh].count()[0]}\nimages',fontsize='x-small');
  plt.savefig(f'{plot_dir}/slanted_images_KDE_rug_plot-{os.getcwd().split(dirSep)[-1]}.pdf',format='pdf',bbox_inches='tight')

def plot_image_widths(matrixList, matrix_rescaled_list, plot_dir):
  '''
  Plots KDE and rug plot of widths of images pre and post rescaling heights.

  Inputs:
  matrixList -- list of image matrices before rescalling heights
  matrix_rescaled_list -- list of image matrices after rescaling heights
  plot_dir -- full directory to folder where plots will be saved
  
  Output: saves plots to plot_dir
  ''' 
  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  # kde plot of widths
  plt.figure()
  sns.kdeplot([x.shape[1] for x in matrixList],cut=0,c='k',label='Original')
  sns.kdeplot([x.shape[1] for x in matrix_rescaled_list],cut=0,c='C1',label='Rescaled')
  sns.rugplot([x.shape[1] for x in matrix_rescaled_list],c='r')
  plt.xlim([70,350])
  plt.legend()
  plt.xlabel('Widths')
  plt.savefig(f'{plot_dir}/KDE_widths_pre_post_height_scaling-{os.getcwd().split(dirSep)[-1]}.pdf',format='pdf',bbox_inches='tight')

def plot_image(matrix, idx, type, plot_dir):
  '''
  Plots an individual mouse image.
  
  Inputs:
  matrix -- raw individual image data matrix (3D)
  idx -- image ID
  type -- which level of the matrix image to plot
      == 1 -- radiance
      == 2 -- mousePixel
      == 3 -- brightfield
  
  plot_dir -- full directory to folder where plots will be saved
  
  Output: saves plots to plot_dir
  ''' 
  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  typeDict = {'radiance':0,
              'mousePixel':1,
              'brightfield':2}

  plt.figure()
  plt.imshow(matrix[:,:,typeDict[type]],cmap='Greys_r')
  plt.title(f'{idx}')
  plt.savefig(f'{plot_dir}/{os.getcwd().split(dirSep)[-1]}-mouseImage_{idx}_{type}.pdf',format='pdf',bbox_inches='tight')



def plot_all_data(data,ax,
                  plot_flag=[True,True,True,True,True],
                  exp_plot=False,
                  x_logscale=True,
                  colors=['green','blue','red','skyblue','pink'],
                  exp_points_flag=False,
                  annotate_exppts=False,
                  show_rates=False,
                  linewidth=0.5,
                  alpha=1,
                  legend_flag=False,
                  legend_phases=False,
                  last_plot=True,
                  aes_labels_flag=True
                  ):
  '''
  Function to plot the growth, decay, and relpase phases of tumor behavior for all mice.

  Inputs:
  data -- dataframe with radiance data to fit (has info for each phase from "Part 1" code)
  plot_flag -- (array) True generates one plot for each phase for all mice
  exp_plot -- True makes plot (if plot_flag True) be experimental plot; False, model fit
  x_logscale -- True makes plot (if plot_flag True) have log scale on x-axis (time)
  colors -- colors for each phase
  exp_points_flag -- True adds (if plot_flag True) experimental points to plot
  annotate_exppts -- True will label points with their day
  show_rates -- True will print the rates on the plot
  linewidth -- size of lines
  alpha -- transparency of lines
  legend_flag -- True will add the legend to the plot
  legend_phases -- True will add color of phases to legend
  last_plot -- True will set plot aesthetics (set to False to plot multiple mice at once)
  ax -- plot axis
  aes_labels_flag -- True will add labels for detection thresholds and number of mice (if greater than 1)
  '''
  index_names = list(data.reset_index().columns)[:-6]
  mice = data.reset_index().MouseID.unique()
  mice_df_list = []
  for mouse in tqdm(mice):
    # df for one mouse at a time
    mouse_df = data.query('MouseID == @mouse')
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
      tg = phase1_df.reset_index().Rate.unique()
      params.add('growth', value=tg, vary=False)
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = phase1_df['Average Radiance'].to_numpy()

      # plot
      if plot_flag[0]:
        if exp_plot: g = sns.lineplot(x=t, y=yexp, color=colors[0], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # experimental lines
        else:
          tmod = np.linspace(min(t),max(t),500) # time range for model
          ymod = tumor_growth(params, tmod) # model values
          g = sns.lineplot(x=tmod, y=ymod, color=colors[0], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # model fits
        if exp_points_flag:
            g = sns.scatterplot(x=t, y=yexp, color=colors[0], linewidth=0, zorder=100, ax=ax) # experimental points
            if annotate_exppts:
                for i,time in enumerate(t):
                    ax.annotate(time,(t[i], yexp[i])) # add annotation (note: time and t[i] are the same)
        if show_rates:
            g.text(2.2-3,2e7,f'{np.round(tg[0],2)}',color=colors[0],fontsize='small')

    # 2. fit decay phase
    if len(phase2_df) >= 1:
      if len(phase1_df) > 0: # need to use last value of the first phase as input into the model for decay phase
        # mouse_df = mouse_df.reset_index()
        dummy_df = mouse_df.iloc[list(phase1_df.index)[-1]:list(phase2_df.index)[-1]+1]
        extra=True
      else: # if don't have values for first phase then it initially starts decreasing
        dummy_df = phase2_df
        extra=False

      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)
      Ti, R, t, ti = get_params(dummy_df, phase='Decay')
      params = Parameters()
      tk = phase2_df.reset_index().Rate.unique()
      params.add('decay', value=tk, vary=False)
      params.add('baseline', value=R, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy() 

      # plot
      if plot_flag[1]:
        if exp_plot: g = sns.lineplot(x=t, y=yexp, color=colors[1], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # experimental lines
        else:
          tmod = np.linspace(min(t),max(t),500) # time range for model
          ymod = tumor_killing(params, tmod) # model values
          g = sns.lineplot(x=tmod, y=ymod, color=colors[1], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # model fits
        if exp_points_flag:
            g = sns.scatterplot(x=t, y=yexp, color=colors[1], linewidth=0, zorder=100, ax=ax) # experimental points
            if annotate_exppts:
                for i,time in enumerate(t):
                    ax.annotate(time,(t[i], yexp[i])) # add annotation (note: time and t[i] are the same)
        if show_rates:
            g.text(3.6-3,2e7,f'{np.round(tk[0],2)}',color=colors[1],fontsize='small')

    # 3. fit relapse phase
    if len(phase3_df) >= 1:
      dummy_df = mouse_df.iloc[list(phase2_df.index)[-1]:list(phase3_df.index)[-1]+1] # need to use last value of the second phase as input into the model for relapse phase
      # dummy_df = remove_outliers(dummy_df) # remove outliers
      # phase3_df = remove_outliers(phase3_df)
      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)  
      Ti, C0, t, ti = get_params(dummy_df, phase='Growth')
      params = Parameters()
      tg = phase3_df.reset_index().Rate.unique()
      params.add('growth', value=tg, vary=False)
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy()

      # plot
      if plot_flag[2]:
        if exp_plot: g = sns.lineplot(x=t, y=yexp, color=colors[2], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # experimental lines
        else:
          tmod = np.linspace(min(t),max(t),500) # time range for model
          ymod = tumor_growth(params, tmod) # model values
          g = sns.lineplot(x=tmod, y=ymod, color=colors[2], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # model fits
        if exp_points_flag:
            g = sns.scatterplot(x=t, y=yexp, color=colors[2], linewidth=0, zorder=100, ax=ax) # experimental points
            if annotate_exppts:
                for i,time in enumerate(t):
                    ax.annotate(time,(t[i], yexp[i])) # add annotation (note: time and t[i] are the same)
        if show_rates:
            g.text(5-3,2e7,f'{np.round(tg[0],2)}',color=colors[2],fontsize='small')

    # 4. fit second decay phase
    if len(phase4_df) >= 1:
      # need to use last value of the third phase as input into the model for decay2 phase
      dummy_df = mouse_df.iloc[list(phase3_df.index)[-1]:list(phase4_df.index)[-1]+1]

      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)
      Ti, R, t, ti = get_params(dummy_df, phase='Decay')
      tk = phase4_df.reset_index().Rate.unique()
      params.add('decay', value=tk, vary=False)
      params.add('baseline', value=R, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy() 

       
      # plot
      if plot_flag[3]:
        if exp_plot: g = sns.lineplot(x=t, y=yexp, color=colors[3], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # experimental lines
        else:
          tmod = np.linspace(min(t),max(t),500) # time range for model
          ymod = tumor_killing(params, tmod) # model values
          g = sns.lineplot(x=tmod, y=ymod, color=colors[3], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # model fits
        if exp_points_flag:
            g = sns.scatterplot(x=t, y=yexp, color=colors[3], linewidth=0, zorder=100, ax=ax) # experimental points
            if annotate_exppts:
                for i,time in enumerate(t):
                    ax.annotate(time,(t[i], yexp[i])) # add annotation (note: time and t[i] are the same)
        if show_rates:
            g.text(6.4-3,2e7,f'{np.round(tk[0],2)}',color=colors[3],fontsize='small')     

    
    
    # 5. fit second relapse phase
    if len(phase5_df) >= 1:
      # need to use last value of the fourth phase as input into the model for relapse2 phase
      dummy_df = mouse_df.iloc[list(phase4_df.index)[-1]:list(phase5_df.index)[-1]+1]
      
      # create set of Parameters
      dummy_df = dummy_df.reset_index().set_index(index_names)  
      Ti, C0, t, ti = get_params(dummy_df, phase='Growth')
      params = Parameters()
      tg = phase5_df.reset_index().Rate.unique()
      params.add('growth', value=tg, vary=False)
      params.add('carrying_capacity', value=C0, vary=False)
      params.add('initial_time', value=ti, vary=False)
      params.add('initial_tumor', value=Ti, vary=False)

      # experimental values
      yexp = dummy_df['Average Radiance'].to_numpy()

      # plot
      if plot_flag[4]:
        if exp_plot: g = sns.lineplot(x=t, y=yexp, color=colors[4], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # experimental lines
        else:
          tmod = np.linspace(min(t),max(t),500) # time range for model
          ymod = tumor_growth(params, tmod) # model values
          g = sns.lineplot(x=tmod, y=ymod, color=colors[4], linewidth=linewidth, alpha=alpha, zorder=10, ax=ax) # model fits
        if exp_points_flag:
            g = sns.scatterplot(x=t, y=yexp, color=colors[4], linewidth=0, zorder=100, ax=ax) # experimental points
            if annotate_exppts:
                for i,time in enumerate(t):
                    ax.annotate(time,(t[i], yexp[i])) # add annotation (note: time and t[i] are the same)
        if show_rates:
            g.text(7.8-3,2e7,f'{np.round(tg[0],2)}',color=colors[4],fontsize='small')
    
    
            
  # add marking for lower/upper detection limits
  g.axhspan(1,100,facecolor='#d9dbde',zorder=-1000) # lower limit is 10
  g.axhspan(1e7,1e8,facecolor='#d9dbde',zorder=-1000) # upper limit is 1e7

  if aes_labels_flag:
      if len(mice) > 1:
          g.text(65,1.5,f'N={len(mice)} mice',fontsize=25)
          g.text(30, 7,'Below Detection Threshold',fontsize=25)
          g.text(30, 2.5e7,'Above Detection Threshold',fontsize=25)
      else:
        # g.text(80,1.5,f'N={len(mice)} mouse')
        g.text((np.max(data.reset_index().Time)-np.min(data.reset_index().Time)-8)/2 , 7,'Below Detection Threshold',fontsize=25)
        g.text((np.max(data.reset_index().Time)-np.min(data.reset_index().Time)-8)/2, 2.5e7,'Above Detection Threshold',fontsize=25)

        # g.text(22, 7,'Below Detection Threshold')
        # g.text(22, 2.5e7,'Above Detection Threshold')


  # plot aesthetics
  if last_plot:
    if plot_flag[0] or plot_flag[1] or plot_flag[2]: # or plot_flag[3] or plot_flag[4]
      g.set_yscale('log')
      if x_logscale:
        g.set_xscale('symlog')
        # g.set_xlim(np.min(data.reset_index().Time)-1, np.max(data.reset_index().Time)+1)
        
        g.set_xlim(-1, np.max(data.reset_index().Time)+1)

        # # set x ticks
        # x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 3)
        # g.xaxis.set_major_locator(x_major)
        # g.set_xticks([10,100])
        # g.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        # x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        # g.xaxis.set_minor_locator(x_minor)
        # # g.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # g.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
      else:
        # g.set_xlim(np.min(data.reset_index().Time)-1, np.max(data.reset_index().Time)+1)
        g.set_xlim(-1, np.max(data.reset_index().Time)+1)
      g.set_ylim(1, 1E8)
      g.set_xlabel('Days Since CAR-T Cell Administration',fontsize=30)
      g.set_ylabel('Average Radiance per Pixel\n(p/sec/cm$^2$/sr/pixel)',fontsize=30);

      # set y ticks
      y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 8)
      g.yaxis.set_major_locator(y_major)
      g.set_yticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8],labels=[r'$10^{0}$','',r'$10^{2}$','',r'$10^{4}$','',r'$10^{6}$','',r'$10^{8}$'])
      y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
      g.yaxis.set_minor_locator(y_minor)
      g.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

      ax.tick_params(axis='both', which='both', labelsize=30)
    


      # if exp_plot: g.set_title('Experimental Plots')
      # else: g.set_title('Model Plots')

    # legend
    if legend_flag:
      if exp_points_flag and ~exp_plot: # experimental points and model fit lines
        line1 = Line2D([0], [0], label='Model Fit', color='k', linestyle='-')
        point1 = Line2D([0], [0], marker='o', color='w', label='Experimental Points', markerfacecolor='k', markersize=15)
        if legend_phases:
          patch1 = mpatches.Patch(color=colors[0], label='Growth Phase')
          patch2 = mpatches.Patch(color=colors[1], label='Decay Phase')
          patch3 = mpatches.Patch(color=colors[2], label='Relapse Phase')
          patch4 = mpatches.Patch(color=colors[3], label='Decay2 Phase')
          patch5 = mpatches.Patch(color=colors[4], label='Relapse2 Phase')
          plt.legend(handles=[patch1,patch2,patch3,patch4,patch5,line1,point1],loc='lower left',bbox_to_anchor=(-0.006, -0.015),fontsize=20,facecolor='w');
        else:
          plt.legend(handles=[line1,point1],loc='lower left',bbox_to_anchor=(-0.006, -0.015),fontsize=20,facecolor='w');
      else:
        line1 = Line2D([0], [0], label='Growth Phase', color=colors[0], linestyle='-')
        line2 = Line2D([0], [0], label='Decay Phase', color=colors[1], linestyle='-')
        line3 = Line2D([0], [0], label='Relapse Phase', color=colors[2], linestyle='-')
        line4 = Line2D([0], [0], label='Decay2 Phase', color=colors[3], linestyle='-')
        line5 = Line2D([0], [0], label='Relapse2 Phase', color=colors[4], linestyle='-')
        plt.legend(handles=[line1,line2,line3,line4,line5],loc='lower left',bbox_to_anchor=(-0.006, -0.015),fontsize=20,facecolor='w');

def plotAvgImgOverTime(df_all_rates,labelDf,matrix,
                       dataType='radiance',region='all',mice='all',cmap='turbo',
                       mouse_thresh=100,nrows=14,ncols=5,
                       plotFlag=True,sectionFlag=True,subplot=False,axs=None,fig=None):
    '''
    Generate plots over time of average value of all images on that day.
    
    Inputs:
         df_all_rates -- dataframe with radiance values calculated from images
         labelDf -- dataframe that has corresponding information for each image matrix in "matrix"
         matrix -- 4D matrix of data for plotting            
            dimension 0 -- rows (y-dimension) of image
            dimension 1 -- columns (x-dimension) of image
            dimension 2 -- type of data (radiance, mousePixel, brightfield)
                matrix[:,:,0,:] = radiance values
                matrix[:,:,1,:] = mousePixel values (0 no, 1 yes)
                matrix[:,:,2,:] = brightfield values
            dimension 3 -- image number (each value is a unique mouse image on a given day -- this value corresponds to the "index" column in labelDf)


         dataType -- 'radiance' or 'mousePixel' or 'brightfield' -- which type of data to plot
         region   -- 'all' or 'snout' or 'lungs' or 'liver' or 'abdomen' or 'bmRm' or 'bmLm' -- which region of the mouse to plot
             'bmRm' = bone marrow (mouse right)
             'bmLm' = bone marrow (mouse left)
         mice -- 'all' or list of mouseIDs to generate average plots
         cmap     -- color map
         mouse_thresh -- minimum number of mice on each day (days with fewer than mouse_thresh mice will not be plotted)
         nrows -- number of rows for subplots
         ncols -- number of columns for subplots
         plotFlag -- True will generate plots
         sectionFlag -- True will add boxes on plot showing different regions
         subplot -- if True, need to include axs; when want to plot radiance values on additional plot
         axs -- list of axes for subplots for each day's plot
         fig -- figure for plot
            
    Outputs:
         avgValsDf -- dataframe with average tumor value of data plotted on each day for selected region
         totValsDf -- dataframe with total tumor value of data plotted on each day for selected region
    '''
    # make ImageID a column
    labelDf = labelDf.reset_index('ImageID')

    # slanted and bad images
    imagesIDs2ignore = loadPickle(f'misc/imagesIDs2ignore-{os.getcwd().split(dirSep)[-1]}.pkl')

    # make sure inputs are valid 
    dataList = ['radiance','mousePixel','brightfield']
    regionList = ['all','snout','lungs','liver','abdomen','bmRm','bmLm']
    if dataType not in dataList:
        raise NameError(f"'{dataType}' not valid dataType. Please choose one of the following: 'radiance', 'mousePixel', or 'brightfield'")
    if region not in regionList:
        raise NameError(f"'{region}' not valid region. Please choose one of the following: 'all', 'head', 'lungs', 'liver', 'bmRm', or 'bmLm'")
    
    
    # dictionaries to deal with inputs
    dataTypeDict = {'radiance':0,'mousePixel':1,'brightfield':2}
    regionDict = { 'all':[(  0, -1),( 0,-1)],
                 'snout':[(  0, 60),(20,165)],
                 'lungs':[( 60,110),(20,165)],
                 'liver':[(110,210),(20,165)],
               'abdomen':[(210,240),(20,165)],
                  'bmRm':[(240,320),(20,92)], # bone marrow (mouse right)
                  'bmLm':[(240,320),(93,165)]} # bone marrow (mouse left)
    
    
    
    # filter data by MouseID --> ImageID
    if mice=='all':
        images=labelDf.ImageID.values
        print(images)
    elif isinstance(mice, list):
        labelDf=labelDf.query('MouseID in @mice') #filter mice
        images=labelDf.ImageID.values
    else:
        raise NameError(f"'{mice}' not valid entry for mice. Please choose one of the following: 'all' or a list of mouseIDs")
   
    # crop matrix depending on input parameter region
    matrixRegion = matrix[regionDict[region][0][0]:regionDict[region][0][1],regionDict[region][1][0]:regionDict[region][1][1],:,:]

    # sort days
    days = [f'D{x}' for x in np.sort([int(x[1:]) for x in df_all_rates.query('MouseID in @mice').reset_index().Day.unique()])]
    
    if plotFlag and (subplot==False):
        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(20*(ncols/6), 5*nrows),sharey=True,sharex=False)
        axs = axs.ravel()
    if plotFlag and (subplot==True):
        axs = axs
    
    counter=0
    dayIdxsList=[]
    avgMatrixList=[]
    dayList=[]
    avg_vals = []
    tot_vals = []
    ## TODO: FIX CASES WHEN HAVE SCALE CHANGES ON DIFFERENT DAYS
    try:
        vmin = min(labelDf.query('ImageID in @images').vmin.values) # get vmin
        vmax = max(labelDf.query('ImageID in @images').vmax.values) # and vmax
    except: pass
    # print(vmin, vmax)
    for i,day in tqdm(enumerate(days),total=len(days)):
        dayIdxs = list(labelDf.query("Day == @day").ImageID.values) # imageIDs
        # remove bad images      
        dayIdxs = [idx for idx in dayIdxs if idx not in imagesIDs2ignore]
        if len(dayIdxs)<mouse_thresh:
            counter+=1 # keep track when below threshold for plotting
        else: # only plot if have >= mouse_thresh number of mice
            dayIdxsList.append(dayIdxs) # keep track of imageIDs on days where above threshold
            avgMatrix = np.nanmean(matrixRegion[:,:,dataTypeDict[dataType],dayIdxs],axis=2)
            avgMatrixList.append(avgMatrix)    
            dayList.append(day) # keep track of days used for plotting
            # plot
            if plotFlag:
                if dataType=='radiance':
                    background = np.nanmean(matrixRegion[:,:,dataTypeDict['brightfield'],dayIdxs],axis=2) # use brightfield as backgroud of plot
                    axs[i-counter].imshow(background,cmap='Greys_r') # plot brightfield as background first
                    mouseMask = np.nanmean(matrixRegion[:,:,dataTypeDict['mousePixel'],dayIdxs],axis=2)>0 # true/false for each pixel - true if on mouse (>0 means have tumor at that pixel for at least one mouse)
                    background[mouseMask] = avgMatrix[mouseMask] # replace pixels on mouse with radiance values              
                    background[~mouseMask] = -1 # make background pixels not on mouse black (ignored by colormap)
                    im = axs[i-counter].imshow(background,cmap=cmap,norm=matplotlib.colors.LogNorm(vmin,vmax)) # normalize color only for radiance
                    # plot color bar -- only once
                    if (subplot==True) & (day == dayList[0]):
                        fig.colorbar(im,ax=axs[-1],orientation='vertical',anchor=(1,0.5),shrink=1) # add colorbar for last image only
                    # if (subplot==True):
                    #     fig.colorbar(im,ax=axs[i-counter],orientation='vertical',anchor=(1,0.5),shrink=1) # add colorbar for each image
 

                    if len(dayIdxs) > 1:
                        axs[i-counter].set_title(f'Day {day[1:]} (N={len(dayIdxs)})');
                    else:
                        axs[i-counter].set_title(f'Day {day[1:]}');
                    
                    if sectionFlag:
                        snout = mpatches.Rectangle((20,1),145,59, linewidth=2, edgecolor='C0', facecolor='none')
                        lungs = mpatches.Rectangle((20,60),145,50, linewidth=2, edgecolor='C1', facecolor='none')
                        liver = mpatches.Rectangle((20,110),145,100, linewidth=2, edgecolor='C2', facecolor='none')
                        abdomen = mpatches.Rectangle((20,210),145,30, linewidth=2, edgecolor='C3', facecolor='none')
                        bmL = mpatches.Rectangle((20,240),72,80, linewidth=2, edgecolor='C4', facecolor='none') # bone marrow left
                        bmR = mpatches.Rectangle((93,240),72,80, linewidth=2, edgecolor='C5', facecolor='none') # bone marrow right
                        axs[i-counter].add_patch(snout)
                        axs[i-counter].add_patch(lungs)
                        axs[i-counter].add_patch(liver)
                        axs[i-counter].add_patch(abdomen)
                        axs[i-counter].add_patch(bmL)
                        axs[i-counter].add_patch(bmR)
                        
                else:
                    axs[i-counter].imshow(avgMatrix,cmap=cmap)
                    if subplot==False:  # plot Day as title for mousePixel/brightfield if not already plotting radiance
                        if len(dayIdxs) == 1:
                            axs[i-counter].set_title(f'Day {day[1:]}');
                        else:
                            axs[i-counter].set_title(f'Day {day[1:]} (N={len(dayIdxs)})');                        
                # don't show x/y ticks
                axs[i-counter].set_yticks([])
                axs[i-counter].set_xticks([])
 
    # remove extra blank plots where don't have enough mice
    if plotFlag and (subplot==False):
        p=i-counter+1
        while(p<nrows*ncols):
            try:axs[p].remove()
            except:pass
            p+=1
    
    
    # create dataframe with average value on each day for selected region and selected mice
    for idxs in dayIdxsList:
        # take average of images on that day
        selected_matrices = np.nanmean(matrix[regionDict[region][0][0]:regionDict[region][0][1],regionDict[region][1][0]:regionDict[region][1][1],:,idxs],axis=3)
        selected_radiance = selected_matrices[:,:,0]
        selected_mousePixel = selected_matrices[:,:,1]
        selected_brightfield = selected_matrices[:,:,2]
        
        zeros = np.zeros(selected_radiance.shape) # matrix of zeros -- will add radiance values on mouse ontop of this
        mouseMask = selected_mousePixel == 1 # true/false for each pixel - true if on mouse
        zeros[mouseMask] = selected_radiance[mouseMask] # replace pixels on mouse with radiance values
        
        tot_vals.append(np.nansum([val for val in zeros.flatten() if val != -1])) # total vals in region
        avg_vals.append(np.nanmean([val for val in zeros.flatten() if val != -1])) # average vals in region

    avgValsDf = pd.DataFrame(avg_vals,index=[int(x[1:]) for x in dayList],columns=[f'avg_{dataType}']) #_{region}
    avgValsDf = avgValsDf.rename_axis('Time')
    
    # total tumor radiance for selectred region and selected mice
    totValsDf = pd.DataFrame(tot_vals,index=[int(x[1:]) for x in dayList],columns=[f'tot_{dataType}']) #_{region}
    totValsDf = totValsDf.rename_axis('Time')
    
    return avgValsDf,totValsDf



def plot_individual_summary_sheet(df_all_rates, labelDf, matrix, plot_dir):
  '''
  Generates a summary plot for each mouse.
  Figure 1 -- radiance plots (model fit with experimental points) colored by region
  Figure 2 -- raw radiance images for mouse on each day
  Figure 3 -- radiance plots for each region quantified in GUI
  Figure 4 -- summary table with information on the mouse/experiment conditions/tumor behavior
  '''

  print('\nGenerating summary sheet for each mouse:')

  tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # disable all other tqdm outputs

  # make dir to save figures if it doesn't already exist
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  # remove error file if already exists -- will create new one now
  try: os.remove(f'{plot_dir}/missing_images.txt')
  except OSError: pass
  try: os.remove(f'{plot_dir}/double_colorscale.txt')
  except OSError: pass

  mice = np.sort((df_all_rates.reset_index().MouseID.unique()))
  for mouse in tqdm(mice,disable=False):
      # set up axes for plots
      ncols = df_all_rates.query('MouseID == @mouse').shape[0]
      nrows=5
      fig = plt.figure(figsize=(np.max([15,20*(ncols/6)]), 5*nrows),layout='constrained',linewidth=25, edgecolor='white')
      plt.subplots_adjust(hspace=0.35)  # add whitespace between subplots in y direction

      gs = GridSpec(nrows, ncols, figure=fig)

      axs1 = fig.add_subplot(gs[0, :]) # row 1 axes # create new subplot axis on first row -- radiance values plots
      axs2 = [] # row 2 axes # radiance images
      for col in range(ncols):
          axs2.append(fig.add_subplot(gs[1,col])) # create new subplot axis on second row for each day -- for image data -- radiance
      axs3 = fig.add_subplot(gs[2, :]) # plot for % of tumor by section
      axs4 = fig.add_subplot(gs[3:, 0:int(np.ceil(ncols/2))]) # axes for summary chart
      
      ### axs1 - plot radiance on first row ###
      plot_all_data(data=df_all_rates.query('MouseID == @mouse'),
                        plot_flag=[True,True,True,True,True],exp_plot=False,x_logscale=False,colors=['green','blue','red','skyblue','pink'],
                        exp_points_flag=True,annotate_exppts=True,show_rates=True,linewidth=0.5,legend_flag=False,legend_phases=False,last_plot=True,aes_labels_flag=False,
                        ax=axs1)

      ### axs2 - plot mice radiance images on second row ###
      plotAvgImgOverTime(dataType='radiance',region='all',mice=[mouse],cmap='turbo',mouse_thresh=0,nrows=14,ncols=5,df_all_rates=df_all_rates,labelDf=labelDf,matrix=matrix,
                        plotFlag=True,sectionFlag=False,subplot=True,axs=axs2,fig=fig)
        
      
      ### axs3 - plot tumor by region on last row ###
      # load data on average tumor over time in each region

      region_df_list = []
      regions2choose = [item for item in os.listdir('outputData/ROI Radiance Calculation') if item.startswith('left')]
      for region_long in regions2choose: # loop through all regions and load in the dataframe
        region_short = "_".join(region_long.split("_")[4:]) # get region name
        if region_short != 'all': # don't include full "all" region
          region_df = loadPickle(f'outputData/ROI Radiance Calculation/{region_long}/BayesianPriors/{os.getcwd().split(dirSep)[-1]}_fit2model_all_alphas_0_0_0_0_0_{region_long}.pkl')
          region_df = region_df.query('MouseID == @mouse') # only get info for individual mouse
          region_df['Region'] = region_short
          region_df_list.append(region_df)

      if len(region_df_list) == 1: df_all_regions = region_df.copy()
      else: df_all_regions = pd.concat(region_df_list)
      
      df_all_regions.to_pickle(f'misc/df_all_regions_DEBUG.pkl')

      # tumor region plot
      g=sns.lineplot(data=df_all_regions.reset_index(),x='Time',y='Average Radiance',marker='o',hue='Region',legend=True,ax=axs3);
      g.set_yscale('log')
      g.set_ylim(1,1e8)
      g.set_xlabel('Days Since CAR-T Cell Administration',fontsize=30)
      g.set_ylabel('Average Radiance per Pixel\n(p/sec/cm$^2$/sr/pixel)',fontsize=30);
      # g.set_xticks(np.arange(0,df.index[-1]+5,5), labels=np.arange(0,df.index[-1]+5,5));
      g.set_xlim(-1, np.max(df_all_regions.reset_index().Time)+1)
      sns.move_legend(g, loc='upper right', bbox_to_anchor=(1, -0.35))
      # add marking for lower/upper detection limits
      g.axhspan(1,100,facecolor='#d9dbde',zorder=-1000) # lower limit is 10
      g.axhspan(1e7,1e8,facecolor='#d9dbde',zorder=-1000) # upper limit is 1e7
      # g.text((np.max(df_all_regions.reset_index().Time)-np.min(df.reset_index().Time)-8)/2 , 7,'Below Detection Threshold')
      # g.text((np.max(df_all_regions.reset_index().Time)-np.min(df.reset_index().Time)-8)/2, 2.5e7,'Above Detection Threshold')

      # set y ticks
      y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 8)
      g.yaxis.set_major_locator(y_major)
      g.set_yticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8],labels=[r'$10^{0}$','',r'$10^{2}$','',r'$10^{4}$','',r'$10^{6}$','',r'$10^{8}$'])
      y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
      g.yaxis.set_minor_locator(y_minor)
      g.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
      axs3.tick_params(axis='both', which='both', labelsize=30)

      
      ### axs4 - parameter summary table ###
      df_all_rates = add_category_labels(df_all_rates) # add a "Category" column
      summary_df = df_all_rates.query('MouseID == @mouse').reset_index()[['Date', 'ExperimentName', 'Researcher', 'CAR_Binding','CAR_Costimulatory', 'Tumor', 'TumorCellNumber', 'TCellNumber','bloodDonorID', 'Perturbation', 'GrowthRate', 'DecayRate','RelapseRate', 'Decay2Rate', 'Relapse2Rate', 'StartGrowth', 'StartDecay', 'StartRelapse','StartDecay2', 'StartRelapse2', 'Categories', 'Group', 'Sample', 'MouseID']].drop_duplicates().T.reset_index().rename({'index':'Parameter',0:'Value'},axis=1)

      # clean up names of parameters
      summary_df = summary_df.replace({'ExperimentName':'Experiment Name',
                                      'CAR_Binding':'CAR Binding',
                                      'CAR_Costimulatory':'CAR Costimulatory',
                                      'TumorCellNumber':'Initial Tumor Burden',
                                      'TCellNumber':'CAR Dose',
                                      'bloodDonorID':'Donor ID',
                                      'GrowthRate':'Growth Rate [day$^{-1}$]',
                                      'DecayRate':'Decay Rate [day$^{-1}$]',
                                      'RelapseRate':'Relapse Rate [day$^{-1}$]',
                                      'Decay2Rate':'2nd Decay Rate [day$^{-1}$]',
                                      'Relapse2Rate':'2nd Relapse Rate [day$^{-1}$]',
                                      'StartGrowth':'Start of Growth [day]',
                                      'StartDecay':'Start of Decay [day]',
                                      'StartRelapse':'Start of Relapse [day]',
                                      'StartDecay2':'Start of 2nd Decay [day]',
                                      'StartRelapse2':'Start of 2nd Relapse [day]',
                                      })

      # remove rows with nans and infinity
      summary_df = summary_df.replace([np.inf, -np.inf], np.nan).dropna()
      # round the rates
      rate_mask = summary_df['Parameter'].str.contains('Rate')
      summary_df.loc[rate_mask, 'Value'] = summary_df.loc[rate_mask, 'Value'].apply(lambda x: round(x, 2))

      # make the start days not have a decimal
      start_mask = summary_df['Parameter'].str.contains('Start of')
      summary_df.loc[start_mask, 'Value'] = summary_df.loc[start_mask, 'Value'].apply(lambda x: int(x))

      # make table as plot
      render_mpl_table(summary_df,ax=axs4)
      axs4.axis('off')
    
      
      ## plotting set up and saving ##
      
      # if have different colorbars across days
      vmin_arr = labelDf.query('MouseID==@mouse').vmin.values
      vmax_arr = labelDf.query('MouseID==@mouse').vmax.values
      if (len(np.unique(vmin_arr))>1) or (len(np.unique(vmax_arr))>1):
          # creating/opening a file
          f = open(f'{plot_dir}/double_colorscale.txt', 'a')
          f.write(f'{mouse}: {exp}-{group}-{sample}\ndays: {labelDf.query("MouseID==@mouse").reset_index().Day.values}\nvmin: {vmin_arr}\nvmax: {vmax_arr}\n\n')
          f.close()

      try:
          # plot aesthetics
          mouse_info = labelDf.query('MouseID == @mouse')
          exp = mouse_info.reset_index().ExperimentName.unique()[0]
          group = mouse_info.reset_index().Group.unique()[0]
          sample = mouse_info.reset_index().Sample.unique()[0]
          plt.suptitle(f'{mouse}: {exp}-{group}-{sample}',y=0.9,fontsize=30)
          plt.subplots_adjust(hspace=0.6)
          # save figures
          plt.savefig(f'{plot_dir}/{mouse}: {exp}-{group}-{sample}.pdf',format='pdf',bbox_inches='tight');
          plt.close(); # prevent plot from showing in Jupyter notebook
      except: # if raw mice images not available (i.e. not in lableDf)
          # get info on mouse where got error
          exp = df_all_rates.query('MouseID == @mouse').reset_index().ExperimentName.unique()[0]
          sample = df_all_rates.query('MouseID == @mouse').reset_index().Sample.unique()[0]
          plt.suptitle(f'{mouse}: {exp}-??-{sample}',y=0.9)   
          # save figures
          plt.savefig(f'{plot_dir}/{mouse}: {exp}-??-{sample}.pdf',format='pdf',bbox_inches='tight');
          plt.close(); # prevent plot from showing in Jupyter notebook
      
          # creating/opening a file
          f = open(f'{plot_dir}/missing_images.txt', 'a')
          f.write(f'{mouse}: {exp}-{sample}\n')
          f.close()

  tqdm.__init__ = partialmethod(tqdm.__init__, disable=False) # re-enable all other tqdm outputs
