import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

def make_bayesian_plots(data, growth_rates, decay_rates, relapse_rates, plot_dir, bayesian_key):
  '''
  Generate plots that show distributions of growth, decay, and relapse rates BEFORE Bayesian Priors correction applied.

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
  plt.text(1.4,0.2,f'N={len(data.reset_index().MouseID.unique())} mice')
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
  plt.text(3.2,-0.17,'outliers to correct with Bayesian Priors',fontsize='x-small')
  plt.text(50,0.47,f'N={len(growth_rates)} mice',fontsize='small')
  g.set_xscale('log')
  g.set_xlim([0.04,300])
  g.set_xlabel('Growth Rate')
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
  plt.text(3,4.32,'outliers to correct with Bayesian Priors',fontsize='x-small')
  plt.text(50,1,f'N={counter} mice',fontsize='small')
  plt.xscale('log')
  plt.xlim([0.04,300])
  plt.xlabel('Growth Rate')
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
  axs[0].text(0.9,1,f'N={counter_outliers} mice',fontsize='small')
  axs[1].text(0.9,1,f'N={counter_normal} mice',fontsize='small')
  plt.savefig(f'{plot_dir}/growth_rsquared_vs_num_data_{bayesian_key}_Bayesian.pdf',format='pdf',bbox_inches='tight')

  ## DECAY PHASE PLOTS ##
  print('Generating Plots for Decay Phase')
  # Swarm #
  fig = plt.figure(figsize=(8, 6))
  g = sns.stripplot(x=decay_rates,label='Decay',color='blue',legend=False)
  bluerect=mpatches.Rectangle((50,-0.15),200,0.3, fill=False,color="blue",linestyle='--',linewidth=2)
  plt.gca().add_patch(bluerect)
  plt.text(50,-0.17,'outliers to remove',fontsize='x-small')
  plt.text(50,0.47,f'N={len(decay_rates)} mice',fontsize='small')
  g.set_xscale('log')
  g.set_xlabel('Decay Rate')
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
  plt.text(50,6.8,'outliers to remove',fontsize='x-small')
  plt.yticks([2,4,6,8,10,12,14,16,18])
  plt.text(0.08,0.75,f'N={counter} mice',fontsize='small')
  plt.xscale('log')
  plt.xlim([0.04,300])
  plt.xlabel('Decay Rate')
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
  plt.text(3.2,-0.17,'outliers to correct with Bayesian Priors',fontsize='x-small')
  plt.text(50,0.47,f'N={len(relapse_rates)} mice',fontsize='small')
  g.set_xscale('log')
  g.set_xlabel('Relapse Rate')
  g.set_title(f'{bayesian_key} Bayesian Priors - Relapse Rate')
  g.set_xlim([0.04,250])
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
  plt.text(3,12,'outliers to correct with Bayesian Priors',fontsize='x-small')
  plt.text(20,0.7,f'N={counter} mice',fontsize='small')
  plt.yticks(np.arange(2,26,2))
  plt.xscale('log')
  plt.xlim([2e-3,300])
  plt.xlabel('Relapse Rate')
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
  plt.xlim([80,300])
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

