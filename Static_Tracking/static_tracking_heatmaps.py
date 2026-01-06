#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import os
import h5py
import static_utils
import pathlib
import pandas as pd
import seaborn as sns
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('pwd', '')


# # Constants and Paths

# In[3]:


# Main data paths for each catheter
prepend = '/data' # Change to empty string after zenodo get
parent_path = prepend + '/processed'
if (not os.path.isdir(parent_path)):
    print("Using precomputed outputs")
    parent_path = '../data/preprocessed'

class YLoc:
    def __init__(self, description=None, input_path_suffix=None, export_path_suffix=None):
        self.description = description
        self.in_suffix = input_path_suffix
        self.out_suffix = export_path_suffix
        
y_locations = [YLoc('Y = 45mm','Y0','Y45mm'), YLoc('Y = 20mm','Y1', 'Y20mm'), YLoc('Y = -5mm', 'Y2', 'Y-5mm')]
catheter_labels = ['C222','C231','C306']

main_paths = {} # input coordinate data from each catheter and position
heatmap_paths = {} # exported heatmap plots
xpmt_date_dict = {
    (catheter_labels[0],'Y0'):'17Aug2021',
    (catheter_labels[1],'Y0'):'15Dec2021',
    (catheter_labels[2],'Y0'):'13Dec2021',
    (catheter_labels[0],'Y1'):'8Nov2021',
    (catheter_labels[1],'Y1'):'21Dec2021',
    (catheter_labels[2],'Y1'):'6Jan2022',
    (catheter_labels[0],'Y2'):'9Nov2021',
    (catheter_labels[1],'Y2'):'9Jan2022',
    (catheter_labels[2],'Y2'):'7Jan2022',
}
for yloc in y_locations:
    main_paths[yloc.in_suffix] = []
    for cath in catheter_labels:
        xpmt_date = xpmt_date_dict[(cath,yloc.in_suffix)]
        main_paths[yloc.in_suffix].append(f'{parent_path}/static/trackTest-{xpmt_date}-{cath}-{yloc.in_suffix}/')
    heatmap_paths[yloc.in_suffix] = f'../reports/figures/static/{yloc.out_suffix}/'

# Tip error data is also exported to HDF5 files
error_path = '../reports/export/static/'

Gt_filename = '1GroundTruthCoords.csv'

geometry_index = 1

max_samples = 170


# In[4]:


def get_catheter_from_path(path):
    for c in catheter_labels:
        if c in path:
            return c
    return None


# # Error Heatmaps
# 
# For each sequence and localization algorithm of interest, plot the heatmap of average tip errors at each position. Each position has recordings from three catheters.

# In[5]:


sequences = ['SRI_Original', 'FH512_noDither_gradSpoiled']
algorithms = ['centroid_around_peak', 'jpng']
aggregate_data = []
aggregate_plot = []

os.makedirs(error_path, exist_ok=True)

for yloc in y_locations:
    for seq in sequences:
        for alg in algorithms:
            main_path = main_paths[yloc.in_suffix]
            path_dct = static_utils.get_catheter_data(main_path, seq, alg, Gt_filename, geometry_index, max_samples)
            # Aggregate
            for path in main_path: # path per catheter
                cath_label = get_catheter_from_path(path)
                coord_stats = path_dct[path][0]
                coord_biases = path_dct[path][1]
                for loc in range(16): # for each grid location
                    bias = coord_stats[loc][2]
                    variance =  coord_stats[loc][3]
                    count = coord_stats[loc][4]
                    gt_x = coord_stats[loc][0]
                    gt_z = coord_stats[loc][1]
                    aggregate_data.append({'Y_Loc': yloc.in_suffix, 'Sequence':seq, 'Algorithm': alg,                                            'Catheter':cath_label, 'Grid_Loc':loc+1, 'Mean_Bias':bias,                                            'Biases':coord_biases[loc], 'Variance':variance, 'Count':count,                                           'GT_X':gt_x, 'GT_Z':gt_z})
            aggregate_plot.append({'Y_Loc': yloc, 'Sequence':seq, 'Algorithm': alg, 'Path_Dict':path_dct})


# In[6]:


agg_df = pd.DataFrame(aggregate_data)


# In[7]:


agg_df[agg_df['Count']==0] # Missed recording from a sequence at one grid location


# In[8]:


agg_df = agg_df.drop(agg_df[agg_df['Count']==0].index)


# In[9]:


# Display heatmap error plots
for plot_combination in aggregate_plot:
    yloc = plot_combination['Y_Loc']
    print(f'Error heatmaps for {yloc.description}')
    seq = plot_combination['Sequence']
    alg = plot_combination['Algorithm']
    path_dct = plot_combination['Path_Dict']
    main_path = main_paths[yloc.in_suffix]
    # Average the error over catheters
    sum_all_caths = path_dct[main_path[0]][0] + path_dct[main_path[1]][0] + path_dct[main_path[2]][0]
    avg_all_caths = sum_all_caths / 3
    if seq == 'SRI_Original' and yloc.in_suffix == 'Y2':
        # A recording is missing - results in one invalid 0 mm error in the path_dct
        print(f'Adjusting for missing recording for {seq}, {alg}, {yloc.in_suffix}')
        avg_all_caths[1, 2] = avg_all_caths[1, 2] * 3/2.0 # recover total sum, then divide by 2 for mean
    avg_err = avg_all_caths[:, 2]
    stddev = np.sqrt(sum_all_caths[:, 3])

    my_data = np.array([avg_all_caths[:, 0], avg_all_caths[:, 1], avg_err]).T

    X = my_data[:, 0]
    Y = my_data[:, 1]
    Z = my_data[:, 2]
    print(f'shape of Z: {np.shape(Z)}')

    heatmap = static_utils.nonuniform_imshow(X, Y, Z, stddev, numeric=True)
    plt.gca().invert_yaxis()
    plt.xlabel('X-Position (mm from isocentre)', fontsize = 20, fontweight = 'bold', labelpad = 10)
    plt.ylabel('Z-Position (mm from isocentre)', fontsize = 20, fontweight = 'bold', labelpad = 10)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    print('Tip Tracking Error @ {2} sequence {0}, algorithm {1}'.format(seq, alg, yloc.description))
    cbar = plt.colorbar(heatmap)
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '\u2265' + '5'])
    cbar.ax.get_yaxis().labelpad = 30
    cbar.ax.tick_params(labelsize = 18)
    cbar.ax.set_ylabel('Error (mm)', rotation = 270, fontsize = 20, fontweight = 'bold')

    if not os.path.isdir('{0}'.format(heatmap_paths[yloc.in_suffix])):
        os.makedirs('{0}'.format(heatmap_paths[yloc.in_suffix]))

    plt.savefig('{0}{1}_{2}_heatmap.png'.format(heatmap_paths[yloc.in_suffix], seq, alg), dpi=300)
    plt.show()
    # Below we save the tip errors from each catheter to hdf5 format
    h5out = '{0}{3}_{1}_{2}.h5'.format(error_path, seq, alg, yloc.in_suffix)
    with h5py.File(h5out, 'w', libver='latest') as f:
        print("Saving error data from each catheter to: " + h5out)
        for k, v in path_dct.items():
            expmt = pathlib.PurePath(k).parts[-1]
            name = 'static_err_' + seq + '_' + alg + '_' + expmt
            f.create_dataset(name, data=v[0])


# # Comparisons across Sequences and Algorithms
# We plan to compare tip errors from the JPNG and CAP peak-finding algorithms, for each sequence. Then we will compare tip errrors from both the HM & 3P sequences, with both JPNG and CAP algorithms. This will result in four comparisons, so we will use a Bonferroni adjustment for the p-value.

# In[10]:


p_val_adj = 0.05 / 4


# In[11]:


p_val_adj


# In[12]:


agg_df  # each row contains the error (bias) and variance for one multi-second recording
        # of one sequence, from one catheter, at one location, processed with one algorithm


# ## JPNG vs CAP
# ### HM
# Compare JPNG and CAP algorithms in the hadamard-multiplexed sequence

# In[13]:


hm_df = agg_df[agg_df['Sequence']=='FH512_noDither_gradSpoiled']


# #### HM mean bias comparison
# Compare JPNG and CAP algorithms in the hadamard-multiplexed sequence, using the mean biases from each 3d grid location - this should be an average of the biases from each catheter:

# In[14]:


hm_cap_mean_errors = hm_df[hm_df['Algorithm']=='centroid_around_peak']['Mean_Bias'].values


# In[15]:


hm_jpng_mean_errors = hm_df[hm_df['Algorithm']=='jpng']['Mean_Bias'].values


# In[16]:


len(hm_cap_mean_errors)


# In[17]:


len(hm_jpng_mean_errors)


# In[18]:


hm_cap_mean_errors = []
hm_jpng_mean_errors = []


# In[19]:


y_locs = hm_df['Y_Loc'].unique() # unique(): returned in order of appearance


# In[20]:


grid_locs = hm_df['Grid_Loc'].unique()


# In[21]:


for y in y_locs:
    for g in grid_locs:
        cap_mean = hm_df[(hm_df['Algorithm']=='centroid_around_peak') & (hm_df['Y_Loc']==y) & (hm_df['Grid_Loc']==g)]['Mean_Bias'].mean()
        jpng_mean = hm_df[(hm_df['Algorithm']=='jpng') & (hm_df['Y_Loc']==y) & (hm_df['Grid_Loc']==g)]['Mean_Bias'].mean()
        hm_cap_mean_errors.append(cap_mean)
        hm_jpng_mean_errors.append(jpng_mean)


# Sanity check that we now have the correct number of 3d locations:

# In[22]:


len(hm_cap_mean_errors)


# In[23]:


np.mean(hm_cap_mean_errors)


# In[24]:


np.std(hm_cap_mean_errors)


# In[25]:


len(hm_jpng_mean_errors)


# In[26]:


np.mean(hm_jpng_mean_errors)


# In[27]:


np.std(hm_jpng_mean_errors)


# In[28]:


w, p = stats.wilcoxon(x=hm_jpng_mean_errors,y=hm_cap_mean_errors,alternative='less')


# In[29]:


p


# In[30]:


import sys
sys.path.append('../')
import Invivo_Tracking.displacement_utils as disp_utils


# In[31]:


ax = sns.boxplot(data=[hm_cap_mean_errors,hm_jpng_mean_errors],orient='v',showmeans=True)
print('Static Experiment HM Sequence: CAP vs JPNG Tip Error')
max_y = max(np.max(hm_cap_mean_errors),np.max(hm_jpng_mean_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.4, star_str, ha='center', fontsize=12)
ax.set_title('HM Sequence: CAP and JPNG Tip Error')
ax.set_xticklabels(['CAP','JPNG'])
plt.savefig('../reports/figures/static/HM-capVsJpng-meantipErr.pdf',dpi=600)
plt.show()


# ### 3P Sequence
# Compare JPNG and CAP algorithms in the three-projection sequence

# In[32]:


proj_df = agg_df[agg_df['Sequence']=='SRI_Original']


# #### 3P mean bias comparison
# Compare JPNG and CAP algorithms in the three-projection sequence, using the mean biases from each 3d grid location - this should be an average of the biases from each catheter:

# In[33]:


proj_cap_mean_errors = []
proj_jpng_mean_errors = []


# In[34]:


for y in y_locs:
    for g in grid_locs:
        cap_mean = proj_df[(proj_df['Algorithm']=='centroid_around_peak') & (proj_df['Y_Loc']==y) & (proj_df['Grid_Loc']==g)]['Mean_Bias'].mean()
        jpng_mean = proj_df[(proj_df['Algorithm']=='jpng') & (proj_df['Y_Loc']==y) & (proj_df['Grid_Loc']==g)]['Mean_Bias'].mean()
        proj_cap_mean_errors.append(cap_mean)
        proj_jpng_mean_errors.append(jpng_mean)


# In[35]:


w, p = stats.wilcoxon(x=proj_jpng_mean_errors,y=proj_cap_mean_errors,alternative='less')


# In[36]:


p


# In[37]:


np.mean(proj_cap_mean_errors)


# In[38]:


np.std(proj_cap_mean_errors)


# In[39]:


np.mean(proj_jpng_mean_errors)


# In[40]:


np.std(proj_jpng_mean_errors)


# In[41]:


ax = sns.boxplot(data=[proj_cap_mean_errors,proj_jpng_mean_errors],orient='v',showmeans=True)
print('Static Experiment 3P Sequence: CAP vs JPNG Tip Error')
max_y = max(np.max(proj_cap_mean_errors),np.max(proj_jpng_mean_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.4, star_str, ha='center', fontsize=12)
ax.set_title('3P Sequence: CAP and JPNG Tip Error')
ax.set_xticklabels(['CAP','JPNG'])
plt.savefig('../reports/figures/static/3P-capVsJpng-meantipErr.pdf',dpi=600)
plt.show()


# ## HM vs 3P
# ### JPNG
# Compare tip errors from hadamard and three-projection sequences for JPNG algorithm.

# In[42]:


jpng_df = agg_df[agg_df['Algorithm']=='jpng'].copy()


# #### JPNG: compare the means 
# Compare mean tip errors from hadamard and three-projection sequences for JPNG algorithm: get the means from each 3d grid location, from all catheters, using Mean_Bias

# In[43]:


jpng_3P_mean_errors = []
jpng_HM_mean_errors = []
for y in y_locs:
    for g in grid_locs:
        proj_biases = jpng_df[(jpng_df['Sequence']=='SRI_Original') & (jpng_df['Y_Loc']==y) & (jpng_df['Grid_Loc']==g)]['Mean_Bias'].ravel()
        hm_biases = jpng_df[(jpng_df['Sequence']=='FH512_noDither_gradSpoiled') & (jpng_df['Y_Loc']==y) & (jpng_df['Grid_Loc']==g)]['Mean_Bias'].ravel()
        proj_biases = proj_biases[~np.isnan(proj_biases)]
        hm_biases = hm_biases[~np.isnan(hm_biases)]
        jpng_3P_mean_errors.append(np.mean(proj_biases))
        jpng_HM_mean_errors.append(np.mean(hm_biases))


# In[44]:


len(jpng_3P_mean_errors)


# In[45]:


np.mean(jpng_3P_mean_errors)


# In[46]:


np.std(jpng_3P_mean_errors)


# In[47]:


np.mean(jpng_HM_mean_errors)


# In[48]:


np.std(jpng_HM_mean_errors)


# In[49]:


w, p = stats.wilcoxon(x=jpng_HM_mean_errors,y=jpng_3P_mean_errors,alternative='less')


# In[50]:


p


# In[51]:


p_val_adj


# In[52]:


ax = sns.boxplot(data=[jpng_3P_mean_errors,jpng_HM_mean_errors],orient='v',showmeans=True)
print('Static Experiment JPNG Algorithm: 3P and HM Mean Tip Error')
max_y = max(np.max(jpng_3P_mean_errors),np.max(jpng_HM_mean_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.4, star_str, ha='center', fontsize=12)
ax.set_title('JPNG Algorithm: 3P and HM Tip Error')
ax.set_xticklabels(['3P','HM'])
plt.savefig('../reports/figures/static/JPNG-3PvsHM-mean_boxplot.pdf',dpi=600)
plt.show()


# ## CAP
# Compare tip errors from hadamard and three-projection sequences for the centroid-around-peak algorithm.

# In[53]:


cap_df = agg_df[agg_df['Algorithm']=='centroid_around_peak'].copy()


# #### CAP: compare the means 
# Compare mean tip errors from hadamard and three-projection sequences for CAP algorithm: get the means from each 3d grid location, from all catheters, using the Mean_Bias column

# In[54]:


cap_3P_mean_errors = []
cap_HM_mean_errors = []
for y in y_locs:
    for g in grid_locs:
        proj_biases = cap_df[(cap_df['Sequence']=='SRI_Original') & (cap_df['Y_Loc']==y) & (cap_df['Grid_Loc']==g)]['Mean_Bias'].ravel()
        hm_biases = cap_df[(cap_df['Sequence']=='FH512_noDither_gradSpoiled') & (cap_df['Y_Loc']==y) & (cap_df['Grid_Loc']==g)]['Mean_Bias'].ravel()
        proj_biases = proj_biases[~np.isnan(proj_biases)]
        hm_biases = hm_biases[~np.isnan(hm_biases)]
        cap_3P_mean_errors.append(np.mean(proj_biases))
        cap_HM_mean_errors.append(np.mean(hm_biases))


# In[55]:


np.mean(cap_3P_mean_errors)


# In[56]:


np.std(cap_3P_mean_errors)


# In[57]:


np.mean(cap_HM_mean_errors)


# In[58]:


np.std(cap_HM_mean_errors)


# In[59]:


w, p = stats.wilcoxon(x=cap_HM_mean_errors,y=cap_3P_mean_errors,alternative='less')


# In[60]:


p


# In[61]:


ax = sns.boxplot(data=[cap_3P_mean_errors,cap_HM_mean_errors],orient='v',showmeans=True)
print('Static Experiment CAP Algorithm: 3P and HM Mean Tip Error')
max_y = max(np.max(cap_3P_mean_errors),np.max(cap_HM_mean_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.4, star_str, ha='center', fontsize=12)
ax.set_title('CAP Algorithm: 3P and HM Tip Error')
ax.set_xticklabels(['3P','HM'])
plt.savefig('../reports/figures/static/CAP-3PvsHM-mean_boxplot.pdf',dpi=600)
plt.show()


# # HDF5 Exports
# Error data has been saved to hdf5 (Hierarchical Data Format) files, one file for each sequence and algorithm combination. These can be read using your own code for further analysis, as shown in the code snippet below.
# 
# We check that we can open and read the most recent hdf5 file. The catheter error arrays are stored as datasets within this file.
# 
# Output the first dataset as an array from this file. This is a 2d array containing the tip errors for this catheter.
# 
# For each grid position:
# the ground truth X coordinate, Z coordinate, bias (tip error) in mm, tip variance in mm (in that order)
# 
# So, there are 16 elements arranged like:
# 
# `[ [GT_x_pos0, GT_z_pos0, bias_pos0, variance_pos0], [GT_x_pos1, GT_z_pos1, bias_pos1, variance_pos1], ... [GT_x_pos15, GT_z_pos15, bias_pos15, variance_pos2] ]`

# In[62]:


with h5py.File(h5out, 'r', libver='latest') as f:
    first_dataset = list(f.keys())[0]

    print("Looking at catheter tip errors for " + first_dataset           + "\n from the file: " + h5out)
    
    # Can also use h5py dataset object: ds_obj = f[a_group_key]
    # The 2d array contains the tip error for this catheter
    ds_arr = f[first_dataset][()]  # returns as a numpy array
    print(ds_arr)

