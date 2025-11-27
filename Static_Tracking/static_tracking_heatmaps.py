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
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('pwd', '')


# # Constants and Paths

# In[3]:


# Main data paths for each catheter (manually input)
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
        
#main_path_222 = parent_path + '/static/trackTest-17Aug2021-C222-Y0/'
#main_path_231 = parent_path + '/static/trackTest-15Dec2021-C231-Y0/'
#main_path_306 = parent_path + '/static/trackTest-13Dec2021-C306-Y0/'

#main_path = [main_path_222, main_path_231, main_path_306]

# Where you would like to save the heatmaps (manually input)
#heatmap_path = '../reports/figures/static/Y45mm/'

# Tip error data is also exported to HDF5 files
error_path = '../reports/export/static/'

Gt_filename = '1GroundTruthCoords.csv'

geometry_index = 1


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

os.makedirs(error_path, exist_ok=True)

for yloc in y_locations:
    print(f'Error heatmaps for {yloc.description}')
    for seq in sequences:
        for alg in algorithms:
            main_path = main_paths[yloc.in_suffix]
            path_dct = static_utils.get_catheter_data(main_path, seq, alg, Gt_filename, geometry_index)
            # Aggregate
            for path in main_path: # path per catheter
                cath_label = get_catheter_from_path(path)
                coord_stats = path_dct[path][0]
                coord_biases = path_dct[path][1]
                for loc in range(16): # for each grid location
                    bias = coord_stats[loc][2]
                    variance =  coord_stats[loc][3]
                    count = coord_stats[loc][4]
                    aggregate_data.append({'Y_Loc': yloc.in_suffix, 'Sequence':seq, 'Algorithm': alg, 'Catheter':cath_label,                              'Grid_Loc':loc+1, 'Mean_Bias':bias, 'Biases':coord_biases[loc], 'Variance':variance, 'Count':count})
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

# In[39]:


p_val_adj = 0.05 / 4


# In[40]:


p_val_adj


# In[6]:


agg_df = pd.DataFrame(aggregate_data)


# In[7]:


agg_df


# In[8]:


agg_df[agg_df['Count']==0] # Missed recording from a sequence at one grid location


# In[9]:


# Missing recording for this combination!
#invalid_rows = agg_df[(agg_df['Y_Loc']=='Y2') & (agg_df['Sequence']=='SRI_Original') & (agg_df['Catheter']=='C306') & (agg_df['Grid_Loc']==2)].index


# In[10]:


agg_df = agg_df.drop(agg_df[agg_df['Count']==0].index)


# In[11]:


agg_df[agg_df['Count']==0]


# In[12]:


agg_df  # each row contains the error (bias) and variance for one multi-second recording
        # of one sequence, from one catheter, at one location, processed with one algorithm


# ## JPNG vs CAP
# ### HM
# Compare JPNG and CAP algorithms in the hadamard-multiplexed sequence

# In[13]:


hm_df = agg_df[agg_df['Sequence']=='FH512_noDither_gradSpoiled']


# In[14]:


hm_cap_errors = hm_df[hm_df['Algorithm']=='centroid_around_peak']['Biases'].apply(pd.Series).values.ravel()


# In[15]:


hm_cap_errors


# In[16]:


hm_jpng_errors = hm_df[hm_df['Algorithm']=='jpng']['Biases'].apply(pd.Series).values.ravel()


# In[17]:


hm_jpng_errors


# In[18]:


np.isnan(hm_jpng_errors).sum() / len(hm_jpng_errors)


# In[19]:


np.array_equal(np.isnan(hm_jpng_errors),np.isnan(hm_cap_errors))


# In[20]:


# Later, check why there are NaNs in our tip error. For now, we know they match between the arrays, so remove them
hm_cap_errors = hm_cap_errors[~np.isnan(hm_cap_errors)]
hm_jpng_errors = hm_jpng_errors[~np.isnan(hm_jpng_errors)]


# In[21]:


len(hm_cap_errors)


# In[22]:


len(hm_jpng_errors)


# In[23]:


w, p = stats.wilcoxon(x=hm_jpng_errors,y=hm_cap_errors,alternative='less')


# In[24]:


p


# In[25]:


import sys
sys.path.append('../')


# In[26]:


import Invivo_Tracking.displacement_utils as disp_utils


# In[41]:


plot, test = disp_utils.plot_displacement_boxplot([hm_cap_errors,hm_jpng_errors], show_scatter=False,                                                  y_label='Tip Error',set_ymax=10, p_value=p_val_adj, alternative='greater')
print('Static Experiment HM Sequence: CAP vs JPNG Tip Error '+ test)
plot.savefig('../reports/figures/static/HM-capVsJpng-tipErr.pdf',dpi=600)
plot.show()


# ### 3P Sequence
# Compare JPNG and CAP algorithms in the three-projection sequence

# In[28]:


proj_df = agg_df[agg_df['Sequence']=='SRI_Original']


# In[29]:


proj_cap_errors = proj_df[proj_df['Algorithm']=='centroid_around_peak']['Biases'].apply(pd.Series).values.ravel()


# In[30]:


proj_jpng_errors = proj_df[proj_df['Algorithm']=='jpng']['Biases'].apply(pd.Series).values.ravel()


# In[31]:


proj_cap_errors


# In[32]:


np.isnan(proj_cap_errors).sum() / len(proj_cap_errors)


# In[33]:


np.array_equal(np.isnan(proj_cap_errors),np.isnan(proj_jpng_errors))


# In[34]:


# Later, check why there are NaNs in our tip error. For now, we know they match between the arrays, so remove them
proj_cap_errors = proj_cap_errors[~np.isnan(proj_cap_errors)]
proj_jpng_errors = proj_jpng_errors[~np.isnan(proj_jpng_errors)]


# In[35]:


w, p = stats.wilcoxon(x=proj_jpng_errors,y=proj_cap_errors,alternative='less')


# In[36]:


p


# In[42]:


plot, test = disp_utils.plot_displacement_boxplot([proj_cap_errors,proj_jpng_errors], show_scatter=False,                                                  y_label='Tip Error',set_ymax=14, p_value=p_val_adj, alternative='greater')
print('Static Experiment 3P Sequence: CAP vs JPNG Tip Error '+ test)
plot.savefig('../reports/figures/static/3P-capVsJpng-tipErr.pdf',dpi=600)
plot.show()


# In[ ]:





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

# In[38]:


with h5py.File(h5out, 'r', libver='latest') as f:
    first_dataset = list(f.keys())[0]

    print("Looking at catheter tip errors for " + first_dataset           + "\n from the file: " + h5out)
    
    # Can also use h5py dataset object: ds_obj = f[a_group_key]
    # The 2d array contains the tip error for this catheter
    ds_arr = f[first_dataset][()]  # returns as a numpy array
    print(ds_arr)

