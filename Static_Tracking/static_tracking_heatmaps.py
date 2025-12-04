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

# In[6]:


p_val_adj = 0.05 / 4


# In[7]:


p_val_adj


# In[8]:


agg_df = pd.DataFrame(aggregate_data)


# In[9]:


agg_df


# In[10]:


agg_df[agg_df['Count']==0] # Missed recording from a sequence at one grid location


# In[11]:


# Missing recording for this combination!
#invalid_rows = agg_df[(agg_df['Y_Loc']=='Y2') & (agg_df['Sequence']=='SRI_Original') & (agg_df['Catheter']=='C306') & (agg_df['Grid_Loc']==2)].index


# In[12]:


agg_df = agg_df.drop(agg_df[agg_df['Count']==0].index)


# In[13]:


agg_df[agg_df['Count']==0]


# In[14]:


agg_df  # each row contains the error (bias) and variance for one multi-second recording
        # of one sequence, from one catheter, at one location, processed with one algorithm


# ## JPNG vs CAP
# ### HM
# Compare JPNG and CAP algorithms in the hadamard-multiplexed sequence

# In[15]:


hm_df = agg_df[agg_df['Sequence']=='FH512_noDither_gradSpoiled']


# In[16]:


hm_cap_errors = hm_df[hm_df['Algorithm']=='centroid_around_peak']['Biases'].apply(pd.Series).values.ravel()


# In[17]:


hm_cap_errors


# In[18]:


hm_jpng_errors = hm_df[hm_df['Algorithm']=='jpng']['Biases'].apply(pd.Series).values.ravel()


# In[19]:


hm_jpng_errors


# In[20]:


np.isnan(hm_jpng_errors).sum() / len(hm_jpng_errors)


# In[21]:


np.array_equal(np.isnan(hm_jpng_errors),np.isnan(hm_cap_errors))


# In[22]:


# NaNs in our tip error are likely due to recordings for multiple coils finishing at slightly different times
# They match between the arrays, so remove them
hm_cap_errors = hm_cap_errors[~np.isnan(hm_cap_errors)]
hm_jpng_errors = hm_jpng_errors[~np.isnan(hm_jpng_errors)]


# In[23]:


len(hm_cap_errors)


# In[24]:


len(hm_jpng_errors)


# In[25]:


w, p = stats.wilcoxon(x=hm_jpng_errors,y=hm_cap_errors,alternative='less')


# In[26]:


p


# In[27]:


import sys
sys.path.append('../')


# In[28]:


import Invivo_Tracking.displacement_utils as disp_utils


# In[29]:


plot, test = disp_utils.plot_displacement_boxplot([hm_cap_errors,hm_jpng_errors], show_scatter=False,                                                  y_label='Tip Error',set_ymax=10, p_value=p_val_adj, alternative='greater')
print('Static Experiment HM Sequence: CAP vs JPNG Tip Error '+ test)
plot.savefig('../reports/figures/static/HM-capVsJpng-tipErr.pdf',dpi=600)
plot.show()


# ### 3P Sequence
# Compare JPNG and CAP algorithms in the three-projection sequence

# In[30]:


proj_df = agg_df[agg_df['Sequence']=='SRI_Original']


# In[31]:


proj_cap_errors = proj_df[proj_df['Algorithm']=='centroid_around_peak']['Biases'].apply(pd.Series).values.ravel()


# In[32]:


proj_jpng_errors = proj_df[proj_df['Algorithm']=='jpng']['Biases'].apply(pd.Series).values.ravel()


# In[33]:


proj_cap_errors


# In[34]:


np.isnan(proj_cap_errors).sum() / len(proj_cap_errors)


# In[35]:


np.array_equal(np.isnan(proj_cap_errors),np.isnan(proj_jpng_errors))


# In[36]:


# NaNs in our tip error are likely due to recordings for multiple coils finishing at slightly different times
# They match between the arrays, so remove them
proj_cap_errors = proj_cap_errors[~np.isnan(proj_cap_errors)]
proj_jpng_errors = proj_jpng_errors[~np.isnan(proj_jpng_errors)]


# In[37]:


w, p = stats.wilcoxon(x=proj_jpng_errors,y=proj_cap_errors,alternative='less')


# In[38]:


p


# In[39]:


plot, test = disp_utils.plot_displacement_boxplot([proj_cap_errors,proj_jpng_errors], show_scatter=False,                                                  y_label='Tip Error',set_ymax=14, p_value=p_val_adj, alternative='greater')
print('Static Experiment 3P Sequence: CAP vs JPNG Tip Error '+ test)
plot.savefig('../reports/figures/static/3P-capVsJpng-tipErr.pdf',dpi=600)
plot.show()


# ## HM vs 3P
# ### JPNG
# Compare tip errors from hadamard and three-projection sequences for JPNG algorithm.

# In[40]:


jpng_df = agg_df[agg_df['Algorithm']=='jpng'].copy()


# We're missing a recording from one catheter & location combination from the 3P sequence:
# - Y_Loc: Y2
# - Catheter: C306
# - Grid_Loc: 2
# 
# To do a paired comparison, we should remove the matching recording from the HM sequence.

# In[41]:


missing_match = jpng_df[(jpng_df['Y_Loc']=='Y2') & (jpng_df['Sequence']=='FH512_noDither_gradSpoiled') &                        (jpng_df['Catheter']=='C306') & (jpng_df['Grid_Loc']==2)].index


# In[42]:


jpng_df = jpng_df.drop(missing_match)


# In[43]:


jpng_df['Biases_Crop'] = jpng_df['Biases'] # Will crop this bias list to the smaller sample size


# In[44]:


def check_nans(biases):
    """ Check if the only NaNs are clustered in the tail, with no non-NaNs after the first NaN
    """
    bias_nans = np.isnan(biases)
    if bias_nans.sum() == 0: # No NaNs
        return True
    first_nan_ind = np.argmax(bias_nans)
    return np.all(bias_nans[first_nan_ind:])

def crop_tails(biases):
    """ Crop everything after the first NaN
    """
    bias_nans = np.isnan(biases)
    if bias_nans.sum() == 0: # No NaNs
        return biases # don't crop
    first_nan_ind = np.argmax(bias_nans)
    return biases[0:first_nan_ind]
                              
def crop_bias_list(bias_list, yloc, gridloc, catheter, shorter_sequence, df):
    # find the combination of location and catheter from a different (shorter) sequence in the dataframe
    # and return the given bias list, shortened so that it matches the number of samples from the other sequence
    other_row = df[(df['Sequence']==shorter_sequence) & (df['Y_Loc']==yloc) & (df['Grid_Loc']==gridloc) &                    (df['Catheter']==catheter)]
    if len(other_row) < 1:
        print('**ERR** Target not found. Not cropping.')
        return bias_list
    if len(other_row) > 1:
        print(f'**WARN** Unexpected {len(other_row)} matches, expected 1')
    target = other_row.iloc[0]['Biases_Crop']
    if (len(target) >= len(bias_list)):
        print(f'**WARN** Target of size {len(target)} not shorter than bias list of size {len(bias_list)}. Not cropping.')
        return bias_list
    return bias_list[0:len(target)]


# In[45]:


nan_tails = jpng_df.apply(lambda x: check_nans(x['Biases']), axis=1)


# In[46]:


np.all(nan_tails)


# In[47]:


jpng_df['Biases_Crop'] = jpng_df.apply(lambda x: crop_tails(x['Biases']),axis=1)


# In[48]:


#crop_bias_list(jpng_df.iloc[0]['Biases'],'Y0',1,'C222','FH512_noDither_gradSpoiled',jpng_df)


# In[49]:


jpng_df['Biases_Crop'] = jpng_df.apply(lambda x: x['Biases_Crop'] if x['Sequence'] != 'SRI_Original' else                                        crop_bias_list(x['Biases_Crop'], x['Y_Loc'], x['Grid_Loc'],                                                       x['Catheter'],'FH512_noDither_gradSpoiled', jpng_df),                                        axis=1)


# In[50]:


jpng_HM_errors = jpng_df[jpng_df['Sequence']=='FH512_noDither_gradSpoiled']['Biases_Crop'].apply(pd.Series).values.ravel()


# In[51]:


jpng_3P_errors = jpng_df[jpng_df['Sequence']=='SRI_Original']['Biases_Crop'].apply(pd.Series).values.ravel()


# In[52]:


len(jpng_HM_errors)


# In[53]:


len(jpng_3P_errors)


# In[54]:


np.isnan(jpng_3P_errors).sum()


# In[55]:


np.isnan(jpng_HM_errors).sum()


# In[56]:


np.isnan(jpng_HM_errors).sum() / len(jpng_HM_errors)


# In[57]:


np.array_equal(np.isnan(jpng_HM_errors),np.isnan(jpng_3P_errors))


# In[58]:


jpng_HM_errors = jpng_HM_errors[~np.isnan(jpng_HM_errors)]
jpng_3P_errors = jpng_3P_errors[~np.isnan(jpng_3P_errors)]


# In[59]:


len(jpng_HM_errors)


# In[60]:


len(jpng_3P_errors)


# In[61]:


import seaborn as sns


# In[62]:


sns.violinplot(data=[jpng_3P_errors, jpng_HM_errors],orient='v')


# Mann-Whitney U tests for independent measures: skip this and do the Wilcoxon test

# In[63]:


u, p = stats.mannwhitneyu(jpng_HM_errors,jpng_3P_errors,alternative='less')


# In[64]:


p


# In[95]:


w, p = stats.wilcoxon(x=jpng_HM_errors,y=jpng_3P_errors,alternative='less')


# In[96]:


p


# In[97]:


ax = sns.boxplot(data=[jpng_3P_errors,jpng_HM_errors],orient='v',showmeans=True)
print('Static Experiment JPNG Algorithm: 3P and HM Tip Error')
max_y = max(np.max(jpng_3P_errors),np.max(jpng_HM_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.5, star_str, ha='center', fontsize=12)
ax.set_title('JPNG Algorithm: 3P and HM Tip Error')
ax.set_xticklabels(['3P','HM'])
plt.savefig('../reports/figures/static/JPNG-3PvsHM-boxplot.pdf',dpi=600)
plt.show()


# ## CAP
# Compare tip errors from hadamard and three-projection sequences for the centroid-around-peak algorithm.

# In[69]:


cap_df = agg_df[agg_df['Algorithm']=='centroid_around_peak'].copy()


# In[70]:


missing_match = cap_df[(cap_df['Y_Loc']=='Y2') & (cap_df['Sequence']=='FH512_noDither_gradSpoiled') &                        (cap_df['Catheter']=='C306') & (cap_df['Grid_Loc']==2)].index


# In[71]:


# drop the hm recording that doesn't have a matching 3p recording
cap_df = cap_df.drop(missing_match)


# In[72]:


cap_df['Biases_Crop'] = cap_df['Biases'] # Will crop this bias list to the smaller sample size


# In[73]:


nan_tails = cap_df.apply(lambda x: check_nans(x['Biases']), axis=1) # check if all the NaNs are in the tails


# In[74]:


np.all(nan_tails)


# In[75]:


cap_df['Biases_Crop'] = cap_df.apply(lambda x: crop_tails(x['Biases']),axis=1) # crop off the NaNs


# In[76]:


# Match sample size between sequences
cap_df['Biases_Crop'] = cap_df.apply(lambda x: x['Biases_Crop'] if x['Sequence'] != 'SRI_Original' else                                        crop_bias_list(x['Biases_Crop'], x['Y_Loc'], x['Grid_Loc'],                                                       x['Catheter'],'FH512_noDither_gradSpoiled', cap_df),                                        axis=1)


# In[77]:


cap_HM_errors = cap_df[cap_df['Sequence']=='FH512_noDither_gradSpoiled']['Biases_Crop'].apply(pd.Series).values.ravel()
cap_3P_errors = cap_df[cap_df['Sequence']=='SRI_Original']['Biases_Crop'].apply(pd.Series).values.ravel()


# In[78]:


len(cap_HM_errors)


# In[79]:


len(cap_3P_errors)


# In[80]:


np.isnan(cap_HM_errors).sum()


# In[81]:


np.isnan(cap_3P_errors).sum()


# In[82]:


np.isnan(cap_3P_errors).sum() / len(cap_3P_errors)


# In[83]:


np.array_equal(np.isnan(cap_HM_errors),np.isnan(cap_3P_errors))


# In[84]:


cap_HM_errors = cap_HM_errors[~np.isnan(cap_HM_errors)]
cap_3P_errors = cap_3P_errors[~np.isnan(cap_3P_errors)]


# In[85]:


len(cap_HM_errors)


# In[86]:


len(cap_3P_errors)


# In[87]:


sns.violinplot(data=[cap_3P_errors, cap_HM_errors],orient='v')


# Mann Whitney U test for independent measures: skip this and go to Wilcoxon test since our samples are now paired

# In[88]:


u, p = stats.mannwhitneyu(cap_HM_errors,cap_3P_errors,alternative='less')


# In[89]:


p


# In[98]:


w, p = stats.wilcoxon(x=cap_HM_errors,y=cap_3P_errors,alternative='less')


# In[99]:


p


# In[100]:


ax = sns.boxplot(data=[cap_3P_errors,cap_HM_errors],orient='v',showmeans=True)
print('Static Experiment CAP Algorithm: 3P and HM Tip Error')
max_y = max(np.max(cap_3P_errors),np.max(cap_HM_errors))
if p < p_val_adj: # draw a significance line
    star_str = disp_utils.get_p_stars(p)
    starline_y = max_y
    ax.plot([0, 0, 1, 1], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
    ax.text(0.49, starline_y+0.5, star_str, ha='center', fontsize=12)
ax.set_title('CAP Algorithm: 3P and HM Tip Error')
ax.set_xticklabels(['3P','HM'])
plt.savefig('../reports/figures/static/CAP-3PvsHM-boxplot.pdf',dpi=600)
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

# In[94]:


with h5py.File(h5out, 'r', libver='latest') as f:
    first_dataset = list(f.keys())[0]

    print("Looking at catheter tip errors for " + first_dataset           + "\n from the file: " + h5out)
    
    # Can also use h5py dataset object: ds_obj = f[a_group_key]
    # The 2d array contains the tip error for this catheter
    ds_arr = f[first_dataset][()]  # returns as a numpy array
    print(ds_arr)

