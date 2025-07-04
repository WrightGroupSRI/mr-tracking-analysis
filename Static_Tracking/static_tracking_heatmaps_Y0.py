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
main_path_222 = parent_path + '/static/trackTest-17Aug2021-C222-Y0/'
main_path_231 = parent_path + '/static/trackTest-15Dec2021-C231-Y0/'
main_path_306 = parent_path + '/static/trackTest-13Dec2021-C306-Y0/'

main_path = [main_path_222, main_path_231, main_path_306]

# Where you would like to save the heatmaps (manually input)
heatmap_path = '../reports/figures/static/Y45mm/'

# Tip error data is also exported to HDF5 files
error_path = '../reports/export/static/'

Gt_filename = '1GroundTruthCoords.csv'

geometry_index = 1


# # Error Heatmaps for Y= 45 mm
# 
# For each sequence and localization algorithm of interest, plot the heatmap of average tip errors at each position. Each position has recordings from three catheters.

# In[4]:


sequences = ['SRI_Original', 'FH512_noDither_gradSpoiled']
algorithms = ['centroid_around_peak', 'jpng']

os.makedirs(error_path, exist_ok=True)

for seq in sequences:
    for alg in algorithms:
        path_dct = static_utils.get_catheter_data(main_path, seq, alg, Gt_filename, geometry_index)
        # Average the error
        sum_all_caths = path_dct[main_path_222] + path_dct[main_path_231] + path_dct[main_path_306]
        avg_all_caths = sum_all_caths / 3
        avg_err = avg_all_caths[:, 2]
        stddev = np.sqrt(sum_all_caths[:, 3])
         
        
        my_data = np.array([avg_all_caths[:, 0], avg_all_caths[:, 1], avg_err]).T
        
        X = my_data[:, 0]
        Y = my_data[:, 1]
        Z = my_data[:, 2]

        heatmap = static_utils.nonuniform_imshow(X, Y, Z, stddev, numeric=True)
        plt.gca().invert_yaxis()
        plt.xlabel('X-Position (mm from isocentre)', fontsize = 18, fontweight = 'bold', labelpad = 10)
        plt.ylabel('Z-Position (mm from isocentre)', fontsize = 18, fontweight = 'bold', labelpad = 10)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        print('Tip Tracking Error @ Y = 45mm sequence {0}, algorithm {1}'.format(seq, alg))
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '\u2265' + '5'])
        cbar.ax.get_yaxis().labelpad = 30
        cbar.ax.tick_params(labelsize = 16) 
        cbar.ax.set_ylabel('Error (mm)', rotation = 270, fontsize = 18, fontweight = 'bold')
        
        if not os.path.isdir('{0}'.format(heatmap_path)):
            os.makedirs('{0}'.format(heatmap_path))

        plt.savefig('{0}{1}_{2}_heatmap.png'.format(heatmap_path, seq, alg), dpi=300)
        plt.show()
        # Below we save the tip errors from each catheter to hdf5 format
        h5out = '{0}Y0_{1}_{2}.h5'.format(error_path, seq, alg)
        with h5py.File(h5out, 'w', libver='latest') as f:
            print("Saving error data from each catheter to: " + h5out)
            for k, v in path_dct.items():
                expmt = pathlib.PurePath(k).parts[-1]
                name = 'static_err_' + seq + '_' + alg + '_' + expmt
                f.create_dataset(name, data=v)   


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

# In[5]:


with h5py.File(h5out, 'r', libver='latest') as f:
    first_dataset = list(f.keys())[0]

    print("Looking at catheter tip errors for " + first_dataset           + "\n from the file: " + h5out)
    
    # Can also use h5py dataset object: ds_obj = f[a_group_key]
    # The 2d array contains the tip error for this catheter
    ds_arr = f[first_dataset][()]  # returns as a numpy array
    print(ds_arr)

