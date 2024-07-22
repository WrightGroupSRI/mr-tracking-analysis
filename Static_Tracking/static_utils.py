import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import scipy.interpolate
from catheter_utils import geometry
from catheter_utils import cathcoords
from catheter_utils import metrics


##### Heatmap Function ######
def nonuniform_imshow(x, y, z, aspect = 1, cmap = plt.cm.rainbow):
    # Create regular grid
    xi, yi = np.linspace(x.min() - 16.6, x.max() + 16.6, 4), np.linspace(y.min() - 16.6, y.max() + 16.6, 4)
    xi, yi = np.meshgrid(xi, yi)

    norm = plc.Normalize(vmin = 0, vmax = 5)
    interp = scipy.interpolate.NearestNDInterpolator(list(zip(x, y)), z)
    zi = interp(xi, yi)

    fig, ax = plt.subplots(figsize=(12, 8))

    hm = ax.imshow(zi, interpolation = 'nearest', cmap = cmap, norm = norm, extent = [x.min() - 16.6, x.max() + 16.6, y.max() + 16.6, y.min() - 16.6])
    ax.scatter(x, y, c = 'k')
    ax.set_aspect(aspect)
    return hm

def get_catheter_data(main_path, sequence, algorithm, gt_filename, geometry_index=1):
    # Get data for each catheter
    path_dct = {}

    for path in main_path:

        coords = []

        # Register every bias (16 in this case)
        for i in range(16):

            # Get ground truth info for each point
            Gt_path = path + str(i + 1) + '/' + gt_filename
            Gt_file = open(Gt_path, 'r')
            reader = Gt_file.readlines()
            distal_gt = []
            proximal_gt = []

            for line in reader[1:]:
                data = line.split(',')
                if (data[-2].strip() == 'dist'):
                    distal_gt.append(np.array([float(data[1]), float(data[2]), float(data[3])]))
                    distal_gt = np.array(distal_gt)
                    distal_index = int(data[-1].strip())

                elif (data[-2].strip() == 'prox'):
                    proximal_gt.append(np.array([float(data[1]), float(data[2]), float(data[3])]))
                    proximal_gt = np.array(proximal_gt)
                    proximal_index = int(data[-1].strip())

            # Determine coil geometry for each point
            geo = geometry.GEOMETRY[geometry_index]
            fit = geo.fit_from_coils_mse(distal_gt, proximal_gt)
            tip_gt = fit.tip

            # Choose relevant sequence directory
            directories = [f.name for f in os.scandir(path + str(i + 1) + '/') if f.is_dir()]
            for directory in directories:
                if directory.startswith(sequence):
                    curr_dir = path + str(i + 1) + '/' + directory + '/'

            # Get error data for current point
            cathcoord_files = cathcoords.discover_files(curr_dir + algorithm + '/')

            if bool(cathcoord_files) == True:
                distal_file = cathcoord_files[0][distal_index]    
                proximal_file = cathcoord_files[0][proximal_index]

                bias = metrics.Bias(distal_file, proximal_file, tip_gt, geo)

                coords.append([tip_gt[0][0], tip_gt[0][2], bias])

            else:
                coords.append([tip_gt[0][0], tip_gt[0][2], 0])

        coords = np.array(coords)

        path_dct[path] = coords
    return path_dct

