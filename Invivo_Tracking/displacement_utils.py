"""Displacement utils
Methods to calculate, plot, and work with displacements from cathcoords files
"""

import matplotlib.pyplot as plt
from catheter_utils import cathcoords
from catheter_utils import geometry
from catheter_utils import metrics
import numpy as np
from collections import namedtuple
from scipy import stats
import warnings
import typing

algos = ['peak','centroid_around_peak','png','jpng']
select_algos = ['centroid_around_peak','jpng']
ArrayStats = namedtuple("ArrayStats", "mean min max std")

def get_p_stars(p_value:float):
    """
    Return a string corresponding to the pvalue star convention
    """
    if p_value < 0: # bad value, return empty string
        return ''
    if p_value >= 0.05:
        return "ns"
    elif p_value < 0.05 and p_value >= 0.01:
        return "*"
    elif p_value < 0.01 and p_value >= 0.001:
        return "**"
    elif p_value < 0.001 and p_value >= 0.0001:
        return "***"
    else:
        return "****"

def get_locs_displacements(file_root, distal_file, proximal_file, geo, selected_algos = algos):
    """
    Reads the localizer output files for the given peak-finder algorithms and
    returns the coil coordinates, fits, and distances of tip from centroid
    for each algorithm
    Returns tuple, each a dict containing an entry for each selected algorithm:
    algo_coords
        - each entry contains the "Cathcoords" tuple (coordinates,snr,times,trigs,resps) for both coils
    algo_fits - each entry contains the "FitFromCoilsResult" tuple (tip, distal, and proximal coordinates)
    tip_distances - each entry contains the distances from the tip to the centroid for that recording & algo over time
    centroids - each entry contains the centroid of each recording & algorithm
    """
    algo_coords = {}
    algo_fits = {}
    tip_distances = {}
    centroids = {}

    for pk_find in selected_algos:
        #print('file root' + str(file_root) + '\t algo: ' + str(pk_find) + '\t distal: ' + str(distal_file))
        distal_path = file_root + '/' + pk_find + '/' + distal_file
        proximal_path = file_root + '/' + pk_find + '/' + proximal_file
        coords_dist, coords_prox = cathcoords.read_pair(distal_path,proximal_path)
        fit = geo.fit_from_coils_mse(coords_dist.coords, coords_prox.coords)
        algo_coords[pk_find] = (coords_dist, coords_prox)
        algo_fits[pk_find] = fit
        tip_distances[pk_find],centroids[pk_find] = cathcoords.get_distances_from_centroid(fit.tip)
    
    return algo_coords, algo_fits, tip_distances, centroids

def getMsecsSinceStart(cath_coords):
    ts = cath_coords.times
    msecs = [0]
    msecs.extend(ts[1:] - ts[0])
    return msecs

def plot_displacements(x_values, displacements,selected_algos = select_algos, y_label='Distance from centroid (mm)', x_label='time (msecs)'):
    for pk_find in selected_algos:
        plt.plot(x_values, displacements[pk_find],marker='.',alpha=0.8,linestyle='--',label=pk_find)

    plt.grid(linestyle='--',which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.legend()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt

def plot_displacement_diff(x_values, displacements, algo_1, algo_2):
    plot_label = algo_1 + ' - ' + algo_2
    plt.plot(x_values, displacements[algo_1] - displacements[algo_2],marker='.',alpha=0.8,linestyle='--')

    plt.grid(linestyle='--',which='both')
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.xlabel('time (msecs)')
    plt.ylabel(plot_label + ' (mm)')
    return plt

def plot_displacement_boxplot(displacements, selected_algos = select_algos, show_scatter = True, y_label="Distances from centroid (mm)", set_ymax=15, p_value=0.05, plot=plt):
    """ Show boxplots comparing displacements from different algorithms
    Runs paired t test / Wilcoxon signed rank test if only two algorithms are passed in
    Returns the plot
    """
    plot_arrays = displacements
    is_dict = type(displacements) == dict
    if (is_dict):
        plot_arrays = [displacements[i] for i in selected_algos]
    bp = plot.boxplot(plot_arrays,notch=True,showmeans=True)
    if (plot == plt):
        plot.xticks(range(1,len(plot_arrays)+1),selected_algos)
    else:
        plot.set_xticks(range(1,len(plot_arrays)+1))
        plot.set_xticklabels(selected_algos)
    plot.grid(linestyle='--',which='both')
    plot.minorticks_on()
    plot.grid(which='minor', alpha=0.3)
    max_y = 0
    if show_scatter:
        for i in range(len(selected_algos)):
            if is_dict:
                y = displacements[selected_algos[i]]
            else:
                y = displacements[i]
            max_y = max(max_y, max(y))
            x = np.random.normal(i+1,0.03,size=len(y))
            plot.plot(x, y, 'r.', alpha=0.4)
    if (len(selected_algos) == 2):
        # check for sig diff
        diff_arr, _, p = test_displacements_means_diff_paired(displacements, [selected_algos], p_value, quiet=True)
        if diff_arr[0] != 0: # reject equal means null hypoth
            # draw a line: using max_y can be too high if there are cropped outliers
            starline_y = max_y
            if (set_ymax != None and starline_y >= set_ymax):
                whisker_bars = [whiskers.get_ydata()[1] for whiskers in bp["whiskers"]]
                # print("whisker_bars ydata shape: " + str(np.shape(whisker_bars)))
                starline_y = np.max(whisker_bars) + 1
            star_str = get_p_stars(p)
            plot.plot([1, 1, 2, 2], [starline_y+0.2, starline_y+0.4, starline_y+0.4, starline_y+0.2], linewidth=1, color='grey')
            plot.text(1.49, starline_y+0.5, star_str, ha='center', fontsize=12)
    if (set_ymax != None and max_y > set_ymax):
        txt='some outliers not shown'
        if (plot == plt):
            plot.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='right', fontsize=12)
    if set_ymax != None:
        if (plot == plt):
            plot.ylim(top=set_ymax)
        else:
            plot.set_ylim(top=set_ymax)
    if (plot == plt):
        plot.xlabel('Algorithm')
        plot.ylabel(y_label)
    else:
        plot.set_xlabel('Algorithm')
        plot.set_ylabel(y_label)
    return plot

def print_displacement_stats(dists, p_value=0.05):
    disp_stats = {}
    for algo in dists:
        mean = np.mean(dists[algo])
        stddev = np.std(dists[algo])
        min_dist = np.min(dists[algo])
        max_dist = np.max(dists[algo])
        disp_stats[algo] = ArrayStats(mean,min_dist,max_dist,stddev)
        norm_stat, norm_p = stats.shapiro(dists[algo])
        normal = 'yes'
        if norm_p < p_value:
            normal = 'no'
        print(algo + '\tcount: ' + str(len(dists[algo])) + '\n\t mean: ' + str(mean) + '\n\t min: ' + str(min_dist) + '\n\t max: ' + str(max_dist) + 
              '\n\t std: ' + str(stddev) + '\n\t normal (shapiro)? ' + normal)
    return disp_stats

def displacements_above_threshold(dists, threshold=5, selected_algos = select_algos, quiet=False):
    above = {}
    for algo in selected_algos:
        aboveThreshold = np.count_nonzero(dists[algo] > threshold)
        pct = aboveThreshold / len(dists[algo]) * 100.0
        if not quiet:
            print(algo + ':\n\t' + str(aboveThreshold) + ' out of ' + str(len(dists[algo])) + ', pct: ' + str(round(pct,2)))
        above[algo] = (aboveThreshold,pct)
    return above

def test_displacements_variances_equal(dists, algo_pairs=[select_algos], p_val=0.05):
    """ tests if the variances in the displacements are equal between the given pairs of algorithms
    Uses the Levene test with median centre
    Returns an array with the results from the test, False if the hypothesis that the variances are equal
    is rejected, True if it cannot be rejected given the p-value
    """
    equal_vars = []
    for algo_pair in algo_pairs:
        if len(algo_pair) != 2:
            print('Expected 2 elements, got an array of size' + str(len(algo_pair)))
            continue
        is_dict = type(dists) == dict
        idxs = [0,1]
        if is_dict:
            idxs = [algo_pair[0], algo_pair[1]]
        res = stats.levene(dists[idxs[0]], dists[idxs[1]])
        equal_vars.append(res.statistic >= p_val)
    return equal_vars

def test_displacements_means_diff_ind(dists, algo_pairs=[select_algos], p_val=0.05):
    """Tests if the means between the displacements of the given pairs of algorithms are different
    only valid for independent samples
    The means of each pair (a,b) are compared
    Uses the two-sided Student-t test if the variances between the two are the same, otherwise
    uses Welch's t-test (using scipy.stats.ttest_ind)
    Returns an array with the results from the test for each pair:
        * -1 : mean(a) < mean(b)
        * 0 : mean(a) = mean(b)
        * 1 : mean(a) > mean(b)
    """
    equal_vars = test_displacements_variances_equal(dists,algo_pairs)
    is_dict = type(dists) == dict
    idxs = [0,1]
    diff_means = []
    for i,algo_pair in enumerate(algo_pairs):
        if len(algo_pair) != 2:
            print('Expected 2 elements, got an array of size' + str(len(algo_pair)))
            continue
        if is_dict:
            idxs = [algo_pair[0], algo_pair[1]]
        diff,p = stats.ttest_ind(dists[idxs[0]],dists[idxs[1]],equal_var=equal_vars[i])
        print('p: ' + str(p))
        if p >= p_val:
            diff_means.append(0)
        else:
            diff_means.append(np.sign(diff))
    return diff_means

def test_displacements_means_diff_paired(dists, algo_pairs=[select_algos], p_val=0.05, quiet=False):
    """Tests if the displacements of the given pairs of algorithms are different
    Uses
      - two-sided paired sample test if both sets are normal (tested using
    Shapiro-Wilk)
      - otherwise uses Wilcoxon's signed rank test
    Returns an array with the results from the test for each pair:
        * -1 : mean(a) < mean(b)
        * 0 : mean(a) = mean(b)
        * 1 : mean(a) > mean(b)
    """
    is_dict = type(dists) == dict
    idxs = [0,1]
    diff_means = []
    p = None
    diff = None
    selected_test = ''
    for i,algo_pair in enumerate(algo_pairs):
        if len(algo_pair) != 2:
            print('Expected 2 elements, got an array of size' + str(len(algo_pair)))
            continue
        if is_dict:
            idxs = [algo_pair[0], algo_pair[1]]
        _, norm_p0 = stats.shapiro(dists[idxs[0]])
        _, norm_p1 = stats.shapiro(dists[idxs[1]])
        if (norm_p0 >= p_val and norm_p1 >= p_val):
            # Normal: use ttest_rel
            selected_test = 'Paired-t'
            diff,p = stats.ttest_rel(dists[idxs[0]],dists[idxs[1]])
        else: # Nonparametric Wilcoxon signed rank test
            selected_test = 'Wilcoxon'
            diffs = dists[idxs[0]] - dists[idxs[1]]
            res = stats.wilcoxon(x=dists[idxs[0]],y=dists[idxs[1]])
            diff = np.mean(diffs)
            p = res.pvalue
        if not quiet:
            print(f'{selected_test} p: {p}')

        if p >= p_val:
            diff_means.append(0)
        else:
            diff_means.append(np.sign(diff))
    return diff_means,selected_test,p

def get_delta_distances(coords):
    """Return the Euclidean distances between successive coordinates in coords
    """
    distances = []
    if (len(coords) < 2):
        warnings.warn("Not enough coordinates, returning empty list")
    else:
        prev_pt = coords[0]
        for i in range(1,len(coords)):
            distances.append(np.linalg.norm(coords[i] - prev_pt))
            prev_pt = coords[i]
    return np.array(distances)

def get_angle(v1, v2):
    """ Return angle between two vectors (degrees,radians)
    """
    v1_n = v1 / np.linalg.norm(v1)
    v2_n = v2 / np.linalg.norm(v2)
    dot = np.dot(v1_n,v2_n)
    rads = np.arccos(dot)
    degs = np.degrees(rads)
    return degs,rads

def get_angles_from_coords(cath_coords):
    """Return a dictionary containing angles between successive coordinates in cath_coords
    cath_coords: dictionary of algorithms to CathCoords tuple (distal, proximal) containing coordinates,
        as returned by get_locs_displacements
    return: {'algorithm':[angles],...}
    """
    algo_angles = {}
    for (algo, coord_tuple) in cath_coords.items():
        algo_angles[algo] = []
        assert(len(coord_tuple)==2)
        dist_coords, prox_coords = coord_tuple[0].coords, coord_tuple[1].coords
        assert(len(dist_coords) >= 2 and len(prox_coords) >= 2 and len(dist_coords)==len(prox_coords))
        for i in range(1,len(dist_coords)):
            v1 = dist_coords[i-1] - prox_coords[i-1]
            v2 = dist_coords[i] - prox_coords[i]
            algo_angles[algo].append(get_angle(v1,v2)[0])
        algo_angles[algo] = np.array(algo_angles[algo])
    return algo_angles

def get_unit_vectors(cath_coords):
    """Return a dictionary containing unit vectors representing the catheter tips
    cath_coords: dictionary of algorithms to CathCoords tuple (distal, proximal) containing coordinates,
        as returned by get_locs_displacements
    return: {'algorithm1':[v0,..],...}
    """
    algo_vecs = {}
    for (algo, coord_tuple) in cath_coords.items():
        algo_vecs[algo] = []
        assert(len(coord_tuple)==2)
        dist_coords, prox_coords = coord_tuple[0].coords, coord_tuple[1].coords
        assert(len(dist_coords) >= 2 and len(prox_coords) >= 2 and len(dist_coords)==len(prox_coords))
        for i in range(0,len(dist_coords)):
            tip_vec = dist_coords[i] - prox_coords[i]
            tip_vec = tip_vec / np.linalg.norm(tip_vec)
            algo_vecs[algo].append(tip_vec)
        algo_vecs[algo] = np.array(algo_vecs[algo])
    return algo_vecs

def get_angles_from_means(mean_vecs, tip_vecs):
    """Return tip angles from the mean tip angle
    Pass in mean unit vector & tip unit vector dictionaries:
    tip_vecs = {'algorithm':[[x0,y0,z0],...]}
    Returns dict of algo:[angles]
    """
    algo_angles = {}
    tip_vectors = []
    for (algo, tip_vectors) in tip_vecs.items():
        algo_angles[algo] = []
        mean_vec = mean_vecs[algo]
        for i in range(0,len(tip_vectors)):
            algo_angles[algo].append(np.degrees(np.arccos(np.dot(tip_vectors[i],mean_vec))))
        algo_angles[algo] = np.array(algo_angles[algo])
    return algo_angles