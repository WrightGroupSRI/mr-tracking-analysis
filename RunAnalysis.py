#!/usr/bin/env python
# coding: utf-8

# # Active catheter tracking notebook: test run
# To run the full pipeline from raw data download through reconstruction and analysis requires resources that may not be available depending on your setup: about 5 GB storage and a few minutes on a single processor, with less time required for a multi-processor system. On resource-limited systems where storage and computation resources may not be available, skip to the [Analysis](#analysis) section, which uses precomputed outputs from the reconstruction and localization if the notebook-processed outputs are not available.
# 
# The notebook is broken down into:
# - [Download](#download) the raw tracking projections from each experiment
# - [Localize](#localize): Reconstruct signals and localize the coils using each algorithm
# - [Verify](#verify) the results against preprocessed data
# - [Analysis](#analysis): Run the static and dynamic analysis notebooks. **SKIP TO HERE** if you don't want to download the full 5GB dataset and reconstruct the raw data on your system.

# # Download dataset from Zenodo <a class="anchor" id="download"></a>
# 
# Download the raw data from the Active Tracking Dataset. For now, we download from the sandbox until the pipeline is fully tested:

# In[ ]:


#!zenodo_get 295280 -g raw.zip --sandbox -o /data
get_ipython().system('wget -P /data https://sandbox.zenodo.org/records/295280/files/raw.zip')


# In[ ]:


get_ipython().system('cd /data && unzip -qn raw.zip')


# In[ ]:


get_ipython().system('cd /code')


# **Warning** Download of the active tracking raw data requires approximately 5 GB of space. If you prefer, skip ahead to the analysis section which uses the precomputed outputs.

# # Run localization algorithms on raw data <a class="anchor" id="localize"></a>
# **Warning:** this step may be slow. To use precomputed outputs instead, skip to the analysis section.
# 
# Test cathy localize:
# - run "cathy localize" on raw projection data: this will generate coordinate output in text files from each of the localization algorithms
# - ex/ cathy localize -d 6 -p 7 input_dir output_dir
#     - runs the algorithm for coils 6 (distal) and 7 (proximal) for the projection files under "input_dir"
#     - output_dir will contain subdirectories for each of the algorithms (peak, centroid, centroid_around_peak, png, jpng)
#     - each algorithm subdirectory will contain a coordinate text output file for each coil
# 
# for this test:
# - input_dir: /data/raw/static/trackTest-13Dec2021-C306-Y0/1/FH512_noDither_gradSpoiled-2021-12-13T13_02_33.756
# - output_dir: /data/localize_c306-y0-1-fh

# In[ ]:


import csv
import pathlib
import os
import multiprocessing as mp
import shutil
import hashlib
from ipywidgets import IntProgress
from IPython.display import display
import numpy as np


# In[ ]:


note_dir = os.getcwd()
note_dir


# In[ ]:


mp.cpu_count()


# In[ ]:


get_ipython().system('cathy localize -d 6 -p 7 /data/raw/static/trackTest-13Dec2021-C306-Y0/1/FH512_noDither_gradSpoiled-2021-12-13T13_02_33.756 /data/localize_c306-y0-1-fh ')


# Test the equivalent regular python method, with option to set which algorithms to run:

# In[ ]:


import cathy.cli as cat


# The full list of raw data directories is in the included csv file.

# In[ ]:


recordings_csv = note_dir + '/data/meta/catheter_raw_recordings.csv'
prepend = '/data/'


# In[ ]:


procdir = prepend+'processed' # processed data should be here if download & localization work

def data_processed(procdir=procdir):
    return os.path.isdir(procdir) and len(os.listdir(procdir)) != 0


# The below cell will reconstruct all the raw data and run the localization algorithms using a single cpu: this will be slow. On non-binder instance: Skip this and run the next two cells to distribute the work across cpus & finish faster.

# %%time
# ```python
# 
# directory_list = [] # list of preprocessed & processed output directory tuples
# doRun = True # True to run cathy localize: ow/ will fill in the directory list but not run cathy
# 
# with open(recordings_csv,'r') as csvfile:
#     rdr = csv.DictReader(csvfile)
#     for row in rdr:
#         raw_dir = pathlib.PurePath(row['Input'])
#         source = pathlib.PurePath(prepend).joinpath(raw_dir)
#         distal = int(row['distal'])
#         proximal = int(row['proximal'])
#         dest = pathlib.PurePath(prepend).joinpath('processed').joinpath(pathlib.PurePath(*raw_dir.parts[1:]))
#         preproc = pathlib.PurePath(prepend).joinpath('preprocessed').joinpath(pathlib.PurePath(*raw_dir.parts[1:]))
#         directory_list.append( (preproc,dest))
#         print('source: ' + str(source))
#         print('dest: ' + str(dest))
#         if doRun:
#             os.makedirs(dest, exist_ok=True)
#             # !cathy localize -d {distal} -p {proximal} {source} {dest}
#             cat.run_localize(str(source), str(dest), distal, proximal, algos=['centroid_around_peak', 'jpng'])
# ```

# In[ ]:


TOL = 0.05
MAX_ITER = 32


# In[ ]:


def call_localize(args):
    assert(len(args[0]) == 6), "Parameter issue: " + str(args)
    src = args[0][0]
    dst = args[0][1]
    dist = args[0][2]
    prox = args[0][3]
    tolerance = args[0][4]
    max_iter = args[0][5]
    print("run src: " + str(src) + ", dest: " + str(dst) + ", distal: " + str(dist) + ", proximal: " + str(prox) +
         ", tol(mm): " + str(tolerance) + ", max iterations: " + str(max_iter) + "\n")
    return cat.run_localize(src, dst, dist, prox, algos=['centroid_around_peak', 'jpng'],output_iterations=True,
                           tol=tolerance, max_iterations=max_iter)
    


# In[ ]:


get_ipython().run_cell_magic('time', '', '# list of preprocessed & processed output directories and coil args\nexperiment_data_tuples = {\'static\':[], \'dynamic\':[], \'vivo\': []}\n\ndoRun = True # True to run cathy localize: ow/ will fill in the data_tuples but not run localize\n\nnew_dir_list = [] #compare reorg processed directory to repo preprocessed directory\nrepo_dir = note_dir + \'/data/\'\n\ncpu_limit_env = os.getenv("CPU_LIMIT")\nif cpu_limit_env is None:\n    cpu_count = mp.cpu_count()\nelse:\n    cpu_count = cpu_limit_env\n# set up the localize arguments based on the spreadsheet:\nwith open(recordings_csv,\'r\') as csvfile:\n    rdr = csv.DictReader(csvfile)\n    for row in rdr:\n        raw_dir = pathlib.PurePath(row[\'Input\'])\n        expmt = \'dynamic\'\n        if \'static\' in str(raw_dir):\n            expmt = \'static\'\n        elif \'vivo\' in str(raw_dir):\n            expmt = \'vivo\'\n        source = pathlib.PurePath(prepend).joinpath(raw_dir)\n        distal = int(row[\'distal\'])\n        proximal = int(row[\'proximal\'])\n        dest = pathlib.PurePath(prepend).joinpath(\'processed\').joinpath(pathlib.PurePath(*raw_dir.parts[1:]))\n        experiment_data_tuples[expmt].append( (str(source), dest, distal, proximal, TOL, MAX_ITER))\n        preproc = pathlib.PurePath(repo_dir).joinpath(\'preprocessed\').joinpath(pathlib.PurePath(*raw_dir.parts[1:]))\n        if os.path.isdir(source):\n            new_dir_list.append( (preproc,dest))\n            os.makedirs(dest, exist_ok=True)\n        else:\n            raise Exception("Missing raw data: Please skip directly to the Analysis section near the end of this notebook")\n\n# data_tuples contains our arguments: we can split the processing across cpus for performance\nprint("Running localize on " + str(len(experiment_data_tuples[\'static\'])) + " static directories, " \\\n     + str(len(experiment_data_tuples[\'dynamic\'])) + " dynamic directories, and " + \\\n      str(len(experiment_data_tuples[\'vivo\'])) + " in vivo directories")\npool = mp.Pool(mp.cpu_count())\n\niterations_overall = {\'jpng\':[]} # accumulate iterations required for each recording\n\niterations_xpmt = { \'static\':{\'jpng\':[]}, \'dynamic\':{\'jpng\':[]}, \'vivo\':{\'jpng\':[]}} #separated by experiment type\n\nif doRun:\n    for xkey in experiment_data_tuples.keys():\n        print("Experiment: " + xkey)\n        for result in pool.map( call_localize, [experiment_data_tuples[xkey][i:i+1] for i in range(0,len(experiment_data_tuples[xkey]))] ):\n            for key in result.keys():\n                if key in iterations_overall:\n                    iterations_overall[key] = np.concatenate((iterations_overall[key],result[key]))\n                    iterations_xpmt[xkey][key] = np.concatenate( (iterations_xpmt[xkey][key],result[key]) )')


# In[ ]:


if data_processed():
    for key in iterations_overall.keys():
        print(f"{key}: mean iterations {np.mean(iterations_overall[key])}\t, median {np.median(iterations_overall[key])},        min {np.min(iterations_overall[key])}, max {np.max(iterations_overall[key])}")
        print(f"percentage of iterations reaching 32: {np.count_nonzero(iterations_overall[key] == 32)/len(iterations_overall[key]) * 100:.2f}")


# In[ ]:


if data_processed():
    for xkey in iterations_xpmt:
        print(f"Experiment set: {xkey}")
        for key in iterations_xpmt[xkey]:
            print(f"\t{key}: mean iterations {np.mean(iterations_xpmt[xkey][key])}\t,             median {np.median(iterations_xpmt[xkey][key])},            min {np.min(iterations_xpmt[xkey][key])}, max {np.max(iterations_xpmt[xkey][key])}")
            print(f"\tpct iterations reaching 32: {np.count_nonzero(iterations_xpmt[xkey][key]==32)/len(iterations_xpmt[xkey][key]) * 100:.2f}")
        


# In[ ]:


import os
print(os.cpu_count())


# # Copy ground truth data to processed directory
# Ground truth files are expected in the processed directory for later analysis. These should be copied from the raw subdirectories to the corresponding processed subdirectories. A list of these files is in meta/gt_files.txt

# In[ ]:


if data_processed():
    gtf = open(note_dir + '/data/meta/gt_files.txt','r')
    gt_list = gtf.readlines()
    for gt_file in gt_list:
        gt_path = pathlib.PurePath(gt_file.rstrip())
        src = pathlib.PurePath(prepend).joinpath(gt_path)
        dest = pathlib.PurePath(prepend).joinpath('processed').joinpath(pathlib.PurePath(*gt_path.parts[1:]))
        dest_dir = pathlib.PurePath(*dest.parts[:-1])
        print('dest dir: ' + str(dest_dir))
        os.makedirs(dest_dir, exist_ok=True)
        print('cp ' + str(src) + ' ' + str(dest))
        shutil.copyfile(src,dest)
    


# # Verify processed outputs <a class="anchor" id="verify"></a>
# Check the processed outputs against the preprocessed. Initial check below will compare files in detail to ensure this method works - the original data only stored an SNR of 0 so we need to ignore this field until we can verify the rest of the data.
# 
# Note we have updated the preprocessed coordinates with output using the current algorithm parameters: the next cell is redundant as we have the checksums in a later step.

# In[ ]:


#```python
import catheter_utils.cathcoords
import glob
import numpy as np

algos = ['centroid_around_peak','jpng']

error_count = 0
EPS = 0.1 #0.001

def distance(c1, c2):
    return np.linalg.norm(c2-c1)

proc_file_list = [] # This will contain a list of the output files
distances = []
for src_test, dst_test in new_dir_list:
    for algo in algos:
        src_dir = pathlib.PurePath(src_test).joinpath(algo)
        dst_dir = pathlib.PurePath(dst_test).joinpath(algo)
        src_files = glob.glob(str(src_dir)+'/cathcoords*.txt')
        for src in src_files:
            srcpath = pathlib.PurePath(src)
            fname = srcpath.parts[-1]
            dstpath = dst_dir.joinpath(fname)
            if (os.path.isfile(dstpath)):
                cc_src = catheter_utils.cathcoords.read_file(srcpath)
                #print("cc_src: " + str(cc_src))
                cc_dst = catheter_utils.cathcoords.read_file(dstpath)
                proc_file_list.append(dstpath)
                # Check each xyz coordinate
                src_coords = cc_src.coords
                dst_coords = cc_dst.coords
                dists = list(map(distance, src_coords, dst_coords))
                times_equal = np.array_equal(cc_src.times,cc_dst.times)
                trigs_equal = np.array_equal(cc_src.trigs,cc_dst.trigs)
                resps_equal = np.array_equal(cc_src.resps,cc_dst.resps)
                if (any(i > EPS for i in dists) or not times_equal or not trigs_equal or not resps_equal):
                    print("Mismatch in: " + str(src))
                    error_count += 1                    
                else:
                    print(".", end = "")
            else:
                print("Missing: " + str(dstpath))
                error_count += 1
print("\nErrors: " + str(error_count))
#```


# # Hash of files
# The processed files should match with the preprocessed data. A checksum file "md5sums.txt" was created for all the cathcoords outputs.

# In[ ]:


def generate_md5(filename, chunk_size=4096):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        # digest = hashlib.file_digest(f, digest) Python >= 3.11 reqd!
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
        return md5.hexdigest()


# ## This was already run to create the md5sums file

# ```python
# proc_md5_sum = repo_dir + 'meta/proc_md5sums.txt'
# with open(proc_md5_sum, "w") as f_out:
#     for proc_file in proc_file_list:
#         md5_string = generate_md5(str(proc_file))
#         f_out.write("{} : {}\n".format(str(proc_file), md5_string))
# ```

# ## Verify processed files against md5sums

# In[ ]:


preproc_md5_sum = note_dir + '/data/meta/proc_md5sums.txt'
error_count = 0

with open(preproc_md5_sum, "r") as f:
    reader = csv.reader(f, delimiter=':')
    records = sum(1 for row in reader)
    f.seek(0)
    bar = IntProgress(min=0, max=records,description='Verifying:')
    display(bar)
    for row in reader:
        proc_file = row[0].strip()
        proc_digest = row[1].strip()
        if (not os.path.isfile(proc_file)):
            print('Coordinate file ' + proc_file + ' missing')
            error_count += 1
        else:
            digest = generate_md5(proc_file)
            bar.value += 1
            if (digest != proc_digest):
                print('Coordinate file ' + proc_file + ' mismatch')
                error_count += 1

print('\nErrors: ' + str(error_count))


# In[ ]:


cpu_limit = os.environ.get("CPU_LIMIT")


# In[ ]:


if cpu_limit is None:
    cpu_limit = os.cpu_count()


# In[ ]:


cpu_limit


# In[ ]:


def get_cpu_quota_within_docker():
    cpu_cores = None
    cfs_period = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        with cfs_period.open('rb') as p, cfs_quota.open('rb') as q:
            p, q = int(p.read()), int(q.read())
            # get the cores allocated by dividing the quota
            # in microseconds by the period in microseconds
            cpu_cores = math.ceil(q / p) if q > 0 and p > 0 else None
    return cpu_cores


# In[ ]:


cpu_cores = get_cpu_quota_within_docker()


# In[ ]:


cpu_cores


# In[ ]:


get_ipython().system('cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us')


# In[ ]:


get_ipython().system('cat /sys/fs/cgroup/cpu/cpu.cfs_period_us')


# # Analysis <a class="anchor" id="analysis"></a>
# The notebooks are in subdirectories. We call each notebook in turn below.
# ## Static Tracking

# In[ ]:


get_ipython().run_line_magic('cd', 'Static_Tracking')
get_ipython().run_line_magic('run', 'static_tracking_heatmaps_Y0.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', 'static_tracking_heatmaps_Y1.ipynb')


# In[ ]:


get_ipython().run_line_magic('run', 'static_tracking_heatmaps_Y2.ipynb')


# In[ ]:


get_ipython().run_line_magic('cd', '../Dynamic_Tracking')
get_ipython().run_line_magic('run', 'dynamic_tracking_analysis.ipynb')


# In[ ]:


get_ipython().run_line_magic('cd', '../Invivo_Tracking')
get_ipython().run_line_magic('run', 'invivo_tracking_analysis.ipynb')


# In[ ]:


get_ipython().system('ls /code/')


# In[ ]:


# Reset to main directory
get_ipython().run_line_magic('cd', '/code')

