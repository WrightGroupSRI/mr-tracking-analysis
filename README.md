# MR Active Tracking Tests

This collects the analysis for catheters in the MR environment tracked under static, dynamic, and in vivo conditions. Two tracking (or localization) algorithms are compared: centroid-around-peak (CAP) and Joint Peak-Normalized Gaussian (JPNG).

These experiments were performed at Sunnybrook Research Institute on a 1.5T MRI scanner.

Details of the setup and experiment can be found in the publication (LINK TO COME).

# How to run the analysis
The easiest way to run the analysis is through binder. You can also run the analysis using a docker image on your local machine.

The *RunAnalysis* notebook downloads and reconstructs the raw data, then calls the individual notebooks for the static, dynamic, and in vivo experiments in turn.

## Get the Data
The full dataset can be found on zenodo (LINK TO COME). This includes the raw MR projection data, which requires about 4.8 GB storage space. **You don't need to download the raw data manually**: this step is included in the notebook. You can skip the data download step by skipping the download cell.

You can run the analysis starting from the raw data or from the preprocessed data available in this repository.

The preprocessed data has been reconstructed from the raw recordings, and the localization algorithms have run on the reconstructed data to compute the tracked coordinates.

## Binder
- [ ] To be tested
## Docker
- [ ] To document
# Code
The notebooks and other code in this repository organize and graph the results from the experiments.

The code for tracking data handling and localization is available in the [cathy](https://github.com/WrightGroupSRI/cathy) suite, a set of Python packages. The localization algorithms are in the localization module of the [catheter_utils](https://github.com/WrightGroupSRI/catheter_utils) package.