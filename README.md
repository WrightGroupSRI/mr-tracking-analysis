# MR Active Tracking Tests

This collects the analysis for catheters in the MR environment tracked under static, dynamic, and in vivo conditions. Two tracking (or localization) algorithms are compared: centroid-around-peak (CAP) and Joint Peak-Normalized Gaussian (JPNG).

These experiments were performed at Sunnybrook Research Institute on a 1.5T MRI scanner.

Details of the setup and experiment can be found in the publication (LINK TO COME).

# How to run the analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WrightGroupSRI/mr-tracking-analysis/HEAD?urlpath=%2Fdoc%2Ftree%2FRunAnalysis.ipynb)

The easiest way to run the analysis is through binder, but currently the full pipeline with raw data reconstruction cannot run on binder: the notebook disconnects partway through. **Instead, on binder, you can start running the notebook from the Analysis section.**

You can also run the analysis using a docker image on your local machine.

The *RunAnalysis* notebook downloads and reconstructs the raw data, then calls the individual notebooks for the static, dynamic, and in vivo experiments in turn.

## Get the Data
The full dataset can be found on zenodo (LINK TO COME). This includes the raw MR projection data, which requires about 4.8 GB storage space. **You don't need to download the raw data manually**: this step is included in the notebook. You can skip the data download step and use the preprocessed data, included in the repository, by skipping the download cell and going to the Analysis section.

You can run the analysis starting from the raw data or from the preprocessed data available in this repository.

The preprocessed data has been reconstructed from the raw recordings, and the localization algorithms have run on the reconstructed data to compute the tracked coordinates.

## Binder
Times out during run: may be exhausting resource limits. Workaround:

- Turn off Settings / Autosave Documents
- Skip to the Analysis section
- Select the Analysis cell and click Run / Run Selected Cell and All Below
## Docker
The docker image will be available from dockerhub (LINK TO COME)
### Build
To build the docker image yourself, you need docker installed. Download or clone the sources. Then, from the terminal in the code directory:
```bash
export DOCKER_BUILDKIT=1
docker build -t mr-tracking .
```
### Run
- From the terminal
```bash
docker run --rm -p 8888:8888 mr-tracking:latest jupyter notebook --ip 0.0.0.0 --no-browser
```
- Open the 127.0.0.1 link shown in the terminal in your web browser
- Open and run **RunAnalysis.ipynb** from the jupyter interface
- From the "Cell" menu, select "Run All" to run the analysis
# Code
The notebooks and other code in this repository organize and graph the results from the experiments.

The code for tracking data handling and localization is available in the [cathy](https://github.com/WrightGroupSRI/cathy) suite, a set of Python packages. The localization algorithms are in the localization module of the [catheter_utils](https://github.com/WrightGroupSRI/catheter_utils) package.
