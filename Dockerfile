################################################
# Builds the mr-tracking docker image
# for reproducible analysis
#
# Run the following commands first:
# 1. eval "$(ssh-agent)"
# 2. ssh-add
# 3. export DOCKER_BUILDKIT=1
#
# Then build the image with:
# docker build --ssh default -t mr-tracking:x.y .
#
# To run the analysis:
# 1. From the terminal:
#    docker run --rm -p 8888:8888 mr-tracking:latest jupyter notebook --ip 0.0.0.0 --no-browser
# 2. Open the 127.0.0.1 link shown in the terminal in your web browser 
# 3. Open and run RunAnalysis.ipynb from the jupyter interface
################################################

FROM mambaorg/micromamba:1.5.8-alpine3.20
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

USER root
RUN apk add --no-cache \
    openssh git \
    tzdata mesa-gl \
    glib libsm-dev libxrender libxext-dev

USER $MAMBA_USER

# ssh keys required for private github repo access
RUN mkdir ~/.ssh && ssh-keyscan -H github.com > ~/.ssh/known_hosts

WORKDIR /opt/dock-cat/

ARG MAMBA_DOCKERFILE_ACTIVATE=1 # For pip and python commands to work

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/catheter_utils.git catheter_utils
RUN pip install -e catheter_utils

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/catheter_ukf.git catheter_ukf
RUN pip install -e catheter_ukf

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/dicom_utils.git dicom_utils
RUN pip install -e dicom_utils

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/dicom_art.git dicom_art
RUN pip install -e dicom_art

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/get_gt.git get_gt
RUN pip install -e get_gt

RUN --mount=type=ssh,mode=0666 git clone git@github.com:WrightGroupSRI/cathy.git cathy
RUN pip install -e cathy

COPY <<EOF /home/$MAMBA_USER/.jupyter/jupyter_notebook_config.py
c = get_config()
### If you want to auto-save .html and .py versions of your notebook:
# modified from: https://github.com/ipython/ipython/issues/8009
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['jupyter', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(['jupyter', 'nbconvert', '--to', 'html', fname], cwd=d)

c.FileContentsManager.post_save_hook = post_save
EOF

USER root
RUN mkdir /data
RUN mkdir /code
RUN chown -R $MAMBA_USER:$MAMBA_USER /home/$MAMBA_USER/.jupyter /data /code
RUN chmod a+w /home/$MAMBA_USER/.jupyter
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER . /code/
WORKDIR /code
