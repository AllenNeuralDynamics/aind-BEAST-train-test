# hash:sha256:c5f04b4e610b821077da854c7308e2c349833c168226c0c73c70dbdf68b1c064
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# BEAST needs Python >=3.10; this CUDA base ships an older Python.
RUN conda install -y python=3.11 \
    && conda clean -ya

# ffmpeg for `beast extract` / video I/O; libsm6/libxext6 for OpenCV.
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# BEAST CLI (provides the `beast` entry point used by run_capsule.py).
# Installed editable from source on purpose: the published `beast-backbones`
# wheel crashes at runtime because beast/__init__.py looks up
# importlib.metadata.version('beast') (distribution is named `beast-backbones`)
# and then falls back to a pyproject.toml that isn't shipped in the wheel. An
# editable install from the cloned repo keeps pyproject.toml next to the
# package so version resolution succeeds.
# Pinned to the commit validated end-to-end (2026-06-13). Bump deliberately.
ARG BEAST_COMMIT=2e56b117fb5457e6f436906789ed75d1764bf47b
RUN git clone https://github.com/paninski-lab/beast.git /opt/beast \
    && git -C /opt/beast checkout ${BEAST_COMMIT} \
    && pip install -e /opt/beast

# JupyterLab so Code Ocean Cloud Workstations can launch. This hand-edited
# Dockerfile blocks CO's automatic IDE install (-> "jupyter: command not found",
# "JupyterLab interactive session port not exposed"), so install it explicitly.
# matplotlib is needed by code/analyze_latents.ipynb and isn't a BEAST dep;
# pyarrow lets pandas read the processed .parquet keypoint/kinematics/event
# assets; pandas/scikit-learn ship with BEAST but are pinned here so the
# notebook runs regardless of upstream dependency changes.
RUN pip install --no-cache-dir jupyterlab matplotlib pandas scikit-learn pyarrow

# Video time-base alignment helpers (session_time <-> video_time <-> behavior_time)
# used by analyze_latents.ipynb to map keypoint event tables onto latent frames.
# Zero runtime deps (pandas only). Pinned to the video_alignment feature branch
# commit; revisit once it merges to main.
RUN pip install --no-cache-dir \
    "git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis.git@640cdca3ea170eccbffe6e327454341454d3f2f1"
