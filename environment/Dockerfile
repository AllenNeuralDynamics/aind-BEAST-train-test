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
