# Pinned on purpose. `latest` silently moves the CUDA, Python and Jupyter versions
# underneath you, which is what made the original image unreproducible.
#
# The toolbox itself has no GPU-dependent code, so the CUDA line here is a
# reproducibility default rather than a requirement. If it doesn't suit your
# driver or GPU generation, swap the tag for any other maintained one:
#   https://hub.docker.com/r/cschranz/gpu-jupyter/tags
FROM cschranz/gpu-jupyter:v1.11_cuda-13.0_ubuntu-24.04_python-only

USER root
# pygraphviz compiles against the graphviz headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz graphviz-dev \
    && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

# Pinned to match requires-python in pyproject.toml. A no-op if the base image
# already ships 3.14; without it, a base on an older Python fails the install.
RUN mamba install -y -c conda-forge python=3.14 && mamba clean -afy

WORKDIR /home/jovyan/benthoz
COPY --chown=${NB_UID}:${NB_GID} . .

# Editable, so a mounted working copy takes precedence at runtime.
RUN pip install --no-cache-dir -e ".[dev,notebook,graphviz]"

# The imagery is not in the image. Mount it at benthoz_data, e.g.
#   docker run --gpus all -p 8888:8888 \
#     -v /path/to/benthoz_data:/home/jovyan/benthoz/benthoz_data \
#     -v "$PWD":/home/jovyan/benthoz benthoz
