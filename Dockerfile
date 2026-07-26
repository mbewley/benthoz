# Pinned on purpose. `latest` silently moves the CUDA, Python and Jupyter versions
# underneath you, which is what made the original image unreproducible.
FROM cschranz/gpu-jupyter:v1.11_cuda-13.0_ubuntu-24.04_python-only

USER root
# pygraphviz compiles against the graphviz headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz graphviz-dev \
    && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

WORKDIR /home/jovyan/benthoz
COPY --chown=${NB_UID}:${NB_GID} . .

# Editable, so a mounted working copy takes precedence at runtime.
RUN pip install --no-cache-dir -e ".[dev,notebook]"

# The imagery is not in the image. Mount it at benthoz_data, e.g.
#   docker run --gpus all -p 8888:8888 \
#     -v /path/to/benthoz_data:/home/jovyan/benthoz/benthoz_data \
#     -v "$PWD":/home/jovyan/benthoz benthoz
