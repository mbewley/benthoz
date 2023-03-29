FROM cschranz/gpu-jupyter:v1.5_cuda-11.6_ubuntu-20.04_python-only
RUN mamba install -y -c conda-forge opencv pytest networkx pygraphviz pytorch-lightning black[d]
