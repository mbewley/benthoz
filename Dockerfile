FROM cschranz/gpu-jupyter:latest
RUN conda install -y -c conda-forge opencv pytest networkx pygraphviz pytorch-lightning black[d]
