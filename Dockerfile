FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y
RUN apt-get install nano tmux -y
ENV LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV PATH=/usr/local/cuda-11.3/bin:$PATH
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN mkdir -p /usr/src/app/bevfusion
WORKDIR /usr/src/app/bevfusion
COPY ./ ./
RUN conda env create -f environment.yml -y
RUN echo "Conda env - ready!"
SHELL ["conda", "run", "-n", "bevfusion", "/bin/bash", "-c"]
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install python=3.8 -y
#RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -y
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
RUN pip install nuscenes-devkit
#RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
RUN pip install numpy==1.21.0
RUN pip install -r requirements.txt
RUN apt-get install ninja-build -y
#RUN export MMCV_WITH_OPS=1
#RUN pip install -v -e .
#RUN export CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
#RUN export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
#RUN python setup.py develop
