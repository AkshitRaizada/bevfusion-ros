conda init
source /root/.bashrc
conda activate bevfusion
python setup.py develop
conda install -c conda-forge mpi4py openmpi
