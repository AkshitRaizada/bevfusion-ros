REMOVE force=True if mmcv installed properly
# BEVFusion Setup
### Docker Installation
(Instructions provided for Ubuntu. For other operating systems refer [this](https://docs.docker.com/engine/install/).) :-
```
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-docker2
```
### BEVFusion Installation :-
```
git clone [AIRL URL]
cd bevfusion
docker build . -t bevfusion
```
### Run Container:-
**Regular run**
```
nvidia-docker run -it --shm-size 16g bevfusion /bin/bash
```
or
**Mount [Nuscenes dataset](https://www.nuscenes.org/download).** Note that this must be done after running the docker build command otherwise it will copy the dataset into the image which wastes space.
Arrange the dataset into a data folder exactly like this:-
```
bevfusion
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-mini
```
```
cd bevfusion
nvidia-docker run -it -v `pwd`/data:/usr/src/app/bevfusion/data --shm-size 16g bevfusion /bin/bash
```
If you have a bag file instead of Nuscenes dataset, you can put it in this folder and it will appear in your image.
### Setup Environment 
```
conda init
source /root/.bashrc
conda activate bevfusion
python setup.py develop
conda install -c conda-forge mpi4py openmpi
```
If you mounted Nuscenes(or custom data in Nuscenes format), you need to run this command once:-
```
python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```
### For multiple terminals:-
```
tmux
```
Press `CTRL+b "` to split horizontally or `CTRL+b %` to split vertically. `CTRL+b ;` to switch between windows.
Refer [link](https://www.shells.com/l/en-US/tutorial/Installing-and-using-tmux-on-Ubuntu-20-04) if you need more commands.
### Run code:-

 - 6 cameras + LiDAR mode
```
python3 tools/visualize_ros.py
```
Publish Nuscenes data
```
python3 tools/ros_publisher.py
```

- 1 camera + LiDAR mode
```
python3 tools/visualize_ros_l1c.py
```
Run ROS bag file
```
rosbag play data/[bagName].bag
rqt
```
