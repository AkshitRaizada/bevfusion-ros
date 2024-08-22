# BEVFusion Setup
### Docker Installation
(Instructions provided for Ubuntu. For other operating systems refer [this](https://docs.docker.com/engine/install/).) :-
### Add Docker's official GPG key:
```
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
**For nvidia-docker:-**
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo systemctl daemon-reload
sudo systemctl restart docker
```
**Configure nvidia-docker**
```
sudo nano /etc/docker/daemon.json
```
And replace all the text in it with
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
For some systems, if nvidia-smi is not working inside the Docker container, you may have to run this:-
```
sudo nano /etc/nvidia-container-runtime/config.toml, then changed no-cgroups = false
```
and change line 13 ```no-cgroups = true``` to ```no-cgroups = false```
Restart docker to see effect of these changes
```
sudo systemctl restart docker
```
### BEVFusion Installation :-
```
git clone https://github.com/AkshitRaizada/bevfusion-ros.git
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
│   │   │   ├── basemap
│   │   │   ├── expansion
│   │   │   ├── prediction
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-mini
```
In the example above, v1.0-mini contains files from Nuscenes Full Dataset(v1.0) and maps contains folders from Map expansion pack (v1.3) merged with the maps folder from v1.0-mini. All of these are available on the aforementioned website.
```
cd bevfusion
nvidia-docker run -it -v `pwd`/data:/usr/src/app/bevfusion/data --shm-size 16g bevfusion /bin/bash
```
If you have a bag file instead of Nuscenes dataset, you can put it in this folder and it will appear in your image.


**Viewable GUI Applications like rqt, rviz, etc.**
```
curl -fsSL https://raw.githubusercontent.com/mviereck/x11docker/master/x11docker | sudo bash -s -- --update
xhost +local:docker
nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v `pwd`/data:/usr/src/app/bevfusion/data --shm-size 16g bevfusion /bin/bash
```
### Setup Environment 
```
conda init
source /root/.bashrc
conda activate bevfusion
python setup.py develop
conda install -c conda-forge mpi4py openmpi
```
**Dataset Preparation** - If you mounted Nuscenes(or custom data in Nuscenes format), you need to run these commands once(these dependencies are specific to dataset preparation and training):-
```
apt-get install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
nano /etc/apt/sources.list
```
Add the following lines to the bottom and save file:-
```
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
```
Run the following in the terminal:-
```
apt-get update
apt-get install gcc-4.9
apt-get upgrade libstdc++6
apt-get dist-upgrade

python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```
### For multiple terminals:-
```
tmux
```
Press `CTRL+b "` to split horizontally or `CTRL+b %` to split vertically. `CTRL+b [→, ←, ↑, ↓]` to switch between windows.
Refer [link](https://www.shells.com/l/en-US/tutorial/Installing-and-using-tmux-on-Ubuntu-20-04) if you need more commands.
### Run code:-
Install pretrained weights if needed:-
```
./tools/download_pretrained.sh
```
 - Training(you must install the dataset preparation dependencies mentioned above before running these commands)
```
pip install yapf==0.40.1 setuptools==59.5.0
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth 
```
 - 6 cameras + LiDAR mode
```
python3 tools/visualize_ros.py
```
Publish Nuscenes data
```
python3 tools/ros_publisher.py
rqt
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
After launching rqt, you can press Perspectives>Import, then select the one of the .perspective files given in the repository
