# BEVFusion Setup
### Docker Installation
(Instructions provided for Ubuntu. For other operating systems refer [this](https://docs.docker.com/engine/install/).) :-
```
```bash
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
```
nvidia-docker run -it --shm-size 16g bevfusion /bin/bash
conda init
source /root/.bashrc
conda activate bevfusion
```
### Run code:-
6 cameras + LiDAR mode
```
python3 tools/visualize_ros.py
```
1 camera + LiDAR mode
```
python3 tools/visualize_ros_l1c.py
```
