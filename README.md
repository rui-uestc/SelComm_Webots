# SelComm_Webots

This repository contains the codes for our paper, which is in submission to RA-L and IROS 2021.

This is the implementation based on Webots. For Stage_ROS version, please refer to another [repository](https://github.com/George-Chia/SelComm_Stage).

## Requirement

- Python 3.6

- [ROS Melodic](http://wiki.ros.org/)

- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

- [Webots 2020b](https://cyberbotics.com/doc/blog/Webots-2020-b-release)

- [PyTorch 1.4](http://pytorch.org/)

  

## Setup

- #### Use Python3 and tf in ROS

```shell
sudo apt-get install python3-dev 
mkdir -p catkin_ws_py3/src
cd catkin_ws_py3/src  
git clone https://github.com/ros/geometry 
git clone https://github.com/ros/geometry2 
cd .. 
virtualenv -p /usr/bin/python3 venv 
source venv/bin/activate 
pip install catkin_pkg pyyaml empy rospkg numpy 
catkin_make --cmake-args -DPYTHON_VERSION=3.6
source devel/setup.bash
```



## How to train

Run the following command::

```
webots 6robots_360lidar_pedestrian.wbt
cd ./pioneer_controller
python run.py
```



## Video Demo

|    Test in the human-crowd scenario     |
| :-------------------------------------: |
| <img src="docs/demo.gif" width="500" /> |

## References

 The authors thank Liu for the open sourced code.

```
@misc{Tianyu2018,
	author = {Tianyu Liu},
	title = {Robot Collision Avoidance via Deep Reinforcement Learning},
	year = {2018},
	publisher = {GitHub},
	journal = {GitHub repository},
	howpublished = {\url{https://github.com/Acmece/rl-collision-avoidance.git}},
	commit = {7bc682403cb9a327377481be1f110debc16babbd}
}
```
