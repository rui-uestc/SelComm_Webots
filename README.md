# SelComm_Webots

This repository contains the codes for our paper, which has been accepted by RA-L and IROS 2021.

This is the implementation based on Webots. For Stage_ROS version, please refer to another [repository](https://github.com/George-Chia/SelComm_Stage).

## Requirement

- Python 3.6

- [ROS Melodic](http://wiki.ros.org/)

- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

- [Webots 2020b](https://cyberbotics.com/doc/blog/Webots-2020-b-release)

- [PyTorch 1.4](http://pytorch.org/)

  

## Setup

Set up your Pycharm according to https://cyberbotics.com/doc/guide/using-your-ide?tab-language=python#pycharm.



## How to train

Run the following command::

```
webots worlds/6robots_360lidar_pedestrian.wbt
cd ./controllers/pioneer_controller/
python run.py
```



## Video Demo

|             The global view             |
| :-------------------------------------: |
| <img src="docs/demo.gif" width="500" /> |

|        The local view        |
| :--------------------------: |
| ![local_1](docs/local_1.gif) |

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
