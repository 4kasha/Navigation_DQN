[//]: # (Image References)

[image1]: ./media/Bnana1.gif "Trained Agent"


# Navigation via DQN - PyTorch implementation

## Introduction

In this project, using a Unity ML-Agents environment you will train an agent to navigate (and collect bananas!) in a large, square world via DQN, Double DQN and Dueling algorithms.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Dependencies

- Python 3.6
- PyTorch 0.4.0
- ML-Agents Beta v0.4

**NOTE** : (_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

## Getting Started

1. Create (and activate) a new environment with Python 3.6 via Anaconda.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name your_env_name python=3.6
	source activate your_env_name
	```
	- __Windows__: 
	```bash
	conda create --name your_env_name python=3.6 
	activate your_env_name
	```

2. Clone the repository, and navigate to the python/ folder. Then, install several dependencies (see `requirements.txt`).
    ```bash
    git clone https://github.com/4kasha/Navigation_DQN.git
    cd Navigation_DQN/python
    pip install .
    ```

3. Download the Unity environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

    **NOTE** : For this project, you will not need to install Unity. The link above provides you a standalone version. Also the above Banana environment is similar to, but **not identical to** the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

4. Place the file in this repository _Navigation_DQN_ and unzip (or decompress) the file. 

## Instructions

- Before running code, change parameters in `train.py`, especially you must change `env_file_name` according to your environment.
- Run the following command to get started with training your own agent!
    ```bash
    python train.py
    ```
- After finishing training weights and scores are saved in the following folder `weights` and `scores` respectively. 


## Tips

- For more details of algolithm description, hyperparameters settings and results, see [Report.md](https://github.com/4kasha/Navigation_DQN/Report.md).
- For the examples of training results, see [Navigation_Results_Example.ipynb](Navigation_Results_Example.ipynb).
- After training you can test the agent with saved weights in the folder `weights`, see [Navigation_Watch_Agent.ipynb](Navigation_Watch_Agent.ipynb). 
- This project is a part of Udacity's [Deep Reinforcement Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

