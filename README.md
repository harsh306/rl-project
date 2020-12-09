### The rl-project [Video](https://www.youtube.com/watch?v=O2swIIuimZA)
RL project at WPI for ML (CS 539) class by Prof. Kyumin Lee

### Project Idea/Proposal [Link](https://github.com/harsh306/rl-project/tree/master/prj_images/docs)


### Literature Survey
Our research is inspired by these two papers.

Progressive GANs [Link](https://arxiv.org/abs/1710.10196)

Author's Abstract: We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. 

![Progressive GANs](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/progan.png)

Teacher Student Curriculum Learning [Link](https://arxiv.org/pdf/1707.00183.pdf)

Author's Abstract: We propose Teacher-Student Curriculum Learning (TSCL), a framework for automatic curriculum learning, where the Student tries to learn a complex task and the Teacher automatically chooses subtasks from a given set for the Student to train on. We describe a family of Teacher algorithms that rely on the intuition that the Student should practice more those tasks on which it makes the fastest progress, i.e. where the slope of the learning curve is highest. In addition, the Teacher algorithms address the problem of forgetting by also choosing tasks where the Student's performance is getting worse. We demonstrate that TSCL matches or surpasses the results of carefully hand-crafted curricula in two tasks: addition of decimal numbers with LSTM and navigation in Minecraft.

![TSCL](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/tscl.png)


### Curriculum startegies 

We leveraged the idea from TSCL method. Teacher is a sampling method that selects the difficulty level of the given game. 
And Student is our DQN agent (NN model),   
#### Pacman

![pacman0](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/pacman/pac0.png)
![pacman1](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/pacman/pac1.png)
![pacman2](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/pacman/pac2.png)

#### Coinrun [Blog](https://openai.com/blog/quantifying-generalization-in-reinforcement-learning/)
Coinrun have multiple levels of games, and here also teacher can select what level should be provided to the student agent to learn based on TSCL stratergies. 

![Coinrun](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/coin_run/coirun2.png)



### Experiments
#### Pacman
Standalone agent learning from 1, 2, and 3 enemy in 5x5 grid in the plots below respectively. 

In y-axis there is the mean-avg reward [100-window]. x-axis is the number of episodes. 

![One](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/1enemy.png)

In y-axis there is the mean-avg reward [100-window]. x-axis is the number of episodes. 
![Two](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/2enemy.png)

In y-axis there is the mean-avg reward [100-window]. x-axis is the number of episodes. 
![Three](https://raw.githubusercontent.com/harsh306/rl-project/master/prj_images/3enemy.png)


#### Coinrun
The mean episode reward for TSCL:

![mer](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/mean_reward.svg)

The probability of each action:
- Action 1
![action1](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param1_mean.svg)

- Action 2
![action2](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param2_mean.svg)

- Action 3
![action3](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param3_mean.svg)

- Action 4
![action4](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param4_mean.svg)


- Action 5
![action5](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param5_mean.svg)


- Action 6
![action5](https://github.com/harsh306/rl-project/blob/master/prj_images/coin_run/actions_param6_mean.svg)

### About the team
#### Harsh Nilesh Pathak [[Bio]](https://sites.google.com/view/harshnpathak/research)

Currently, I am Data Scientist at Expedia and also a Ph.D. student at WPI. I am doing my research on continuation methods for Deep Learning Optimization with Prof. Randy Paffenroth.
I have worked on a diverse set of applications of computer vision for example, GANs, Image classification, compression and object detection. Also, in NLP I have done industry projects of one plus year length.
These include Text classification, Named Entity Recognition, Text similarity and Learning to rank frameworks.

### Yichuan Li
Yichuan Li is a Ph.D. student at Worcester Polytechnic Institue, now he is affiliated with Infolabs under the supervision of Prof. Kyumn Lee.
Research interests: Data Mining, Machine Learning, Social Computing.

### Thejus Jose

I am a Masters student in the Robotics Engineering program at Worcester Polytechnic Institute. My current research is focused on robot learning through human demonstration under the guidance of Prof. Jane Li. In the past, I have worked on system identification, motion planning and human robot interaction

### Paurvi Dixit
Master's Student at Worcester Polytechnic Institue.

### Qihuan Aixinjueluo 
Student at Worcester Polytechnic Institue.

## Extras
- How to use Github and usual flow, please read this [Link](https://guides.github.com/introduction/git-handbook/)
- Github Setup [Link](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
- Prefer using Pycharm Community for Free [Link](https://www.jetbrains.com/pycharm/download/)   
