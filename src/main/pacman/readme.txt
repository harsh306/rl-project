This code runs the TSCL Algorithm with our own Pacman game created using OpenCV based on Sentdex Tutorials on youtube
(https://www.youtube.com/watch?v=t3fbETsIBCY&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=5&ab_channel=sentdex)


To run TSCL-Pacman, simply run tscl_pacman.py


-tscl_pacman.py: This file contains the definitions of the different Teacher algorithms and the main file for the project

-pacman_model.py: This file contains the interface between the teacher and the student. Classes:
--PacmanTeacherEnvironment: This class defines the teacher parameters and the step function for the Teacher.
  The teacher algorithm only has access to the PacmanTeacherEnvironment which talks to the PacmanModel class
--PacmanModel: This class defines the parameters of the Pacman Student Model. Some of its functions & params and
  inherited from pacman_agent.py.

-pacman_agent.py: This file contains all the definitions related to the student agent including the model, update functions,
the environment definitions, etc. This code can be run on its own to train a DQN pacman agent.

TODO:
1. Write code to allow tensorboard logging
2. Fine-tune hyperparams to get good performance
