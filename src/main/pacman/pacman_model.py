
import numpy as np
from keras.callbacks import TensorBoard

from .pacman_agent import PacmanEnv, PacmanPlayer


class PacmanModel(PacmanPlayer):
    # Model definitions can be found in pacman_agent.py
    def __init__(self, max_enemies=5, optimizer_lr=0.001, logdir=None):
        PacmanPlayer.__init__(self,'each')
        self.max_enemies = max_enemies
        self.render_ep = 0 # Renders 1 episode from the subtask list

        self.optimizer_lr = optimizer_lr #Learning Rate
        self.logdir = logdir

        self.epochs = 0
        # self.make_model()
        if logdir:
            self.callbacks = [TensorBoard(log_dir=self.logdir)]
        else:
            self.callbacks = []
    #
    # def make_model(self): # To be filled in by Qi. Define the neural network for the RL agent.
    #     self.model = 1 #Save the neural network in self.model

    def generate_data(self, dist, size):
        """
        # Generates a problem set given the distribution over subtasks
        :param dist:
        :param size:
        :return:
        """

        problem_set = []
        while len(problem_set) <= size:
            problem_set.append(1 + np.random.choice(len(dist), p=dist))
        print('Problem Set: ', problem_set)
        return problem_set

    def train_epoch(self, train_data, val_data):
        """
        Takes the problem set and trains the agent over all subtasks in it.
        Returns rewards for training and val data
        :param train_data:
        :param val_data:
        :return:
        """
        # rewards_list = np.ones(self.max_enemies) * (-np.inf)
        rewards_list = [[] for _ in range(self.max_enemies)]
        epoch_history = []
        num_eps = len(train_data)
        num_val_eps = len(val_data)
        for ep in range(num_eps):
            print("Running Episode No. ", ep," with ",train_data[ep]," enemies")
            env = PacmanEnv(model=self.model, num_enemies=train_data[ep], ACTION_SPACE_SIZE=self.ACTION_SPACE_SIZE)

            render = False
            if ep % self.RENDER_EVERY == 0:
            # if self.render_ep == ep:
                render = True

            ep_history, ep_reward = env.runEpisode(render)

            [epoch_history.append(hist) for hist in ep_history]
            # We save all reward obtained for a certain subtask in an epoch and later avg them
            # rewards_list[train_data[ep]-1] = max(rewards_list[train_data[ep]-1], ep_reward)
            rewards_list[train_data[ep] - 1].append(ep_reward)

            if self.update_after == 'each':
                self.update_model(epoch_history)
            elif self.update_after != 'all':
                print('update_after has to be either all or each')
                assert False

        if self.update_after == 'all':
            self.update_model(epoch_history)

        ##################### Running validation episodes to find out how well the model performs.######################
        val_rewards_list = [[] for _ in range(self.max_enemies)]
        for val_ep in range(num_val_eps):
            env = PacmanEnv(model=self.model, num_enemies=val_data[val_ep], ACTION_SPACE_SIZE=self.ACTION_SPACE_SIZE)
            _, val_reward = env.runEpisode(render=False)
            val_rewards_list[val_data[val_ep] - 1].append(val_reward)
        val_rewards_list = np.mean(val_rewards_list,axis=1)
        ################################################################################################################


        # Average all rewards
        for n, subtask_rewards in enumerate(rewards_list):
            if subtask_rewards == []:
                rewards_list[n] = -500
            else:
                rewards_list[n] = np.mean(subtask_rewards)
        print("===========================END OF EPOCH===========================================")
        return np.array(rewards_list), val_rewards_list


class PacmanTeacherEnvironment:
    def __init__(self, model, train_size, val_size, val_dist, val_threshold=None, writer=None):
        self.model = model
        self.num_subtasks = model.max_enemies
        self.train_size = train_size
        self.val_data = model.generate_data(val_dist, val_size)
        self.writer = writer
        self.val_threshold = -100 if not val_threshold else val_threshold
        self.epochs = 0

    def step(self, train_dist):
        """
        This function steps the teacher i.e. it generates a problem set based on the probability distribution over num_enemies
        and provides this problem set to the student to train  on it.
        This function returns the average reward obtained for each subtask (num of enemies) based on the current problem set.
        :param train_dist:
        :return:
        """

        print("Training on", train_dist)
        train_data = self.model.generate_data(train_dist, self.train_size)
        reward, val_reward = self.model.train_epoch(train_data, self.val_data)

        train_done = False
        val_done = all(i>0 for i in [1,2,3])
        self.epochs += 1

        # if self.writer:
        #     for k, v in history.items():
        #         add_summary(self.writer, "model/" + k, v[-1], self.model.epochs)
        #     for i in range(self.num_actions):
        #         # add_summary(self.writer, "train_accuracies/task_%d" % (i + 1), train_accs[i], self.model.epochs)
        #         add_summary(self.writer, "valid_accuracies/task_%d" % (i + 1), val_accs[i], self.model.epochs)

        return reward, train_done, val_done
