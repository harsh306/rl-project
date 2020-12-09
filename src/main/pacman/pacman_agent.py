import copy
import random
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tqdm import tqdm

MODEL_NAME = 'model_name'
class Blob:
    # Definitions related to each blob in the environment (walls, enemies, player, food).
    def __init__(self, *args):

        if len(args) == 1:
            self.map_size = args[0]
            self.x = np.random.randint(0, self.map_size)
            self.y = np.random.randint(0, self.map_size)
        elif len(args) == 3:
            self.map_size = args[0]
            self.x = args[1]
            self.y = args[2]
        else:
            print("ERROR: Incorrect Number of Args. Should be 1 or 3.")
            assert False

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice, walls):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(walls, x=1, y=0)
        elif choice == 1:
            self.move(walls, x=-1, y=0)
        elif choice == 2:
            self.move(walls, x=0, y=1)
        elif choice == 3:
            self.move(walls, x=0, y=-1)
        else:
            pass

    def move(self, walls, x=False, y=False):

        temp_x = self.x
        temp_y = self.y

        self.x += x
        self.y += y

        # If we hit a wall, fix
        for wall in walls:
            if self == wall:
                self.x = temp_x
                self.y = temp_y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.map_size - 1:
            self.x = self.map_size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.map_size - 1:
            self.y = self.map_size - 1


class PacmanEnv:
    # Definitions related to the environment
    def __init__(self, model, num_enemies, MAX_ITERS=15, start_q_table=None,
                 map_file='map_no_walls.txt', ACTION_SPACE_SIZE=4):
        self.SIZE = 5
        self.MODEL_NAME = '2x32'
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
        self.MAX_ITERS = MAX_ITERS

        # self.student_agent = student
        self.model = model

        self.MOVE_PENALTY = 0.01  # 0.2
        self.ENEMY_PENALTY = 1  # 0.8
        self.FOOD_REWARD = 1

        self.SHOW_EVERY = 9000  # how often to play through env visually.

        self.PLAYER_COLOR = (255, 50, 0)
        self.FOOD_COLOR = (0, 255, 0)  # food key in dict
        self.ENEMY_COLOR = (0, 0, 255)  # enemy key in dict
        self.WALL_COLOR = (255, 255, 255)
        self.episode_step = 0

        # with open(map_file, 'r') as f:
        #     self.map = np.zeros((self.SIZE, self.SIZE), dtype=float)
        #     data = f.readlines()
        #
        #     A_row = 0
        #     for line in data:
        #         list = line.strip('\n').split(' ')
        #         self.map[A_row:] = list[0:self.SIZE]  #
        #         A_row += 1

        self.walls = []
        # for i in range(len(self.map)):
        #     for j in range(len(self.map[0])):
        #         if self.map[i][j] == 1:
        #             self.walls.append(Blob(self.SIZE,i, j))

        self.player = Blob(self.SIZE)
        while self.player in self.walls:
            self.player = Blob(self.SIZE)

        self.food = Blob(self.SIZE)
        while (self.food in self.walls) or (self.food == self.player):
            self.food = Blob(self.SIZE)

        self.enemies = []
        for i in range(num_enemies):
            self.enemies.append(Blob(self.SIZE))
            while (self.enemies[i] in self.walls) or (self.enemies[i] == self.player) or (self.enemies[i] == self.food):
                self.enemies[i] = Blob(self.SIZE)

    def runEpisode(self, render, epsilon):  # Runs a single episode
        current_state = np.array(self.get_image())
        ep_history = []
        ep_reward = 0
        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(self.get_qs(current_state))  # big change
            else:
                # Get random action
                action = np.random.randint(0, self.ACTION_SPACE_SIZE)

            new_state, reward, done = self.step(action, done, current_state)
            ep_reward += reward
            ep_history.append((current_state, action, new_state, reward, done))

            if render:
                self.render()

            current_state = new_state

        return ep_history, ep_reward

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0)[0]

    def step(self, action, done=False, current_state=None):  # Called in runEpisode, steps the environment
        self.episode_step += 1
        self.player.action(action, self.walls)
        reward = 0
        new_observation = current_state  # np.array(self.get_image())

        for i in range(len(self.enemies)):
            if self.player == self.enemies[i]:
                reward = -self.ENEMY_PENALTY
                done = True

        if reward == 0:
            if self.player == self.food:
                reward = self.FOOD_REWARD
                done = True
            else:
                reward = -self.MOVE_PENALTY

        if self.episode_step >= self.MAX_ITERS:
            done = True
        return new_observation, reward, done

    def get_image(self): # Returns an image of the game board
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.FOOD_COLOR  # sets the food location tile to green color
        env[self.player.x][self.player.y] = self.PLAYER_COLOR  # sets the player tile to blue
        for enemy in self.enemies:
            env[enemy.x][enemy.y] = self.ENEMY_COLOR  # sets the enemy location to red
        for wall in self.walls:
            env[wall.x][wall.y] = self.WALL_COLOR  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.float32(img))
        cv2.waitKey(1)

    def plotRewards(self, episode_rewards):
        moving_avg = np.convolve(episode_rewards, np.ones((self.SHOW_EVERY,)) / self.SHOW_EVERY, mode='valid')

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {self.SHOW_EVERY}ma")
        plt.xlabel("episode #")
        plt.show()

class PacmanPlayer:
    # Definitions related to the student RL agent: model, update function
    def __init__(self, update_after='each', epsilon=0.85, EPS_DECAY=0.998, OBSERVATION_SPACE_SIZE=(5, 5, 3),
                 ACTION_SPACE_SIZE=4,
                 ep_number=0, RENDER_EVERY=200, DISCOUNT=0.50, BATCH_SIZE=500):
        #Update after: All = all subtasks; Each = each subtask

        self.update_after = update_after
        self.OBSERVATION_SPACE_SIZE = OBSERVATION_SPACE_SIZE
        self.ACTION_SPACE_SIZE = ACTION_SPACE_SIZE
        self.ep_number = ep_number
        self.RENDER_EVERY = RENDER_EVERY  # render every 1000 episodes
        self.DISCOUNT = DISCOUNT
        self.BATCH_SIZE = BATCH_SIZE
        self.epsilon = epsilon
        self.EPSILON_DECAY = EPS_DECAY
        self.MIN_EPSILON = 0.05
        self.logdir = "logs"
        self.REPLAY_BUFFER_SIZE = 10 * self.BATCH_SIZE
        self.epoch_history = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        self.REWARDS = []
        self.MEAN_REWARDS = []
        self.WINDOW_SIZE = 100

        self.model = self.create_model_simple()
        self.target_model = self.create_model_simple()
        # self.model = tf.keras.models.load_model('model')
        # self.target_model = tf.keras.models.load_model('model')

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(16, (3, 3),
                         input_shape=self.OBSERVATION_SPACE_SIZE))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(BatchNormalization())

        model.add(Conv2D(16, (3, 3),
                         input_shape=self.OBSERVATION_SPACE_SIZE))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(BatchNormalization())

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(8, activation='linear'))
        model.add(BatchNormalization())

        model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Nadam(lr=0.0007), metrics=['mae'])
        return model

    def create_model_simple(self):
        model = Sequential()
        model.add(Conv2D(4, (3, 3),
                         input_shape=self.OBSERVATION_SPACE_SIZE))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(BatchNormalization())

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(8, activation='linear'))
        model.add(BatchNormalization())

        model.add(Dense(self.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Nadam(lr=0.005), metrics=['mae'])
        return model

    def logMeanReward(self):
        if len(self.REWARDS) >= self.WINDOW_SIZE:
            self.MEAN_REWARDS.append(np.mean(self.REWARDS[-self.WINDOW_SIZE:]))
            return self.MEAN_REWARDS[-1]

    def train_epochs(self, num_enemies, num_eps): # This function is only used when pacman_agent.py is run on its own
        rewards_list = []
        total_reward = 0
        # epoch_history = []
        # for ep in range(1, num_eps+1):
        for ep in tqdm(range(1, num_eps + 1), ascii=True, unit='episodes'):
            env = PacmanEnv(model=self.model, num_enemies=num_enemies, ACTION_SPACE_SIZE=self.ACTION_SPACE_SIZE)

            render = False
            if ep%self.RENDER_EVERY == 0:
                render = True

            ep_history, ep_reward = env.runEpisode(render, self.epsilon)
            if self.epsilon > self.MIN_EPSILON and len(self.epoch_history) > 2 * self.BATCH_SIZE:
                self.epsilon *= self.EPSILON_DECAY

            # [self.epoch_history.append(hist) for hist in ep_history]

            self.epoch_history.extend(ep_history)
            self.REWARDS.append(ep_reward)
            print(f"Episode No. {ep}, Epsilon: {self.epsilon}, Reward: {ep_reward}, "
                  f"Mean {np.mean(self.REWARDS)}, Last window Avg {self.logMeanReward()}")

            if self.update_after == 'each':
                self.update_model(self.epoch_history)
            elif self.update_after != 'all':
                print('update_after has to be either all or each')
                assert False
            if ep % 100 == 0:
                self.target_model = tf.keras.models.clone_model(self.model)

            # rewards_list.append(ep_reward)
            # total_reward += ep_reward
        if self.update_after == 'all':
            self.update_model(self.epoch_history)


    def update_model(self, epoch_history): # Updates the model given the results of an epoch
        # print("============================Updating Model=================================")
        if len(epoch_history) < 2 * self.BATCH_SIZE:
            return
            # batch = epoch_history
        else:
            batch = random.sample(epoch_history, self.BATCH_SIZE)
            #batch = epoch_history[-self.BATCH_SIZE:]
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in batch]) / 255.0
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[2] for transition in batch]) / 255.0
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, new_current_state, reward, done) in enumerate(batch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # print("Current State: ", current_state)
            # print("New State: ", new_current_state)
            # print("Old Q", current_qs_list[index])

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]

            old_q = current_qs[action]
            current_qs[action] = copy.deepcopy(new_q)

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

            # print(f"Index: {index}, Action: {action}, Reward: {reward}, Done: {done}, Old Q: {old_q}, New Q: {new_q}")
            # print("New Q", current_qs_list[index])

        # Fit on all samples as one batch, log only on terminal state
        x_train = np.array(X) / 255.0
        y_train = np.array(y)
        self.model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, verbose=2, shuffle=False,
                       callbacks=None)

        # print("============================/Updating Model=================================")
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255.0)[0]



if __name__ == "__main__":
    player = PacmanPlayer()
    player.train_epochs(3, 2000)
    # plt.figure(1)
    # plt.plot(player.REWARDS)
    plt.figure(2)
    plt.plot(player.MEAN_REWARDS)
    plt.savefig("model_plots/"+MODEL_NAME)
    plt.show()
    player.model.save("model_logs/"+MODEL_NAME)
    # run 1 200
    # run 2 100
    # run 3 400
