import argparse
import csv
import os

import numpy as np
from collections import deque, defaultdict

from src.main.pacman.pacman_model import PacmanModel, PacmanTeacherEnvironment
from src.main.pacman.pacman_curriculum import gen_curriculum_baseline, gen_curriculum_naive, gen_curriculum_mixed, gen_curriculum_combined
# from src.tensorboard_utils import create_summary_writer, add_summary

class args:
    def __init__(self, run_id, csv_file=None, teacher='sampling',curriculum='combined',policy='egreedy',epsilon=0.1,
                 temperature=0.0004,bandit_lr=0.1,window_size=10,abs=False,max_timesteps=20000,max_enemies=5,invert=True,
                 hidden_size=128,batch_size=4096,train_size=100,val_size=4096,optimizer_lr=0.001,clipnorm=2,logdir='pacman_logs'):

        # parser.add_argument('--teacher', choices=['naive', 'online', 'window', 'sampling'], default='sampling')
        # parser.add_argument('--curriculum', choices=['uniform', 'naive', 'mixed', 'combined'], default='combined')
        # parser.add_argument('--policy', choices=['egreedy', 'boltzmann', 'thompson'], default='thompson')
        # parser.add_argument('--epsilon', type=float, default=0.1)
        # parser.add_argument('--temperature', type=float, default=0.0004)
        # parser.add_argument('--bandit_lr', type=float, default=0.1)
        # parser.add_argument('--window_size', type=int, default=10)
        # parser.add_argument('--abs', action='store_true', default=False)
        # parser.add_argument('--no_abs', action='store_false', dest='abs')
        # parser.add_argument('--max_timesteps', type=int, default=20000)
        # parser.add_argument('--max_digits', type=int, default=9)
        # parser.add_argument('--invert', action='store_true', default=True)
        # parser.add_argument('--no_invert', action='store_false', dest='invert')
        # parser.add_argument('--hidden_size', type=int, default=128)
        # parser.add_argument('--batch_size', type=int, default=4096)
        # parser.add_argument('--train_size', type=int, default=40960)
        # parser.add_argument('--val_size', type=int, default=4096)
        # parser.add_argument('--optimizer_lr', type=float, default=0.001)
        # parser.add_argument('--clipnorm', type=float, default=2)
        # parser.add_argument('--logdir', default='addition')
        # parser.add_argument('run_id')
        # parser.add_argument('--csv_file')
        self.csv_file = csv_file
        self.run_id = run_id
        self.teacher = teacher
        self.curriculum = curriculum
        self.policy = policy
        self.epsilon = epsilon
        self.temperature = temperature
        self.bandit_lr = bandit_lr
        self.window_size = window_size
        self.abs = abs
        self.max_timesteps = max_timesteps
        self.max_enemies = max_enemies
        self.invert = invert,
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.optimizer_lr = optimizer_lr
        self.clipnorm = clipnorm
        self.logdir = logdir

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def __call__(self, Q):
        # find the best action with random tie-breaking
        idx = np.where(Q == np.max(Q))[0] #Location of first occurence of max(Q)
        assert len(idx) > 0, str(Q)
        a = np.random.choice(idx)

        # create a probability distribution
        p = np.zeros(len(Q))
        p[a] = 1

        # Mix in a uniform distribution, to do exploration and
        # ensure we can compute slopes for all tasks
        p = p * (1 - self.epsilon) + self.epsilon / p.shape[0] # This is done to ensure total probability is 1

        assert np.isclose(np.sum(p), 1)
        return p


class BoltzmannPolicy:
    def __init__(self, temperature=1.):
        self.temperature = temperature

    def __call__(self, Q):
        e = np.exp((Q - np.max(Q)) / self.temperature)
        p = e / np.sum(e)

        assert np.isclose(np.sum(p), 1)
        return p


# HACK: just use the class name to detect the policy
class ThompsonPolicy(EpsilonGreedyPolicy):
    pass


def estimate_slope(x, y):
    assert len(x) == len(y)
    A = np.vstack([x, np.ones(len(x))]).T
    c, _ = np.linalg.lstsq(A, y)[0]
    return c


class NaiveSlopeBanditTeacher:
    def __init__(self, env, policy, lr=0.1, window_size=10, abs=False, writer=None):
        print("Using Naive Teacher")
        self.env = env
        self.policy = policy
        self.lr = lr
        self.window_size = window_size
        self.abs = abs
        self.Q = np.zeros(env.num_subtasks)
        self.writer = writer

    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps // self.window_size):
            p = self.policy(np.abs(self.Q) if self.abs else self.Q)
            scores = [[] for _ in range(len(self.Q))]
            for i in range(self.window_size):
                r, train_done, val_done = self.env.step(p)

                if val_done:
                    return self.env.model.epochs

                for a, score in enumerate(r):
                    if not np.isnan(score):
                        scores[a].append(score)
            s = [estimate_slope(list(range(len(sc))), sc) if len(sc) > 1 else 1 for sc in scores]
            self.Q += self.lr * (s - self.Q)

            # if self.writer:
            #     for i in range(self.env.num_subtasks):
            #         add_summary(self.writer, "Q_values/task_%d" % (i + 1), self.Q[i], self.env.model.epochs)
            #         add_summary(self.writer, "slopes/task_%d" % (i + 1), s[i], self.env.model.epochs)
            #         add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs


class OnlineSlopeBanditTeacher:
    def __init__(self, env, policy, lr=0.1, abs=False, writer=None):
        print("Using Online Teacher")
        self.env = env
        self.policy = policy
        self.lr = lr
        self.abs = abs
        self.Q = np.zeros(env.num_subtasks)
        self.prevr = np.zeros(env.num_subtasks)
        self.writer = writer

    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps):
            p = self.policy(np.abs(self.Q) if self.abs else self.Q)
            r, train_done, val_done = self.env.step(p)

            if val_done:
                return self.env.model.epochs

            s = r - self.prevr

            # safeguard against not sampling particular action at all
            s = np.nan_to_num(s)
            self.Q += self.lr * (s - self.Q)
            self.prevr = r

            # if self.writer:
            #     for i in range(self.env.num_subtasks):
            #         add_summary(self.writer, "Q_values/task_%d" % (i + 1), self.Q[i], self.env.model.epochs)
            #         add_summary(self.writer, "slopes/task_%d" % (i + 1), s[i], self.env.model.epochs)
            #         add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs


class WindowedSlopeBanditTeacher:
    def __init__(self, env, policy, window_size=10, abs=False, writer=None):
        print("Using Windowed Teacher")
        self.env = env
        self.policy = policy
        self.window_size = window_size
        self.abs = abs
        self.scores = [deque(maxlen=window_size) for _ in range(env.num_subtasks)]
        self.timesteps = [deque(maxlen=window_size) for _ in range(env.num_subtasks)]
        self.writer = writer

    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps):
            slopes = [estimate_slope(timesteps, scores) if len(scores) > 1 else 1 for timesteps, scores in zip(self.timesteps, self.scores)]
            p = self.policy(np.abs(slopes) if self.abs else slopes)
            r, train_done, val_done = self.env.step(p)

            if val_done:
                return self.env.model.epochs

            for a, s in enumerate(r):
                if not np.isnan(s):
                    self.scores[a].append(s)
                    self.timesteps[a].append(t)

            # if self.writer:
            #     for i in range(self.env.num_subtasks):
            #         add_summary(self.writer, "slopes/task_%d" % (i + 1), slopes[i], self.env.model.epochs)
            #         add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs


class SamplingTeacher:
    def __init__(self, env, policy, window_size=10, abs=False, writer=None):
        print("Using Sampling Teacher")
        self.env = env
        self.policy = policy
        self.window_size = window_size
        self.abs = abs
        self.writer = writer
        self.dscores = deque(maxlen=window_size)
        self.prevr = np.zeros(self.env.num_subtasks)

    def teach(self, num_timesteps=2000):
        for t in range(num_timesteps):
            # find slopes for each task
            if len(self.dscores) > 0:
                if isinstance(self.policy, ThompsonPolicy):
                    slopes = [np.random.choice(drs) for drs in np.array(self.dscores).T]
                else:
                    slopes = np.mean(self.dscores, axis=0)
            else:
                slopes = np.ones(self.env.num_subtasks)
            print("slopes: ",slopes)
            p = self.policy(np.abs(slopes) if self.abs else slopes)
            r, train_done, val_done = self.env.step(p)

            if val_done:
                return self.env.model.epochs

            # log delta score
            print("r: ", r)
            print("prevr: ", self.prevr)
            dr = r - self.prevr
            print("dr: ",dr)
            self.prevr = r
            self.dscores.append(dr)
            print("dscores: ",self.dscores)

            # if self.writer:
            #     for i in range(self.env.num_subtasks):
            #         add_summary(self.writer, "slopes/task_%d" % (i + 1), slopes[i], self.env.model.epochs)
            #         add_summary(self.writer, "probabilities/task_%d" % (i + 1), p[i], self.env.model.epochs)

        return self.env.model.epochs


if __name__ == '__main__':
    args = args(run_id='pacman0')

    logdir = os.path.join(args.logdir, args.run_id)
    # writer = create_summary_writer(logdir)

    model = PacmanModel(args.max_enemies)

    val_dist = gen_curriculum_baseline(args.max_enemies+1)[-1] # This is just a uniform distribution
    env = PacmanTeacherEnvironment(model, args.train_size, args.val_size, val_dist)  # Provide writer as arg when tensorboard_utils is debugged

    if args.teacher != 'curriculum':
        if args.policy == 'egreedy':
            policy = EpsilonGreedyPolicy(args.epsilon)
        elif args.policy == 'boltzmann':
            policy = BoltzmannPolicy(args.temperature)
        elif args.policy == 'thompson':
            assert args.teacher == 'sampling', "ThompsonPolicy can be used only with SamplingTeacher."
            policy = ThompsonPolicy(args.epsilon)
        else:
            assert False

    if args.teacher == 'naive':
        teacher = NaiveSlopeBanditTeacher(env, policy, args.bandit_lr, args.window_size, args.abs) # Provide writer as arg when tensorboard_utils is debugged
    elif args.teacher == 'online':
        teacher = OnlineSlopeBanditTeacher(env, policy, args.bandit_lr, args.abs) # Provide writer as arg when tensorboard_utils is debugged
    elif args.teacher == 'window':
        teacher = WindowedSlopeBanditTeacher(env, policy, args.window_size, args.abs) # Provide writer as arg when tensorboard_utils is debugged
    elif args.teacher == 'sampling':
        teacher = SamplingTeacher(env, policy, args.window_size, args.abs) # Provide writer as arg when tensorboard_utils is debugged
    else:
        assert False

    epochs = teacher.teach(args.max_timesteps)

    print("Finished after", epochs, "epochs.")
