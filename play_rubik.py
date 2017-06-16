from rubik import Rubik
import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if not done:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == "__main__":
    rubik = Rubik()
    RL = QLearningTable(actions=list(range(rubik.n_actions)))
    for episode in range(10):
        # initial observation
        state = rubik.state
        while True:
            # fresh env
            # RL choose action based on observation
            action = RL.choose_action(str(state))

            # RL take action and get next observation and reward
            state_, reward, done = rubik.take_action(action)

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_), done)

            # swap observation
            state = state_
            # break while loop when end of this episode
            if done:
                print 'Solved!'
                print RL.q_table
                rubik.reset()
                rubik.update_rubik()
                break
            if rubik.count == 200:
                print 'reach the max movements'+ str(rubik.count)
                rubik.reset()
                rubik.update_rubik()
                break
    RL.q_table.to_csv('q_table.csv')