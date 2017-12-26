import gym
import numpy as np


class QTableFrozenLake:

    def __init__(self, lr=.8, y=.95, n_epochs=2000):
        """
        Initialise a Q table for the simple Frozen Lake scenario available at https://gym.openai.com
        :param lr: Learning rate for the Q-table
        :param y: Reward discount (see Bellman equation)
        :param n_epochs: How many iterations to for the algorithm (each iteration does 100 simulations)
        """
        self.env = gym.make('FrozenLake-v0')
        # Initialize table with all zeros
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.lr = lr
        self.y = y
        self.n_epochs = n_epochs

    def learn_q_table(self):
        # create lists to contain total rewards and steps per episode
        reward_list = []
        for i in range(self.n_epochs):
            # Reset environment and get first new observation
            state = self.env.reset()
            reward_counter = 0
            # The Q-Table learning algorithm
            for j in range(1, 100):
                # Choose an action by greedily (with noise) picking from Q table
                action = np.argmax(self.q_table[state, :] + np.random.randn(1, self.env.action_space.n)*(1./(i+1)))
                # Get new state and reward from environment
                next_state, r, done, _ = self.env.step(action=action)
                # Update Q-Table with new knowledge (Bellman equation)
                self.q_table[state, action] = self.q_table[state, action] + \
                                              self.lr*(r + self.y*np.max(self.q_table[next_state, :]) - self.q_table[state, action])
                reward_counter += r
                state = next_state
                if done:
                    break
            reward_list.append(reward_counter)

        print("Average score over time: " + str(sum(reward_list)/self.n_epochs))
        print("Final Q-Table Values: \n {t}".format(t=self.q_table))


if __name__ == '__main__':
    q_frozen = QTableFrozenLake()
    q_frozen.learn_q_table()
