import gym
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SingleLayerModel:

    def __init__(self, observation_space, action_space):
        tf.reset_default_graph()
        # Create the model
        # Input is a one hot vector encoding [1x16]
        self.inputs = tf.placeholder(shape=[1, observation_space], dtype=tf.float32)
        # [16, 4] matrix
        self.weights = tf.Variable(tf.random_uniform([observation_space, action_space], 0, 0.01))
        self.q_pred = tf.matmul(self.inputs, self.weights)
        # prediction is a [1, 4] vector representing the 4 possible actions
        self.predict = tf.argmax(self.q_pred, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targuet_q = tf.placeholder(shape=[1, action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.targuet_q - self.q_pred))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.gradient_update = trainer.minimize(loss)

        self.initialize_vars = tf.global_variables_initializer()


class TwoLayerModel:

    def __init__(self, observation_space, action_space):
        """
        Same as Single Layer model but added an extra fully connected layer to see if performance improves
        :param observation_space:
        :param action_space:
        """
        tf.reset_default_graph()
        # Create the model

        # first layer
        self.inputs = tf.placeholder(shape=[1, observation_space], dtype=tf.float32)
        # [16, 16] matrix
        extra_weights = tf.Variable(tf.random_uniform([observation_space, observation_space], 0, 0.01))
        int_pred = tf.nn.sigmoid(tf.matmul(self.inputs, extra_weights))

        # Second layer
        # [16, 4] matrix
        self.weights = tf.Variable(tf.random_uniform([observation_space, action_space], 0, 0.01))
        self.q_pred = tf.matmul(int_pred, self.weights)

        # prediction is a [1, 4] vector representing the 4 possible actions
        self.predict = tf.argmax(self.q_pred, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targuet_q = tf.placeholder(shape=[1, action_space], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(self.targuet_q - self.q_pred))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.gradient_update = trainer.minimize(loss)

        self.initialize_vars = tf.global_variables_initializer()


class QNetworkFrozenLake:

    def __init__(self, e=.1, y=.99, n_epochs=2000):
        """
        Initialise the environment on which the QNetwork will learn
        :param e: chance of random action
        :param y: reward discount
        :param n_epochs: how many iterations to run the learning algorithm for (each iteration does 100 simulations)
        """
        self.env = gym.make('FrozenLake-v0')
        # params
        self.y = y
        self.e = e
        self.n_epochs = n_epochs

    def learn_q_network(self, model, plot_lists=True):
        # create lists to contain total rewards and steps per epoch
        j_list = []
        r_list = []
        with tf.Session() as sess:
            sess.run(model.initialize_vars)
            for i in range(self.n_epochs):
                # Reset environment and get first new observation
                state = self.env.reset()
                r_counter = 0
                # Q-Network
                for j in range(1, 100):
                    # Choose an action by greedily (with e chance of random action) picking from the Q-network
                    a, all_q = sess.run([model.predict, model.q_pred],
                                        feed_dict={model.inputs: np.identity(self.env.observation_space.n)[state:state+1]})
                    if np.random.rand(1) < self.e:
                        # do a random action
                        a[0] = self.env.action_space.sample()
                    # Get new state and reward given decided action to take
                    next_state, reward, done, _ = self.env.step(a[0])

                    # Obtain the Q' values by feeding the new state through our network
                    q_prime = sess.run(model.q_pred,
                                       feed_dict={model.inputs: np.identity(16)[next_state:next_state+1]})
                    # Obtain maxQ' and set our target value for chosen action.
                    max_q_prime = np.max(q_prime)
                    all_q[0, a[0]] = reward + self.y * max_q_prime
                    # Train our network using target and predicted Q values
                    sess.run([model.gradient_update, model.weights],
                             feed_dict={model.inputs: np.identity(16)[state:state+1], model.targuet_q: all_q})
                    r_counter += reward
                    state = next_state
                    if done:
                        # Reduce chance of random action as we train the model -- since it's predictions are better
                        self.e = 1./((i/50) + 10)
                        break
                print('Epoch: {t}'.format(t=i))
                print('Accumulated reward: {t}'.format(t=r_counter))
                j_list.append(j)
                r_list.append(r_counter)

        print("Percent of succesful episodes: {t}".format(t=sum(r_list)/self.n_epochs))

        if plot_lists:
            plt.plot(r_list)
            plt.show()

            plt.plot(j_list)
            plt.show()


if __name__ == '__main__':
    frozen_lake_sim = QNetworkFrozenLake()
    model = SingleLayerModel(observation_space=frozen_lake_sim.env.observation_space.n,
                             action_space=frozen_lake_sim.env.action_space.n)
    frozen_lake_sim.learn_q_network(model=model)
