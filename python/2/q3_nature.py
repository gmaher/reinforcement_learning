import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################

        with tf.variable_scope(scope):
            print out
            out = layers.convolution2d(out,
                num_outputs=32,
                kernel_size=(8,8),
                stride=4,
                reuse=reuse,
                scope='conv1')

            print out
            out = layers.convolution2d(out,
                num_outputs=64,
                kernel_size=(4,4),
                stride=2,
                reuse=reuse,
                scope='conv2')

            print out
            out = layers.convolution2d(out,
                num_outputs=64,
                kernel_size=(3,3),
                stride=1,
                reuse=reuse,
                scope='conv3')

            print out
            s = out.get_shape().as_list()
            print s
            out = tf.reshape(out,shape=[-1,s[1]*s[2]*s[3]],name='reshape1')
            out = layers.fully_connected(inputs=out, num_outputs=512,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc1')

            print out
            out = layers.fully_connected(inputs=out, num_outputs=num_actions,
                weights_initializer=layers.xavier_initializer(), reuse=reuse,
                scope='fc2', activation_fn=None)

            print out
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    with tf.device('/gpu:0'):
        env = EnvTest((80, 80, 1))

        # exploration strategy
        exp_schedule = LinearExploration(env, config.eps_begin,
                config.eps_end, config.eps_nsteps)

        # learning rate schedule
        lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
                config.lr_nsteps)

        # train model
        model = NatureQN(env, config)
        model.run(exp_schedule, lr_schedule)
