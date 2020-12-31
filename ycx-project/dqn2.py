# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training import checkpoint_management


np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQN:
    def __init__(self,
                 lr=1e-3,
                 reward_decay=0.9,
                 e_greedy=0.3,
                 replace_target_iter=100,
                 memory_size=800,
                 batch_size=30
                 ):
        self.n_features=6
        self.double_q=True
        self.prioritized=True
        self.lr = lr
        self.reward_decay = reward_decay
        self.gamma = reward_decay
        self.e_greedy = e_greedy
        self.epsilon_increment = None
        self.epsilon_max= self.e_greedy
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, 6*1*2+2))
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory_counter =0

        self.build_net()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        self.cost_his = []

    def reshape_value(self, s):
        pass
        """
        B: 100-200M
        B_:16-256M
        D: 10-20ms
        D_:50-90ms
        d_sum:40-80ms
        """
        B = (s[..., :24] - 100) / 100 #100~200
        B_ = (s[..., 24:25] - 100) / 100 #100~200
        D = (s[..., 25:49] - 1) / 9 #1-10
        D_ = (s[..., 49:50] - 1) / 9
        d_sum = s[..., 50:51]/8
        p=(s[...,51:52]-50)/50  #50, 100
        cap=s[...,52:53]
        S=tf.concat([B, B_, D, D_, d_sum,p,cap],axis=3)
        # print(S)
        return S  # 79


    def build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None,1,6,1], name='s')
        self.s_reshaped = self.reshape_value(self.s)

        # print("s_reshaped")
        # print(self.s_reshaped)

        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, 11], name='q_target')

        w_initializer = tf.random_normal_initializer(stddev=0.01)
        b_initializer = tf.constant_initializer(0.0)
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.compat.v1.variable_scope('eval_net'):
            with tf.compat.v1.variable_scope('conv_net1'):
                w1 = tf.compat.v1.get_variable('w1', [1, 3, 1, 64],#1*6,窗口1*3，=1*4， 通道64
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b1 = tf.compat.v1.get_variable('b1', [64],
                                     initializer=b_initializer,
                                     collections=['eval_net', 'variables'])
                conv1 = tf.nn.conv2d(self.s_reshaped, w1, strides=[1,1, 1, 1], padding='VALID')
                h1 = tf.nn.relu(conv1 + b1)

            with tf.compat.v1.variable_scope('conv_net2'):
                w2 = tf.compat.v1.get_variable('w2', [1, 4, 64, 128],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b2 = tf.compat.v1.get_variable('b2', [128],
                                     initializer=b_initializer,
                                     collections=['eval_net', 'variables'])
                conv2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='VALID')
                h2 = tf.reshape(tf.nn.relu(conv2 + b2), [-1, 128])

            with tf.compat.v1.variable_scope('fc_net1'):
                w3 = tf.compat.v1.get_variable('w3', [128, 256],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b3 = tf.compat.v1.get_variable('b3', [256],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

            with tf.compat.v1.variable_scope('fc_net2'):
                w4 = tf.compat.v1.get_variable('w4', [256, 11],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                b4 = tf.compat.v1.get_variable('b4', [11],
                                     initializer=w_initializer,
                                     collections=['eval_net', 'variables'])
                self.q_eval = tf.matmul(h3, w4) + b4

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            #print("self.loss:", self.q_target-self.q_eval)

        with tf.compat.v1.variable_scope('train'):
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None,1,6,1], name='s_')
        self.s_reshaped_ = self.reshape_value(self.s_)

        with tf.compat.v1.variable_scope('target_net'):
            with tf.compat.v1.variable_scope('conv_net1'):
                w1 = tf.compat.v1.get_variable('w1',  [1, 3, 1, 64],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b1 = tf.compat.v1.get_variable('b1', [64],
                                     initializer=b_initializer,
                                     collections=['target_net', 'variables'])
                conv1 = tf.nn.conv2d(self.s_reshaped_, w1, strides=[1, 1, 1, 1], padding='VALID')
                h1 = tf.nn.relu(conv1 + b1)

            with tf.compat.v1.variable_scope('conv_net2'):
                w2 = tf.compat.v1.get_variable('w2', [1, 4, 64, 128],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b2 = tf.compat.v1.get_variable('b2', [128],
                                     initializer=b_initializer,
                                     collections=['target_net', 'variables'])
                conv2 = tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='VALID')
                h2 = tf.reshape(tf.nn.relu(conv2 + b2), [-1, 128])

            with tf.compat.v1.variable_scope('fc_net1'):
                w3 = tf.compat.v1.get_variable('w3', [128, 256],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b3 = tf.compat.v1.get_variable('b3', [256],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

            with tf.compat.v1.variable_scope('fc_net2'):
                w4 = tf.compat.v1.get_variable('w4', [256, 11],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                b4 = tf.compat.v1.get_variable('b4', [11],
                                     initializer=w_initializer,
                                     collections=['target_net', 'variables'])
                self.q_next = tf.matmul(h3, w4) + b4##matmul

    def store_transition(self, s, a, r, s_):
         if self.prioritized:    # prioritized replay
            transition = np.hstack((s.reshape((6,)), [a, r], s_.reshape((6,))))
            self.memory.store(transition)    # have high priority for newly arrived transition
            self.memory_counter += 1
         else:
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = np.hstack((s.reshape((6,)), [a, r], s_.reshape((6,))))
            self.memory_counter += 1

    def choose_action(self, observation0, larger_greedy=0.0):
        obsera=observation0.reshape((6,1),order='A')
        observation1 = obsera[np.newaxis, :]###一个observation为一个整体，所以加上一个维度
        observation = observation1[np.newaxis, :]
        # print(observation)
        # print("以上observation")
        if np.random.uniform() < max(self.e_greedy, larger_greedy):
            action_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
           # print('actionvalue：',action_values)
            action = np.argmax(action_values)
           # print('chooseaction：',action)
        else:
            action = np.random.randint(0, 10)
            print('randomly',action)
        return action

    def replace_target_net_params(self):
        target_net_params = tf.compat.v1.get_collection('target_net')
        eval_net_params = tf.compat.v1.get_collection('eval_net')
        self.sess.run([tf.compat.v1.assign(t, e)
                       for t, e in zip(target_net_params, eval_net_params)])

    def learn(self):
        # Check to replace target net params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_net_params()
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                 sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:].reshape([self.batch_size, 1,6,1]),    # next observation
                       self.s: batch_memory[:, -self.n_features:].reshape([self.batch_size, 1,6,1])})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features].reshape([self.batch_size, 1,6,1])})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self.train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features].reshape([self.batch_size, 1,6,1]),
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self.train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features].reshape([self.batch_size, 1,6,1]),
                                                    self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.e_greedy = self.e_greedy + self.epsilon_increment if self.e_greedy < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save(self, ckpt_file='ckpt/dqn.ckpt'):
        if not checkpoint_management.checkpoint_exists(os.path.dirname(ckpt_file)):
            os.makedirs(os.path.dirname(ckpt_file))
        self.saver.save(self.sess, ckpt_file)

    def load(self, ckpt_dir='ckpt'):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(ckpt_dir, ckpt_name))
            #print '[SUCCESS] Checkpoint loaded.'
        else:
            print ('[WARNING] No checkpoint found.')

if __name__ == '__main__':
    agent = DQN()


