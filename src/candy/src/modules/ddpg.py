import tensorflow as tf
from modules.module import Module
from modules.replaybuffer import ReplayBuffer

class ActorCritic(Module):
    def __init__(self, name, args, reuse=False):
        super(ActorCritic, self).__init__(name, args, reuse)

    def _build_graph(self):
        self.actor
        self.critic

class Agent(Module):
    def __init__(self, name, args, reuse=False):
        # hyperparameters
        self.gamma = args[name]['gamma']
        self.tau = args[name]['tau']
        self.noise_decay = 1 + 5e-6
        # env info
        self.state_size = args[name]['state_size']
        self.action_size = args[name]['action_size']
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=args['batch_size'])

        super(Agent, self).__init__(name, args, reuse)

    def _build_graph(self):
        with tf.variable_scope('main', reuse=self.reuse):
            self.actor_critic_main = ActorCritic(self._name, self._args, reuse=reuse)

        with tf.variable_scope('target', reuse=self.reuse):
            self.actor_critic_target = ActorCritic(name, args, reuse=reuse)

        self.loss = self._loss()

    def _loss(self):
        with tf.name_scope('loss'):
            targets = rewards + self.gamma * self.actor_critic_target.critic

            with tf.name_scope('critic_loss'):
                critic_loss = tf.losses.mean_squared_error(tf.stop_gradient(targets), self.actor_critic_main)

            with tf.name_scope('target_loss'):
                actor_loss = - tf.reduce_mean(self.actor_critic_main.critic)

    def _optimize(self, loss):
        learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999

        with tf.name_scope('optimizer'):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

        opt_op = self._optimizer.minimize(loss)
        
        return opt_op

    def act(self, sess, state):
        # add noise to parameters
        saved_params = []
        for param in self.actor_main.parameters():
            saved_params.append(copy.deepcopy(param))
            param = param + torch.normal(mean=0.0, std=torch.ones_like(param) / (10 * self.noise_decay))

        self.noise_decay *= 1 + 5e-6

        self.actor_main.eval()
        with torch.no_grad():
           action = self.actor_main(state).cpu().numpy()
        self.actor_main.train()
        # restore parameters
        for param, saved_param in zip(self.actor_main.parameters(), saved_params):
            param = saved_param

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.buffer.sample_size + 100:
            self._learn()

    def _learn(self):
        targets = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, self.actor_target(next_states)).detach()
        critic_loss = F.mse_loss(self.critic_main(states, actions), targets)
        actor_loss = -self.critic_main(states, self.actor_main(states)).mean()
        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update the target networks
        self._moving_average(self.actor_main, self.actor_target)
        self._moving_average(self.critic_main, self.critic_target)

    def _moving_average(self, main, target):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)