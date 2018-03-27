from collections import namedtuple
import numpy as np
import sys
from network_all import ActorCriticAgent
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.util import general_n_step_advantage, combine_first_dimensions
import tensorflow as tf
from absl import flags

PPORunParams = namedtuple("PPORunParams", ["lambda_par", "batch_size", "n_epochs"])


class Runner(object):
    def __init__(
            self,
            envs,
            agent: ActorCriticAgent,
            n_steps=5,
            discount=0.99,
            do_training=True,
            ppo_par: PPORunParams = None,
            episodes=3
    ):
        self.envs = envs
        self.agent = agent # assign a variable agent to the AC which comes from the module of your choice
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.ppo_par = ppo_par
        self.batch_counter = 0
        self.episode_counter = 0
        self.episodes = episodes

    def reset(self):
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def _log_score_to_tb(self, score):
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        print("episode %d ended. Score %f" % (self.episode_counter, score))
        self._log_score_to_tb(score)
        self.episode_counter += 1

    def run_batch(self):
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.n_envs, self.n_steps), dtype=np.float32)

        latest_obs = self.latest_obs # state(t0)
        rnn_state = self.agent.state_init
        for n in range(self.n_steps):
            action_ids, spatial_action_2ds, value_estimate,rnn_state = self.agent.step(latest_obs, rnn_state)
            #print('step: ', n, action_ids, spatial_action_2ds, value_estimate)  # for debugging

            # Store actions and value estimates for all steps (this is being done in parallel with the other envs)
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids, spatial_action_2ds))
            # Do action, return it to environment, get new obs and reward, store reward
            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = self.envs.step(actions_pp)
            latest_obs = self.obs_processer.process(obs_raw) # state(t+1)
            mb_rewards[:, n] = [t.reward for t in obs_raw]

            #Check for all convolutional lstm vizdoom if last state is true
            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(t)
        # Get the Vt+1 and use it as a future reward from st+1 as we dont know the actual reward (bootstraping here with net's predicitons)
        mb_values[:, -1] = self.agent.get_value(latest_obs, rnn_state)


        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            lambda_par= 1.0
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()
        }
        # We combine all experiences from every env
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        else:
            self.agent.train(full_input, rnn_state) # You might want to reset the state between forward and backward pass

        self.latest_obs = latest_obs #state(t) = state(t+1)
        self.batch_counter += 1
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_trained_batch(self):
        # This function enables testing from a trained model. It loads the weights and runs the model for some episodes.
        latest_obs = self.latest_obs # state(t0)
        rnn_state = self.agent.state_init
        while self.episode_counter <= (self.episodes - 1):
            action_ids, spatial_action_2ds, value_estimate, rnn_state = self.agent.step(latest_obs, rnn_state) # (MINE) AGENT STEP = INPUT TO NN THE CURRENT
            #print('action: ', actions.FUNCTIONS[action_ids[0]].name, 'on', 'x=', spatial_action_2ds[0][0], 'y=', spatial_action_2ds[0][1], 'Value=', value_estimate[0]) # for debugging
            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = self.envs.step(actions_pp)
            latest_obs = self.obs_processer.process(
                obs_raw)

            # Check for all observations if last state is true
            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(t)

        self.latest_obs = latest_obs # state(t) = state(t+1)
        self.batch_counter += 1
        #print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()
