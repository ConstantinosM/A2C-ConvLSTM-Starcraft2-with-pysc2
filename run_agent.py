# Fully Convolutional LSTM Network (the 3rd in DeepMind's paper)
# Architecture is the same as the FullyConv but we add a ConvLSTM after the state concatenation.
import logging
import sys
import os
import shutil
import sys
from datetime import datetime
from functools import partial
import tensorflow as tf
from absl import flags
from network_all import ActorCriticAgent
from actorcritic.runner import Runner, PPORunParams
from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 1, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 3, "Number of complete episodes to run for testing")
flags.DEFINE_integer("n_steps_per_batch", None,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("replay_dir", "_replays", "Save SC2 Replay")
flags.DEFINE_string("model_name", "ConvLSTM_MVT", "Name for checkpoints and tensorboard summaries")
flags.DEFINE_integer("K_batches", 300,
    "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
flags.DEFINE_string("map_name", "MoveToBeacon_beta", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", False,
    "if should train the model, if false then save only episode score summaries"
                     ) # If False then it runs on testing mode
flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 0.5, "good value might depend on the environment")
flags.DEFINE_float("entropy_weight_spatial", 0.001,
    "entropy of spatial action distribution loss weight")
flags.DEFINE_float("entropy_weight_action", 0.001, "entropy of action-id distribution loss weight")
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo") # PPO is removed!
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")

FLAGS(sys.argv)

full_chekcpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training(agent):
    agent.save(full_chekcpoint_path)
    agent.flush_summaries()
    sys.stdout.flush()


def main():
    if FLAGS.training:
        check_and_handle_existing_folder(full_chekcpoint_path)
        check_and_handle_existing_folder(full_summary_path)

    env_args = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        screen_size_px=(FLAGS.resolution,) * 2, # It makes it x,y by 'multiplying' by two here
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize,
        replay_dir=FLAGS.replay_dir
    )
    #Create multiple parallel environements (or a single instance for testing agent mode)
    if FLAGS.training:
        envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
    else:
        envs = SingleEnv(make_sc2env(**env_args))

    tf.reset_default_graph()
    sess = tf.Session()

    if FLAGS.n_steps_per_batch is None:
        n_steps_per_batch = 8
    else:
        n_steps_per_batch = FLAGS.n_steps_per_batch

    agent = ActorCriticAgent(
        mode="a2c",
        sess=sess,
        spatial_dim=FLAGS.resolution,
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        all_summary_freq=FLAGS.all_summary_freq,
        summary_path=full_summary_path,
        max_gradient_norm=FLAGS.max_gradient_norm,
    )
    # Build Agent
    agent.build_model()
    if os.path.exists(full_chekcpoint_path):
        agent.load(full_chekcpoint_path) #(MINE) LOAD!!!
    else:
        agent.init()

    ppo_par = None

    runner = Runner(
        envs=envs,
        agent=agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
        ppo_par=ppo_par,
        episodes=FLAGS.episodes
    )

    runner.reset() # Reset env which means you get first observation

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches
    else:
        n_batches = -1


    if FLAGS.training:
        i = 0

        try:
            while True:
                if i % 500 == 0:
                    _print(i)
                if i % 4000 == 0:
                    _save_if_training(agent)
                    print('...model_saved!')
                runner.run_batch()  # HERE WE RUN MAIN LOOP FOR TRAINING
                i += 1
                if 0 <= n_batches <= i:
                    break
        except KeyboardInterrupt:
            pass
    else: # Test the agent
        try:
            runner.run_trained_batch()  # HERE WE RUN MAIN LOOP FOR TESTING
        except KeyboardInterrupt:
            pass

    print("Okay. Work is done")
    if FLAGS.training:
        _save_if_training(agent)
    if not FLAGS.training:
        envs.env.save_replay(FLAGS.replay_dir)

    envs.close()


if __name__ == "__main__":
    main()
