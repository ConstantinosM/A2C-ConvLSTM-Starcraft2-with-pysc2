# A2C-ConvLSTM-Starcraft2
A2C with ConvLSTM agent playing Starcraft 2 (DeepMind's FullyConv LSTM)

Synchronous Advantage Actor Critic (synchronous variation of the [A3C](https://arxiv.org/abs/1602.01783) with [Convolutional LSTM](https://arxiv.org/abs/1506.04214) playing Starcraft 2 using DeepMind's API [pysc2](https://github.com/deepmind/pysc2/).

The code is based on [pekaalto's](https://github.com/pekaalto/sc2aibot) FullyConv Net, although there are some modifications of the original version, and there is the ConvLSTM added after the state concatenation. Please note that there is no PPO active here and the code is for experimentation purposes.

## Dependencies
- Python 3
- pysc2
- Tensorflow
