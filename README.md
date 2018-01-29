# Parallel-Wavenet

Parallel wavenet has been implemented, partial codes will be placed here soon.

# Citings

Citing 1: Parallel WaveNet: Fast High-Fidelity Speech Synthesis

Citing 2: WAVENET: A GENERATIVE MODEL FOR RAW AUDIO

Citing 3: Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders

Citing 4: TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS 

Citing 5: PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications

Citing 6: https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth

Citing 7: https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L3254

Citing 8: https://github.com/openai/pixel-cnn

Citing 9: https://github.com/keithito/tacotron

# Notes

You should read citing6's codes first, then you can implement the original wavenet.

We use mel-scale spectrogram transforming from real wav as local conditions for convenience. You can train a tacotron model to get predicted mel-scale spectrogram.

A good teacher network is VERY VERY VERY important for training the student network.

# Teacher training Step

1. Replace casual conv1d in citing6(masked.py) with Keras's implement. Refer to citing7.

2. Implement a datafeeder to provide mel and wav. Refer to citing9's datafeeder.py.

3. Using discretized mixture of logistics distribution instead of 256-way categorical distribution. Refer ro citing8's nn.py.

4. Training with Adam.

# Student training Step

1. Modify Teacher's datafeeder to provider white noises Z. One mixture logistic, np.random.logistic(size=wav.shape)

2. Modify h512_bo16.py.

3. Add power loss and cross entropy loss.

4. Restore teacher weights, and then train student.
