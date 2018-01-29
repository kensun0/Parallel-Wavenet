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

As to parallel wavenet's Teacher, you should use discretized mixture of logistics distribution instead of 256-way categorical distribution.


# Step

1. Replace casual conv1d in citing6(masked.py) with Keras's implement.

2. Modify datafeeder
