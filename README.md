# torch-lrcn
torch-lrcn provides a framework in Torch7 for action recognition using Long-term Recurrent Convolutional Networks. The LRCN model was [proposed by Jeff Donahue et. al in this paper](http://arxiv.org/pdf/1411.4389v3.pdf). Find more information about their Caffe code and experiments [here](http://www.eecs.berkeley.edu/~lisa_anne/LRCN_video).

Note that currently this library does **not** support fine-grained action detection (i.e. a specific label for each frame). The detection accuracy it computes is simply the frame accuracy using only a single label for each video.

# Installation
## System setup
You need `ffmpeg` accessible via command line. Find installation guides [here](https://trac.ffmpeg.org/wiki/CompilationGuide).

## Lua setup
All code is written in [Lua](http://www.lua.org/) using [Torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [torch/image](https://github.com/torch/image)
- [clementfarabet/lua---ffmpeg](https://github.com/clementfarabet/lua---ffmpeg)

After installing Torch, you can install / update these packages by running the following:

```bash
# Install using Luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install image
luarocks install ffmpeg
```

We also need [@jcjohnson](https://github.com/jcjohnson)'s [LSTM module](https://github.com/jcjohnson/torch-rnn/blob/master/LSTM.lua), which is already included in this repository.

### CUDA support
Because training takes awhile, you will want to use CUDA to get results in a reasonable amount of time. To enable GPU acceleration with CUDA, you'll first need to install CUDA 6.5 or higher. Find CUDA installations [here](https://developer.nvidia.com/cuda-downloads).

Then you need to install following Lua packages for CUDA:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update the Lua packages by running:

```bash
luarocks install cutorch
luarocks install cunn
```

# Usage
Training and testing a model requires some text files. The scripts assume that there are valid text files detailed below, and that all videos have the same native resolution.

## Step 1: Ready the data
The training step requires a text file for each of the training, validation, and testing splits. The structure of these text files is identical.

Example line: `<path to video> <label>`

Example file:
```
/path/to/video1.avi 1
/path/to/video2.avi 4
...
/path/to/video10.avi 3
```

## Step 2: Train the model
With the text files ready, we can begin training using `train.lua`. This will take quite some time because it is training a CNN and LSTM step.

You can run the training script, at minimum, like this:

```bash
th train.lua -trainList train.txt -valList val.txt -testList test.txt -numClasses 101 -videoHeight 240 -videoWidth 320
```

By default, this will dump 8 random frames at 5 FPS in native resolution representing semi-equally sized chunks for each video, train for 30 epochs, and save checkpoints to the trained models with names like `checkpoints/checkpoint_3.t7`. This also runs with CUDA by default. Run on CPU with `-cuda 0`. The default values are tuned to fit on an NVIDIA GPU with 4GB VRAM.

Some important parameters for training to tune are:
- `-scaledHeight`: optional downscaling
- `-scaledWidth`: optional downscaling
- `-desiredFPS`: FPS rate to convert videos to
- `-seqLength`: number of frames for each video
- `-batchSize`: number of videos per batch
- `-numEpochs`: number of epochs to train for
- `-learningRate`: learning rate
- `-lrDecayFactor`: multiplier for the learning rate decay
- `-lrDecayEvery`: decay the learning rate after every `n` epochs

An example of a more specific run:

```bash
th train.lua -trainList train.txt -valList val.txt -testList test.txt -numClasses 101 -videoHeight 240 -videoWidth 320 -scaledHeight 224 -scaledWidth 224 -seqLength 16 -batchSize 4 -numEpochs 15
```

<!--There are many more flags you can use to configure training; [read about them here](doc/flags.md#training).-->

## Step 3: Test the model
After training a model, you can compute the action recognition and detection accuracies using a model you trained. Do this by running `test.lua` as such:

```bash
th test.lua -checkpoint checkpoints/checkpoint_final.t7
```

By default, this will load the trained checkpoint `checkpoints/checkpoint_final.t7` from the training step and then compute the action detection and recognition accuracies for the test split. This also runs with CUDA by default. Run on CPU with `-cuda 0`.

The list of parameters is:
- `-checkpoint`: path to a checkpoint file (default: '')
- `-split`: name of split to test on (default: 'test')
- `-cuda`: run with CUDA (default: 1)

# Acknowledgments
- J. Donahue, L. A. Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, and T. Darrell. Long-term recurrent convolutional networks for visual recognition and description. In CVPR, 2015.
- [Justin Johnson](https://github.com/jcjohnson) for his [torch-rnn](https://github.com/jcjohnson/torch-rnn) library, which this library was heavily modeled after.
- Serena Yeung for the project idea, direction, and advice.
- Stanford University [CS 231N](http://cs231n.stanford.edu/) course staff for granting funds for AWS EC2 testing.

# TODOs
- Separate data preprocessing into its own step.
- Parallelize data loading.
- Write more documentation in a doc folder about training flags.
- Implement fine grained action detection..
- Add unit tests.
# dissertation
# CodeandScripts
