## Frame Mining: a Free Lunch for Learning Robotic Manipulation from 3D Point Clouds

  - [Installation](#installation)
  - [Example Training with FM-MA](#example-training-with-fm-ma)

### Installation

For this repo, we require CUDA=11.3. If you haven't had CUDA=11.3 locally yet, download the runfile from NVIDIA at [this link](https://developer.nvidia.com/cuda-11.3.0-download-archive) and install it.

To install, first create an Anaconda environment with python=3.8:

```
conda create -n py38 python=3.8
```

Then install pytorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pytorch3d
```

Install SAPIEN:

```
pip install sapien.whl
```

Install our code base:
```
cd {this_directory}/mani_skill
pip install -e .
cd {this_directory}/pyrl
pip install ninja
pip install -e .
pip install protobuf==3.19.0
cd {this_directory}/torchsparse
pip install -e .
```

### Example Training with FrameMiner-Mixaction (FM-MA)

First, `cd {this_directory}/pyrl`

Example script can be found at `./script/ppo.sh`


