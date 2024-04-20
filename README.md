
# ZunshineTu-ai-agent
This is a Reinforcement Learning (RL) agent using private and shared world models. It is compatible with both Linux and Windows, given that CUDA drivers are installed.

## Requirements:

* Linux/Windows: CUDA Drivers installed ([compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward))
* Linux: [Docker Engine](https://docs.docker.com/engine/install/), [nvidia-docker2 installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker), "nvidia-smi" docker test on previous link needs to work.
* Windows:
  * [WSL nbody benchmark needs to work](https://docs.docker.com/desktop/windows/wsl/#gpu-support)
  * Docker Desktop must be installed even if you intend on using docker from the command prompt in WSL.
  * ["Does nvidia-docker2 support Windows?"](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#is-microsoft-windows-supported)

To use:

```bash
$ git clone https://github.com/ZunshineTu/ZunshineTu-ai-agent.git
$ cd ZunshineTu-ai-agent
$ ./build.sh build
$ ./build.sh dev
root@f3537fad7977:/app# time python agent.py
...
RUN ...-new
time:   0:00:00:21    steps:2712    t/s:0.00781960    ms:256     |     attn: net io out ar    al:16    am:4     |     a-clk:0.001    a-spd:160.0    aug:SP     |     action:4e-06

real	0m28.174s
user	0m42.520s
sys	0m9.661s
root@f3537fad7977:/app#
```