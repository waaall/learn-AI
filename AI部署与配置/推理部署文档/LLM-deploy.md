
# 一、LLM 部署基础知识

##  部署框架
| 框架/平台            | 核心优势                                | 跨平台支持 (CPU/GPU)        | 部署灵活性 & 异构兼容性      | 代表性应用场景                        |
| ---------------- | ----------------------------------- | ---------------------- | ------------------ | ------------------------------ |
| **ONNX Runtime** | 推理优化、**标准模型格式支持**、硬件后端抽象            | **广泛** (x86, ARM, GPU) | **高** (EP机制灵活切换后端) | 边缘设备、云原生推理、多硬件环境统一部署           |
| **Triton**       | **高性能推理**、多框架支持、**动态批处理**           | **广泛** (x86, ARM, GPU) | **高** (支持多种硬件和平台)  | 高并发云服务、复杂模型管道、混合负载 (CPU/GPU协同) |
| **LMDeploy**     | **大模型（LLM）专用**、**推理性能优化**、量化        | 支持 (GPU为主)             | 中 (专注LLM高效推理)      | 大语言模型低延迟推理、消费级GPU部署、降低显存消耗79   |
| **FastDeploy**   | **端到端部署**、**多硬件适配** (国产芯片)、工具链完整    | **广泛** (x86, ARM, GPU) | **高** (国产芯片深度适配)   | 国产化环境、全场景（云边端）部署、快速原型验证        |
| **OpenVINO**     | **Intel硬件深度优化** (CPU, iGPU)、计算机视觉优势 | x86, Intel GPU         | 中 (Intel生态最佳)      | Intel平台高性能推理、边缘AI、计算机视觉应用      |

## onnxruntime
- [microsoft-onnxruntime](https://github.com/microsoft/onnxruntime)
- [onnxruntime-doc](https://onnxruntime.ai/docs/)

### onnxruntime-gpu和cuda的版本关系
- [onnx-CUDA-ExecutionProvider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
#### CUDA 12.x 

|ONNX Runtime|CUDA|cuDNN|Notes|
|---|---|---|---|
|1.20.x|12.x|9.x|Avaiable in PyPI. Compatible with PyTorch >= 2.4.0 for CUDA 12.x.|
|1.19.x|12.x|9.x|Avaiable in PyPI. Compatible with PyTorch >= 2.4.0 for CUDA 12.x.|
|1.18.1|12.x|9.x|cuDNN 9 is required. No Java package.|
|1.18.0|12.x|8.x|Java package is added.|
|1.17.x|12.x|8.x|Only C++/C# Nuget and Python packages are released. No Java package.|

#### CUDA 11.x 

|ONNX Runtime|CUDA|cuDNN|Notes|
|---|---|---|---|
|1.20.x|11.8|8.x|Not available in PyPI. See [Install ORT](https://onnxruntime.ai/docs/install) for details. Compatible with PyTorch <= 2.3.1 for CUDA 11.8.|
|1.19.x|11.8|8.x|Not available in PyPI. See [Install ORT](https://onnxruntime.ai/docs/install) for details. Compatible with PyTorch <= 2.3.1 for CUDA 11.8.|
|1.18.x|11.8|8.x|Available in PyPI.|
|1.17  <br>1.16  <br>1.15|11.8|8.2.4 (Linux)  <br>8.5.0.96 (Windows)|Tested with CUDA versions from 11.6 up to 11.8, and cuDNN from 8.2 up to 8.9|
|1.14  <br>1.13|11.6|8.2.4 (Linux)  <br>8.5.0.96 (Windows)|libcudart 11.4.43  <br>libcufft 10.5.2.100  <br>libcurand 10.2.5.120  <br>libcublasLt 11.6.5.2  <br>libcublas 11.6.5.2  <br>libcudnn 8.2.4|
|1.12  <br>1.11|11.4|8.2.4 (Linux)  <br>8.2.2.26 (Windows)|libcudart 11.4.43  <br>libcufft 10.5.2.100  <br>libcurand 10.2.5.120  <br>libcublasLt 11.6.5.2  <br>libcublas 11.6.5.2  <br>libcudnn 8.2.4|
|1.10|11.4|8.2.4 (Linux)  <br>8.2.2.26 (Windows)|libcudart 11.4.43  <br>libcufft 10.5.2.100  <br>libcurand 10.2.5.120  <br>libcublasLt 11.6.1.51  <br>libcublas 11.6.1.51  <br>libcudnn 8.2.4|
|1.9|11.4|8.2.4 (Linux)  <br>8.2.2.26 (Windows)|libcudart 11.4.43  <br>libcufft 10.5.2.100  <br>libcurand 10.2.5.120  <br>libcublasLt 11.6.1.51  <br>libcublas 11.6.1.51  <br>libcudnn 8.2.4|
|1.8|11.0.3|8.0.4 (Linux)  <br>8.0.2.39 (Windows)|libcudart 11.0.221  <br>libcufft 10.2.1.245  <br>libcurand 10.2.1.245  <br>libcublasLt 11.2.0.252  <br>libcublas 11.2.0.252  <br>libcudnn 8.0.4|
|1.7|11.0.3|8.0.4 (Linux)  <br>8.0.2.39 (Windows)|libcudart 11.0.221  <br>libcufft 10.2.1.245  <br>libcurand 10.2.1.245  <br>libcublasLt 11.2.0.252  <br>libcublas 11.2.0.252  <br>libcudnn 8.0.4|


## 国产GPU

### 华为NPU-CANN框架
- [CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/index/index.html)
- [Atlas 推理卡 NPU驱动和固件安装指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100493509/426cffd9)
- [华为软件包](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01|23710424|251366513|22892968|251168373)
- [ascend gitee 仓库](https://gitee.com/ascend)
	- [torch-npu](https://gitee.com/ascend/pytorch)

![](./scripts/assets/华为NPU框架.png)

CANN（Compute Architecture for Neural Networks）是华为为昇腾（Ascend）AI处理器打造的**全栈软件平台和异构计算架构**。它的核心使命是充分发挥昇腾AI处理器的强大算力，简化AI应用开发与部署，并充当上层深度学习框架（如TensorFlow, PyTorch, MindSpore）与底层昇腾硬件之间的“桥梁”。

官方文档详细介绍Host和Device的概念以及标准/开放形态，是因为这直接关系到开发者**如何利用CANN进行应用开发、资源调配和性能优化**，是实现其“连接AI框架与硬件的关键角色”这一目标的基础。

CANN官方文档介绍Host（主机）和Device（设备）的概念，以及标准形态（EP模式）与开放形态的区别，主要是因为这关系到**开发模式、资源利用和性能优化**：

*   **Host（主机）**：通常指与昇腾AI处理器相连的**X86或ARM服务器**。它负责整体的控制流、业务逻辑，以及不适合在AI处理器上运行的计算（如某些自定义预处理或后处理）。
*   **Device（设备）**：指安装了**昇腾AI处理器的硬件板卡**（如Atlas 300I推理卡），通过PCIe接口与Host服务器连接。它专注于提供强大的神经网络（NN）计算能力。

这种划分源于异构计算的常见模式，即**专用协处理器（Device）与通用主机（Host）协同工作**。

CANN支持两种主要的Device工作形态，以适应不同的场景需求：

1.  **标准形态（EP模式）**：
    *   􀋾 **Device作为被动协处理器**：在此形态下，昇腾AI处理器工作于**EP（Endpoint）模式**。它作为PCIe总线上的一个从设备，**其上的CPU资源通常仅能通过Host调用**。
    *   􀋾 **常见开发流程**：AI应用程序（如模型推理的所有步骤）主要**运行在Host侧**。Host通过CANN提供的接口（如AscendCL）调用Device的算力。
    *   􀋾 **适用场景**：**大多数推理场景**，开发相对简单，资源管理集中于Host。

2.  **开放形态**：
    *   􀋾 **释放Device侧CPU算力**：此形态下，开发者可以**利用Device板载的Control CPU的通用计算能力**。
    *   􀋾 **开发流程变化**：需要为Device侧的CPU**编译专用的应用程序**（通常使用华为的HCC编译器），并将其放入Device的文件系统镜像中。
    *   􀋾 **主要优势**：
        *   **降低Host负载**：将一些计算任务（如图像/视频预处理）卸载到Device端执行。
        *   **减少数据传输**：数据在Device内部处理，避免了在Host和Device之间的大量数据传输，从而**降低延迟、提升整体效率**。
    *   􀋾 **适用场景**：对**延迟敏感**或希望**最大化利用Device资源**的应用。


官方文档详细介绍Host、Device及两种形态，主要是因为：

*   **明确开发环境配置**：让开发者清楚知道**代码的不同部分将在何处运行**（Host的X86/ARM CPU 还是 Device的NPU或Control CPU），从而正确设置编译环境、链接库和部署路径。
*   **理解性能瓶颈与优化方向**：数据传输 between Host and Device 往往是性能瓶颈之一。了解开放形态的存在，就知道可以通过**将更多计算任务卸载到Device端**来减少数据交换，从而提升性能。
*   **实现资源高效利用**：引导开发者根据实际需求选择合适的形态，避免Device侧CPU资源的闲置，实现**异构计算资源的精细化利用和效率最大化**。
*   **避免概念混淆**：清晰区分Host和Device，有助于理解CANN的工具链（如哪些工具用在Host上分析Device状态）。

CANN 是华为昇腾AI生态的软件核心，它通过一系列工具和接口，让开发者能高效利用昇腾芯片的强大算力。

理解**Host（控制与通用计算）、Device（专用AI计算）** 的概念及**标准（集中控制）、开放（分布式利用）** 两种形态，对于在昇腾平台上进行**高效的应用开发、性能优化和资源管理至关重要**。这直接反映了CANN的设计哲学：**充分发挥异构计算优势，提供灵活且高性能的AI计算解决方案**。#### CANN安装

```bash
# conda 在线自动安装 CANN
conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
conda install ascend::cann-toolkit
```

#### CANN组件

| CANN组件/功能类别   | 关键组成部分/技术举例                                 | 主要功能简介                                       |
| :------------ | :------------------------------------------ | :------------------------------------------- |
| **统一编程接口**    | AscendCL (Ascend Computing Language)        | 提供设备管理、内存管理、任务调度等API，是开发者直接调用的主要接口。          |
| **基础计算库与算子**  | ACL (Ascend Computing Library), AOL算子库      | 提供高效的基础数学运算（如BLAS）和深度优化过的AI算子（如卷积、矩阵运算）。     |
| **深度学习框架支持**  | torch_npu (PyTorch适配), MindSpore            | 实现与主流深度学习框架的无缝集成，允许框架调用昇腾硬件。                 |
| **模型转换与部署工具** | ATC (Ascend Tool Chain)                     | 将其他框架（如ONNX, Caffe）训练的模型转换为昇腾处理器可执行的格式（.om）。 |
| **编译与执行引擎**   | 图编译器, Runtime运行时                            | 将计算图转为硬件可执行指令，并进行深度优化（如算子融合、内存优化）。           |
| **性能调优工具**    | AOE (Ascend Optimization Engine), Profiling | 自动或辅助进行性能调优，例如优化算子调度策略。                      |
| **高级开发与调试支持** | Ascend C语言, msdebug调试工具                     | 支持开发者进行底层算子开发和调试。                            |
| **异构计算管理**    | 调度不同计算单元（NPU, CPU等）                         | 智能分配任务到合适的计算单元，以实现最佳计算性能。                    |
#### CANN依赖

| 类别                                                                                                          | 名称                  | 版本要求                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :---------------------------------------------------------------------------------------------------------- | :------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 安装时所需工具                                                                                                     | Python              | Python3.7._x_至3.11.4版本。<br><br>如果需安装NNAL软件包的Python库，请安装Python3.10._x_或3.11._x_版本。<br><br>如果需安装TensorFlow，请安装要求的Python版本：<br><br>- TensorFlow1.15配套的Python版本是：Python3.7._x_（3.7.5~3.7.11）。<br>- TensorFlow2.6.5配套的Python版本是：Python3.7._x_（3.7.5~3.7.11）、Python3.8._x_、Python3.9._x_。<br><br>安装失败、版本不满足或者未包含动态库libpython3._x_.so请参考[编译安装Python](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_0061.html#ZH-CN_TOPIC_0000002366267590)操作。 |
| python3-pip                                                                                                 | 与已安装的Python版本配套使用。  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 运行时所需工具                                                                                                     | gcc                 | >=7.3.0，以系统源提供的版本为准。                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| g++                                                                                                         | 与已安装gcc版本配套使用。      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 运行时所需Python第三方库                                                                                             | numpy               | 大于等于1.19.2，小于等于1.24。<br><br>Python3.7.x时推荐安装numpy 1.21.6版本。                                                                                                                                                                                                                                                                                                                                                                                                                              |
| decorator                                                                                                   | >=4.4.0             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| sympy                                                                                                       | >=1.5.1             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| cffi                                                                                                        | >=1.12.3            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| protobuf                                                                                                    | 3.20._x_            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| attrs<br><br>cython<br><br>pyyaml<br><br>pathlib2<br><br>scipy<br><br>requests<br><br>psutil<br><br>absl-py | 无版本要求，安装的版本以pip源为准。 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 软件包运行时依赖                                                                                                    | glibc               | 运行NNAL加速库时，glibc版本需大于等于2.17，执行**ldd --version**可以查询glibc版本（主流Linux系统均满足glibc版本要求，若不满足要求，**建议通过重装新版本的系统解决**，不推荐直接升级glibc，直接升级glibc可能导致系统崩溃）。                                                                                                                                                                                                                                                                                                                                              |

#### [onnxruntime-cann](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
```bash
pip install onnxruntime-cann
```

```python
import numpy as np
import onnxruntime as ort

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "enable_cann_graph": True,
        },
    ),
    "CPUExecutionProvider",
]

model_path = '<path to model>'

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.int64)
x_ortvalue = ort.OrtValue.ortvalue_from_numpy(x, "cann", 0)

io_binding = sess.io_binding()
io_binding.bind_ortvalue_input(name="input", ortvalue=x_ortvalue)
io_binding.bind_output("output", "cann")

sess.run_with_iobinding(io_binding)

return io_binding.get_outputs()[0].numpy()
```



## docker 部署onnx-cuda

docker不依赖宿主机的cuda，但却依赖宿主机的显卡驱动。另外，docker 运行还需要安装[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)(win需要wsl2安装后额外配置，下文介绍)；且docker run 指定gpu。

![wsl-cuda](./scripts/assets/wsl-cuda.png)
### docker cuda 支持

#### 1. [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 这是 **Linux 上的通用方案**。
- 如果你在 **裸机 Linux**（比如 Ubuntu server）上跑 Docker，要让容器访问 GPU，就必须装 nvidia-container-toolkit。
- 它负责把宿主机上的 GPU 驱动、CUDA 库 mount 到容器里。
- 没有它，Linux 上的 Docker 就看不到 GPU。
##### 1.1 安装


##### 1.2 配置


#### 2. [CUDA on WSL](https://developer.nvidia.com/cuda/wsl)

要注意的是：wsl2和里面的ubuntu发行版是不同的，wsl2是虚拟机(特殊的)，docker和ubuntu都可以运行在上面。

- 这是 **NVIDIA + Microsoft** 合作的方案，让 WSL2 (Linux 子系统) 可以调用 Windows 驱动里的 GPU。
- 本质上：**让 WSL2 里的 Linux 看见显卡**。

一般这个默认自动安装的（windows装了比较新版本的显卡驱动）；可以用指令来验证：
```bash
wsl -- nvidia-smi
```

#### 3. [GPU support in Docker Desktop for Windows](https://docs.docker.com/desktop/features/gpu/)

- 在 Windows 上的 Docker Desktop **自带对 GPU 的支持**，不需要在 Windows 上额外装 NVIDIA Container Toolkit。(using wsl2 backend)
- 它会自动对接 WSL2 里的 GPU (上面第 1 步)，然后让容器里可以看到 GPU。
- 所以在 Windows Docker Desktop 上跑 GPU 容器，流程是：
    1. Windows 装好 NVIDIA 驱动 (>= 470)。
    2. 直接在容器run的时候加上 `--gpus all` 就能用 GPU。

#### 4. [docker compose cuda](https://docs.docker.com/compose/how-tos/gpu-support/)
下面是官方推荐：
```dockerfile
# swarm 的 docker-compose.yml
services:
  myservice-add-name:
		巴拉巴拉
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
# 正常运行
docker compose up -d
```

### docker pytorch cuda

显卡驱动可以向下兼容cuda版本，所以宿主机的显卡驱动要比较新，然后接下来基础镜像有四种方案：
- 拉一个python，然后容器内装cuda和pip的onnxruntime
- 拉一个[cuda](https://hub.docker.com/r/nvidia/cuda/tags)，然后容器内装python
- 拉一个[PyTorch-cuda-runtime](https://hub.docker.com/r/pytorch/pytorch/tags)，然后里面装onnxruntime
- 拉一个[python-cuda-onnx](https://hub.docker.com/r/microsoft/azureml-onnxruntimefamily)，只需要自己w装库[onnx-cuda-docker博客（已过时）](https://blog.csdn.net/weixin_42939529/article/details/122006947)

虽然看似最后一种方案最好，但是这个镜像是微软做的，只很少几个版本的组合，现在也不再支持，如果不符合自己的版本，可能自己的软件出现兼容性问题，它现在提供[onnx-github-Dockerfile.cuda](https://github.com/microsoft/onnxruntime/blob/main/dockerfiles)可以自行构建（网络问题……构建也很慢）。

坑最少的还是第三种方案，但是也需要注意两点：
1. onnxruntime的版本所需要的一定要和pytorch所带的cuda和cudnn版本一致。
2. 代码中的导入方式不太一样，具体见onnx官网doc

### [tensorflow-docker-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags)


# 二、具体 LLM 部署平台

vllm、llama.cpp、ollama、openllm

vllm是支持并发最好的，llama.cpp是支持平台最多的，ollama是最简单性能也是最差的。

## [vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)Nvidia部署

```
┌─────────────────────────────────────────────┐
│           vllm/vllm-openai 容器              │
│  ┌───────────────────────────────────────┐  │
│  │  CUDA Runtime (cuDNN, cuBLAS等)        │  │  ← 镜像自带
│  │  PyTorch, vLLM, Python                │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│        nvidia-container-toolkit             │  ← 宿主机安装
├─────────────────────────────────────────────┤
│           NVIDIA Driver                     │  ← 宿主机安装
├─────────────────────────────────────────────┤
│              GPU 硬件                        │
└─────────────────────────────────────────────┘
```


- linux 需要先安装显卡驱动 和 nvidia-container-toolkit。比如ubuntu可以通过apt安装，但一般都会比较老，会有兼容性问题，去英伟达官方搜索。
- windows需要安装nvidia驱动和wsl2(不建议在windows部署)


### vllm 参数

#### gpu-memory-utilization

vLLM 启动时会检查当前 free memory 是否足够，并通过 profile 计算能给 KV cache 留多少。所以不要让 gpu-memory-utilization * 总的 GPU 显存 大于所需要的 模型 + 常驻 buffer + KV cache 。这些大小可以用如下的指令推测：

```bash
docker logs vllm-qwen36-35b-a3b 2>&1 | grep -Ei "kv cache|gpu blocks|maximum concurrency|available memory|profile"
(EngineCore pid=475) INFO 05-25 15:28:40 [gpu_model_runner.py:5920] Encoder cache will be initialized with a budget of 65536 tokens, and profiled with 4 image items of the maximum feature size.
(EngineCore pid=475) INFO 05-25 15:35:14 [gpu_worker.py:462] Available KV cache memory: 16.42 GiB
(EngineCore pid=475) INFO 05-25 15:35:14 [gpu_worker.py:477] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.8000 is equivalent to --gpu-memory-utilization=0.7989 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.8011. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=475) INFO 05-25 15:35:14 [kv_cache_utils.py:1710] GPU KV cache size: 774,084 tokens
(EngineCore pid=475) INFO 05-25 15:35:14 [kv_cache_utils.py:1711] Maximum concurrency for 65,536 tokens per request: 11.81x
(EngineCore pid=475) INFO 05-25 15:35:21 [core.py:299] init engine (profile, create kv cache, warmup model) took 401.63 s (compilation: 98.54 s)

docker logs vllm-qwen3-embed-4b 2>&1 | grep -Ei "kv cache|gpu blocks|maximum concurrency|available memory|profile"
(EngineCore pid=374) INFO 05-25 15:29:28 [gpu_worker.py:462] Available KV cache memory: -6.54 GiB
(EngineCore pid=374) INFO 05-25 15:29:28 [gpu_worker.py:477] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.1500 is equivalent to --gpu-memory-utilization=0.1480 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.1520. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=374) ERROR 05-25 15:29:28 [core.py:1140] ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.
(EngineCore pid=374) ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.
(EngineCore pid=208) INFO 05-25 15:30:18 [gpu_worker.py:462] Available KV cache memory: 5.7 GiB
(EngineCore pid=208) INFO 05-25 15:30:18 [gpu_worker.py:477] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.1500 is equivalent to --gpu-memory-utilization=0.1480 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.1520. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=208) INFO 05-25 15:30:18 [kv_cache_utils.py:1710] GPU KV cache size: 41,504 tokens
(EngineCore pid=208) INFO 05-25 15:30:18 [kv_cache_utils.py:1711] Maximum concurrency for 8,192 tokens per request: 5.07x
(EngineCore pid=208) INFO 05-25 15:30:20 [core.py:299] init engine (profile, create kv cache, warmup model) took 12.33 s (compilation: 7.22 s)

```




`--gpu-memory-utilization 0.8` 在 vLLM 里**不是 "总固定占 0.8"**，它的语义是：

> vLLM 自己能用的预算 ≈ `total_gpu_mem × 0.8 − 启动时刻 GPU 上其他进程已占用的显存`

RTX PRO 6000 是 96G，所以预算上限 ≈ 76.8G。但你这张卡是三模型共用：

| 场景 | 启动时刻其他模型已占 | vLLM 拿到的预算 | 观测显存 |
|---|---|---|---|
| vLLM 先起，干净卡 | ~0 | ~76G | **76G** ✓ |
| Embed-4B 已加载 | ~11G | ~65G | **65G** ✓ |
| Embed + Whisper 都在工作 | ~20G | ~56G | **56G** ✓ |

这部分预算在启动时通过一次 profile_run 测出 `model + activation peak`，剩余的全部一次性预分配给 KV cache。所以一旦启起来就**不会再涨**，看到的浮动主要来自：

1. **启动时其他进程占多少**（最大变量，正好解释你的 56/65/76）
2. 运行时 activation buffer 的瞬时峰值（小，几百 MB 到几 GB）
3. PyTorch caching allocator 不会主动还给 OS

主要是vllm占了的显存不还回来，所以，原则如下：
- 首先先理论计算所有服务的大致显存占用。
- 然后单个服务启动测试 `docker logs <name> 2>&1 | grep -Ei "kv cache|gpu blocks|maximum concurrency|available memory|profile"` 显存分布，进一步确定每个服务的显存分配。
- 不要让 gpu-memory-utilization * 总的 GPU 显存 大于所需要的 模型 + 常驻 buffer + KV cache。
- 先启动固定占用显存的服务，再启动占用显存小的VLLM推理服务，最后启动占用显存大的VLLM推理服务。



### 查看vllm服务参数/状态/性能

#### 1. 查看 vLLM 运行时参数

##### 方法一：API 端点查询

vLLM 提供了多个 API 端点可以查看运行时信息：

```bash
# 基本模型信息
curl http://localhost:8123/v1/models | jq

# 详细配置信息（vLLM 特有）
curl http://localhost:8123/v1/model_info | jq

# 服务器健康检查
curl http://localhost:8123/health
```

##### 方法二：进入容器查看日志

```bash
# 查看启动日志（包含详细配置）
docker logs vllm-qwen3-4090-awq

# 实时跟踪日志
docker logs -f vllm-qwen3-4090-awq
```

启动日志会显示类似这样的关键信息：

```
INFO: Model config: ...
INFO: KV cache data type: auto
INFO: GPU memory utilization: 0.90
INFO: Maximum number of batched tokens: 8192
INFO: Number of GPU blocks: XXXX
INFO: Number of CPU blocks: XXXX
```

##### 方法三：Python 脚本查询详细信息

```python
import requests
import json

BASE_URL = "http://localhost:8123"

def get_model_info():
    """获取模型基本信息"""
    resp = requests.get(f"{BASE_URL}/v1/models")
    print("=== 模型列表 ===")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def get_detailed_info():
    """获取详细配置（vLLM 特有端点）"""
    endpoints = [
        "/v1/model_info",
        "/metrics",  # Prometheus 格式的指标
    ]

    for ep in endpoints:
        try:
            resp = requests.get(f"{BASE_URL}{ep}")
            print(f"\n=== {ep} ===")
            if "json" in resp.headers.get("content-type", ""):
                print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            else:
                # metrics 是文本格式
                print(resp.text[:2000])  # 截断显示
        except Exception as e:
            print(f"{ep}: {e}")

def get_metrics_parsed():
    """解析 Prometheus 指标中的 KV Cache 信息"""
    resp = requests.get(f"{BASE_URL}/metrics")
    lines = resp.text.split('\n')

    print("\n=== KV Cache 相关指标 ===")
    kv_keywords = ['kv_cache', 'gpu_cache', 'cache_block', 'prefix_cache']
    for line in lines:
        if any(kw in line.lower() for kw in kv_keywords):
            print(line)

    print("\n=== GPU 内存相关 ===")
    mem_keywords = ['gpu_memory', 'memory_usage']
    for line in lines:
        if any(kw in line.lower() for kw in mem_keywords):
            print(line)

if __name__ == "__main__":
    get_model_info()
    get_detailed_info()
    get_metrics_parsed()
```

##### 方法四：进入容器执行诊断

```bash
# 进入容器
docker exec -it vllm-qwen3-4090-awq bash

# 在容器内查看 GPU 状态
nvidia-smi

# 查看 Python 环境中的 vLLM 配置
python -c "import vllm; print(vllm.__version__)"
```

---

#### 2. vLLM Benchmark 测试

##### vllm bench 的部署方式

**两种方式都可以**：

- **本地 Python**：直接 pip install vllm 后使用
- **Docker 内执行**：进入已有容器或启动新容器

##### 方式一：本地 Python 安装（推荐用于 benchmark）

```bash
# 创建虚拟环境
conda create -n vllm-bench python=3.11 -y
conda activate vllm-bench

# 安装 vllm（与你的 Docker 版本一致）
pip install vllm==0.12.0
```

##### 方式二：在 Docker 容器内执行

```bash
# 进入正在运行的容器
docker exec -it vllm-qwen3-4090-awq bash

# 或者启动新容器专门做 benchmark
docker run --rm -it --gpus all \
    -v "D:/dev_software/AI_models/huggingface/Qwen3-30B-A3B-AWQ-4bit:/models/qwen3-awq:ro" \
    vllm/vllm-openai:v0.12.0 \
    bash
```

---

#### 3. Benchmark 命令详解

##### 3.1 离线 Throughput 测试（不需要服务运行）

```bash
# 在容器内或本地 Python 环境执行
vllm bench throughput \
    --model /models/qwen3-awq \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 100 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --quantization awq
```

##### 3.2 在线延迟测试（需要服务运行）

先确保你的 Docker 服务已启动，然后：

```bash
# 测试延迟
vllm bench latency \
    --model Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
    --base-url http://localhost:8123 \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 50
```

---

#### 4. 查看详细 KV Cache 配置的方法

##### 方法一：启动时添加详细日志

修改你的 docker-compose.yaml，添加日志参数：

```yaml
command:
  # ... 其他参数 ...
  - "--log-level"
  - "debug"
  # 或者使用
  # - "-v"  # verbose 模式
```

##### 方法二：查看 /metrics 端点

```bash
# 获取所有指标
curl http://localhost:8123/metrics | grep -E "(cache|block|memory)"
```

关键指标解释：

|指标|含义|
|---|---|
|`vllm:num_gpu_blocks_total`|GPU 上 KV Cache 总块数|
|`vllm:num_cpu_blocks_total`|CPU 上 KV Cache 总块数|
|`vllm:gpu_cache_usage_perc`|GPU Cache 使用率|
|`vllm:prefix_cache_hit_rate`|Prefix Cache 命中率|
|`vllm:num_preemption_total`|抢占次数（KV Cache 不足时发生）|

##### 方法三：使用 vLLM 内部 API（需要修改启动方式）

如果你想获取更详细的配置，可以用 Python 直接加载模型查看：

```python
"""
离线查看模型配置（不启动服务）
"""
from vllm import LLM
from vllm.config import CacheConfig

# 只初始化，不加载权重（快速查看配置）
llm = LLM(
    model="/models/qwen3-awq",  # 本地路径
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    quantization="awq",
    # 仅用于查看配置，实际推理时去掉
    enforce_eager=True,
)

# 查看配置
print("=== Model Config ===")
print(f"Hidden size: {llm.llm_engine.model_config.hf_config.hidden_size}")
print(f"Num layers: {llm.llm_engine.model_config.hf_config.num_hidden_layers}")
print(f"Num KV heads: {llm.llm_engine.model_config.hf_config.num_key_value_heads}")
print(f"Head dim: {llm.llm_engine.model_config.hf_config.hidden_size // llm.llm_engine.model_config.hf_config.num_attention_heads}")

print("\n=== Cache Config ===")
cache_config = llm.llm_engine.cache_config
print(f"Block size: {cache_config.block_size}")
print(f"Num GPU blocks: {cache_config.num_gpu_blocks}")
print(f"Num CPU blocks: {cache_config.num_cpu_blocks}")
print(f"Cache dtype: {cache_config.cache_dtype}")

print("\n=== Scheduler Config ===")
scheduler_config = llm.llm_engine.scheduler_config
print(f"Max num seqs: {scheduler_config.max_num_seqs}")
print(f"Max num batched tokens: {scheduler_config.max_num_batched_tokens}")

# 计算 KV Cache 大小
num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
num_kv_heads = llm.llm_engine.model_config.hf_config.num_key_value_heads
head_dim = llm.llm_engine.model_config.hf_config.hidden_size // llm.llm_engine.model_config.hf_config.num_attention_heads
dtype_bytes = 2  # FP16/BF16 = 2 bytes

kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
print(f"\n=== KV Cache 计算 ===")
print(f"KV Cache per token: {kv_cache_per_token / 1024:.2f} KB")
print(f"KV Cache for 8192 tokens: {kv_cache_per_token * 8192 / 1024 / 1024 / 1024:.2f} GB")
```

---

#### 5. 完整 Benchmark Docker Compose（独立测试容器）

如果你想有一个专门用于 benchmark 的配置：

```yaml
# docker-compose.bench.yaml
services:
  vllm-bench:
    image: vllm/vllm-openai:v0.12.0
    container_name: vllm-bench
    volumes:
      - "D:/dev_software/AI_models/huggingface/Qwen3-30B-A3B-AWQ-4bit:/models/qwen3-awq:ro"
    gpus: all
    ipc: host
    shm_size: "16gb"
    entrypoint: ["bash"]
    stdin_open: true
    tty: true
```

使用：

```bash
# 启动 benchmark 容器
docker compose -f docker-compose.bench.yaml up -d

# 进入容器
docker exec -it vllm-bench bash

# 在容器内运行 benchmark
vllm bench throughput \
    --model /models/qwen3-awq \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 50 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# 退出后清理
docker compose -f docker-compose.bench.yaml down
```

---

#### 总结

| 需求          | 推荐方法                                 |
| ----------- | ------------------------------------ |
| 快速查看运行参数    | `docker logs`+`/metrics`端点        |
| KV Cache 监控 | `/metrics`端点 + Python 脚本            |
| 吞吐量测试       | `vllm bench throughput`（Docker 内或本地） |
| 延迟测试        | `vllm bench latency`或 Python 脚本     |
| 详细配置查看      | Python 直接加载 LLM 对象                   |


## vllm 华为显卡部署

首先要[确认操作系统和硬件的兼容性](https://www.hiascend.com/hardware/compatibility)。（这个链接的兼容性好像更严格）

[安装华为显卡相关](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition)的：
- [显卡驱动 & 固件](https://www.hiascend.com/hardware/firmware-drivers/community)
- CANN-toolkit
- CANN-kernals
- pytorch(CANN兼容版本)。

再[安装vllm相关](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)。一定要注意[版本兼容](https://docs.vllm.ai/projects/ascend/zh-cn/latest/community/versioning_policy.html)问题。比如下表：

| vLLM Ascend | vLLM    | Python          | Stable CANN | PyTorch/torch_npu   |
| ----------- | ------- | --------------- | ----------- | ------------------- |
| v0.13.0rc1  | v0.13.0 | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0 / 2.8.0       |
| v0.11.0     | v0.11.0 | >= 3.9 , < 3.12 | 8.3.RC2     | 2.7.1 / 2.7.1.post1 |
| v0.12.0rc1  | v0.12.0 | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0 / 2.8.0       |
| v0.11.0rc3  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1.post1 |
| v0.11.0rc2  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1       |
| v0.11.0rc1  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC1     | 2.7.1 / 2.7.1       |

所以我如果下载8.3.RC1（华为官网推荐这个版本），那就要安装pytorch 2.7.1、vLLM v0.11.0 和 vllm-ascend v0.11.0rc1。注意如果使用docker，那么CANN是容器中的，宿主机只是显卡驱动和显卡固件要兼容，

### 准备

- 教程(选择-软件安装-安装指南): https://www.hiascend.com/document/detail/zh/CANNCommunityEdition
- 系统: 统信UOS（CentOS（RPM）体系）
- server: 华为 泰山服务器
- CPU: 鲲鹏920
- GPU: Atlas 300I Duo
- ssh信息: root@192.168.50.117 -p 36406

#### 下载文件

驱动、固件、CANN-toolkit、CANN-kernels

- Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run
- Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run

- Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
- Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run

#### 下载链接

- [ 驱动 & 固件 下载](https://www.hiascend.com/hardware/firmware-drivers/community)
- [CANN 下载](https://www.hiascend.com/zh/developer/download/community/result?module=cann)（最好是 [pytorch 和 CANN](https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=7.2.0&cann=8.3.RC1&product=2&model=17)都下载安装了, pytorch需要基于CANN, 后者类似CUDA）
- [cann-driver仓库(可以参考实现,但不刚需)](https://gitcode.com/cann/driver)

### 安装驱动 & 固件



1. 拷贝

```bash
scp -P 36406 Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
scp -P 36406 Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run root@192.168.50.117:/home/LLM_project/CANN/
```

2. 安装

```bash
cd /home/LLM_project/CANN
id HwHiAiUser

# 如果没有该用户
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m -s /bin/bash HwHiAiUser

chmod +x Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run
chmod +x Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run

./Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run --check
./Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run --check

# 如果check ok
./Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run --full --install-for-all
./Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run --full

# 重启
reboot

# 检查是否有驱动
npu-smi info
```

### 安装 CANN


1. 拷贝

```bash
scp -P 36406 Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
scp -P 36406 Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
```

2. 安装

```bash
cd /home/LLM_project/CANN
chmod +x Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --check

# 如果 check All good. （如果是升级，就 --upgrade）
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --install

# 安装完成后，若显示后文信息，则说明软件安装成功：xxx install success
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 cann-kernels
chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux.run
./Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run --check

# 如果 check All good. （kernels升级也要用install upgrade有问题）
./Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run --install

# 刚才设置的环境变量需要在 bashrc/zshrc 中生效
# echo '. /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
# source ~/.bashrc
echo '. /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.zshrc
source ~/.zshrc

# 查看 CANN 版本信息
cat /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info
```


3. 安装NNAL神经网络加速库（可选）

NNAL神经网络加速库中提供了ATB（Ascend Transformer Boost）加速库和SiP（AscendSiPBoost）信号处理加速库。

加速库安装之前，需已安装同一版本的Toolkit并配置环境变量。

1. 增加对软件包的可执行权限。
```bash
chmod +x Ascend-cann-nnal_<version>_linux-aarch64.run
```

2. 安装软件包（安装命令支持`--install-path=<path>`等参数，具体使用方式请参见[参数说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0043.html)）。
```bash
./Ascend-cann-nnal_8.5.0.alpha002_linux-aarch64.run --install
```
如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：“/usr/local/Ascend”，非root用户：`_${HOME}_/Ascend`，`_${HOME}_`为当前用户目录。

3. 配置环境变量，当前以root用户安装后的默认路径为例，请用户根据加速库对应的set_env.sh的实际路径进行替换。需注意，不支持同时配置ATB和SiP的环境变量脚本。

- ATB加速库：
```bash
source /usr/local/Ascend/nnal/atb/set_env.sh

# 如果不报错
echo '. /usr/local/Ascend/nnal/atb/set_env.sh' >> ~/.zshrc
source ~/.zshrc
```

- SiP加速库：
```bash
source /usr/local/Ascend/nnal/asdsip/set_env.sh

# 如果不报错
echo '. /usr/local/Ascend/nnal/asdsip/set_env.sh' >> ~/.zshrc
source ~/.zshrc

# 如果报错
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
```

上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

### 部署 vllm-ascend

python 版本和docker版本是相互独立的，但都依赖上述安装的系统级软件包。docker其实就是在容器里安装好了python，所以使用docker是更方便的。

#### docker 部署

- [注意看其中的docker部分](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)
- [vllm-ascend-docker](https://quay.io/repository/ascend/vllm-ascend?tab=tags)
- [cann-py-docker](https://hub.docker.com/r/ascendai/cann/tags)
- [ascend-docker国内快速](https://quay.io/organization/ascend)

```bash
docker pull quay.io/ascend/vllm-ascend:v0.11.0rc1-310p
```

检查docker内部信息：

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc1-310p

docker run --rm ${IMAGE} bash -lc '
set -e
ARCH=$(uname -m)
FILE=/usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/ascend_toolkit_install.info
if [ -f "$FILE" ]; then
  echo "==> $FILE"
  grep -E "^(package_name|version|innerversion|path)=" "$FILE"
else
  echo "not found: $FILE"
  echo "try list:"
  ls -la /usr/local/Ascend/ascend-toolkit/latest/ || true
  find /usr/local/Ascend/ascend-toolkit/latest -maxdepth 2 -name ascend_toolkit_install.info -print -exec sh -c "echo ==== {}; grep -E \"^(package_name|version|innerversion|path)=\" {}" \; || true
fi
'
```

package_name=Ascend-cann-toolkit
version=8.3.RC1
innerversion=V100R001C23SPC001B235
path=/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux

#### python 部署(与docker部署替代关系)

除了下文这种方式，还可以自己根据官方支持的docker镜像来安装自己的python镜像。
- [cann-py-docker](https://hub.docker.com/r/ascendai/cann/tags)
- [ascend-docker国内快速](https://quay.io/organization/ascend)

```bash
docker pull quay.io/ascend/cann:8.3.rc1-310p-ubuntu22.04-py3.11

```
##### 安装 pytorch


1. 先安装python


版本不能随便安装，要结合所有库的支持版本的交集，比如vllm 0.11.0 要求python版本不低于3.9，那么python 3.8就不行。各种都要搜集。我综合各种条件决定安装 python 3.11.13。

```bash
# 查看 python 支持的版本
yum list | grep python

# 已当前系统支持的python39为例
sudo yum module enable -y python39
sudo yum install -y python39 python39-pip python39-setuptools python39-wheel

# 安装uv
pip3.9 install uv

# 安装虚拟环境
mkdir -p ~/.python-user
uv venv --python /usr/bin/python3.9 ~/.python-user/default

# 虚拟环境临时生效
source ~/.python-user/default/bin/activate
```

上述版本如果不能满足要求，可以选择源码安装（但是gcc 版本不太行，安装gcc高版本要见《llama.cpp 本地编译(华为)》）
```bash
# 安装 python 编译依赖
sudo yum install -y \
  gcc-toolset-12-gcc \
  gcc-toolset-12-gcc-c++ \
  gcc-toolset-12-libstdc++-devel

sudo yum install -y make wget \
  zlib-devel bzip2-devel xz-devel \
  readline-devel sqlite-devel \
  openssl-devel libffi-devel \
  ncurses-devel tk-devel gdbm-devel
```

2. 再安装 pytorch

- [华为官网-安装pytorch教程](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)
	再次强调，注意版本！

3. 部署 pip 版 vllm

- [注意看其中的pip部分](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)

### vllm 华为 bug

- [LinearOperation CreateOperation failed](https://github.com/Ascend/pytorch/issues/94)
- [vllm-ascend:v0.11.0rc2 qwen3-next-80B OOM](https://github.com/vllm-project/vllm-ascend/issues/4474)



## llama.cpp 华为显卡部署

同样是有两种方式，推荐是docker，因为本地gcc相关的环境容易改出问题

###  1-1 llama.cpp docker (华为)

- [llama.cpp-docker](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)

```bash
git clone https://git.ustc.edu.cn/ustc-os-lab/llama.cpp.git
cd llama.cpp

# 编辑这个dockerfile 修改基础镜像的版本适配本机安装的CANN, 然后 :wq 保存
vim .devops/llama.cpp.cann.Dockerfile

docker build -f .devops/cann.Dockerfile -t llama-cpp-cann:full --target full .
```

###  1-2 llama.cpp 本地编译(华为)

- [llama.cpp-CANN](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

```bash
# clone 代码 (国内镜像)
git clone https://git.ustc.edu.cn/ustc-os-lab/llama.cpp.git
cd llama.cpp

# 查看可用版本
yum list | grep gcc-toolset

# 安装
sudo yum install -y \
  gcc-toolset-12-gcc \
  gcc-toolset-12-gcc-c++ \
  gcc-toolset-12-libstdc++-devel

# gcc-toolset-12 (加入环境变量, 如果想要持久生效加入 .zshrc)  
export GCC12_ROOT=/opt/UOS/gcc-toolset-12/root/usr
export PATH="$GCC12_ROOT/bin:$PATH"

# 库环境变量可能会有问题: 某些程序本来应该加载系统的库版本，但因为把 toolset 的 libstdc++ 放前面，程序加载了不同版本的 libstdc++.so.6，少数情况下会出现兼容性问题
export LD_LIBRARY_PATH="$GCC12_ROOT/lib64:${LD_LIBRARY_PATH}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh --force

cmake -B build \
-DGGML_CANN=ON \
-DCMAKE_BUILD_TYPE=Release \
-DSOC_TYPE=ascend${CHIP_TYPE} \

cmake --build build --config Release -j$(nproc)

# 运行可能还有问题, 因为有些依赖的动态库可能版本不兼容

```

### 2. llama.cpp 模型

- [unsloth-qwen3](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)

- 它的入口通常是 llama-cli

- CANN 后端对量化格式有限制
    仓库的 CANN 指南明确写了：目前只支持 **FP16 / Q4_0 / Q8_0**。
    这点很关键：Hugging Face 上很多 GGUF 是 Q4_K_M、Q5_K_M、IQ* 等，你在 Ascend/CANN 上不一定能用（至少这份指南口径是不支持）。

- lama.cpp 通用要求：模型必须是 GGUF
    llama.cpp 官方 README：llama.cpp 需要模型是 **GGUF**；其他格式要用仓库里的 convert_*.py 脚本转换。

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /home/LLM_project/models/GUFF/Qwen3-30B-A3B-Q4_0

hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_0.gguf \
   --local-dir /home/LLM_project/models/GUFF/Qwen3-30B-A3B-Q4_0
```


### 3-1 docker 部署

- 《AI-config/llama-cpp-docker/llama-cpp-ascend.yml》

```bash
docker compose -f llama-cpp-ascend.yml up -d
```

### 3-2 本地运行

```bash
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

### 4 llama.cpp MoE bug


```
load_tensors:        CANN0 model buffer size =    48.80 MiB   ← NPU 只放了 48MB
load_tensors:  CPU_AARCH64 model buffer size = 15390.00 MiB  ← 模型主体还在 CPU
```

虽然显示 `offloaded 49/49 layers`，但实际上 **MoE 的专家层没有真正卸载到 NPU**。

这是 llama.cpp CANN 后端对 **MoE 架构支持不完整** 的问题。Qwen3-30B-A3B 有 128 个专家，这些专家权重仍在 CPU 上。

#### 验证方式

```
graph splits = 483
```

483 次 CPU/NPU 切换，说明大量计算在 CPU 和 NPU 之间来回跳。

#### 解决方案

**方案 1：换非 MoE 模型**（推荐）

在 310P3 上跑 **Qwen2.5-14B** 或 **Qwen2.5-7B** 这类 Dense 模型会快很多：

```bash
# 下载 Qwen2.5-14B-Instruct Q4_K_M（约 8.5GB，完全放得下）
```

**方案 2：继续用 vLLM**

你之前用的 vLLM 对华为 NPU 和 MoE 架构支持应该更成熟。如果一定要跑 Qwen3-30B-A3B，vLLM 可能是更好的选择。

**方案 3：等 llama.cpp 更新**

CANN 后端对 MoE 的支持还在完善中，可以关注 [llama.cpp CANN 相关 issue](https://github.com/ggerganov/llama.cpp/issues)。

---



## tensorflow 华为显卡

- [ascend-tensorflow仓库](https://gitee.com/ascend/tensorflow)


## 模型与显卡的性能指标

### 模型

模型本身的参数：
- 模型的参数量、量化bit位数、上下文、激活的参数量。

模型使用的参数：
- 单访问 tokens/s
- 多用户并发总 tokens/s
- first token 延迟
- 100token输入 & 500token 输出单用户耗时

### 显卡

单卡的性能（q4 q8 的tops）、单卡的显存容量、单卡的显存带宽

多卡的通信瓶颈、有无nvlink的差距；nvlink的支持情况。


## llm部署优化


原则上应该是尽量单卡能运行模型，然后并发多卡独立部署再用ngi 做负载均衡


单卡够用时的选择，不需要 tensor-parallel 的情况；如果单卡显存足够，`--tensor-parallel-size 2` 反而可能**降低性能**，因为：

- 两卡之间有通信开销（NVLink/PCIe）
- 同步等待会增加延迟

想用两张卡提高并发，正确做法是**启动两个独立实例**，每个实例用一张卡：

```bash
# 实例1：使用 GPU 0，端口 8000
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000

# 实例2：使用 GPU 1，端口 8001
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8001
```

然后用 Nginx 做负载均衡：
```nginx
upstream vllm_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backend;
    }
}
```

|场景|推荐方案|原因|
|---|---|---|
|单卡放不下模型|`--tensor-parallel-size 2`|必须拆分|
|单卡够用，想提高并发|两个独立实例 + 负载均衡|吞吐量更高，无通信开销|
|单卡够用，想降低单次延迟|`--tensor-parallel-size 2`|可能略有提升，但不明显|

### vLLM 配置侧的关键优化点


0) 先把“版本与已知坑”固定住

- Qwen3-Next 官方明确要求 vllm>=0.10.2。 
- vLLM 的 Qwen3-Next recipes 里提到：如果遇到 CUDA illegal memory access，可加 --compilation_config.cudagraph_mode=PIECEWISE。 
- vLLM 的核心吞吐来自连续批处理、PagedAttention、chunked prefill 等能力。 


1) 你这种“每次几千 token”的负载，第一优先级是避免 preemption（抢占/重算）

抢占会导致重算，直接把尾延迟打爆。vLLM 文档建议的处理手段很明确：


- 提高 gpu_memory_utilization 给 KV cache 更多显存
- 或降低 max_num_seqs / max_num_batched_tokens 减少同批并发占用的 KV 空间 


实操建议（思路，不是唯一答案）

  - 单实例独占 GPU：gpu_memory_utilization 往 0.92~0.97 区间试（视你显存余量而定）。 
- 只要监控里出现频繁 preempt/recompute，就不要继续“加并发”，而是先把 KV 空间扩出来（或把上下文/批次控下来）。


2) 开启并调好 chunked prefill（你的场景几乎必开）


RAG 长 prompt 的典型问题是：长 prefill 会压住 decode，短请求被“堵车”。chunked prefill 的目的就是把长 prefill 切块，和 decode 交错调度，从而同时改善吞吐与延迟。


你要关注的不只是 --enable-chunked-prefill，还包括“长 prompt 并发 prefill 的上限”，避免一堆超长 prompt 同时进来把 GPU 步长全占了：

- --max-num-partial-prefills
- --max-long-partial-prefills
- --long-prefill-token-threshold（多版本文档/参数解释一致） 

经验法则：

- RAG 高并发时，把 “long partial prefills” 设得比 “partial prefills” 更保守，让短请求更容易插队，p95 会明显好看。 

3) 打开 Automatic Prefix Caching，并让你的请求真正“吃到缓存”

vLLM 的 APC（自动前缀缓存）会缓存已处理前缀对应的 KV blocks，新请求如果共享同一前缀即可跳过那段 prefill，属于典型“几乎白给”的优化，且不改变输出。

但注意：APC 的收益取决于前缀共享率。想吃到缓存，你需要在 RAG 应用侧配合（后面会讲）。

另外，如果你多实例扩容，要尽量让同前缀请求路由到同一实例，否则缓存命中率会被打散。Ray Serve 就提供了面向 prefix caching 的路由策略思路（强调“缓存命中比完美负载均衡更重要”）。



4) 用 

max_num_batched_tokens

 / 

max_num_seqs

 控吞吐-延迟的杠杆


这俩是 vLLM 服务端最核心的“批量化闸门”：


- max_num_seqs 控同一轮调度里最多并发序列数
- max_num_batched_tokens 控每轮最多处理的总 token 预算
    文档与参数列表里明确给出了这两个概念。 

建议调参顺序（适用于你的“长输入+多用户”）：

1. 先把 max_model_len（上下文上限）设到你业务真正需要的值，别为了“以防万一”拉满；这会直接决定 KV 预算天花板。
2. 开 chunked prefill 后，先用偏保守的 max_num_seqs 保 p95，再逐步加 max_num_batched_tokens 拉吞吐。
3. 一旦出现 preemption 或 GPU KV cache usage 接近 1 且排队加长，就回退并发或缩短上下文。

4) 监控一定要上：用 /metrics 盯住“是不是在浪费算力”


vLLM 的指标文档把指标分为 server-level 与 request-level 两类，非常适合用来定位是“KV 不够”“长 prompt 堵车”“缓存没命中”还是“调度参数太保守”。

并且 vLLM 的 OpenAI server 会暴露 Prometheus 格式指标，官方也有 Prometheus+Grafana 的示例。


你这种场景建议重点盯：

- vllm:gpu_cache_usage_perc（KV 使用率）与 vllm:gpu_prefix_cache_hit_rate（前缀缓存命中） 
- TTFT、TPOT（每 token 延迟）相关的 request-level 直方图
- preemption/recompute 相关计数（如果有，就说明 KV/并发配置不合理） 

6) 两张 GPU 时可考虑“Prefill/Decode 解耦”，但要当成高级选项


如果你经常有“极长 prompt + 同时很多短请求”，单引擎即便 chunked prefill 也可能出现明显干扰。vLLM 提供了把 prefill 和 decode 放到不同实例/GPU 的实验性方案（KV 在两者间传输）。

这种思路也被社区文章用来解释“长 prompt 阻塞短请求”，并给出拆分部署的方向。

但它是“工程换收益”：会引入 KV 传输与更多运维复杂度，且文档明确是 experimental。

建议你先把：APC + chunked prefill + 并发闸门 + RAG 侧减 prompt/减调用 做到位，再评估是否需要上解耦。


# LLM 推理部署问题


## 理论问题


## 实践问题

### tensorflow内存overflow问题
用tf2onnx，然后用onnx推理就没有内存overflow的问题了。
