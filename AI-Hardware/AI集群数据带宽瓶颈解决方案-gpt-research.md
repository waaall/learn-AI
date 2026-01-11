# 现代 AI 训练集群中的数据带宽瓶颈与解决方案

## 1. NVIDIA 的"专有超节点 (Supernode) scale-up"方案

**NVLink/NVSwitch 互连**：NVIDIA 倾向通过 NVLink 和 NVSwitch 在单机或单柜内部将多块 GPU 组成高速、低延迟的紧耦合 GPU "超节点"[Designing and implementing Enterprise RAG Platform on Azure AI Foundry](https://i-banerjee83.medium.com/architecting-predictable-ai-data-centers-a-systems-view-of-nvidia-dgx-superpod-c3a210546e42)。
在典型的 DGX 系统中，8 张 A100/H100 GPU 通过 NVSwitch 实现全互联，每个 GPU 与任意其他 GPU 间都有均等的大带宽低时延连接，以构成单一逻辑 GPU 复合体。
这使得节点内的 collective 操作（如 AllReduce）直接经由 NVLink 进行，避免传统 PCIe 分层拓扑的瓶颈。
DGX SuperPOD 架构即遵循这一思路：节点内 GPU 经 NVLink/NVSwitch 打造高速通信域，节点间则通过 InfiniBand 等高速网络通信。
正如 NVIDIA 工程文档所述："DGX 节点内 GPU 通过 NVLink/NVSwitch 形成独立于 CPU 的高带宽低延迟 GPU 数据结构；多个 DGX 节点之间通过 InfiniBand 进行通信，实现分布式训练"

**跨节点网络与 SuperPOD**：当需要扩展到多节点集群时，NVIDIA 利用 InfiniBand 或高端以太网（如 RoCE）将多个超节点互联起来，确保跨节点通信性能。
NVIDIA 的 DGX SuperPOD 参考架构即在单机柜内使用 NVLink/NVSwitch，将数个 GPU 打造成一个超节点，然后通过 Quantum HDR/NDR InfiniBand 骨干网络连接多个超节点[Architecting Predictable AI Data Centers: A Systems View of NVIDIA DGX SuperPOD](https://i-banerjee83.medium.com/architecting-predictable-ai-data-centers-a-systems-view-of-nvidia-dgx-superpod-c3a210546e42)。

**持续演进的整柜超节点**：最近一两年，NVIDIA 正将超节点的一体化规模从单机扩展到整机架。例如最新发布的 Vera Rubin NVL72 架构，在一个全液冷机柜内实现 72 个 GPU 全互联，总内部通信带宽达 260 TB/s[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)。

| 特性                            | 第四代 (NVLink 4)    | 第五代 (NVLink 5)     | 第六代 (NVLink 6) |
| :------------------------------ | :------------------- | :-------------------- | :---------------- |
| **NVLink 带宽 (每 GPU)**  | 900 GB/s             | 1,800 GB/s            | 3,600 GB/s        |
| **最大链路数量 (每 GPU)** | 18                   | 18                    | 36                |
| **支持的 NVIDIA 架构**    | NVIDIA Hopper™ 架构 | NVIDIA Blackwell 架构 | NVIDIA Rubin 平台 |

在 Rubin NVL72 中，NVLink Switch 芯片构建了非阻塞全互联架构，每对 GPU 之间的链路带宽高达 3.6 TB/s，使 72 卡可作为单一加速器使用。 NVIDIA NVLink 第六代技术在架级规模上的突破，随着 H100 "NVL" 双卡等新品出现，以及下一代 Blackwell GPU 引入 NVLink 6，总线速率翻倍，NVIDIA 正推动超节点概念从服务器级走向机架级甚至多机架。这一专有 scale-up 策略通过极高的内部带宽和统一内存语义，使超大模型训练在单体架构内的通信开销显著降低。

## 2. AMD 的"GPU 互联 + 开放机架级形态"方案

**Infinity Fabric (XGMI) 节点互联**：AMD 采用 Infinity Fabric 技术（外部称为 XGMI）实现 GPU 间高速互联。例如，在 Instinct MI300X 加速器平台中，每颗 GPU 通过 7 条 XGMI 链路连接至同节点内其余 7 个 GPU，形成全互联网状拓扑，提供高带宽、低时延的对等通信。
每对 MI300X GPU 间单链路理论带宽达 64 GB/s，单 GPU 聚合互联带宽可达 448 GB/s（实际有效约 315~336 GB/s）。
这种 8-GPU 全互联结构确保 AMD GPU 节点内部可以类似 NVIDIA NVSwitch 的方式实现统一的大带宽互连，为多卡训练提供紧耦合支持[xGMI Overview and Performance Expectations](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)。

| 项目                                     | 规格说明                | 理论最大值              | 实测性能                         |
| :--------------------------------------- | :---------------------- | :---------------------- | :------------------------------- |
| **每GPU聚合单向总带宽**            | 7条xGMI链路 × 每条带宽 | 448 GB/s (7 × 64 GB/s) | 315 - 336 GB/s (7 × 45-48 GB/s) |
| **GPU-GPU单向带宽 (单条xGMI链路)** | 单条链路带宽            | 64 GB/s                 | 45 - 48 GB/s                     |

**开放的机架级扩展**：在更大规模上，AMD 正积极拥抱开放标准，推动基于以太网生态的机架级 scale-up/scale-out。2025 年 AMD 发布了**"Helios"开放机架级 AI 平台**：采用 OCP Open Rack Wide 机架规范，将 Instinct GPU、EPYC CPU 与 Pensando 智能网卡融合，并整合开放互连标准 UALink 和 Ultra Ethernet (UEC)。
Helios 参考设计通过 UALink 构建整机架 GPU 互联，可在单机架内连接 72 个 GPU 为一体，使其作为统一资源池运作[AMD Delivering Open Rack Scale AI Infrastructure](https://www.amd.com/en/blogs/2025/amd-delivering-open-rack-scale-ai-infrastructure-to-unlock-agentic-ai.html)。Helios 提供高达 260 TB/s 级别的机架内部总带宽，与 NVIDIA 同级别超节点相当[hpe_amd_helios_racks](https://www.theregister.com/2025/12/02/hpe_amd_helios_racks/)。
AMD 宣布将与 OEM 合作，将 Helios 平台于 2026 年推向市场，例如 HPE 已计划推出基于 Helios 架构的整机架 AI 系统。该系统使用 "scale-up Ethernet" 实现 UALink over Ethernet，通过 Juniper/Broadcom 的 102.4Tbps 专用交换芯片，实现开放标准下的超节点互联。
AMD 此举旨在避免专有互连的厂商锁定，以100% 开放标准实现 Pod 级别的加速器互联。此外，AMD 与 Broadcom 合作开发支持 XGMI 的 PCIe Gen5 交换芯片，可突破传统 8-GPU 节点限制，将更多 GPU 高效互连到同一系统[Broadcom and AMD Collaborate to Enhance AI Infrastructure](https://www.liqid.com/blog/broadcom-and-amd-collaborate-to-enhance-ai-infrastructure)。
总体而言，AMD 路线通过现有标准（PCIe/CXL/以太网）扩展 GPU 互联，并参与制定 UALink 等新规范，为产业提供更开放的机架级 AI 基础设施[hpe_amd_helios_racks](https://www.theregister.com/2025/12/02/hpe_amd_helios_racks/)。

```text
下一代AMD Instinct MI400系列GPU。预计将提供高达432 GB的HBM4内存、40 petaflops的MXFP4性能以及每秒300 GB的横向扩展带宽。
这些GPU将为训练海量模型和大规模分布式推理带来机架级AI性能领导力。

采用UALink™的开放纵向扩展。借助"Helios"参考设计，性能可在最多72个GPU上轻松扩展——这得益于UALink。UALink标准是一种开放标准，将实现纵向扩展结构中的客户选择与互操作性。
在"Helios"中，我们以多种方式使用UAL，包括互连GPU和横向扩展NIC，以及通过以太网隧道化以互连GPU。这连接了机架中的每个GPU，使其能够作为一个统一系统进行通信——提供突破性的机架级性能。

第六代AMD EPYC "Venice" CPU。由开创性的"Zen 6"架构驱动，预计该系列CPU将提供高达256个核心、高达1.7倍的性能提升以及1.6 TB/s的内存带宽，以帮助在整个"Helios"机架范围内维持最大性能。

AMD Pensando "Vulcano" AI NIC。用于AI横向扩展的下一代NIC符合UEC 1.0标准，并支持PCIe®和UALink接口，用于直接连接CPU和GPU。
与上一代相比，它还将支持800G网络吞吐量，以及预计每GPU高达8倍的横向扩展带宽。"Vulcano"对于实现高密度集群内快速无缝的数据传输至关重要，有效消除了大规模AI部署的通信瓶颈。

注：
1. MI350-026A - 基于AMD性能实验室在2025年9月的计算，针对128 GPU的AMD Instinct MI355X机架，以确定在比较FP64、FP32、FP16、OCP FP8、FP8、MXFP6、FP6、MXFP4和FP4数据类型（适用于矩阵、张量、向量和稀疏性）时的峰值理论精度性能，与类似配置的NVIDIA Grace Blackwell GB200 NVL72 72 GPU机架相比。服务器制造商可能采用不同配置，从而产生不同结果。结果可能因使用最新驱动程序和优化而异。
2. MI350-027 - AMD性能实验室在2025年5月基于已发布的AMD Instinct MI350X / MI355X OAM 128xGPU机架与NVIDIA Blackwell GB200 72xGPU (NVL72)机架的内存容量规格进行的计算。服务器制造商可能采用不同配置，从而产生不同结果。结果可能因使用最新驱动程序和优化而异。
3. GD-247A：基于截至2025年6月的AMD工程预测或早期测量的初步性能估计，可能发生变化。
4. VEN-001：基于AMD对顶级2P第六代EPYC CPU的内部估计以及截至2025年6月3日的第五代EPYC测量结果进行的SPECrate®2017_int_base比较。基于截至2025年6月6日的AMD工程预测或测量的初步性能估计，可能发生变化。
5. VEN-003：基于PCI-SIG发布声明的PCIe代次比较，https://pcisig.com/pci-express-6.0-specification。截至2025年6月3日，配备128通道PCIe Gen 6的2P第六代EPYC CPU与配备128通道PCIe Gen 5的第五代EPYC CPU的比较。PCIe是PCI-SIG公司的注册商标。
```

## 3. 云厂商自研加速器的 "Pod 级互联" 路线（以 Google TPU 为例）

**Pod 内专用互联**：云厂商（如 Google）的自研 AI 加速器通常将一个 Pod（或称 SuperPod）视作基本扩展单元。Pod 内采用专用高速互连（如 TPU ICI）将大量芯片联结为一个整体，外部呈现为可按需划分的资源池（Slice）。
Google 的 TPU 架构文档将 TPU Pod、Slice 以及 ICI (InterChip Interconnect) 作为核心概念加以描述：一个 TPU Pod 即通过 ICI 将一定数量的 TPU 芯片组成的连续互联域，不同芯片间可直接高速通信[Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。
Slice 则指 Pod 内通过 ICI 连接的一组 TPU 芯片，可看作一个全局互联分区；多个 slice 则需借助数据中心网络（DCN）跨slice通信[Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。
换言之，单个 TPU Pod 内部以专有互连实现高带宽、低延迟的负载-存储级通信，把许多加速芯片凝聚为"一台逻辑超级计算机"[tpu-codesigned-ai-stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)。

```text
Ironwood 架构的核心理念是系统级协同设计，它将整个 TPU pod 视为一个单一的、统一的超级计算机，而非一系列独立的加速器。该架构基于定制互连技术，支持大规模远程直接内存访问 (RDMA)，使数千个芯片能够以高带宽和低延迟直接交换数据，绕过主机 CPU。Ironwood 拥有总计 1.77 PB 的可直接访问 HBM 容量，每个芯片包含八个 HBM3E 堆栈，HBM 峰值带宽为 7.4 TB/s，容量为 192 GiB。
```

**TPU SuperPod 规模与带宽**：Google 持续升级 TPU Pod 的规模与互连能力。最新一代第七代 TPU（代号 Ironwood）的 SuperPod 可扩展到 9,216 颗 TPU 芯片在同一高速互联域内运行。
每颗 TPU v7 芯片配备大幅升级的 ICI：双向带宽提升到 1.2 TB/s，支持大规模同步通信[TPU: Why Google Doesn’t Wait in Line for NVIDIA GPUs](https://medium.com/@jiminlee-ai/tpu-why-google-doesnt-wait-in-line-for-nvidia-gpus-2-2-2267e4ed686f)。
Google 工程博客强调，这种架构使上千芯片能够"像单个巨型并行处理器一样运行"，充分利用 RDMA 式直连互访能力，实现大模型训练的高吞吐、高效率互联。
此外，Google TPU Pod 引入了光交换 (OCS) 技术来动态重构光互连，以在多达数千芯片的3D Torus网络中提供可重配置、容错的全互联结构。文档披露，新一代 TPU SuperPod 拥有 1.77PB HBM 内存和超 11,000 TB/s 总互连带宽[tpu-codesigned-ai-stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)。

**资源池化与切片**：TPU Pod 对外可按 Slice 划分资源，一个 Slice 内部芯片通过 ICI 全带宽互通，而跨 Slice 则通过数据中心以太网/光网络连接，实现更大规模并行（即 Google 所称 Multi-slice 模式）[Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。
综上，云厂商通过专用 Pod 级互联，将海量加速器凝聚为统一集群，并以Slice为粒度对外提供弹性调度，这种方法有效突破单服务器限制，满足大规模 AI 训练对通信带宽和规模的要求。

## 4. 开放的"加速器到加速器"互联标准：UALink

**定位与规模**：UALink（Ultra Accelerator Link）是当前国际上面向 AI Pod 内部加速器互联的重要开放标准之一。其宗旨是在 GPU/AI 加速器之间建立低延迟、高带宽的统一直连互联，以替代各家专有方案[ualink](https://ualinkconsortium.org/about-ualink/)。
根据 UALink 联盟发布的 1.0 规范，UALink 支持在单一 AI Pod 内连接最多 1024 个加速器，并提供每通道 200Gbps（即每 lane 200G）规模的高速链路[ualinkconsortium.org](https://ualinkconsortium.org)。这使其成为业界最快的开放互连之一，可满足下一代大规模 AI 集群对内部通信的需求[ualinkconsortium.org](https://ualinkconsortium.org)。
UALink 技术采用存储级加载/存储语义（memory semantic），支持加速器之间直接进行缓存一致的内存访问和原子操作，实现加速器-加速器间像访问本地内存一样通信。

**技术细节**：UALink 1.0 基于 IEEE P802.3dj PHY（即 200G SerDes）实现物理层，与 PCIe 6.0 等速率协同发展[ualink-faq](https://ualinkconsortium.org/faq/)。
它定义了高效的直连交换结构，可通过专用交换芯片构建 Pod 内的全互联拓扑，同时提供负载/存储直接访问模式的软件支持。

**开放生态**：UALink 联盟于 2024 年成立，汇聚 AMD、Intel、Astera 等众多芯片和系统厂商推动规范制定。
UALink 规范于 2025 年4月开放发布，非联盟成员也可自由下载使用，联盟成员则可共同制定演进路线。目前 UALink 正积极与 OCP 等社区合作，规划后续对更大规模（超过1024节点）以及 128G SerDes 的支持。
总体而言，UALink 提供了一个厂商无关的超节点互联方案，可作为 NVLink 等专有方案的替代。

## 5. 内存语义与资源池化的开放标准：CXL（及 GenZ、OpenCAPI、CCIX）

**CXL 统一负载/存储语义互连**：Compute Express Link (CXL) 是目前业界主流的 CPU-设备/内存 高速互连开放标准，旨在实现远端内存/设备的缓存一致、低延迟访问，并支持内存扩展与池化。
CXL 基于 PCIe 物理层，提供三种子协议：CXL.io (常规 I/O)、CXL.mem (内存直访) 和 CXL.cache (缓存一致)[Compute_Express_Link-wikipedia](https://en.wikipedia.org/wiki/Compute_Express_Link)。这使 CPU 可以像本地一样访问加速器或内存扩展设备的内存。

**CXL 3.0/3.1 新特性**：2022 年发布的 CXL 3.0 (及后续3.1) 进一步引入交换和织网 (fabric) 能力，实现多级级联交换和复杂拓扑支持，从而构建大规模内存池和可组合系统。
CXL 3.x 支持多主机共享内存资源，允许数百台服务器通过 CXL 交换机访问同一组外部内存设备，且各主机与设备间保持缓存一致。
比如，CXL 3.0 已支持外部内存池在多主机间按需分配，每个主机的缓存与设备内存协调一致，从而实现真正的内存池化[Hundreds of servers could share external memory pools across Panmnesia CXL fabric](https://blocksandfiles.com/2024/08/01/panmnesia-cxl-fabric/)。
此外，CXL 3.0 将总线速率提升至 64 GT/s (双向 256 GB/s@x16)[Compute_Express_Link-wikipedia](https://en.wikipedia.org/wiki/Compute_Express_Link)；CXL 3.1 则增加了面向 fabric 扩展的改进（如端口路由、可靠性和安全增强）。

**生态融合**：早期曾有多个类似标准（Gen-Z、OpenCAPI、CCIX 等）探索内存语义互连。但随着 2021-2022 年业界整合，主要贡献者纷纷转向 CXL。Gen-Z 联盟已于 2022 年宣布将规范和资产并入 CXL 联盟，独立运作终止——彼时约 70% Gen-Z 成员已加入 CXL。
同样，IBM 主导的 OpenCAPI 和 Arm/Xilinx 等提出的 CCIX 也不再推出新版本，其技术成果部分吸收进 CXL 标准[Compute_Express_Link-wikipedia](https://en.wikipedia.org/wiki/Compute_Express_Link)。
这表明整个行业正收敛到以 CXL 为唯一内存池化/组合接口标准[FINALLY, A COHERENT INTERCONNECT STRATEGY: CXL ABSORBS GEN-Z](https://www.nextplatform.com/2021/11/23/finally-a-coherent-interconnect-strategy-cxl-absorbs-gen-z/)[FINALLY, A COHERENT INTERCONNECT STRATEGY: CXL ABSORBS GEN-Z](https://www.nextplatform.com/2021/11/23/finally-a-coherent-interconnect-strategy-cxl-absorbs-gen-z/)。

**产品落地**：目前已有厂商发布基于 CXL 的内存池化与组合式基础设施产品。例如三星、MemVerge 等展示了采用 CXL 2.0 开放接口的 2TB 内存池系统，通过 CXL Switch 将 8 条 256GB CXL 内存模块连至多台主机，共享总带宽达 2,048 GB/s[h3platform.com](https://h3platform.com)[h3platform.com](https://h3platform.com)。
Microchip、Astera 等公司也推出了 CXL 交换芯片和内存扩展解决方案，用于构建服务器间共享内存池[h3platform.com](https://h3platform.com)[Hundreds of servers could share external memory pools across Panmnesia CXL fabric](https://blocksandfiles.com/2024/08/01/panmnesia-cxl-fabric/)。这些进展表明，借助 CXL 标准，远端内存/设备池化正从概念走向商用，为数据中心提供弹性组合的新范式[h3platform.com](https://h3platform.com)[h3platform.com](https://h3platform.com)。

**注**: OpenCAPI、CCIX 等更早期的缓存一致互连（如 IBM Power 的 OMI/OpenCAPI，Xilinx CCIX 等）目前多被视为特定平台的过渡技术，随着 CXL 的普及，这些规范在主流数据中心中的角色正被 CXL 所取代[Compute_Express_Link-wikipedia](https://en.wikipedia.org/wiki/Compute_Express_Link)。

## 6. 集群 Scale-Out 的主流互联与"通信卸载"：InfiniBand、Slingshot、SHARP

**InfiniBand 与 HPE Slingshot**：当 AI 训练扩展到大量节点时，跨节点通信（尤其 All-Reduce、All-to-All 等集体通信）的带宽与延迟成为瓶颈。业界目前有两大成熟的集群级互联方案：NVIDIA (Mellanox) InfiniBand 和 HPE Cray Slingshot。InfiniBand 是专为 HPC/AI 设计的高性能通信网络，提供硬件级 RDMA 支持和极低延迟，高端产品 (NDR) 速率达 400 Gbps，并具备先进的拥塞控制机制和 QoS。
Slingshot 则是 Cray 开发的 以太网兼容 高速互连，工作在 200 Gbps 级，以太网物理层之上集成 HPC优化特性（如自适应路由、端到端拥塞控制等）[Slingshot](https://www.glennklockwood.com/garden/Slingshot)。Slingshot 能在保持以太生态兼容的同时，实现与 HDR InfiniBand 相当的低延迟和高吞吐，被用于 Frontier 等超算系统[CRAY’S SLINGSHOT INTERCONNECT IS AT THE HEART OF HPE’S HPC AND AI AMBITIONS](https://www.nextplatform.com/2022/01/31/crays-slingshot-interconnect-is-at-the-heart-of-hpes-hpc-and-ai-ambitions/)。
总的来说，InfiniBand 和 Slingshot 代表了当前大规模 HPC/AI 集群网络的最高水平，两者都针对全局通信密集的工作负载进行了专门优化。

**集体通信卸载 (In-Network Computing)**：为进一步缓解超大规模训练中的通信开销，业界引入了"将部分 collective 运算下沉至网络"的创新。NVIDIA 提出的 SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 技术是这一方向的典型代表。
在传统集群中，All-Reduce 等操作需要各节点反复交换数据并参与规约计算。而 SHARP 则将规约算子的执行下放到 InfiniBand 交换机硬件上：网络交换芯片直接对来自不同节点的数据进行聚合、求和等运算，然后只将结果发送出去。如此可将数据传输量减半，并避免多轮等待同步，从而显著降低延迟和 CPU/GPU 开销。
SHARP 已经过三代演进：第一代针对 MPI 小消息，在 100Gb EDR IB 交换机上实现；第二代在 200Gb HDR IB 上支持更大消息和 AI 工作负载；第三代随 400Gb NDR IB 推出，支持多租户环境下更复杂的 in-network 计算。实测表明，SHARP 在诸如 BERT 训练等任务上带来 10~20% 的性能提升[Advancing Performance with NVIDIA SHARP In-Network Computing](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)。

**综合效果**：通过 InfiniBand 的硬件 RDMA 和 SHARP 协同，大规模分布式训练的 All-Reduce 等通信瓶颈得到极大缓解。
同样地，Slingshot 网络结合先进的拥塞管理，也能够高效支撑全域通信，不过目前以太网阵营在 in-network 规约计算方面仍在追赶。
总体而言，"通信卸载"已成为解决大模型训练通信瓶颈的工程主流方向之一——网络不仅传输数据，还协助处理部分数据规约，以最大化整体吞吐[Advancing Performance with NVIDIA SHARP In-Network Computing](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)。

## 7. 华为"灵衢"架构 (UnifiedBus, UB-Mesh)

**背景概述**：华为在 2025 年推出了号称**"灵衢"的超级互联架构，其英文名为 UnifiedBus (UB)，旨在为 AI 超级算力集群提供一个统一的"Pod 级到数据中心级"互连方案[华为发布全球最强算力超节点和集群](https://www.huawei.com/cn/news/2025/9/hc-lingqu-ai-superpod)。
灵衢立足于华为昇腾 AI 体系，为解决大型 AI 基础设施中的互连瓶颈而研发。在华为看来，现有技术（如 PCIe、NVLink、以太/IP 等）存在协议林立、转换开销大、规模受限的问题。
灵衢 UB-Mesh 则尝试**"一种互连统一替代所有"**：无论节点内还是节点间通信，都采用同一种协议/总线，减少不同总线间转换，提高效率和可靠性[Huawei to open-source its UB-Mesh data center-scale interconnect](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-to-open-source-its-ub-mesh-data-center-scale-interconnect-soon-details-technical-aspects-one-interconnect-to-rule-them-all-is-designed-to-replace-everything-from-pcie-to-tcp-ip)。

**技术指标**：根据华为在 Hot Chips 2025 公开的数据，UnifiedBus 面向 "SuperNode"级架构，可将多达 100 万颗处理器（CPU/GPU/NPU 等）通过统一互连组成一个逻辑上共享内存的大系统。它提供的每芯片带宽高达 10 Tbps（约合 1.25 TB/s），远超当前 PCIe 5.0/6.0 和 NVLink 等链路能力。同时，通信延迟显著降低——UB-Mesh 目标在跨整中心连接下实现 ~150 ns 量级的 hop 延迟。更重要的是，灵衢采用同步负载/存储语义，即整个百万级别系统可以像单机一样执行负载存储指令，支持全局缓存一致（这类似于CXL的远端负载存储理念，但扩展到更大尺度）[Huawei to open-source its UB-Mesh data center-scale interconnect](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-to-open-source-its-ub-mesh-data-center-scale-interconnect-soon-details-technical-aspects-one-interconnect-to-rule-them-all-is-designed-to-replace-everything-from-pcie-to-tcp-ip)。

**拓扑与可靠性**：UB-Mesh 采用 Clos 上层 + 多维Mesh下层的混合拓扑结构，将机架级、机房级连接起来。机架内部通过电连接（高速铜背板/短距光）构建二维/三维Mesh联结数十节点，机架之间则通过多级Clos光纤交换横联。
由于大规模采用光互连，灵衢在链路层引入了错误重传、光模块备份、跨控制器交叉连接等机制，以克服长距离光链路较高误码率，保证百万节点级系统的可靠运行。
此外，设计了热备份机架机制：如某机架故障，备用机架自动接管，以提高整个系统 MTBF，应对超大规模下节点失效频繁的问题[Huawei to open-source its UB-Mesh data center-scale interconnect](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-to-open-source-its-ub-mesh-data-center-scale-interconnect-soon-details-technical-aspects-one-interconnect-to-rule-them-all-is-designed-to-replace-everything-from-pcie-to-tcp-ip)。

**性能与成本优势**：华为声称 UB-Mesh 架构可在规模增加时实现亚线性的成本增长，相比传统互连在数万节点时互连成本甚至高过计算芯片本身，UB 能将成本曲线压平。
例如在 8192 节点规模下结合 Clos+Mesh 拓扑验证了方案的可行性和经济性。


### 与 NVLink/NVSwitch 的目标边界不同

#### NVLink/NVSwitch 的强项与边界

NVLink/NVSwitch 典型是为 加速器域的 scale-up 服务：把一组 GPU（乃至整机架 NVL72 这种规模）做成高带宽、低延迟、全互联（all-to-all）的通信域，用于张量并行/流水并行/专家并行等训练通信热点。

从 NVIDIA 官方页面与 NVL72 产品页信息看，Vera Rubin NVL72 以 NVLink 6 Switch 提供每 GPU 3.6 TB/s 的 scale-up 带宽，并给出 单机架 260 TB/s 级的 GPU 互联带宽表述。

Tom’s Hardware 对 CES 2026 报道也复述了 NVLink 6 与 260 TB/s 机架级互联的关键信息（并给出发售节奏等）。

但 NVLink 的“边界”也很清晰：它主要解决加速器域的 scale-up，把跨节点/跨机架的 scale-out 仍交给 InfiniBand/以太网等网络体系（或至少是另一套网络/交换域），因此系统往往天然是“两张网/两套域”的组合。

#### UB-Mesh 的“野心”更像“把两张网合成一张”

Tom’s Hardware 对 Hot Chips 2025 的报道写得很直白：UB-Mesh 试图用“一种互联协议”覆盖 节点内与节点间，并提到华为计划对外开放该协议供免费使用。

这意味着 UB 的发力点不止是“让 NPU 之间更快”，而是希望减少“节点内总线 + 节点间网络”的语义裂缝与协议转换成本。

一个很现实的推论是：UB 真正的难点不在“把带宽做高”，而在于把一致性、可靠性、容错、运维分域、升级演进等数据中心级问题纳入同一互联体系里。openEuler 的 UB Service Core 白皮书之所以强调控制面、资源池化、HA，就是在补这部分系统工程。

### 与 CXL 对比
#### CXL 的设计中心是“主机扩展与组合式资源”

CXL 从一开始的主战场就是 CPU ↔ 设备/内存 的一致性访问与内存扩展，基于 PCIe 物理层演进；到了 3.x，进一步强调交换与更复杂拓扑，并把设备间 P2P 纳入能力边界。CXL 联盟 2025 年的 3.x 介绍材料明确提到 CXL 3.0/3.1 的 P2P 通信能力（含 CXL.mem 方向的增强）。

换句话说，CXL 更像是在标准化“可组合系统”的底座：内存池、设备池、主机间共享资源等。

#### UB 更像把“内存语义”推到超节点乃至数据中心互联

华为公开稿件把 UB 描述为面向 SuperPoD 的互联协议，并希望形成开放生态。

而 Tom’s Hardware 的 UB-Mesh 报道把它描述为意图替代从 PCIe 到 TCP/IP 的多种互联形态，强调“统一互联”的覆盖面。

因此两者虽然都触及“内存语义/一致性”这类关键词，但在系统分工上差异很大：

- CXL 更偏“主机中心”的扩展与池化（尤其在服务器/机箱/机架内的组合式架构）。
- UB 更偏“把超节点内与节点间互联统一起来”，把“互联域”从单机/机箱扩大到更大规模的计算域（至少在目标叙事上如此）。

这几套标准/联盟的共同点是：都曾尝试解决“缓存一致/内存语义互联”的一部分拼图，但产业最终明显向 CXL 收敛。NextPlatform 在 2021 年的文章就讨论了 Gen-Z 向 CXL 的整合趋势与“互联战略收敛”。

一些 EDA/验证领域的行业文章也提到 OpenCAPI、Gen-Z 等资产向 CXL 方向转移、CXL 3.0 成为焦点。

|维度|华为 UnifiedBus / UB-Mesh|NVIDIA NVLink/NVSwitch（以 NVL72 为代表）|CXL 3.x（含 fabric / P2P）|
|---|---|---|---|
|首要目标|试图统一节点内与节点间互联，并配套系统软件/资源池化|加速器域极致 scale-up（GPU 全互联通信域）|主机与设备/内存的一致性互联，面向组合式资源与内存池|
|互联域边界|叙事上覆盖超节点到更大互联域；落地依赖系统软件与生态推进|以 NVLink 域为核心，scale-out 仍需网络体系配合|以 PCIe/CXL 拓扑为核心，强调交换、P2P、内存共享能力|
|带宽侧重点|更像“全栈统一”而非单点带宽竞赛（公开信息更多在体系与规范）|NVLink 6：每 GPU 3.6 TB/s；机架域 260 TB/s 级|更关注一致性语义、共享与拓扑扩展；带宽受 PCIe/CXL 代际与实现影响|
|开放性|规范发布与系统软件（openEuler）开源在推进中|互联技术体系高度 NVIDIA 化（生态强但锁定更强）|联盟标准、跨厂商，开放生态清晰|


---

## 参考文献

1. Banerjee, I. Architecting Predictable AI Data Centers: NVIDIA DGX SuperPOD Systems (Medium, 2026) – 关于 DGX 超级节点内 NVLink/NVSwitch 及节点间 InfiniBand 架构[i-banerjee83.medium.com](https://i-banerjee83.medium.com)[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。
2. NVIDIA Official – NVIDIA NVLink and NVSwitch Overview (2025) – 第六代 NVLink/NVSwitch 支持 72 GPU 全互联 (Vera Rubin NVL72) 总带宽 260TB/s[nvidia.com](https://nvidia.com)[nvidia.com](https://nvidia.com)。
3. AMD ROCm Blog – MI300X xGMI Fully-Connected Topology (2025) – MI300X 8-GPU 节点 7×XGMI 全互联，单 GPU 7链路总带宽 448GB/s（实测 ~336GB/s）[rocm.blogs.amd.com](https://rocm.blogs.amd.com)[rocm.blogs.amd.com](https://rocm.blogs.amd.com)。
4. AMD Newsroom – AMD "Helios" Rack-Scale AI Platform (OCP Summit 2025) – Helios 采用 OCP Open Rack，与 UALink、UEC 结合构建72 GPU 整机架超节点[amd Helios](https://www.amd.com/en/newsroom/press-releases/2025-10-14-amd-showcases-helios-rack-scale-platform-built-o.html)。
5. The Register – HPE backs AMD Helios rack with UALink Ethernet (Dec 2025) – HPE 将于2026推出 Helios 架构机架，72×MI455X GPU/架，260TB/s 带宽，采用 UALink over Ethernet (Broadcom Tomahawk6)[hpe_amd_helios_racks](https://www.theregister.com/2025/12/02/hpe_amd_helios_racks/)[hpe_amd_helios_racks](https://www.theregister.com/2025/12/02/hpe_amd_helios_racks/)。
6. Google Cloud – TPU System Architecture (v4/v6 Documentation) – 定义 TPU Pod、Slice、Multislice、ICI 等概念[Google TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。
7. Lee, J. – Why Google Doesn't Wait for NVIDIA GPUs (TPU v7 Ironwood) (Medium, 2025) – TPU Pod 扩展至9216芯片，ICI 带宽提升到1.2TB/s 双向[TPU: Why Google Doesn’t Wait in Line for NVIDIA GPUs](https://medium.com/@jiminlee-ai/tpu-why-google-doesnt-wait-in-line-for-nvidia-gpus-2-2-2267e4ed686f)。
8. Google Cloud Blog – Inside the Ironwood TPU stack (2025) – TPU Ironwood 9216-chip Superpod 互联架构，光交换+3D Torus，全系统带宽超11000TB/s[tpu-codesigned-ai-stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)。
9. UALink Consortium – UALink FAQ (2025) – UALink 1.0 支持 1024 加速器，200G/lane 互连，低延迟高带宽开放加速器互联标准[ualink](https://ualinkconsortium.org/about-ualink/)。
10. UALink Consortium – About UALink (2024) – UALink 旨在满足 AI 日益增长的算力需求，提供 Pod 内数百加速器低延迟/高带宽负载存储语义互连；1.0规范基于IEEE 802.3dj，支持1K规模[ualink-faq](https://ualinkconsortium.org/faq/)。
11. NextPlatform – CXL Absorbs Gen-Z (2021) – Gen-Z 联盟解散并入 CXL，业界内存语义标准加速收敛到单一协议；CXL 成为连接 CPU-加速器和远程内存的首选[Compute_Express_Link-wikipedia](https://en.wikipedia.org/wiki/Compute_Express_Link)。
12. Blocks & Files – CXL 3.0 Memory Pooling (2024) – CXL 3.0/3.1 引入交换式 fabric，实现多主机共享外部内存池，缓存一致，支持大规模内存池化[Hundreds of servers could share external memory pools across Panmnesia CXL fabric](https://blocksandfiles.com/2024/08/01/panmnesia-cxl-fabric/)。
13. Huawei Press Release – Huawei Unveils UnifiedBus for SuperPods (HC2025) – 华为发布灵衢 (UnifiedBus) 协议，克服现有光电互连物理限制，实现超级节点统一互连，并开放 UB2.0 规范[华为发布全球最强算力超节点和集群](https://www.huawei.com/cn/news/2025/9/hc-lingqu-ai-superpod)。
14. [UB Service Core](https://www.openeuler.org/zh/projects/ub-service-core/)
15. [UnifiedBus](https://www.unifiedbus.com/zh)
16. Tom's Hardware – Huawei UB-Mesh Interconnect Details (Aug 2025) – 华为 UB-Mesh 技术细节：旨在统一替代 PCIe/CXL/NVLink/TCPIP，全局负载存储语义，支持百万级处理器，单芯片带宽1.25TB/s，150ns级延迟[Huawei to open-source its UB-Mesh data center-scale interconnect](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-to-open-source-its-ub-mesh-data-center-scale-interconnect-soon-details-technical-aspects-one-interconnect-to-rule-them-all-is-designed-to-replace-everything-from-pcie-to-tcp-ip)。
17. NVIDIA Developer Blog – Advancing Performance with SHARP (2024) – NVIDIA SHARP 将 All-Reduce/广播等集体通信从服务器卸载到 InfiniBand 交换机执行，在网内完成规约运算，减少一半数据传输并降低延迟[Advancing Performance with NVIDIA SHARP In-Network Computing](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)。
18. [Slingshot](https://www.glennklockwood.com/garden/Slingshot)

