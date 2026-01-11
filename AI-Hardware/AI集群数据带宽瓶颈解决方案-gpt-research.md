# 现代 AI 训练集群中的数据带宽瓶颈与解决方案

## 1. NVIDIA 的"专有超节点 (Supernode) scale-up"方案

**NVLink/NVSwitch 互连**：NVIDIA 倾向通过 NVLink 和 NVSwitch 在单机或单柜内部将多块 GPU 组成高速、低延迟的紧耦合 GPU "超节点"[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。在典型的 DGX 系统中，8 张 A100/H100 GPU 通过 NVSwitch 实现全互联，每个 GPU 与任意其他 GPU 间都有均等的大带宽低时延连接，以构成单一逻辑 GPU 复合体[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。这使得节点内的 collective 操作（如 AllReduce）直接经由 NVLink 进行，避免传统 PCIe 分层拓扑的瓶颈[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。DGX SuperPOD 架构即遵循这一思路：节点内 GPU 经 NVLink/NVSwitch 打造高速通信域，节点间则通过 InfiniBand 等高速网络通信[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。正如 NVIDIA 工程文档所述："DGX 节点内 GPU 通过 NVLink/NVSwitch 形成独立于 CPU 的高带宽低延迟 GPU 数据结构；多个 DGX 节点之间通过 InfiniBand 进行通信，实现分布式训练"[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。

**跨节点网络与 SuperPOD**：当需要扩展到多节点集群时，NVIDIA 利用 InfiniBand 或高端以太网（如 RoCE）将多个超节点互联起来，确保跨节点通信性能[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。NVIDIA 的 DGX SuperPOD 参考架构即在单机柜内使用 NVLink/NVSwitch，将数个 GPU 打造成一个超节点，然后通过 Quantum HDR/NDR InfiniBand 骨干网络连接多个超节点[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。

**持续演进的整柜超节点**：最近一两年，NVIDIA 正将超节点的一体化规模从单机扩展到整机架。例如最新发布的 Vera Rubin NVL72 架构，在一个全液冷机柜内实现 72 个 GPU 全互联，总内部通信带宽达 260 TB/s[nvidia.com](https://nvidia.com)。在 Rubin NVL72 中，NVLink Switch 芯片构建了非阻塞全互联架构，每对 GPU 之间的链路带宽高达 3.6 TB/s，使 72 卡可作为单一加速器使用[nvidia.com](https://nvidia.com)。这体现了 NVIDIA NVLink 第六代技术在架级规模上的突破[nvidia.com](https://nvidia.com)。随着 H100 "NVL" 双卡等新品出现，以及下一代 Blackwell GPU 引入 NVLink 6，总线速率翻倍，NVIDIA 正推动超节点概念从服务器级走向机架级甚至多机架[nvidia.com](https://nvidia.com)[nvidia.com](https://nvidia.com)。这一专有 scale-up 策略通过极高的内部带宽和统一内存语义，使超大模型训练在单体架构内的通信开销显著降低[nvidia.com](https://nvidia.com)。

## 2. AMD 的"GPU 互联 + 开放机架级形态"方案

**Infinity Fabric (XGMI) 节点互联**：AMD 采用 Infinity Fabric 技术（外部称为 XGMI）实现 GPU 间高速互联。例如，在 Instinct MI300X 加速器平台中，每颗 GPU 通过 7 条 XGMI 链路连接至同节点内其余 7 个 GPU，形成全互联网状拓扑，提供高带宽、低时延的对等通信[rocm.blogs.amd.com](https://rocm.blogs.amd.com)。每对 MI300X GPU 间单链路理论带宽达 64 GB/s，单 GPU 聚合互联带宽可达 448 GB/s（实际有效约 315~336 GB/s）[rocm.blogs.amd.com](https://rocm.blogs.amd.com)[rocm.blogs.amd.com](https://rocm.blogs.amd.com)。这种 8-GPU 全互联结构确保 AMD GPU 节点内部可以类似 NVIDIA NVSwitch 的方式实现统一的大带宽互连，为多卡训练提供紧耦合支持[rocm.blogs.amd.com](https://rocm.blogs.amd.com)。

**开放的机架级扩展**：在更大规模上，AMD 正积极拥抱开放标准，推动基于以太网生态的机架级 scale-up/scale-out。2025 年 AMD 发布了**"Helios"开放机架级 AI 平台**：采用 OCP Open Rack Wide 机架规范，将 Instinct GPU、EPYC CPU 与 Pensando 智能网卡融合，并整合开放互连标准 UALink 和 Ultra Ethernet (UEC)[amd.com](https://amd.com)[amd.com](https://amd.com)。Helios 参考设计通过 UALink 构建整机架 GPU 互联，可在单机架内连接 72 个 GPU 为一体，使其作为统一资源池运作[amd.com](https://amd.com)。Helios 提供高达 260 TB/s 级别的机架内部总带宽，与 NVIDIA 同级别超节点相当[theregister.com](https://theregister.com)。AMD 宣布将与 OEM 合作，将 Helios 平台于 2026 年推向市场，例如 HPE 已计划推出基于 Helios 架构的整机架 AI 系统[theregister.com](https://theregister.com)[theregister.com](https://theregister.com)。该系统使用 "scale-up Ethernet" 实现 UALink over Ethernet，通过 Juniper/Broadcom 的 102.4Tbps 专用交换芯片，实现开放标准下的超节点互联[theregister.com](https://theregister.com)。AMD 此举旨在避免专有互连的厂商锁定，以100% 开放标准实现 Pod 级别的加速器互联[theregister.com](https://theregister.com)。此外，AMD 与 Broadcom 合作开发支持 XGMI 的 PCIe Gen5 交换芯片，可突破传统 8-GPU 节点限制，将更多 GPU 高效互连到同一系统[liqid.com](https://liqid.com)[liqid.com](https://liqid.com)。总体而言，AMD 路线通过现有标准（PCIe/CXL/以太网）扩展 GPU 互联，并参与制定 UALink 等新规范，为产业提供更开放的机架级 AI 基础设施[amd.com](https://amd.com)[theregister.com](https://theregister.com)。

## 3. 云厂商自研加速器的 "Pod 级互联" 路线（以 Google TPU 为例）

**Pod 内专用互联**：云厂商（如 Google）的自研 AI 加速器通常将一个 Pod（或称 SuperPod）视作基本扩展单元。Pod 内采用专用高速互连（如 TPU ICI）将大量芯片联结为一个整体，外部呈现为可按需划分的资源池（Slice）。Google 的 TPU 架构文档将 TPU Pod、Slice 以及 ICI (InterChip Interconnect) 作为核心概念加以描述：一个 TPU Pod 即通过 ICI 将一定数量的 TPU 芯片组成的连续互联域，不同芯片间可直接高速通信[docs.cloud.google.com](https://docs.cloud.google.com)。Slice 则指 Pod 内通过 ICI 连接的一组 TPU 芯片，可看作一个全局互联分区；多个 slice 则需借助数据中心网络（DCN）跨slice通信[docs.cloud.google.com](https://docs.cloud.google.com)。换言之，单个 TPU Pod 内部以专有互连实现高带宽、低延迟的负载-存储级通信，把许多加速芯片凝聚为"一台逻辑超级计算机"[cloud.google.com](https://cloud.google.com)。

**TPU SuperPod 规模与带宽**：Google 持续升级 TPU Pod 的规模与互连能力。最新一代第七代 TPU（代号 Ironwood）的 SuperPod 可扩展到 9,216 颗 TPU 芯片在同一高速互联域内运行[medium.com](https://medium.com)。每颗 TPU v7 芯片配备大幅升级的 ICI：双向带宽提升到 1.2 TB/s，支持大规模同步通信[medium.com](https://medium.com)。Google 工程博客强调，这种架构使上千芯片能够"像单个巨型并行处理器一样运行"，充分利用 RDMA 式直连互访能力，实现大模型训练的高吞吐、高效率互联[cloud.google.com](https://cloud.google.com)[cloud.google.com](https://cloud.google.com)。此外，Google TPU Pod 引入了光交换 (OCS) 技术来动态重构光互连，以在多达数千芯片的3D Torus网络中提供可重配置、容错的全互联结构[cloud.google.com](https://cloud.google.com)。文档披露，新一代 TPU SuperPod 拥有 1.77PB HBM 内存和超 11,000 TB/s 总互连带宽[cloud.google.com](https://cloud.google.com)。

**资源池化与切片**：TPU Pod 对外可按 Slice 划分资源，一个 Slice 内部芯片通过 ICI 全带宽互通，而跨 Slice 则通过数据中心以太网/光网络连接，实现更大规模并行（即 Google 所称 Multi-slice 模式）[docs.cloud.google.com](https://docs.cloud.google.com)。综上，云厂商通过专用 Pod 级互联，将海量加速器凝聚为统一集群，并以Slice为粒度对外提供弹性调度，这种方法有效突破单服务器限制，满足大规模 AI 训练对通信带宽和规模的要求[cloud.google.com](https://cloud.google.com)[medium.com](https://medium.com)。

## 4. 开放的"加速器到加速器"互联标准：UALink

**定位与规模**：UALink（Ultra Accelerator Link）是当前国际上面向 AI Pod 内部加速器互联的重要开放标准之一。其宗旨是在 GPU/AI 加速器之间建立低延迟、高带宽的统一直连互联，以替代各家专有方案[ualinkconsortium.org](https://ualinkconsortium.org)。根据 UALink 联盟发布的 1.0 规范，UALink 支持在单一 AI Pod 内连接最多 1024 个加速器，并提供每通道 200Gbps（即每 lane 200G）规模的高速链路[ualinkconsortium.org](https://ualinkconsortium.org)。这使其成为业界最快的开放互连之一，可满足下一代大规模 AI 集群对内部通信的需求[ualinkconsortium.org](https://ualinkconsortium.org)。UALink 技术采用存储级加载/存储语义（memory semantic），支持加速器之间直接进行缓存一致的内存访问和原子操作，实现加速器-加速器间像访问本地内存一样通信[ualinkconsortium.org](https://ualinkconsortium.org)[ualinkconsortium.org](https://ualinkconsortium.org)。

**技术细节**：UALink 1.0 基于 IEEE P802.3dj PHY（即 200G SerDes）实现物理层，与 PCIe 6.0 等速率协同发展[ualinkconsortium.org](https://ualinkconsortium.org)。它定义了高效的直连交换结构，可通过专用交换芯片构建 Pod 内的全互联拓扑，同时提供负载/存储直接访问模式的软件支持[ualinkconsortium.org](https://ualinkconsortium.org)[ualinkconsortium.org](https://ualinkconsortium.org)。

**开放生态**：UALink 联盟于 2024 年成立，汇聚 AMD、Intel、Astera 等众多芯片和系统厂商推动规范制定[ualinkconsortium.org](https://ualinkconsortium.org)[ualinkconsortium.org](https://ualinkconsortium.org)。UALink 规范于 2025 年4月开放发布，非联盟成员也可自由下载使用，联盟成员则可共同制定演进路线[ualinkconsortium.org](https://ualinkconsortium.org)。目前 UALink 正积极与 OCP 等社区合作，规划后续对更大规模（超过1024节点）以及 128G SerDes 的支持[ualinkconsortium.org](https://ualinkconsortium.org)。总体而言，UALink 提供了一个厂商无关的超节点互联方案，被视为类似华为"灵衢"那样的开放 Pod 级互联主线，可作为 NVLink 等专有方案的替代[theregister.com](https://theregister.com)[ualinkconsortium.org](https://ualinkconsortium.org)。

## 5. 内存语义与资源池化的开放标准：CXL（及 GenZ、OpenCAPI、CCIX）

**CXL 统一负载/存储语义互连**：Compute Express Link (CXL) 是目前业界主流的 CPU-设备/内存 高速互连开放标准，旨在实现远端内存/设备的缓存一致、低延迟访问，并支持内存扩展与池化[en.wikipedia.org](https://en.wikipedia.org)。CXL 基于 PCIe 物理层，提供三种子协议：CXL.io (常规 I/O)、CXL.mem (内存直访) 和 CXL.cache (缓存一致)[en.wikipedia.org](https://en.wikipedia.org)。这使 CPU 可以像本地一样访问加速器或内存扩展设备的内存。

**CXL 3.0/3.1 新特性**：2022 年发布的 CXL 3.0 (及后续3.1) 进一步引入交换和织网 (fabric) 能力，实现多级级联交换和复杂拓扑支持，从而构建大规模内存池和可组合系统[blocksandfiles.com](https://blocksandfiles.com)[blocksandfiles.com](https://blocksandfiles.com)。CXL 3.x 支持多主机共享内存资源，允许数百台服务器通过 CXL 交换机访问同一组外部内存设备，且各主机与设备间保持缓存一致[blocksandfiles.com](https://blocksandfiles.com)。比如，CXL 3.0 已支持外部内存池在多主机间按需分配，每个主机的缓存与设备内存协调一致，从而实现真正的内存池化[blocksandfiles.com](https://blocksandfiles.com)。此外，CXL 3.0 将总线速率提升至 64 GT/s (双向 256 GB/s@x16)[en.wikipedia.org](https://en.wikipedia.org)；CXL 3.1 则增加了面向 fabric 扩展的改进（如端口路由、可靠性和安全增强）[blocksandfiles.com](https://blocksandfiles.com)。

**生态融合**：早期曾有多个类似标准（Gen-Z、OpenCAPI、CCIX 等）探索内存语义互连。但随着 2021-2022 年业界整合，主要贡献者纷纷转向 CXL。Gen-Z 联盟已于 2022 年宣布将规范和资产并入 CXL 联盟，独立运作终止——彼时约 70% Gen-Z 成员已加入 CXL[en.wikipedia.org](https://en.wikipedia.org)。同样，IBM 主导的 OpenCAPI 和 Arm/Xilinx 等提出的 CCIX 也不再推出新版本，其技术成果部分吸收进 CXL 标准[en.wikipedia.org](https://en.wikipedia.org)。这表明整个行业正收敛到以 CXL 为唯一内存池化/组合接口标准[nextplatform.com](https://nextplatform.com)[nextplatform.com](https://nextplatform.com)。

**产品落地**：目前已有厂商发布基于 CXL 的内存池化与组合式基础设施产品。例如三星、MemVerge 等展示了采用 CXL 2.0 开放接口的 2TB 内存池系统，通过 CXL Switch 将 8 条 256GB CXL 内存模块连至多台主机，共享总带宽达 2,048 GB/s[h3platform.com](https://h3platform.com)[h3platform.com](https://h3platform.com)。Microchip、Astera 等公司也推出了 CXL 交换芯片和内存扩展解决方案，用于构建服务器间共享内存池[h3platform.com](https://h3platform.com)[blocksandfiles.com](https://blocksandfiles.com)。这些进展表明，借助 CXL 标准，远端内存/设备池化正从概念走向商用，为数据中心提供弹性组合的新范式[h3platform.com](https://h3platform.com)[h3platform.com](https://h3platform.com)。

**注**: OpenCAPI、CCIX 等更早期的缓存一致互连（如 IBM Power 的 OMI/OpenCAPI，Xilinx CCIX 等）目前多被视为特定平台的过渡技术，随着 CXL 的普及，这些规范在主流数据中心中的角色正被 CXL 所取代[en.wikipedia.org](https://en.wikipedia.org)。

## 6. 集群 Scale-Out 的主流互联与"通信卸载"：InfiniBand、Slingshot、SHARP

**InfiniBand 与 HPE Slingshot**：当 AI 训练扩展到大量节点时，跨节点通信（尤其 All-Reduce、All-to-All 等集体通信）的带宽与延迟成为瓶颈。业界目前有两大成熟的集群级互联方案：NVIDIA (Mellanox) InfiniBand 和 HPE Cray Slingshot。InfiniBand 是专为 HPC/AI 设计的高性能通信网络，提供硬件级 RDMA 支持和极低延迟，高端产品 (NDR) 速率达 400 Gbps，并具备先进的拥塞控制机制和 QoS。Slingshot 则是 Cray 开发的 以太网兼容 高速互连，工作在 200 Gbps 级，以太网物理层之上集成 HPC优化特性（如自适应路由、端到端拥塞控制等）[glennklockwood.com](https://glennklockwood.com)。Slingshot 能在保持以太生态兼容的同时，实现与 HDR InfiniBand 相当的低延迟和高吞吐，被用于 Frontier 等超算系统[glennklockwood.com](https://glennklockwood.com)[nextplatform.com](https://nextplatform.com)。总的来说，InfiniBand 和 Slingshot 代表了当前大规模 HPC/AI 集群网络的最高水平，两者都针对全局通信密集的工作负载进行了专门优化。

**集体通信卸载 (In-Network Computing)**：为进一步缓解超大规模训练中的通信开销，业界引入了"将部分 collective 运算下沉至网络"的创新。NVIDIA 提出的 SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) 技术是这一方向的典型代表[developer.nvidia.com](https://developer.nvidia.com)。在传统集群中，All-Reduce 等操作需要各节点反复交换数据并参与规约计算。而 SHARP 则将规约算子的执行下放到 InfiniBand 交换机硬件上：网络交换芯片直接对来自不同节点的数据进行聚合、求和等运算，然后只将结果发送出去[developer.nvidia.com](https://developer.nvidia.com)。如此可将数据传输量减半，并避免多轮等待同步，从而显著降低延迟和 CPU/GPU 开销[developer.nvidia.com](https://developer.nvidia.com)[developer.nvidia.com](https://developer.nvidia.com)。SHARP 已经过三代演进：第一代针对 MPI 小消息，在 100Gb EDR IB 交换机上实现；第二代在 200Gb HDR IB 上支持更大消息和 AI 工作负载；第三代随 400Gb NDR IB 推出，支持多租户环境下更复杂的 in-network 计算[developer.nvidia.com](https://developer.nvidia.com)[developer.nvidia.com](https://developer.nvidia.com)。实测表明，SHARP 在诸如 BERT 训练等任务上带来 10~20% 的性能提升[developer.nvidia.com](https://developer.nvidia.com)[developer.nvidia.com](https://developer.nvidia.com)。

**综合效果**：通过 InfiniBand 的硬件 RDMA 和 SHARP 协同，大规模分布式训练的 All-Reduce 等通信瓶颈得到极大缓解[developer.nvidia.com](https://developer.nvidia.com)。同样地，Slingshot 网络结合先进的拥塞管理，也能够高效支撑全域通信，不过目前以太网阵营在 in-network 规约计算方面仍在追赶。总体而言，"通信卸载"已成为解决大模型训练通信瓶颈的工程主流方向之一——网络不仅传输数据，还协助处理部分数据规约，以最大化整体吞吐[developer.nvidia.com](https://developer.nvidia.com)[developer.nvidia.com](https://developer.nvidia.com)。

## 7. 华为"灵衢"架构 (UnifiedBus, UB-Mesh)

**背景概述**：华为在 2025 年推出了号称**"灵衢"的超级互联架构，其英文名为 UnifiedBus (UB)，旨在为 AI 超级算力集群提供一个统一的"Pod 级到数据中心级"互连方案[huawei.com](https://huawei.com)[tomshardware.com](https://tomshardware.com)。灵衢立足于华为昇腾 AI 体系，为解决大型 AI 基础设施中的互连瓶颈而研发。在华为看来，现有技术（如 PCIe、NVLink、以太/IP 等）存在协议林立、转换开销大、规模受限的问题[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。灵衢 UB-Mesh 则尝试**"一种互连统一替代所有"**：无论节点内还是节点间通信，都采用同一种协议/总线，减少不同总线间转换，提高效率和可靠性[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。

**技术指标**：根据华为在 Hot Chips 2025 公开的数据，UnifiedBus 面向 "SuperNode"级架构，可将多达 100 万颗处理器（CPU/GPU/NPU 等）通过统一互连组成一个逻辑上共享内存的大系统[tomshardware.com](https://tomshardware.com)。它提供的每芯片带宽高达 10 Tbps（约合 1.25 TB/s），远超当前 PCIe 5.0/6.0 和 NVLink 等链路能力[tomshardware.com](https://tomshardware.com)。同时，通信延迟显著降低——UB-Mesh 目标在跨整中心连接下实现 ~150 ns 量级的 hop 延迟[tomshardware.com](https://tomshardware.com)。更重要的是，灵衢采用同步负载/存储语义，即整个百万级别系统可以像单机一样执行负载存储指令，支持全局缓存一致（这类似于CXL的远端负载存储理念，但扩展到更大尺度）[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。

**拓扑与可靠性**：UB-Mesh 采用 Clos 上层 + 多维Mesh下层的混合拓扑结构，将机架级、机房级连接起来[tomshardware.com](https://tomshardware.com)。机架内部通过电连接（高速铜背板/短距光）构建二维/三维Mesh联结数十节点，机架之间则通过多级Clos光纤交换横联[tomshardware.com](https://tomshardware.com)。由于大规模采用光互连，灵衢在链路层引入了错误重传、光模块备份、跨控制器交叉连接等机制，以克服长距离光链路较高误码率，保证百万节点级系统的可靠运行[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。此外，设计了热备份机架机制：如某机架故障，备用机架自动接管，以提高整个系统 MTBF，应对超大规模下节点失效频繁的问题[tomshardware.com](https://tomshardware.com)。

**性能与成本优势**：华为声称 UB-Mesh 架构可在规模增加时实现亚线性的成本增长，相比传统互连在数万节点时互连成本甚至高过计算芯片本身，UB 能将成本曲线压平[tomshardware.com](https://tomshardware.com)。例如在 8192 节点规模下结合 Clos+Mesh 拓扑验证了方案的可行性和经济性[tomshardware.com](https://tomshardware.com)。

**开放生态**：灵衢 UnifiedBus 被视为华为对标国际开放标准（如 UALink、Ultra Ethernet）的自主方案。值得一提的是，华为已在 2025 年 Huawei Connect 大会上公布了 UnifiedBus 2.0 的技术规范，并计划将 UB 协议开源免费提供给业界[huawei.com](https://huawei.com)[tomshardware.com](https://tomshardware.com)。Huawei 表示希望行业伙伴采用该协议开发产品，共建开放生态[huawei.com](https://huawei.com)。UB-Mesh 旨在替代 PCIe、CXL、以太网/IP 等各种总线协议，实现**"数据中心级统一互连"**[tomshardware.com](https://tomshardware.com)。Forrester 等分析指出，UB 的目标是超过现有 NVLink 及以太RoCE网络的扩展能力，为未来 AI 基础设施提供一个中国自研的标准[forrester.com](https://forrester.com)[forrester.com](https://forrester.com)。总体而言，华为灵衢架构代表了一种激进的技术路径：通过统一通信协议和超高速链路，打破传统节点边界，将 AI 集群真正提升到Pod/数据中心级别的统一算力池[huawei.com](https://huawei.com)[tomshardware.com](https://tomshardware.com)。目前这套方案仍在推动中，其能否被更广泛采用或标准化还有待观察[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。但无疑，它反映了业界为解决 AI 训练带宽/延迟瓶颈所做的前沿探索。

---

## 参考文献

1. Banerjee, I. Architecting Predictable AI Data Centers: NVIDIA DGX SuperPOD Systems (Medium, 2026) – 关于 DGX 超级节点内 NVLink/NVSwitch 及节点间 InfiniBand 架构[i-banerjee83.medium.com](https://i-banerjee83.medium.com)[i-banerjee83.medium.com](https://i-banerjee83.medium.com)。

2. NVIDIA Official – NVIDIA NVLink and NVSwitch Overview (2025) – 第六代 NVLink/NVSwitch 支持 72 GPU 全互联 (Vera Rubin NVL72) 总带宽 260TB/s[nvidia.com](https://nvidia.com)[nvidia.com](https://nvidia.com)。

3. AMD ROCm Blog – MI300X xGMI Fully-Connected Topology (2025) – MI300X 8-GPU 节点 7×XGMI 全互联，单 GPU 7链路总带宽 448GB/s（实测 ~336GB/s）[rocm.blogs.amd.com](https://rocm.blogs.amd.com)[rocm.blogs.amd.com](https://rocm.blogs.amd.com)。

4. AMD Newsroom – AMD "Helios" Rack-Scale AI Platform (OCP Summit 2025) – Helios 采用 OCP Open Rack，与 UALink、UEC 结合构建72 GPU 整机架超节点[amd.com](https://amd.com)[amd.com](https://amd.com)。

5. The Register – HPE backs AMD Helios rack with UALink Ethernet (Dec 2025) – HPE 将于2026推出 Helios 架构机架，72×MI455X GPU/架，260TB/s 带宽，采用 UALink over Ethernet (Broadcom Tomahawk6)[theregister.com](https://theregister.com)[theregister.com](https://theregister.com)。

6. Google Cloud – TPU System Architecture (v4/v6 Documentation) – 定义 TPU Pod、Slice、Multislice、ICI 等概念[docs.cloud.google.com](https://docs.cloud.google.com)[docs.cloud.google.com](https://docs.cloud.google.com)。

7. Lee, J. – Why Google Doesn't Wait for NVIDIA GPUs (TPU v7 Ironwood) (Medium, 2025) – TPU Pod 扩展至9216芯片，ICI 带宽提升到1.2TB/s 双向[medium.com](https://medium.com)。

8. Google Cloud Blog – Inside the Ironwood TPU stack (2025) – TPU Ironwood 9216-chip Superpod 互联架构，光交换+3D Torus，全系统带宽超11000TB/s[cloud.google.com](https://cloud.google.com)[cloud.google.com](https://cloud.google.com)。

9. UALink Consortium – UALink FAQ (2025) – UALink 1.0 支持 1024 加速器，200G/lane 互连，低延迟高带宽开放加速器互联标准[ualinkconsortium.org](https://ualinkconsortium.org)。

10. UALink Consortium – About UALink (2024) – UALink 旨在满足 AI 日益增长的算力需求，提供 Pod 内数百加速器低延迟/高带宽负载存储语义互连；1.0规范基于IEEE 802.3dj，支持1K规模[ualinkconsortium.org](https://ualinkconsortium.org)。

11. NextPlatform – CXL Absorbs Gen-Z (2021) – Gen-Z 联盟解散并入 CXL，业界内存语义标准加速收敛到单一协议；CXL 成为连接 CPU-加速器和远程内存的首选[en.wikipedia.org](https://en.wikipedia.org)。

12. Blocks & Files – CXL 3.0 Memory Pooling (2024) – CXL 3.0/3.1 引入交换式 fabric，实现多主机共享外部内存池，缓存一致，支持大规模内存池化[blocksandfiles.com](https://blocksandfiles.com)。

13. Huawei Press Release – Huawei Unveils UnifiedBus for SuperPods (HC2025) – 华为发布灵衢 (UnifiedBus) 协议，克服现有光电互连物理限制，实现超级节点统一互连，并开放 UB2.0 规范[huawei.com](https://huawei.com)[huawei.com](https://huawei.com)。

14. Tom's Hardware – Huawei UB-Mesh Interconnect Details (Aug 2025) – 华为 UB-Mesh 技术细节：旨在统一替代 PCIe/CXL/NVLink/TCPIP，全局负载存储语义，支持百万级处理器，单芯片带宽1.25TB/s，150ns级延迟[tomshardware.com](https://tomshardware.com)[tomshardware.com](https://tomshardware.com)。

15. NVIDIA Developer Blog – Advancing Performance with SHARP (2024) – NVIDIA SHARP 将 All-Reduce/广播等集体通信从服务器卸载到 InfiniBand 交换机执行，在网内完成规约运算，减少一半数据传输并降低延迟[developer.nvidia.com](https://developer.nvidia.com)。
