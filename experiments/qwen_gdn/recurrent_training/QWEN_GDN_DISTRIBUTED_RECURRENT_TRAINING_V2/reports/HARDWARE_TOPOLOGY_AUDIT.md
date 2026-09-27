# Hardware topology audit

- HARDWARE_TOPOLOGY_GATE: **PASS**.
- NVLINK_AVAILABLE: **NO**; direct PyTorch peer access: **NO**.
- Selected pair: **[0, 1]**. Both 0/1 and 2/3 are PHB pairs on the same NUMA node; 0/1 were idle at selection.
- No Megatron Core or Transformer Engine installation was performed. PyTorch distributed/NCCL availability is recorded in `configs/distributed_environment.json`.

## Raw `nvidia-smi topo -m`

```text
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	SYS	SYS	0-11,24-35	0		N/A
GPU1	PHB	 X 	SYS	SYS	0-11,24-35	0		N/A
GPU2	SYS	SYS	 X 	PHB	12-23,36-47	1		N/A
GPU3	SYS	SYS	PHB	 X 	12-23,36-47	1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

## Raw `nvidia-smi nvlink -s`

```text
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-a78176ed-f97f-c54f-6201-5a01b570f422)
NVML: Unable to retrieve NVLink information as all links are inActive
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-596e6591-b065-bb91-fd35-a1e90ba8f9e3)
NVML: Unable to retrieve NVLink information as all links are inActive
GPU 2: NVIDIA GeForce RTX 3090 (UUID: GPU-b7ed673d-18ba-39ca-d7bc-c846af4f6b31)
NVML: Unable to retrieve NVLink information as all links are inActive
GPU 3: NVIDIA GeForce RTX 3090 (UUID: GPU-30b07aba-a728-7649-4899-e09d33f8bc75)
NVML: Unable to retrieve NVLink information as all links are inActive
```
