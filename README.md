<p align="center">
  <img src="docs/assets/logo2.png" alt="SkyReels Logo" width="50%">
</p>

# SkyReels V1: Human-Centric Video Foundation Model

<p align="center">
🤗 <a href="https://huggingface.co/collections/Skywork/skyreels-v1-67b34676ff65b4ec02d16307" target="_blank">Hugging Face</a> · 👋 <a href="https://www.skyreels.ai/" target="_blank">Playground</a>
</p>

---
Welcome to the SkyReels V1 repository! Here, you'll find the Text-to-Video & Image-to-Video model weights and inference code for our groundbreaking video foundation model.

## 🔥🔥🔥 News!!

* Feb 18, 2025: 👋 We release the inference code and model weights of [SkyReels-V1 Text2Video Model](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V).
* Feb 18, 2025: 👋 We release the inference code and model weights of [SkyReels-V1 Image2Video Model](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V).
* Feb 18, 2025: 🔥 We also release [SkyReels-A1](https://github.com/SkyworkAI/SkyReels-A1). This is an open-sourced and effective framework portrait image animation model.

## 🎥 Demos
<div align="center">
<video src="https://github.com/user-attachments/docs/assets/2dbd116a-033d-4f7e-bd90-78a3da47cd9c" width="70%"> </video>
</div>

## 📑 TODO List

- SkyReels-V1 (Text2Video Model)
  - [x] Checkpoints
  - [x] Inference Code
  - [x] Web Demo (Gradio)
  - [x] User-Level GPU Inference on RTX4090
  - [x] Parallel Inference on Multi-GPUs
  - [ ] Prompt Rewrite && Prompt Guidance
  - [ ] CFG-distilled Model
  - [ ] Lite Model
  - [ ] 720P Version
  - [ ] ComfyUI

- SkyReels-V1 (Image2Video Model)
  - [x] Checkpoints
  - [x] Inference Code
  - [x] Web Demo (Gradio)
  - [x] User-Level GPU Inference on RTX4090
  - [x] Parallel Inference on Multi-GPUs
  - [ ] Prompt Rewrite && Prompt Guidance
  - [ ] CFG-distilled Model
  - [ ] Lite Model
  - [ ] 720P Version
  - [ ] ComfyUI

## 🌟 Overview

SkyReels V1 is the first and most advanced open-source human-centric video foundation model. By fine-tuning <a href="https://huggingface.co/tencent/HunyuanVideo">HunyuanVideo</a> on O(10M) high-quality film and television clips, SkyReels V1 offers three key advantages:

1. **Open-Source Leadership**: Our Text-to-Video model achieves state-of-the-art (SOTA) performance among open-source models, comparable to proprietary models like Kling and Hailuo.
2. **Advanced Facial Animation**: Captures 33 distinct facial expressions with over 400 natural movement combinations, accurately reflecting human emotions.
3. **Cinematic Lighting and Aesthetics**: Trained on high-quality Hollywood-level film and television data, each generated frame exhibits cinematic quality in composition, actor positioning, and camera angles.

## 🔑 Key Features

### 1. Self-Developed Data Cleaning and Annotation Pipeline

Our model is built on a self-developed data cleaning and annotation pipeline, creating a vast dataset of high-quality film, television, and documentary content.

- **Expression Classification**: Categorizes human facial expressions into 33 distinct types.
- **Character Spatial Awareness**: Utilizes 3D human reconstruction technology to understand spatial relationships between multiple people in a video, enabling film-level character positioning.
- **Action Recognition**: Constructs over 400 action semantic units to achieve a precise understanding of human actions.
- **Scene Understanding**: Conducts cross-modal correlation analysis of clothing, scenes, and plots.

### 2. Multi-Stage Image-to-Video Pretraining

Our multi-stage pretraining pipeline, inspired by the <a href="https://huggingface.co/tencent/HunyuanVideo">HunyuanVideo</a> design, consists of the following stages:

- **Stage 1: Model Domain Transfer Pretraining**: We use a large dataset (O(10M) of film and television content) to adapt the text-to-video model to the human-centric video domain.
- **Stage 2: Image-to-Video Model Pretraining**: We convert the text-to-video model from Stage 1 into an image-to-video model by adjusting the conv-in parameters. This new model is then pretrained on the same dataset used in Stage 1.
- **Stage 3: High-Quality Fine-Tuning**: We fine-tune the image-to-video model on a high-quality subset of the original dataset, ensuring superior performance and quality.

## 📊 Benchmark Results
We evaluate the performance of our text-to-video model using <a href="https://github.com/Vchitect/VBench">VBench</a>, comparing it with other outstanding open-source models.

Based on the benchmark results, SkyReels V1 demonstrates SOTA performance among open-source Text-to-Video (T2V) models. Specifically, our model achieves an overall score of 82.43, which is higher than other open-source models such as VideoCrafter-2.0 VEnhancer (82.24) and CogVideoX1.5-5B (82.17). Additionally, our model achieves the highest scores in several key metrics, including Dynamic Degree and Multiple Objects, indicating our model's superior ability to handle complex video generation tasks.
| Models                    | Overall | Quality Score | Semantic Score | Image Quality | Dynamic Degree | Multiple Objects | Spatial Relationship |  
|---------------------------|---------|---------------|----------------|---------------|----------------|------------------|----------------------|
| OpenSora V1.3             | 77.23   | 80.14         | 65.62          | 56.21         | 30.28          | 43.58            | 51.61                |
| AnimateDiff-V2            | 80.27   | 82.90         | 69.75          | 70.1          | 40.83          | 36.88            | 34.60                |
| VideoCrafter-2.0 VEnhancer| 82.24   | 83.54         | 77.06          | 65.35         | 63.89          | 68.84            | 57.55                |
| CogVideoX1.5-5B           | 82.17   | 82.78         | 79.76          | 65.02         | 50.93          | 69.65            | 80.25                |
| HunyuanVideo 540P         | 81.23   | 83.49         | 72.22          | 66.31         | 51.67          | 70.45            | 63.46                |
| SkyReels V1 540P (Ours)   | **82.43** | **84.62**     | 73.68          | 67.15         | **72.5**       | **71.61**        | 70.83                |    


## 📦 Model Introduction
| Model Name      | Resolution | Video Length | FPS | Download Link |
|-----------------|------------|--------------|-----|---------------|
| SkyReels-V1-Hunyuan-I2V | 544px960p  | 97           | 24  | 🤗 [Download](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V) |
| SkyReels-V1-Hunyuan-T2V | 544px960p  | 97           | 24  | 🤗 [Download](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-T2V) |


## 🚀 SkyReels Infer Introduction

SkyReelsInfer is a highly efficient video generation inference framework that enables accurate and swift production of high-quality videos, making video generation inference significantly faster without any loss in quality.

**Multi-GPU Inference Support**: The framework accommodates Context Parallel, CFG Parallel, and VAE Parallel methodologies, facilitating rapid and lossless video production to meet the stringent low-latency demands of online environments.

**User-Level GPU Deployment**: By employing model quantization and parameter-level offload strategies, the system significantly reduces GPU memory requirements, catering to the needs of consumer-grade graphics cards with limited VRAM.

**Superior Inference Performance**: Demonstrating exceptional efficiency, the framework achieves a 58.3% reduction in end-to-end latency compared to HunyuanVideo XDiT, setting a new benchmark for inference speed.

**Excellent Usability**: Built upon the open-source framework Diffusers and featuring a non-intrusive parallel implementation approach, the system ensures a seamless and user-friendly experience.

## 🛠️ Running Guide

Begin by cloning the repository:
```shell
git clone https://github.com/SkyworkAI/SkyReels-V1
cd skyreelsinfer
```

### Installation Guide for Linux

We recommend Python 3.10 and CUDA version 12.2 for the manual installation.

```shell
# Install pip dependencies
pip install -r requirements.txt
```

When sufficient VRAM is available (e.g., on A800), the lossless version can be run directly.

**Note: When generating videos, the prompt should start with "FPS-24, " as we referenced the controlling the fps training method from <a href=https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models>Moviegen</a> during training.**

```shell
SkyReelsModel = "Skywork/SkyReels-V1-Hunyuan-T2V"
python3 video_generate.py \
    --model_id ${SkyReelsModel} \
    --task_type t2v \
    --guidance_scale 6.0 \
    --height 544 \
    --width 960 \
    --num_frames 97 \
    --prompt "FPS-24, A cat wearing sunglasses and working as a lifeguard at a pool" \
    --embedded_guidance_scale 1.0
```

### User-Level GPU Inference (RTX4090)

We list the height/width/frame settings we recommend in the following table.
|      Resolution       |           h/w=9:16           |    h/w=16:9     |     h/w=1:1     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|
|         544p          |        544px960px97f        |  960px544px97f |  720px720px97f |

#### Using Command Line

```shell
# SkyReelsModel: If using i2v, switch to Skywork/SkyReels-V1-Hunyuan-I2V.
# quant: Enable FP8 weight-only quantization
# offload: Enable offload model
# high_cpu_memory: Enable pinned memory to reduce the overhead of model offloading.
# parameters_level: Further reduce GPU VRAM usage.
# task_type:The task type is designated to support both t2v and i2v. For the execution of an i2v task, it is necessary to input --image.
SkyReelsModel = "Skywork/SkyReels-V1-Hunyuan-T2V"
python3 video_generate.py \
    --model_id ${SkyReelsModel} \
    --task_type t2v \
    --guidance_scale 6.0 \
    --height 544 \
    --width 960 \
    --num_frames 97 \
    --prompt "FPS-24, A cat wearing sunglasses and working as a lifeguard at a pool" \
    --embedded_guidance_scale 1.0 \
    --quant \
    --offload \
    --high_cpu_memory \
    --parameters_level
```
The example above shows generating a 544px960px97f 4s video on a single RTX 4090 with full VRAM optimization, peaking at 18.5G VRAM usage. At maximum VRAM capacity, a 544px960px289f 12s video can be produced (using `--sequence_batch`, taking ~1.5h on one RTX 4090; adding GPUs greatly reduces time).

#### 🚀 Parallel Inference on Multiple GPUs

```shell
# SkyReelsModel: If using i2v, switch to Skywork/SkyReels-V1-Hunyuan-I2V.
# quant: Enable FP8 weight-only quantization
# offload: Enable offload model
# high_cpu_memory: Enable pinned memory to reduce the overhead of model offloading.
# gpu_num: Number of GPUs used.
SkyReelsModel = "Skywork/SkyReels-V1-Hunyuan-T2V"
python3 video_generate.py \
    --model_id ${SkyReelsModel} \
    --guidance_scale 6.0 \
    --height 544 \
    --width 960 \
    --num_frames 97 \
    --prompt "FPS-24, A cat wearing sunglasses and working as a lifeguard at a pool" \
    --embedded_guidance_scale 1.0 \
    --quant \
    --offload \
    --high_cpu_memory \
    --gpu_num $GPU_NUM
```

## Performance Comparison

This test aims to compare the end-to-end latency of SkyReelsInfer and HunyuanVideo XDiT for 544p video processing on both the A800 (high-performance computing GPU) and RTX 4090 (consumer-grade GPU). The results will demonstrate the superior inference performance of SkyReelsInfer in terms of speed and efficiency.

### Testing Parameters

|      Resolution       |           video size           |    transformer step    |     guidance_scale     |
|:---------------------:|:----------------------------:|:---------------:|:---------------:|
|         540p          |        544px960px97f        |  30 |  6 |


### User-Level GPU Inference (RTX4090)

In practice, Hunyuanvideo XDIT cannot perform inference on the RTX 4090 due to insufficient VRAM. To address this issue, we implemented fixes based on the official offload, FP8 model weights, and VAE tiling. These include:  
a) Optimizing the model loading and initialization logic to avoid fully loading the FP16 model into memory.  
b) Reducing the VAE tiling size to alleviate memory usage.
For the deployment of SkyReelsInfer on the RTX 4090, the following measures will be implemented to ensure sufficient VRAM availability and efficient inference:  
a) **Model Quantization**: Apply FP8 weight-only quantization to ensure the model can be fully loaded into memory.  
b) **Offload Strategy**: Enable parameter-level offloading to further reduce VRAM usage.  
c) **Multi-GPU Parallelism**: Activate context parallelism, CFG parallelism, and VAE parallelism for distributed processing.  
d) **Computation Optimization**: Optimize attention layer calculations using SegaAttn and enable Torch.Compile for transformer compilation optimization (supporting both 4-GPU and 8-GPU configurations).


|      GPU NUM      |           hunyuanvideo + xdit    |           SkyReelsInfer   | 
|:---------------------:|:----------------------------:|:----------------------------:|
|         1          |        VRAM OOM        |        889.31s        |
|         2          |        VRAM OOM        |        453.69s        |
|         4          |        464.3s        |        293.3s        |
|         8          |        Cannot split video sequence into ulysses_degree x ring_degree        |        159.43s        |

The table above summarizes the end-to-end latency test results for generating 544p 4-second videos on the RTX 4090 using HunyuanVideo XDIT and SkyReelsVideoInfer. The following conclusions can be drawn:  
- Under the same RTX 4090 resource conditions (4 GPUs), the SkyReelsInfer version reduces end-to-end latency by **58.3%** compared to HunyuanVideo XDIT (293.3s vs. 464.3s).  
- The SkyReelsInfer version features a more robust deployment strategy, supporting inference deployment across **1 to 8 GPUs** at the user level.


### A800
Based on the A800 (80G), the primary testing focused on comparing the performance differences between HunyuanVideo XDIT and SkyReelsInfer without compromising output quality.

|      GPU NUM      |           hunyuanvideo + xdit    |           SkyReelsInfer   | 
|:---------------------:|:----------------------------:|:----------------------------:|
|         1          |        884.20s       |        771.03s        |
|         2          |        487.22s        |        387.01s        |
|         4          |        263.48s        |        205.49s        |
|         8          |        Cannot split video sequence into ulysses_degree x ring_degree        |        107.41s        |

The table above summarizes the end-to-end latency test results for generating 544p 4-second videos on the A800 using HunyuanVideo XDIT and SkyReelsVideoInfer. The following conclusions can be drawn:

Under the same A800 resource conditions, the SkyReelsInfer version reduces end-to-end latency by 14.7% to 28.2% compared to the official HunyuanVideo version.

The SkyReelsInfer version features a more robust multi-GPU deployment strategy.

## Acknowledgements
We would like to thank the contributors of <a href="https://huggingface.co/tencent/HunyuanVideo">HunyuanVideo</a> repositories, for their open research and contributions.

## Citation

```bibtex
@misc{SkyReelsV1,
  author = {SkyReels-AI},
  title = {Skyreels V1: Human-Centric Video Foundation Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SkyworkAI/SkyReels-V1}}
}
```
