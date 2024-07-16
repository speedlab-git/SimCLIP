# Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust Visual Language Models

## Abstract

<p align="justify">Vision-language models (VLMs) have achieved remarkable performance on multimodal tasks but remain vulnerable to adversarial attacks targeting the vision component. We propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. We demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while maintaining high clean accuracy across diverse downstream tasks. Notably, our approach does not require any additional training or fine-tuning of the VLM itself. Simply replacing the original vision encoder with our fine-tuned encoder is sufficient to provide robustness against adversarial attacks. This work underscores the criticality of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications.</p>

## Contents

1. [Install](#installation-guides)
2. [Dataset](#dataset)
3. [Adversarial Training](#adversarial-training)
4. [Models](#models)
5. [Evaluation](#evaluation)

## Installation

1. Clone this repository and navigate to the SimCLIP folder:

```
git clone https://github.com/speedlab-git/SimCLIP.git
cd SimCLIP
```

2. We recommend you to use [Anaconda](https://www.anaconda.com/products/distribution) to maintain installed packages and the environment. We use **Python 3.11** for our training and evaluation. Install required packages using the following commands:

```
conda create -n simclip python=3.11 -y
conda activate simclip
pip install -r requirements.txt
```

### Prerequisites

- List of prerequisites

### Installation Steps

1. Step 1
2. Step 2
3. Step 3

## Results

### Experiment 1

- Description of Experiment 1 results

### Experiment 2

- Description of Experiment 2 results

```

```
