# Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust Visual Language Models

## Abstract

<embed src="utils/advllm.pdf" type="application/pdf">

<p align="justify">Vision-language models (VLMs) have achieved remarkable performance on multimodal tasks but remain vulnerable to adversarial attacks targeting the vision component. We propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. We demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while maintaining high clean accuracy across diverse downstream tasks. Notably, our approach does not require any additional training or fine-tuning of the VLM itself. Simply replacing the original vision encoder with our fine-tuned encoder is sufficient to provide robustness against adversarial attacks. This work underscores the criticality of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications.</p>

## Contents

1. [Installation](#installation-guides)
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

## Dataset

### Adversarial training dataset

We adversarially pre-train CLIP on the ImageNet dataset. Please download the ImageNet dataset from [here](https://www.image-net.org/download.php) or use the following command:

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

Download the ImageNet dataset and extract the training and validation data using the provided script in `bash` folder:

```
./bash/imagenet/extract_ILSVRC.sh
```

### Evaluation dataset

<p align="justify">For evaluating the robustness and performance of our fine-tuned CLIP vision encoder, we utilize a diverse set of datasets tailored for different tasks. For Visual Question Answering (VQA)
tasks, we employ the OKVQA and VizWiz datasets, which provide challenging benchmarks for assessing the model's ability to understand and answer questions based on visual content.
For image captioning tasks, we use the COCO and Flickr30k datasets, which are widely recognized for their comprehensive annotations and variety of images. The following table provides download links for each dataset we used in our experiments:<p>

| Dataset Name | Download Link                                                                         |
| ------------ | ------------------------------------------------------------------------------------- |
| OKVQA        | [Download OKVQA](https://okvqa.allenai.org/download.html)                             |
| COCO         | [Download COCO](https://cocodataset.org/#download)                                    |
| Flickr30k    | [Download Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) |
| VizWiz       | [Download VizWiz](https://vizwiz.org/tasks-and-datasets/)                             |

#Adversarial training
