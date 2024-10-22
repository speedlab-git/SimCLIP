
<h1 align="center">Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models</h1>

<p align="center">
  <b>Md Zarif Hossain</b> · <b>Ahmed Imteaj</b>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2407.14971"><img src="https://img.shields.io/badge/arXiv-2407.14971-red"></a>
  <img src="https://img.shields.io/badge/Paper%20Status-In--Review-yellow">
  <a href="https://github.com/speedlab-git/SimCLIP"><img src="https://img.shields.io/badge/Code-Repository-blue"></a>
</p>

<hr>


<!-- ![system architecture](./utils/arch.png) -->

<p align="justify">Vision-language models (VLMs) have achieved significant strides in recent years, particularly in multimodal tasks. However, VLMs remain susceptible to adversarial attacks, specially on their vision components. This paper introduces Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder, preserving fine-grained semantic details of visual representations even in unsusceptible adversarial situations. Sim-CLIP is built upon Siamese architecture to learn Interpretable and Meaningful visual representations. Through adversarial fine-tuning of the CLIP encoder, Sim-CLIP enhances resilience against adversarial attacks while capturing nuanced semantic details of perturbed images, distinguishing it from state-of-the-art CLIP models. Sim-CLIP achieves such robustness without requiring large batch sizes, which demand significant resources and can affect generalization, or additional momentum encoders, which add complexity and computational overhead. We present comprehensive performance benchmark of Sim-CLIP compared to state-of-the-art CLIP models across a wide range of datasets and various downstream VLM applications (e.g., image captioning, question-answering, zero-shot tasks). Our results expose the vulnerabilities of existing methods under unsusceptible adversarial conditions and demonstrate the superior robustness of Sim-CLIP while retaining nuanced details of the perturbed images.</p>

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Adversarial Training](#adversarial-training)
- [Models](#models)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Overview

<p align="center">
  <img src="./utils/arch_robust.png" width="950" alt="accessibility text">
</p>
<p align="justify">
Adversarial attacks on vision-language models (VLMs) pose a significant challenge as they can compromise the model's ability to accurately interpret and respond to visual inputs. Previous approaches such as FARE have attempted to enhance the robustness of VLMs by minimizing the discrepancy between the embeddings of clean and adversarially perturbed images using ℓ<sub>2</sub> loss. However, this method struggles with high-dimensional data leading to inefficiencies and a failure to capture the full semantic meaning of image features. To address these limitations we introduce Sim-CLIP, an unsupervised adversarial fine-tuning approach for the CLIP vision encoder that leverages a Siamese architecture. By generating perturbed views of input images and employing a cosine similarity loss our method encourages the model to learn features invariant to adversarial perturbations. Additionally, we incorporate a stop-gradient mechanism to prevent loss collapse ensuring stable and effective training. This results in a model that not only exhibits improved robustness against adversarial attacks but also retains the semantic richness of image features outperforming previous methods in maintaining meaningful patterns essential for downstream tasks.
</p>

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
If you are using windows, please use `linux subsystem (WSL)`

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

After downloading the ImageNet dataset, extract the training and validation data using the provided script in `bash` folder:

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

<!-- https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main -->

## Adversarial training

In this repository, we provide scripts for running adversarial training with `FARE` and `TeCoA` alongside our proposed method, `Sim-CLIP`. We have provided bash scripts for easier execution of these training methods. Each script is tailored to run the respective training method with the necessary configurations.

### 1. Sim-CLIP<sup>4</sup>

```
python -m train.adversarial_training_simclip --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC --template std --output_normalize False --steps 10000 --warmup 1400 --batch_size 64 --loss l2 --opt adamw --lr 1e-3 --wd 1e-5 --attack pgd --attack_loss l2 --norm linf --eps 4 --iterations_adv 10 --stepsize_adv 1 --wandb True --output_dir "output directory" --experiment_name SimCLIP4 --log_freq 10
```

or execute the bash script(you can specify the training parameters inside). Make sure you are in the `SimCLIP` folder

```
./bash/training/simclip_train.sh
```

### 2. FARE<sup>4</sup>

```
python -m train.adversarial_training_clip --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC --template std --output_normalize False --steps 10000 --warmup 1400 --batch_size 64 --loss l2 --opt adamw --lr 1e-5 --wd 1e-4 --attack pgd --inner_loss l2 --norm linf --eps 4 --iterations_adv 10 --stepsize_adv 1 --wandb False --output_dir "output directory" --experiment_name FARE4 --log_freq 10
```

```
./bash/training/fare_train.sh
```

### 3. TeCoA<sup>4</sup>

```
python -m train.adversarial_training_clip_up --clip_model_name ViT-L-14 --pretrained openai --dataset imagenet --imagenet_root /c/CodesSpring24/Data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC --template std --output_normalize False --steps 10000 --warmup 1400 --batch_size 64 --loss ce --opt sgd --lr 1e-3 --wd 1e-5 --attack pgd --inner_loss ce --norm linf --eps 4 --iterations_adv 10 --stepsize_adv 1 --wandb True --output_dir "output directory" --experiment_name TeCOA4 --log_freq 10
```

```
./bash/training/tecoa_train.sh
```
We utilize 2x A6000 GPU for our adversarial training and inference

### **Note:**

- Set `--imagenet_root` with the path of your downloaded ImageNet dataset. Set `eps 2` to obtain Sim-CLIP<sup>2</sup>, FARE<sup>2</sup> and TeCoA<sup>2</sup> models
- We recommend a dual GPU setup with a total of 32 GB VRAM. If you are facing any issues with the GPU running out of memory, please reduce the `batch size`
- Modify the `output_dir` parameter to specify the directory to save the model checkpoints

## Models

| Model Name           | Type   | Proposed By                                                  | Download Link                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| CLIP                 | Clean  | [OpenAI](https://arxiv.org/pdf/2103.00020)                   | [Load CLIP model](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel)        |
| Sim-CLIP<sup>4</sup> | Robust | Our Method                                                   | [Download Sim-CLIP<sup>4</sup>](https://huggingface.co/hossainzarif19/SimCLIP/blob/main/simclip4.pt)        |
| Sim-CLIP<sup>2</sup> | Robust | Our Method                                                   | [Download Sim-CLIP<sup>2</sup>](https://huggingface.co/hossainzarif19/SimCLIP/blob/main/simclip2.pt)        |
| FARE<sup>4</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>4</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |
| FARE<sup>2</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>2</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |
| TeCoA<sup>4</sup>    | Robust | [Mao et al. (2023)](https://arxiv.org/abs/2212.07016)        | [Download TeCoA<sup>4</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978) |
| TeCoA<sup>2</sup>    | Robust | [Mao et al. (2023)](https://arxiv.org/abs/2212.07016)        | [Download TeCoA<sup>2</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978) |

## Usage

To use these models, you can load them using the provided code. For example, to load the Sim-CLIP<sup>4</sup> model, you can use the following code snippet:

```
import torch
import open_clip
model, _, image_processor = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device='gpu'
        )

checkpoint = torch.load('simclip4.pt path', map_location=torch.device('gpu'))
model.vision_encoder.load_state_dict(checkpoint)
```

## Evaluation

### Zero-shot Classification

Acquire the classification dataset by visiting the Huggingface CLIP_benchmark repository at [Huggingface CLIP_benchmark](https://huggingface.co/clip-benchmark). Configure the models for evaluation in `CLIP_benchmark/benchmark/models.txt` and specify the datasets in `CLIP_benchmark/benchmark/datasets.txt`. Then execute

```

cd CLIP_benchmark
./bash/run_benchmark_adv.sh

```

### Down-stream tasks evaluation (Untargeted Attacks)

Before proceeding with Down-stream tasks evaluations, download validation annotations set from [Huggingface openflamingo repository](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main)

### Captioning Tasks

- OpenFlamingo

  To evaluate the OpenFlamingo 9B model, first download the model from [here](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main). Then, supply the downloaded annotation set and flamingo checkpoint paths in `/bash/of_eval_9B_coco.sh` . Set the `--vision_encoder_pretrained` parameter to `openai` or provide the path to a fine-tuned CLIP model checkpoint (e.g., Sim-CLIP). Finally, run the evaluation script.

```

./bash/of_eval_9B_coco.sh

```

```
./bash/of_eval_9B_Flickr.sh

```

- LLaVA

  The LLaVA model checkpoint will be automatically downloaded from repository. Update the dataset path with the location of your downloaded dataset and then execute the following command:

```

./bash/LLaVA_eval_coco.sh

```

### Visual Question Answering Tasks

- For VQA, provide the path of OKVQA dataset in the script and then execute the following commands:

For LLaVA run

```

./bash/LLaVA_eval_okvqa.sh

```

For Flamingo run

```

./bash/of_eval_9B_okvqa.sh

```

## Targeted attacks

To perform targeted attacks with the LLaVA model on the COCO or Flickr30k dataset, please run these steps:

```
./bash/eval_targeted.sh
```

**Note**: Default target strings can be updated in `run_evaluation.py`

For targeted attacks on custom images, update `vlm_eval/run_evaluation_qualitative.py` with your images and captions, then execute:

```
python -m vlm_eval.run_evaluation_qualitative --precision float32 --attack apgd --eps 2 --steps 10000 --vlm_model_name LLaVA --vision_encoder_pretrained openai --verbose
```

**Note**:
To increase the strength of the attack, modify the `--attack` parameter with higher steps in the bash script. A higher attack step size results in a stronger attack.

## Results

<p align="justify">
The tables provided illustrate the robust performance of various vision-language models (VLMs) using different versions of the CLIP model across two main tasks: image captioning and visual question answering (VQA). For the image captioning tasks on COCO and Flickr30k datasets, we evaluate the models using the CIDEr score, which quantifies the consensus between a candidate caption and reference captions. Higher CIDEr scores indicate a better capture of semantic meaning and relevance in the generated captions. For VQA tasks on VizWiz and OKVQA datasets, we report accuracy to measure the model's ability to correctly interpret and answer questions based on visual content.
From the results, Sim-CLIP models, particularly Sim-CLIP<sup>4</sup>, consistently outperform other robust models like FARE and TECOA across most datasets in both clean and adversarial settings. This suggests that Sim-CLIP's adversarial fine-tuning approach not only enhances robustness against adversarial attacks but also preserves or even improves the model's ability to understand and generate semantically meaningful responses in complex multimodal tasks.
In adversarial scenarios, where models are evaluated at an epsilon value of 4/255, Sim-CLIP models demonstrate superior resilience, maintaining higher performance metrics compared to other models. This robustness is crucial for practical applications where models might encounter manipulated or adversarially altered inputs.
</p>

- ### Clean Evaluation

<table>
    <thead>
        <tr>
            <th>VLM</th>
            <th>Vision Encoder</th>
            <th>COCO</th>
            <th>Flickr30k</th>
            <th>VizWiz</th>
            <th>OKVQA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7">Open Flamingo</td>
            <td>Sim-CLIP-2</td>
            <td>85.6</td>
            <td>56.3</td>
            <td>21.8</td>
            <td>35.1</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>84.3</td>
            <td>53.1</td>
            <td>22.1</td>
            <td>34.5</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>74.5</td>
            <td>48.2</td>
            <td>22.3</td>
            <td>33.6</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>81.6</td>
            <td>54.5</td>
            <td>20.0</td>
            <td>32.0</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>81.4</td>
            <td>51.8</td>
            <td>16.4</td>
            <td>31.8</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>71.0</td>
            <td>45.6</td>
            <td>19.3</td>
            <td>31.0</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>60.5</td>
            <td>41.7</td>
            <td>18.0</td>
            <td>28.2</td>
        </tr>
        <tr>
            <td rowspan="7">LLaVA 1.5</td>
            <td>Sim-CLIP-2</td>
            <td>109.1</td>
            <td>66.3</td>
            <td>45.8</td>
            <td>54.5</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>108.1</td>
            <td>65.1</td>
            <td>43.8</td>
            <td>54.1</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>85.3</td>
            <td>60.3</td>
            <td>41.6</td>
            <td>55.3</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>103.6</td>
            <td>65.0</td>
            <td>41.3</td>
            <td>54.6</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>105.1</td>
            <td>64.0</td>
            <td>41.7</td>
            <td>54.5</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>79.1</td>
            <td>57.7</td>
            <td>39.4</td>
            <td>50.0</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>75.2</td>
            <td>50.3</td>
            <td>37.5</td>
            <td>48.1</td>
        </tr>
    </tbody>
</table>

- ### Adversarial evaluation at $\epsilon$ = 4/255 radii

<table>
    <thead>
        <tr>
            <th>VLM</th>
            <th>Vision Encoder</th>
            <th>COCO</th>
            <th>Flickr30k</th>
            <th>VizWiz</th>
            <th>OKVQA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7">Open Flamingo</td>
            <td>Sim-CLIP-2</td>
            <td>58.4</td>
            <td>35.1</td>
            <td>13.6</td>
            <td>19.7</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>53.5</td>
            <td>34.3</td>
            <td>12.3</td>
            <td>17.1</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>40.3</td>
            <td>27.4</td>
            <td>10.6</td>
            <td>15.3</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>60.5</td>
            <td>39.2</td>
            <td>15.7</td>
            <td>22.0</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>56.1</td>
            <td>37.6</td>
            <td>13.7</td>
            <td>19.2</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>50.3</td>
            <td>32.9</td>
            <td>14.7</td>
            <td>20.5</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>5.6</td>
            <td>3.8</td>
            <td>1.8</td>
            <td>0</td>
        </tr>
        <tr>
            <td rowspan="7">LLaVA 1.5</td>
            <td>Sim-CLIP-2</td>
            <td>69.4</td>
            <td>42.3</td>
            <td>33.4</td>
            <td>33.9</td>
        </tr>
        <tr>
            <td>FARE-2</td>
            <td>68.3</td>
            <td>41.6</td>
            <td>30.1</td>
            <td>31.8</td>
        </tr>
        <tr>
            <td>TECOA-2</td>
            <td>61.1</td>
            <td>36.1</td>
            <td>25.3</td>
            <td>24.1</td>
        </tr>
        <tr>
            <td>Sim-CLIP-4</td>
            <td>72.9</td>
            <td>46.3</td>
            <td>35.2</td>
            <td>36.2</td>
        </tr>
        <tr>
            <td>FARE-4</td>
            <td>71.7</td>
            <td>43.6</td>
            <td>33.7</td>
            <td>34.0</td>
        </tr>
        <tr>
            <td>TECOA-4</td>
            <td>65.5</td>
            <td>39.4</td>
            <td>28.4</td>
            <td>27.2</td>
        </tr>
        <tr>
            <td>CLIP</td>
            <td>13.5</td>
            <td>10.0</td>
            <td>3.2</td>
            <td>0.0</td>
        </tr>
    </tbody>
</table>

## Acknowledgements

We extend our gratitude to the developers and contributors of the following repositories for their invaluable resources and tools that have significantly aided in the development and evaluation of our project:

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [RobustVLM](https://github.com/chs20/RobustVLM)
