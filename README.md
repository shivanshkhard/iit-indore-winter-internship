# iit-indore-winter-internship
project on computer vision image to image translation using advance GANs
Frequency-Aware Semantic Consistency for Unpaired Image-to-Image Translation
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)
![GAN](https://img.shields.io/badge/Model-GAN-orange.svg)
![ViT](https://img.shields.io/badge/Backbone-ViT-purple.svg)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-black.svg)
![DINO](https://img.shields.io/badge/DINO-ViT-blueviolet.svg)
![Wavelet](https://img.shields.io/badge/Frequency-Wavelet%20DWT-teal.svg)
![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)


This repository contains the implementation of a Frequency-Aware GAN framework for unpaired image-to-image translation, inspired by the research paper:

“Towards Semantically Continuous Unpaired Image-to-Image Translation via Margin Adaptive Contrastive Learning and Wavelet Transform”
Expert Systems With Applications, Elsevier (2024)

The project focuses on semantic consistency preservation, frequency-domain modeling, and contrastive learning for high-quality unpaired image translation.

🔍 Overview

Unpaired image-to-image translation aims to learn a mapping between two visual domains without paired training data.
Traditional GAN-based approaches often suffer from:

Semantic distortions

Texture inconsistency

Mode collapse

This work addresses these issues by integrating:

Wavelet-based frequency decomposition

Margin Adaptive Contrastive Learning (MACL)

Frequency Feature Transform Normalization (FFTN)

CLIP and DINO-based semantic guidance

🧠 Key Contributions

✅ Frequency-aware generator using Haar Wavelet Transform

✅ Adaptive Frequency Fusion (AFF) for low/high-frequency integration

✅ FFTN blocks for frequency-guided feature normalization

✅ CLIP-based patch-level contrastive loss (MACL)

✅ DINO-ViT global semantic consistency loss

✅ Stable training with PatchGAN discriminator

✅ Automatic checkpoint resuming support

🏗️ Architecture
Generator (MACL Generator)

Encoder–Decoder CNN

Haar DWT/IDWT for frequency decomposition

FFTN-based residual blocks

Adaptive Frequency Fusion (AFF)

Discriminator

PatchGAN with Spectral Normalization

Semantic Guidance

CLIP ViT-B/32 → patch-level contrastive features

DINO ViT-Base → global semantic consistency

📂 Project Structure
macl_net/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   ├── resblocks.py
│   ├── fftn.py
│   ├── aff.py
│   ├── clip_encoder.py
│   └── dino_encoder.py
│
├── losses/
│   ├── gan_loss.py
│   ├── macl_loss.py
│   └── global_vit_loss.py
│
├── utils/
│   ├── wavelet.py
│   └── image_utils.py
│
├── data/
│   └── unpaired_dataset.py
│
├── checkpoints/
├── outputs/
│
├── train.py
├── plot_results.py
├── requirements.txt
└── README.md

🗂️ Dataset

We use the Horse ↔ Zebra unpaired dataset, originally introduced with CycleGAN.

Domain A: Horse images

Domain B: Zebra images

No paired samples required

Ensure dataset structure:

dataset/
├── horse/
└── zebra/

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/macl-net.git
cd macl-net

2️⃣ Install Dependencies
pip install -r requirements.txt


⚠️ CUDA-enabled PyTorch is strongly recommended.

🚀 Training
Start Training (with Auto-Resume)
python train.py


Automatically resumes from latest checkpoint

Supports GPU acceleration

Saves outputs and checkpoints periodically

Force Fresh Training from Epoch 0
rm -rf checkpoints/*
python train.py

🔁 Checkpoint Resume Logic

Automatically loads latest macl_epoch_*.pth

Training continues from the last completed epoch

Safe to interrupt and resume anytime

📊 Evaluation & Visualization
Plot Training Curves
python plot_results.py


Generates:

Generator loss

Discriminator loss

MACL loss

Global semantic loss

Output Samples

Generated images are saved in:

outputs/fake_<step>.png

🖼️ Results

Progressive improvement in zebra texture patterns

Strong semantic alignment with input horse structure

Reduced distortion compared to baseline GANs

Stable convergence with frequency-aware learning

📖 Reference

If you use this work, please cite:

H. Zhang, Y.-J. Yang, and W. Zeng,
"Towards Semantically Continuous Unpaired Image-to-Image Translation
via Margin Adaptive Contrastive Learning and Wavelet Transform,"
Expert Systems With Applications, 2024.

🙏 Acknowledgements

This work was carried out under the guidance of
Prof. Surya Prakash, IIT Indore

Special thanks to Mr. Prasant Phatak (PhD Scholar)
for technical mentoring and research guidance.

📜 License

This project is intended for research and academic use only.

✨ Author

Shivansh Gupta
B.Tech (AI & Data Science)
Dual Degree, IIT Madras (Data Science)
Intern, IIT Indore
