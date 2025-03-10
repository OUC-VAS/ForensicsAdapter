# Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection (CVPR 2025)
**👥 Authors: Xinjie Cui, Yuezun Li, Ao Luo, Jiaran Zhou, Junyu Dong**

📄 [Paper](https://arxiv.org/abs/2411.19715) | ⚖️ [Pre-trained Weights](XXXXX)

![Pipeline of the proposed Forensics Adapter. ](https://github.com/OUC-VAS/ForensicsAdapter/blob/main/figures/structure.png)

## 🚀 Start

- [⚙️ Environment Setup](#environment-setup)
- [📂 Dataset](#dataset)
- [🏋️ Training](#training)
- [🧪 Testing](#testing)

## Environment Setup
Ensure your environment meets the following requirements:

- 🐍 Python 3.7
- 🔥 PyTorch 1.11
- 🚀 CUDA 11.3

Install dependencies:

```bash
conda create -n FA python=3.7.2
conda activate FA
sh install.sh
```

## Dataset

We use multiple datasets for training and evaluation:

- FF++
- DFDC
- DFDCP
- DFD
- CD1/CD2

Follow the official guidelines to download and preprocess the data.

## Training

Start training with the following command:

```bash
python train.py 
```

## Testing

Run the following command to test the model:

```bash
python test.py 
```

