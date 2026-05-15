# Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection (CVPR 2025)
**👥 Authors: Xinjie Cui, [Yuezun Li](https://yuezunli.github.io/) (corresponding author), Ao Luo, Jiaran Zhou, Junyu Dong**




![Pipeline of the proposed Forensics Adapter. ](https://github.com/OUC-VAS/ForensicsAdapter/blob/main/figures/archi.png)

---

## 📚 Resources  
| **Section**       | **Content**                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| 📄 **Paper**       | [arXiv Preprint](https://arxiv.org/abs/2411.19715)                       |
| ⚖️ **Model Weights** | [Google Drive](https://drive.google.com/file/d/1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm/view?usp=sharing) |

---

## 📊 Benchmark Comparison  
## 🖼️ Frame-Level Comparison 
🏆 **Champion Method Alert**: Our approach establishes new state-of-the-art on all frame-level benchmarks!


| Method         | Venue      | CDF-v1  | CDF-v2  | DFDC  | DFDCP  | DFD  | Avg. 📈 |
|----------------|------------|---------|---------|-------|--------|------|-------|
| SPSL       | CVPR'21    | 0.815   | 0.765   | 0.704 | 0.741  | 0.812| 0.767 |
| SRM       | CVPR'21    | 0.793   | 0.755   | 0.700 | 0.741  | 0.812| 0.760 |
| Reece       | CVPR'22    | 0.768   | 0.732   | 0.713 | 0.734  | 0.812| 0.752 |
| SBI        | CVPR'22    | -       | 0.813   | -     | 0.799  | 0.774| -     |
| UCF        | ICCV'23    | 0.779   | 0.753   | 0.719 | 0.759  | 0.807| 0.763 |
| ED          | AAAI'24    | 0.818   | 0.864   | 0.721 | 0.851  | -    | -     |
| LSDA      | CVPR'24    | 0.867   | 0.830   | 0.736 | 0.815  | 0.880| 0.826 |
| CFM        | TIFS'24    | -       | 0.828   | -     | 0.758  | 0.915| -     |
| **Ours**       | **CVPR'25**      | 🥇 **0.914** | 🥇 **0.900** | 🥇 **0.843** | 🥇 **0.890** | 🥇 **0.933** | 🥇 **0.896** |

## 🎥 Video-Level Comparison

🏆 **Champion Method Alert**: Our approach achieves new SOTA performance across all video-level benchmarks!

| Method             | Venue      | CDF-v2  | DFDC  | DFDCP  |
|--------------------|------------|---------|-------|--------|
| TALL          | ICCV'23    | 0.908   | 0.768 | -      |
| AltFreezing    | CVPR'23    | 0.895   | -     | -      |
| SeeABLE       | ICCV'23    | 0.873   | 0.759 | 0.863  |
| IID           | CVPR'23    | 0.838   | -     | 0.812  |
| TALL++        | IJCV'24    | 0.920   | 0.785 | -      |
| SAM            | CVPR'24    | 0.890   | -     | -      |
| SBI           | CVPR'22    | 0.932   | 0.724 | 0.862  |
| CADDM       | CVPR'23    | 0.939   | 0.739 | -      |
| SFDG         | CVPR'23    | 0.758   | 0.736 | -      |
| LAA-NET       | CVPR'24    | 0.954   | -     | 0.869  |
| LSDA          | CVPR'24    | 0.911   | 0.770 | -      |
| CFM        | TIFS'24    | 0.897   | -     | 0.802  |
| **Ours**           | **CVPR'25**      | 🥇 **0.957** | 🥇 **0.872** | 🥇 **0.929** |

---


## 🚀 Start

- [⏳ Environment Setup](#-environment-setup)
- [📂 Dataset](#-dataset)
- [🏋️ Training](#-training)
- [🧪 Testing](#-testing)
- [📝 Citation](#-citation)


## ⏳ Environment Setup
Ensure your environment meets the following requirements:

- 🐍 Python 3.9
- 🔥 PyTorch 1.11
- 🚀 CUDA 11.3

Install dependencies:

```bash
git clone https://github.com/OUC-VAS/ForensicsAdapter.git
cd ForensicsAdapter
conda create -n FA python=3.9
conda activate FA
sh install.sh
```

## 📂 Dataset

We use multiple datasets for training and evaluation:

- FF++
- DFDC
- DFDCP
- DFD
- CD1/CD2

The dataset downloading and processing procedures can be referred to the implementation provided in [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) .


## 🏋️ Training
Make sure to modify the relevant configurations in the train.yaml file before training.

Start training with the following command:

```bash
python train.py 
```

## 🧪 Testing
Make sure to modify the relevant configurations in the test.yaml file before testing.

Start testing with the following command:


```bash
python /data/cuixinjie/FA/test.py 
```
## 📝 Citation

If our work is useful for your research, please cite it as follows:

```bibtex
@InProceedings{Cui_2025_CVPR,
    author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
    title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {19207-19217}
}

