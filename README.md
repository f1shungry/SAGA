# SAGA: Structural Aggregation Guided Alignment with Dynamic View and Neighborhood Order Selection for Multiview Graph Domain Adaptation

**[ICLR 2026]** Official PyTorch implementation of **SAGA**.

**SAGA ** is a novel framework for **Multi-view Graph Domain Adaptation (MGDA)**. Unlike traditional graph domain adaptation methods that assume single-view graph structures, SAGA effectively handles multi-relational graphs by dynamically aligning structural information across both views and neighborhood hops.

The framework is illustrated as follows：

![](.\framework.jpg)

## Dataset

Our framework supports two types of graph datasets: **ACM** and **MAG**. Both are **citation networks** commonly used for graph domain adaptation tasks.

### Dataset Types

#### **ACM Dataset**
- A citation network dataset divided into two subsets: **ACM1** and **ACM2**
- Contains **3 label categories**
- Each subset has **two views**:
  - **PAP (Paper–Author–Paper)**: Meta-paths connecting papers through shared authors
  - **PSP (Paper–Subject–Paper)**: Meta-paths connecting papers through shared subjects
- These different views provide complementary structural information for domain adaptation

#### **MAG Dataset**
- Derived from the Microsoft Academic Graph, a large-scale citation network
- Divided into **six country-specific subgraphs**:
  - **CN** (China)
  - **US** (United States)
  - **JP** (Japan)
  - **DE** (Germany)
  - **FR** (France)
  - **RU** (Russia)
- Contains **10 label categories** 
- Each subgraph has **two views**:
  - **PAP (Paper–Author–Paper)**: Meta-paths connecting papers through shared authors
  - **PP (Paper–Paper)**: Direct paper-paper citation relationships

### Source-Target Pairs

In domain adaptation, we designate one dataset as the **source** (rich labels) and another as the **target** (limited/unlabeled data). 

#### **Important Notes:**
- ✅ **Within-domain adaptation**: ACM1 ↔ ACM2, or between any two MAG subgraphs
- ❌ **Cross-domain adaptation NOT supported**: ACM cannot be paired with MAG datasets due to different feature spaces, label spaces (3 vs 10 classes), and distribution discrepancies

#### **Available Source-Target Pairs:**

| Source → Target | Description                               |
| --------------- | ----------------------------------------- |
| **ACM1-ACM2**   | ACM1 as source, ACM2 as target            |
| **ACM2-ACM1**   | ACM2 as source, ACM1 as target            |
| **CN-US**       | China as source, United States as target  |
| **US-CN**       | United States as source, China as target  |
| **JP-CN**       | Japan as source, China as target          |
| **CN-JP**       | China as source, Japan as target          |
| **DE-FR**       | Germany as source, France as target       |
| **FR-DE**       | France as source, Germany as target       |
| **RU-US**       | Russia as source, United States as target |
| **US-RU**       | United States as source, Russia as target |
| **DE-CN**       | Germany as source, China as target        |
| **CN-DE**       | China as source, Germany as target        |

> **Note**: The naming convention `X-Y` means using **X as the source domain** and **Y as the target domain**.

---

### Dataset Download

The complete dataset used in this work is available for download via Google Drive:

**Download Link**: [SAGA Dataset](https://drive.google.com/file/d/1YcWmDl6byEAwEGAEXX1Fc7KlBEmBZUDu/view?usp=sharing)

After downloading, please ensure the dataset files are placed in the following directory structure:

```
./data/
├── ACM1/
├── ACM2/
├── CN/
├── DE/
├── FR/
├── JP/
├── RU/
├── US/
```

## Environment Dependencies

```
matplotlib==3.10.8
numpy==2.4.2
scikit_learn==1.8.0
scipy==1.17.0
torch==2.8.0+cu128
```

The `requirements.txt` file is provided for your reference. Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Run Code

### Basic Usage

Run the model by specifying the source-target dataset pair using the `-d` or `--dataset` argument:

```bash
python main.py -d DATASET_PAIR
```

Replace `DATASET_PAIR` with one of the following:
- `ACM1-ACM2` or `ACM2-ACM1` (ACM datasets with PAP+PSP views, 3 classes)
- `CN-US`, `US-CN`, `JP-CN`, `CN-JP`, `DE-FR`, `FR-DE`, `RU-US`, `US-RU`, `CN-DE`, `DE-CN` (MAG subgraphs with PAP+PP views, 10 classes)

### Usage Examples

#### **1. Basic usage – only specify dataset (recommended)**
```bash
python main.py -d ACM2-ACM1
```

#### **2. Specify dataset and override default parameters**
```bash
python main.py -d CN-US --batchsize 5000 --nb_epochs 100 --lr 0.0005
```

#### **3. Specify GPU device and random seed**
```bash
python main.py -d DE-FR --gpu 1 --seed 1234
```

#### **4. Full parameter customization**
```bash
python main.py -d ACM1-ACM2 --batchsize 2000 --hidden_dim 512 --embed_dim 256 --nb_epochs 200 --l2_coef 1e-5 --lr 0.0001 --dropout 0.1
```

### Parameter Overview

| Argument         | Type  | Default            | Description                                 |
| ---------------- | ----- | ------------------ | ------------------------------------------- |
| `-d, --dataset`  | str   | **Required**       | Source-target dataset pair (see list above) |
| `--gpu`          | int   | 0                  | GPU device ID                               |
| `--seed`         | int   | 0                  | Random seed for reproducibility             |
| `--batchsize`    | int   | *dataset-specific* | Batch size                                  |
| `--hidden_dim`   | int   | 256                | Hidden layer dimension                      |
| `--embed_dim`    | int   | 128                | Embedding dimension                         |
| `--nb_epochs`    | int   | *dataset-specific* | Number of training epochs                   |
| `--l2_coef`      | float | 1e-4               | L2 regularization coefficient               |
| `--lr`           | float | 0.001              | Learning rate                               |
| `--dropout`      | float | *dataset-specific* | Dropout rate                                |
| `--tau`          | float | 1.0                | Temperature parameter                       |
| `--filter_alpha` | float | 0.0                | Filter alpha parameter                      |

> **Note**: Parameters marked as *dataset-specific* will automatically use optimized default values based on your selected dataset pair. You can still override them manually if needed.

---

