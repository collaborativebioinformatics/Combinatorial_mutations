# FedMut: Federated Prediction of Combinatorial Mutation Effects

**Predicting Deep Mutational Scanning (DMS) scores using Pre-trained Protein Language Models in a Federated Environment.**

## Problem

Predicting the functional effects of combinatorial mutations is a critical challenge in protein engineering and evolutionary biology. While Deep Mutational Scanning (DMS) provides ground-truth fitness landscapes, the data is often:

1. **Siloed:** Distributed across different institutions (hospitals, academic labs, industry).
2. **Sparse:** The combinatorial space of mutations is too vast to measure experimentally.

This project addresses these challenges by leveraging **Federated Learning (FL)** to train a predictive model across distributed datasets without sharing raw patient or proprietary sequence data.

## Introduction

We aim to predict the DMS score (fitness/stability) of mutant sequences. By utilizing [NVIDIA BioNeMo](https://github.com/NVIDIA/bionemo-framework) pre-trained models as a feature extractor, we benefit from representations learned on massive protein databases while keeping the computational cost of local training low.

## Dataset

We utilize the **[ProteinGym](https://proteingym.org/)** benchmark, a comprehensive collection of deep mutational scanning assays. ProteinGym allows us to evaluate zero-shot and supervised learning performance across diverse protein families.

For the Federated Learning simulation, we partition the ProteinGym data into semantically meaningful groups to simulate real-world data silos:

### Federated Clients Simulation

We simulate 4 distinct clients based on their `taxon`, each representing a specific domain and utilizing a representative dataset from ProteinGym:

| Node | Client Type | Simulation Scenario | Key Dataset (ProteinGym) |
| --- | --- | --- | --- |
| **Client 1** | **Human** | Clinical Hospital / Oncology | `P53_HUMAN` (Tumor suppressor) |
| **Client 2** | **Virus** | Virology Lab / Pandemic Prep | `SPIKE_SARS2` (Viral entry) |
| **Client 3** | **Prokaryote** | Antibiotic Resistance Lab | `BLAT_ECOLX` (Beta-lactamase) |
| **Client 4** | **Eukaryote** | Academic Bio-Foundry | `GAL4_YEAST` (Transcription factor) |

![data](./figures/federated_clients.png)

## Methodology

### Model Structure

We utilize a **Transfer Learning** approach with a frozen backbone and a trainable regression head.

1. **Frozen Backbone (BioNeMo):**
* We use a pre-trained Protein Language Model (e.g., ESM-2 or MegaMolBART via NVIDIA BioNeMo) as the encoder.
* **Input:** Amino acid sequence of the mutant (e.g., `M1A, ...`).
* **Output:** Per-residue or whole-sequence embeddings ().
* *Note:* All weights in this encoder are **frozen** to reduce communication overhead and computational requirements on edge clients.


2. **Trainable Prediction Head:**
* **Pooling Layer:** Aggregates the sequence embedding (using Mean Pooling or the `<CLS>` token representation) into a fixed-size vector.
* **Regression MLP:** A multi-layer perceptron stacked on top of the embeddings.
* Layer 1: Linear () + ReLU + Dropout (0.1)
* Layer 2: Linear ()


* **Output:** A single scalar value representing the predicted DMS score.


<!-- 
### Training Process (Federated)

We use the **FedAvg** (Federated Averaging) algorithm.

1. **Initialization:** The central server initializes the weights of the **Prediction Head** and distributes them to all 4 clients. The BioNeMo backbone is pre-loaded on all clients.
2. **Local Training:**
* Each client trains the Prediction Head on their local data (e.g., Client 1 trains only on `P53`).
* Loss Function: Mean Squared Error (MSE) between predicted and actual DMS scores.
* Optimizer: AdamW.


3. **Aggregation:**
* Clients send *only* the updated weights (gradients) of the Prediction Head back to the server.
* The server averages these weights to create a new global model.


4. **Distribution:** The updated global model is sent back to clients for the next round. -->

## How to use

### Prerequisites

* Python 3.8+
* PyTorch
* NVIDIA BioNeMo Framework (or HuggingFace Transformers for ESM-2)
* Flower (flwr) for Federated Learning orchestration

### Installation

```bash
git clone https://github.com/your-username/Combinatorial_mutations.git
cd Combinatorial_mutations
pip install -r requirements.txt
```

### Running the Simulation
<!-- 
**1. Start the Server:**
The server orchestrates the aggregation of model weights.

```bash
python server.py --rounds 10

```

**2. Start the Clients:**
Open separate terminals for each client to simulate distributed nodes.

```bash
# Terminal 1: Human/Clinical Node
python client.py --client_id 1 --dataset P53_HUMAN

# Terminal 2: Virus Node
python client.py --client_id 2 --dataset SPIKE_SARS2

# Terminal 3: Prokaryote Node
python client.py --client_id 3 --dataset BLAT_ECOLX

# Terminal 4: Eukaryote Node
python client.py --client_id 4 --dataset GAL4_YEAST

``` -->

## Results

* **Metric:** Spearman's Rank Correlation () between predicted and ground-truth DMS scores.
* **Global Performance:** After  rounds of federated training, the global model achieves competitive performance compared to centrally trained baselines, while preserving data privacy.

| Model | Client 1 (Human) | Client 2 (Virus) | Client 3 (Prokaryote) | Client 4 (Eukaryote) | Average  |
| --- | --- | --- | --- | --- | --- |
| Local Training Only | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| **Federated (FedMut)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

## Acknowledgements

* [ProteinGym](https://proteingym.org/) for the benchmarking datasets.
* [NVIDIA BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/) for the foundational protein models.

## Team Members

- Ustha
- Maggie
- Sihyun
- Jiayi
- Bhanvi
- Ramith
- Sumeet
