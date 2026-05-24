# CaN: Core-aware Neural Attributed Hypergraph Generation

This repository provides a refactored PyTorch implementation of CaN for attributed hypergraph generation. The code follows the original experimental pipeline while organizing the implementation into reusable modules.

## Repository Structure

```text
CaN/
├── can/
│   ├── cli.py
│   ├── data.py
│   ├── generation.py
│   ├── hypergraph.py
│   ├── models.py
│   ├── training.py
│   └── utils.py
├── evaluation.py
├── run_can.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.x are recommended. A CUDA-enabled GPU is recommended for training.

## Data Format

The datasets need to be obtained from their original public sources.

### Node attributes

The attribute file should contain one node per line. The line index is treated as the node ID. Do not include an additional node-ID column. The file should be comma-separated, as shown below.

```text
1,0,0,1
0,1,0,1
1,1,0,0
```


### Hyperedges

The hyperedge file should contain one hyperedge per line. Each line lists only the member node IDs of this hyperedge. Do not include an additional hyperedge-ID column. The file should be comma-separated, as shown below.

```text
0,1
2,3,4
5,6
7,8,9,10
```


## Training

```bash
python run_can.py \
  --mode train \
  --attr_path data/attribute.txt \
  --edge_path data/hyperedge.txt \
  --out_edge_path out/generated_edges.txt \
  --epoch_node 1000 \
  --epoch_member 1000 \
  --member_steps_per_epoch 20 \
  --model_path out/can_model.pt
```

The refactored training flow keeps the original logic: it first trains the structural feature allocator and dynamic member assignment model, then trains the hyperedge structural predictor, saves a checkpoint, and generates one verification sample.

## Generation

```bash
python run_can.py \
  --mode gen \
  --attr_path data/attribute.txt \
  --model_path out/can_model.pt \
  --out_edge_path out/generated_edges.txt \
  --num_samples 5
```

When `--num_samples` is greater than 1, output files are saved with numeric suffixes.

## Evaluation

The evaluation script reports the structure--attribute consistency metrics used in the paper: T2, T3, T4, HE, HOHE, and NHS.

```bash
python evaluation.py \
  --gt_structure data/hyperedge.txt \
  --gt_attribute data/attribute.txt \
  --gen_structure out/generated_edges.txt
```
