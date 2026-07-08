# FaceMesh2HPO

FaceMesh2HPO is a research codebase for hierarchical classification of facial phenotypic descriptors aligned with the Human Phenotype Ontology (HPO) from 3D face meshes derived from 2D facial photographs. The repository accompanies the manuscript *Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO)*.

## Overview

The project introduces a phenotype-centered pipeline that predicts facial HPO terms instead of directly predicting syndromes. The approach uses a cascaded model tree, where each node represents one HPO term and passes a reduced point mask to its descendants based on feature importance.

Core characteristics of the method include:

- 3D face meshes with 478 landmarks extracted from 2D facial images.
- A hierarchical HPO model tree with one classifier per phenotype term.
- Cascading feature elimination based on Integrated Gradients.
- Optional demographic metadata such as age, sex, and ethnicity.
- PointNet-based models adapted to variable numbers of mesh points.
- Evaluation with 5-fold stratified cross-validation and external validation.

## Installation

```bash
git clone https://github.com/hcmlab/FaceMesh2HPO.git
cd FaceMesh2HPO
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

A typical end-to-end workflow may include:

```bash
python main.py ablation <params>
python main.py train <params>
python main.py export_onnx <params>
```

Training details:

- Up to 25 epochs per model.
- Learning rate of 0.0001.
- Learning-rate reduction on plateau with patience 5.
- Early stopping with patience 5.
- Training seed 42.
- Minimum sample size of 50 for an HPO term model.
- Minimum of 2 input points required for model training.

## Evaluation

The manuscript reports model assessment using:

- AUROC as the primary performance metric.
- Matthews Correlation Coefficient (MCC) for selecting the best model for inference.
- F1-score, precision, and recall.
- Prevalence and detection prevalence.
- Independent multi-expert external validation across seen and unseen disorders.

## Citation

If you use this codebase in academic work, cite the accompanying manuscript.

```bibtex
@misc{hellmann_facemesh2hpo_2026,
  doi = {10.48550/ARXIV.2607.05585},
  url = {https://arxiv.org/abs/2607.05585},
  author = {Hellmann,  Fabio and Hustinx,  Alexander and Solomon,  Benjamin D. and Consortium,  GestaltMatcher Database and Hsieh,  Tzung-Chien and Krawitz,  Peter and André,  Elisabeth},
  keywords = {Computer Vision and Pattern Recognition (cs.CV),  Artificial Intelligence (cs.AI),  Machine Learning (cs.LG),  FOS: Computer and information sciences},
  title = {Hierarchical Classification via Cascading Feature Elimination: Application to Human Phenotype Ontology-Aligned Facial Phenotyping (FaceMesh2HPO)},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
