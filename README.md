# TumorTrack   
**Automated MRI-Based Tumor Detection & Explainability**

[![Docker Image Size](https://img.shields.io/docker/image-size/jayesh422x/tumortrack)](https://hub.docker.com/r/jayesh422x/tumortrack)
[![License](https://img.shields.io/github/license/jayeshgit65/TumorTrack-Automated-MRI-Based-Tumor-Detection-and-Explainability-System)](LICENSE)

---

## Project Overview

TumorTrack is a complete pipeline for detecting brain tumors from MRI scans using deep learning. It features:
- **Multi-class classification** (glioma, meningioma, pituitary, no-tumor) with state-of-the-art accuracy (~86%)
- **Explainable AI** visualizations (Grad‑CAM) for clinician-friendly interpretability
- **Streamlit-powered UI** to upload MRI scans and display predictions + heatmaps
- **Robust CI/CD** with GitHub Actions to lint, test, build Docker images, and run security scans (Trivy)

---

## Features

| Feature | Description |
|--------|-------------|
| **Deep Learning** | Utilizes pre-trained CNN (VGG) fine-tuned on MRI datasets |
| **Explainability** | Generates Grad‑CAM heatmaps to highlight tumor regions |
| **Dockerized Deployment** | Run locally or in the cloud with a single `docker run` command |
| **Automated CI/CD** | Ensures code quality, security, and reproducible builds |

---

## Quick Start

### Prerequisites
- Docker installed
- Optional: Python 3.11

### Local Docker Run
docker pull jayesh422x/tumortrack:latest
docker run -p 8501:8501 jayesh422x/tumortrack:latest

---

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
