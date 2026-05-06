---
title: Sugarcane Disease Detector
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# 🌾 Sugarcane Disease Detection — MoE v4

Upload a sugarcane leaf photo to detect common diseases affecting sugarcane crops. This application uses a powerful **Mixture of Experts (MoE)** model architecture to accurately classify leaf conditions into five categories: **Mosaic, Rust, RedRot, YellowLeaf, or Healthy**.

## 🚀 Live Demo

The application is deployed and available to use on Hugging Face Spaces:
👉 **[Sugarcane Disease Detector App](https://huggingface.co/spaces/vishnu0107/sugarcane-disease-detector)**

## 🧠 Model Checkpoints

The pre-trained Mixture of Experts (MoE) models can be found on the Hugging Face Hub:
👉 **[vishnu0107/sugarcane_moe](https://huggingface.co/vishnu0107/sugarcane_moe)**

## 💻 Running the Application Locally

To run this Gradio application on your local machine, follow these steps:

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need `git` and `git-lfs` to clone the repository and download the model weights.

### 1. Clone the Repository

```bash
git clone https://huggingface.co/spaces/vishnu0107/sugarcane-disease-detector
cd sugarcane-disease-detector
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(Note: Ensure `gradio` and the appropriate deep learning framework (e.g., `torch`, `transformers`) used by the model are included in your `requirements.txt`.)*

### 3. Run the App

```bash
python app.py
```

The application will start, and you can access the interface by opening the local URL provided in your terminal (typically `http://localhost:7860`).