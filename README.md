# How to Execute the Code

This repository contains Python scripts designed for fine-tuning, probing, and comparing model attentions of various pre-trained models. Below, you'll find instructions on how to execute each script via the command line, including the necessary arguments.

## Prerequisites

Before running the scripts, ensure you have Python installed on your system. It's recommended to use a virtual environment for Python projects to manage dependencies effectively.

## Scripts

### 1. Fine-Tuning (`fine-tuning.py`)

This script fine-tunes a selected model on a specified dataset.

**Usage:**

```bash
python fine-tuning.py --training_size <PERCENTAGE> --model_name <MODEL>

