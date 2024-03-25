# LLMs in the Biomedical Domain

This repository supports the research paper "LLMs in the Biomedical Domain" and contains Python scripts specifically designed for fine-tuning, probing, and comparing attentions of various pre-trained language models in the biomedical sector. These scripts are essential for researchers looking to understand the intricacies of language model behavior within this specialized domain.

## Getting Started

Before executing the scripts, it's crucial to navigate to the correct directory based on the task you wish to perform:

- For Natural Language Inference (NLI) tasks, enter the `NLI` directory.
- For Named Entity Recognition (NER) tasks, move to the `NER` directory.

Each directory contains scripts tailored for these tasks, using the following pre-trained models:

- **BERT**: The original BERT model, known for its effectiveness in a wide range of NLP tasks.
- **BioBERT**: A domain-specific version of BERT pre-trained on biomedical literature, optimized for biomedical NLP tasks.
- **GPT-2 Medium**: A medium-sized variant of the GPT-2 model, offering a balance between performance and computational efficiency.
- **BioGPT**: Similar to BioBERT, BioGPT is fine-tuned for the biomedical domain, providing insights into GPT-style generative models' capabilities in this field.

Below, you'll find instructions on how to execute each script via the command line, including the necessary arguments for fine-tuning, probing, and attention comparison tasks. Ensure you are in the correct task-specific directory (`NLI` or `NER`) before proceeding with the script execution.

### 1. Fine-Tuning (`fine-tuning.py`)

This script fine-tunes a selected model on a specified dataset.

#### Usage:

```bash
python fine-tuning.py --training_size <PERCENTAGE> --model_name <MODEL>
Arguments:

--training_size: Size of the training set as a percentage. Valid choices are 0, 10, 30, 50, and 100.
--model_name: Name of the model to fine-tune. Valid choices are bert, biobert, gpt2, and biogpt.

```

### 2. Probing Task (`probing.py`)

Executes a probing task to analyze the encoding capabilities of pre-trained or fine-tuned models

**Usage:**

```bash
python probing.py --training_size <PERCENTAGE> --model_name <MODEL>
--training_size: Specifies the size of the training set as a percentage. This allows for experimentation with varying amounts of training data to see how it affects model performance. Valid choices are 0, 10, 30, 50, and 100.
--model_name: Determines the model to be used for the probing task. This script supports a range of models tailored for biomedical tasks, including bert, biobert, gpt2, and biogpt, allowing for comparative analysis across different architectures and training regimes.

```
### 3. Attention Comparison (`attention.py`)

This script compares the attentions between pre-trained and fine-tuned models, providing insights into how attention mechanisms vary as a result of fine-tuning on specific datasets.

**Usage:**

```bash
python attention.py --training_sizes <PERCENTAGES> --model_name <MODEL>
--training_sizes: Sizes of the training sets as percentages, separated by commas (e.g., 10,30). Please ensure no spaces are included between commas and numbers.
--model_name: Name of the model to compare attentions. Valid choices are bert, biobert, gpt2, and biogpt.

