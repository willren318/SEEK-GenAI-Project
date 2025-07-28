# COMP6713-SEEK-Project

A comprehensive machine learning project comparing **open models**, **proprietary models**, and **fine-tuned models** for job advertisement analysis tasks, evaluating performance across **accuracy**, **latency**, and **cost** dimensions.

## Project Overview

This project implements and evaluates multiple approaches to solve three core job advertisement analysis tasks:

1. **Work Arrangement Classification (WA)**: Categorize jobs as Remote, Hybrid, or OnSite
2. **Salary Extraction (SA)**: Extract structured salary information from job postings  
3. **Seniority Classification (SE)**: Classify jobs into 6 seniority levels (Internship/Trainee, Entry-Level/Junior, Mid-Level Professional, Senior Individual Contributor, Manager/Supervisor, Executive/Director)

## Model Comparison Framework

### 🤖 Model Types Evaluated

#### Proprietary Models
- **Claude**: 3 Haiku, 3.7 Sonnet
- **DeepSeek**: Chat, Reasoner (with cache optimization)

#### Open Models  
- **Llama**: 3.1-8B, 3.1-70B, 4-Maverick
- **Together AI**: Llama variants via Together API

#### Fine-tuned Models
- **Mistral 7B + LoRA**: Custom adapters for each task (SA, SE, WA)
- **Phi-3 Mini + LoRA**: 4-bit quantized with LoRA fine-tuning

#### Additional Approaches
- **RAG (Retrieval Augmented Generation)**: FAISS-based similarity search with k-NN voting
- **Rule-based Models**: Baseline implementations for comparison

### 📊 Evaluation Metrics

- **Accuracy**: Primary classification performance metric
- **Latency**: Response time measurement for speed comparison
- **Cost**: Token usage and API costs for economic analysis
- **Additional**: F1-score, precision, recall, confusion matrices

## Project Structure

```
CODE/
├── llm_evaluation/          # Proprietary & open model evaluation
│   ├── models/             # Model-specific evaluators (Claude, DeepSeek, Llama)
│   ├── tasks/              # Task implementations (WA, SA, SE)
│   ├── prompts/            # Optimized prompt templates
│   └── run_experiments.py  # Main evaluation runner
├── fine_tuning_evaluation/  # Mistral 7B fine-tuning
│   ├── mistral_task_*.ipynb # Task-specific training notebooks
│   └── mistral_*.py        # Model and data handling
├── seniority_prediction_v2/ # Phi-3 Mini fine-tuning  
│   ├── models/             # Phi-3 with LoRA configuration
│   ├── training/           # Training pipeline with 4-bit quantization
│   └── experiments/        # Template selection and optimization
├── RAG/                    # Retrieval Augmented Generation
│   ├── embedding_utils.py  # Sentence transformer embeddings
│   ├── index_utils.py      # FAISS index management
│   └── predict.py          # RAG-based prediction pipeline
├── mistral-7b-lora-*/      # Trained LoRA adapters
├── EDA/                    # Exploratory data analysis
└── utils/                  # Shared utilities and mappings
```

## Quick Start

### 1. LLM Evaluation (Proprietary & Open Models)

```bash
cd CODE/llm_evaluation

# Claude evaluation
export ANTHROPIC_API_KEY=your_key_here
python run_experiments.py --model claude --claude-variant claude-3-haiku --task work_arrangement --limit 50

# DeepSeek evaluation  
export DEEPSEEK_API_KEY=your_key_here
python run_experiments.py --model deepseek --task salary --limit 100

# Llama evaluation
python run_experiments.py --model llama --task seniority --limit 200
```

### 2. Fine-tuned Model Evaluation

```bash
# Mistral 7B LoRA evaluation
cd CODE/fine_tuning_evaluation
jupyter notebook mistral_task_sa.ipynb  # For salary task

# Phi-3 Mini LoRA training
cd CODE/seniority_prediction_v2
python train_phi3_structured.py
```

### 3. RAG-based Prediction

```bash
cd CODE/RAG
python predict.py --test-data path/to/test_data.csv --k 10 --final-k 3
```

### 4. View Results in MLflow

```bash
cd CODE/llm_evaluation
mlflow ui
# Open http://localhost:5000 for experiment tracking and comparison
```

## Key Features

### 🚀 Performance Optimizations
- **4-bit Quantization**: Efficient fine-tuning on consumer hardware
- **LoRA Adapters**: Parameter-efficient fine-tuning
- **Mixed Precision Training**: Faster training with FP16
- **FAISS Indexing**: Fast similarity search for RAG

### 📈 Comprehensive Evaluation
- **MLflow Integration**: Experiment tracking and visualization
- **Cost Analysis**: Token usage and API cost calculation
- **Latency Measurement**: Response time profiling
- **Error Analysis**: Confusion matrices and per-class metrics

### 🎯 Task-Specific Optimizations
- **Prompt Engineering**: Optimized templates for each task
- **Data Preprocessing**: HTML cleaning and text normalization
- **Cache Optimization**: DeepSeek cache hit/miss tracking
- **Template Selection**: Automated prompt template evaluation

## Results Summary

The project provides comprehensive comparisons across:
- **Model Performance**: Accuracy benchmarks across all approaches
- **Economic Efficiency**: Cost per prediction analysis  
- **Speed Analysis**: Latency comparisons for real-time applications
- **Resource Usage**: Memory and computational requirements

Detailed results are available in the MLflow tracking system and individual component README files.

## Installation

```bash
# Core dependencies
pip install -r CODE/llm_evaluation/requirements.txt
pip install -r CODE/RAG/requirements.txt  
pip install -r CODE/seniority_prediction_v2/requirements.txt

# For fine-tuning (additional GPU dependencies)
pip install torch transformers accelerate peft bitsandbytes
```

## Contributing

This project is part of COMP6713 coursework. See `INDIVIDUAL EFFORT.md` for contribution details and `REPORT.md` for comprehensive analysis and findings.
