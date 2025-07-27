# RAG-based Seniority Classification

This folder implements a Retrieval-Augmented Generation (RAG) approach to job seniority classification, using a combination of:
1. Dense vector embeddings of job advertisements
2. FAISS for efficient similarity search
3. Cross-encoder reranking
4. K-nearest neighbor voting for final prediction

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

The requirements.txt file include:
- faiss-cpu (or faiss-gpu for GPU support)
- sentence-transformers
- transformers
- torch
- pandas
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Usage

### Basic Usage

To predict seniority levels on a test/validation dataset:

```bash
python predict.py --test-data path/to/test_data.csv
```

This will:
1. Load the pre-built index and embeddings of the training set
2. Embed the test data
3. Find similar job ads from the training set
4. Make predictions using majority voting
5. Output metrics and save results to the 'results' directory

### Advanced Options

```bash
python predict.py --test-data path/to/test_data.csv \
                 --train-data path/to/custom_train_data.csv \
                 --embedding-model all-mpnet-base-v2 \
                 --rebuild-index \
                 --k 15 \
                 --final-k 5 \
                 --output-dir custom_results
```

### Arguments

- `--train-data`: Path to training data CSV (default: '../MISC/split_seniority-dev-data_into_train_validatoin/train_set.csv')
- `--test-data`: Path to test/validation data (required)
- `--output-dir`: Directory to save results (default: 'results')
- `--model-dir`: Directory to save/load models and indices (default: 'models')
- `--embedding-model`: Name or path of embedding model (default: 'BAAI/bge-large-en-v1.5')
- `--reranker-model`: Name or path of reranker model (default: 'BAAI/bge-reranker-large')
- `--k`: Number of initial neighbors to retrieve (default: 10)
- `--final-k`: Number of neighbors after reranking (default: 3)
- `--no-rerank`: Skip reranking step (use top-k from initial retrieval)
- `--rebuild-index`: Force rebuild FAISS index even if it exists

## Methodology

1. **Text Preparation**: Combine job title, summary, classification, and details into a single text, with title emphasized
2. **Embedding**: Convert text to dense vectors using a pre-trained sentence transformer
3. **Indexing**: Store training set embeddings in a FAISS index for efficient retrieval
4. **Retrieval**: For each test example, find k most similar jobs from training set
5. **Reranking**: Use a cross-encoder to rerank the initial results for higher precision
6. **Prediction**: Use majority voting on the labels of the top-k similar examples

## Files

- `data_utils.py`: Data loading and preprocessing
- `embedding_utils.py`: Embedding model handling and vector generation
- `index_utils.py`: FAISS index creation and management
- `search_utils.py`: Retrieval and reranking functionality
- `evaluate.py`: Metrics calculation and visualization
- `predict.py`: Main script for making predictions

## Strengths

1. Learning from examples in the training data
2. Leveraging semantic similarity rather than strict keyword matching
3. Using multiple similar examples (k-NN) to make more robust predictions
