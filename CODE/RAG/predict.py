#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import time
import json
from tqdm import tqdm

# Import custom modules
import data_utils
import embedding_utils
import index_utils
import search_utils
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Predict seniority levels using RAG approach')
    parser.add_argument('--train-data', type=str, required=False,
                        default='../../MISC/split_seniority-dev-data_into_train_validation/train_set.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test/validation data CSV')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load models and indices')
    parser.add_argument('--embedding-model', type=str, default='intfloat/e5-large-v2',
                        help='Name or path of embedding model')
    parser.add_argument('--reranker-model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                        help='Name or path of the reranker model')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of initial neighbors to retrieve')
    parser.add_argument('--final-k', type=int, default=3,
                        help='Number of neighbors to use after reranking')
    parser.add_argument('--no-rerank', action='store_true',
                        help='Skip reranking step')
    parser.add_argument('--rebuild-index', action='store_true',
                        help='Force rebuild of FAISS index even if it exists')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index of test data to process (default: 0)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Ending index of test data to process (default: None, process all rows)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output and model directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define paths for saved models
    index_path = os.path.join(args.model_dir, 'faiss_index.bin')
    train_embeds_path = os.path.join(args.model_dir, 'train_embeddings.npy')
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = data_utils.load_dataset(args.test_data)
    print(f"Loaded {len(test_df)} test examples")
    
    # Apply data row filtering if specified
    original_test_count = len(test_df)
    if args.start_index > 0 or args.end_index is not None:
        test_df = test_df.iloc[args.start_index:args.end_index]
        print(f"Processing subset of test data: rows {args.start_index} to {args.end_index if args.end_index else 'end'}")
        print(f"Selected {len(test_df)} out of {original_test_count} examples")
    
    # Preprocess test data
    test_df = data_utils.prepare_dataset(test_df)
    
    # Load embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = embedding_utils.get_embedding_model(args.embedding_model)
    
    # Load training data and build index if needed
    train_df, index = None, None
    
    # Check if index exists and rebuild flag is not set
    if os.path.exists(index_path) and os.path.exists(train_embeds_path) and not args.rebuild_index:
        print("Loading existing FAISS index and embeddings")
        index = index_utils.load_index(index_path)
        train_embeddings = embedding_utils.load_embeddings(train_embeds_path)
        train_df = data_utils.load_dataset(args.train_data)
        train_df = data_utils.prepare_dataset(train_df)
    else:
        # If one or both files don't exist, or rebuild flag is set, regenerate everything
        should_rebuild_embeddings = not os.path.exists(train_embeds_path) or args.rebuild_index
        should_rebuild_index = not os.path.exists(index_path) or args.rebuild_index

        print(f"Loading training data from {args.train_data}")
        train_df = data_utils.load_dataset(args.train_data)
        train_df = data_utils.prepare_dataset(train_df)
        
        # Load or generate embeddings
        if should_rebuild_embeddings:
            print("Generating embeddings for training data (this may take a while)")
            train_embeddings = embedding_utils.embed_dataset(train_df, embedding_model)
            print(f"Saving embeddings to {train_embeds_path}")
            embedding_utils.save_embeddings(train_embeddings, train_embeds_path)
        else:
            print(f"Loading existing embeddings from {train_embeds_path}")
            train_embeddings = embedding_utils.load_embeddings(train_embeds_path)

        # Load or build index
        if should_rebuild_index:
            print("Building FAISS index")
            index = index_utils.build_and_save_index(train_embeddings, index_path)
        else:
            print(f"Loading existing FAISS index from {index_path}")
            index = index_utils.load_index(index_path)
    
    # Initialize reranker if not skipped
    reranker = None
    if not args.no_rerank:
        print(f"Loading reranker model: {args.reranker_model}")
        reranker = search_utils.Reranker(args.reranker_model)
    
    # Generate embeddings for test data
    print("Generating embeddings for test data")
    test_embeddings = embedding_utils.embed_dataset(test_df, embedding_model)
    
    # Make predictions
    print(f"Making predictions (k={args.k}, final_k={args.final_k})")
    predictions = []
    retrieved_neighbors = []
    all_retrieval_results = []
    all_rerank_results = []
    
    for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df))):
        query_text = row['combined_text']
        query_embedding = test_embeddings[i]
        query_id = row.get('job_id', f'query_{i}')
        query_label = row.get('y_true', 'unknown')
        
        # Retrieve similar documents and get prediction
        job_ids, labels, texts, scores, retrieval_results, rerank_results = search_utils.retrieve_and_rerank(
            query_text, query_embedding, index, train_df, train_embeddings,
            reranker, args.k, args.final_k
        )
        
        # Add query information to results 
        for item in retrieval_results:
            item['query_id'] = query_id
            item['query_label'] = query_label
            item['query_text'] = query_text
            all_retrieval_results.append(item)
            
        for item in rerank_results:
            item['query_id'] = query_id
            item['query_label'] = query_label
            item['query_text'] = query_text
            all_rerank_results.append(item)
        
        prediction = search_utils.predict_by_majority_vote(labels)
        predictions.append(prediction)
        retrieved_neighbors.append(job_ids)
    
    # Add predictions to test dataframe
    test_df['y_pred'] = predictions
    
    # Save detailed retrieval and reranking results
    retrieval_df = pd.DataFrame(all_retrieval_results)
    rerank_df = pd.DataFrame(all_rerank_results)
    
    retrieval_path = os.path.join(args.output_dir, 'retrieval_details.csv')
    rerank_path = os.path.join(args.output_dir, 'rerank_details.csv')
    
    print(f"Saving detailed retrieval results to {retrieval_path}")
    retrieval_df.to_csv(retrieval_path, index=False)
    
    print(f"Saving detailed reranking results to {rerank_path}")
    rerank_df.to_csv(rerank_path, index=False)
    
    # Calculate metrics
    metrics = evaluate.calculate_metrics(test_df['y_true'], test_df['y_pred'])
    
    # Print and visualize results
    evaluate.print_metrics(metrics, report_path=os.path.join(args.output_dir, 'classification_report.txt'))
    
    # Save confusion matrix plot
    plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    evaluate.plot_confusion_matrix(metrics, save_path=plot_path)
    
    # Save predictions to file
    output_path = os.path.join(args.output_dir, 'predictions.csv')
    evaluate.save_predictions(test_df, output_path)
    
    print("Prediction complete!")

if __name__ == "__main__":
    main()
