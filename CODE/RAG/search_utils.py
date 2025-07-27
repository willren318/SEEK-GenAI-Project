import numpy as np
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

# https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
# Popular, high-performance reranker fine-tuned on MS MARCO passage ranking data
# This model consistently ranks in the top performers for passage ranking tasks
# while being much more efficient than larger models

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Select device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        # MiniLM has smaller context window than Longformer
        self.max_length = 512

    def rerank(self, query, documents, batch_size=4, verbose=False):
        """Rerank documents by similarity to the query"""
        # For long queries and documents, we'll use a sliding window approach
        # and take the maximum score across all windows
        tokenizer = self.tokenizer
        max_length = self.max_length - 50  # Leave room for special tokens
        
        # Check if the query is long and needs chunking
        query_tokens = len(tokenizer.encode(query))
        long_query = query_tokens > max_length
        
        if long_query and verbose:
            print(f"Query length: {query_tokens} tokens (exceeds {max_length})")
            print("Using sliding window approach for long query")
        
        # Get query chunks if needed
        query_chunks = [query]
        if long_query:
            query_chunks = self._get_text_chunks(query, max_length)
            if verbose:
                print(f"Split query into {len(query_chunks)} chunks")
        
        scores = []
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            doc_scores = []
            
            # Check if document is long
            doc_tokens = len(tokenizer.encode(doc))
            long_doc = doc_tokens > max_length
            
            if verbose and long_doc:
                print(f"Document {doc_idx}: {doc_tokens} tokens (exceeds {max_length})")
            
            # Get document chunks if needed
            doc_chunks = [doc]
            if long_doc:
                doc_chunks = self._get_text_chunks(doc, max_length)
                if verbose:
                    print(f"Split document {doc_idx} into {len(doc_chunks)} chunks")
            
            # Compare all query chunks with all document chunks
            # and take the maximum score as the document's score
            chunk_scores = []
            
            for q_chunk in query_chunks:
                for d_chunk in doc_chunks:
                    # Create inputs for this specific query-doc chunk pair
                    inputs = tokenizer(
                        [q_chunk], 
                        [d_chunk],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=self.max_length,
                        return_token_type_ids=False
                    ).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                        # Use sigmoid to convert single score to 0-1 range
                        chunk_score = torch.sigmoid(logits[0, 0]).item()
                    
                    chunk_scores.append(chunk_score)
            
            # Take maximum score across all chunk pairs
            doc_score = max(chunk_scores) if chunk_scores else 0.0
            scores.append(doc_score)
            
            if verbose:
                print(f"Document {doc_idx} - Max score: {doc_score:.4f} from {len(chunk_scores)} chunk pairs")
            
            # Process in smaller batches to avoid memory issues
            if (doc_idx + 1) % batch_size == 0:
                if verbose:
                    print(f"Processed {doc_idx + 1}/{len(documents)} documents")
        
        return np.array(scores)

    def _get_text_chunks(self, text, max_tokens=450):
        """Split text into chunks of approximately max_tokens
        
        This is a simpler version for reranking that aims to create
        chunks with maximum semantic coherence.
        """
        # Simple approach: split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Rough token estimate (this is approximate)
            sentence_tokens = len(sentence.split())
            
            # Check if adding this sentence exceeds our chunk token limit
            if current_length + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk and start a new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def search_index(query_embedding, index, k=12):
    """Search the FAISS index for similar vectors"""
    distances, indices = index.search(
        np.array([query_embedding]).astype('float32'), 
        k
    )
    return distances[0], indices[0]


def retrieve_and_rerank(query_text, query_embedding, index, train_df, train_embeddings, reranker=None, k=12, final_k=3):
    """Retrieve similar documents and optionally rerank them"""
    # Initial retrieval
    distances, indices = search_index(query_embedding, index, k)
    
    # Get corresponding job ids and texts
    retrieved_job_ids = train_df.iloc[indices]['job_id'].tolist()
    retrieved_texts = train_df.iloc[indices]['combined_text'].tolist()
    retrieved_labels = train_df.iloc[indices]['y_true'].tolist()
    
    # Convert distances to similarity scores (1 - normalized distance)
    # FAISS uses L2 distance, so smaller is better
    max_dist = max(distances) if distances.size > 0 else 1.0
    similarity_scores = [1.0 - (dist/max_dist) for dist in distances]
    
    # Create initial retrieval results
    initial_results = []
    for i in range(len(retrieved_job_ids)):
        initial_results.append({
            'job_id': retrieved_job_ids[i],
            'label': retrieved_labels[i],
            'similarity_score': similarity_scores[i],
            'distance': distances[i],
            'text': retrieved_texts[i][:100] + "...", # Truncate for display
            'query_text': query_text[:100] + "..."
        })
    
    # If reranker is provided, rerank the results
    rerank_results = []
    if reranker:
        rerank_scores = reranker.rerank(query_text, retrieved_texts)
        rerank_indices = np.argsort(-rerank_scores)[:final_k]  # Sort by highest score
        
        # Reorder based on reranking
        top_indices = [indices[i] for i in rerank_indices]
        top_job_ids = [retrieved_job_ids[i] for i in rerank_indices]
        top_texts = [retrieved_texts[i] for i in rerank_indices]
        scores = [rerank_scores[i] for i in rerank_indices]
        
        # Create reranking results
        for i in range(len(rerank_indices)):
            idx = rerank_indices[i]
            rerank_results.append({
                'job_id': retrieved_job_ids[idx],
                'label': retrieved_labels[idx],
                'original_rank': idx + 1,
                'rerank_score': rerank_scores[idx],
                'similarity_score': similarity_scores[idx],
                'text': retrieved_texts[idx][:100] + "...", # Truncate for display
                'query_text': query_text[:100] + "..."
            })
    else:
        # Just take top-k based on initial retrieval
        top_indices = indices[:final_k]
        top_job_ids = retrieved_job_ids[:final_k]
        top_texts = retrieved_texts[:final_k]
        top_labels = train_df.iloc[top_indices]['y_true'].tolist()
        scores = similarity_scores[:final_k]  # Use similarity scores instead of placeholder
        
        # Create results from initial order when not reranking
        for i in range(final_k):
            rerank_results.append({
                'job_id': retrieved_job_ids[i],
                'label': retrieved_labels[i],
                'original_rank': i + 1,
                'rerank_score': 0.0,  # No reranking
                'similarity_score': similarity_scores[i],
                'text': retrieved_texts[i][:100] + "...", # Truncate for display
                'query_text': query_text[:100] + "..."
            })
    
    # Get labels if not already retrieved
    if 'top_labels' not in locals():
        top_labels = train_df.iloc[top_indices]['y_true'].tolist()

    for i in range(len(top_job_ids)):
        print(f"Job ID: {top_job_ids[i]}")
        print(f"Label: {top_labels[i]}")
        print(f"Similarity: {scores[i]:.4f}")
        print(f"Text: {top_texts[i][:100]}")
        print("--------------------------------")
    
    return top_job_ids, top_labels, top_texts, scores, initial_results, rerank_results

def predict_by_majority_vote(labels):
    """Make a prediction based on majority vote of retrieved documents"""
    label_counts = Counter(labels)
    return label_counts.most_common(1)[0][0]
