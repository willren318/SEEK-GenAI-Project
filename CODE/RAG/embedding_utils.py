from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from transformers import AutoTokenizer
import re

# https://huggingface.co/intfloat/e5-large-v2   
def get_embedding_model(model_name='intfloat/e5-large-v2'):
    """Load a sentence transformer embedding model"""
    model = SentenceTransformer(model_name)
    # Check and report the model's maximum sequence length
    print(f"Original model max sequence length: {model.max_seq_length}")
    # Note: E5 models use fixed position embeddings that cannot be safely extended
    # without retraining. We'll use the model's default sequence length.
    return model

def generate_embeddings(texts, model, batch_size=32):
    """Generate embeddings for a list of texts"""
    return model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=True  # Normalize for cosine similarity
    )

def generate_embeddings_with_chunking(texts, model, max_tokens=512, chunk_overlap=50, batch_size=32):
    """Generate embeddings for texts by chunking long documents and averaging chunk embeddings
    
    Args:
        texts: List of text documents
        model: SentenceTransformer model
        max_tokens: Maximum tokens per chunk (default: 512 - E5 model limit)
        chunk_overlap: Number of overlapping tokens between chunks (default: 50)
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings, one per document
    """
    # Use e5-large-v2 tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    all_embeddings = []
    
    # Pre-process all texts to ensure none exceed token limits
    processed_texts = []
    doc_lengths = []  # Track which documents needed chunking
    
    print("Pre-checking document lengths...")
    for i, text in enumerate(texts):
        # Count tokens
        token_count = len(tokenizer.encode(text))
        
        if token_count <= max_tokens:
            # Document is short enough, no chunking needed
            processed_texts.append(text)
            doc_lengths.append(1)  # Mark as single document
        else:
            # Document needs chunking
            chunks = split_text_to_chunks(text, max_tokens, chunk_overlap, tokenizer)
            processed_texts.extend(chunks)
            doc_lengths.append(len(chunks))  # Store number of chunks
            print(f"Document {i}: Split into {len(chunks)} chunks ({token_count} tokens)")
    
    # Encode all processed texts in one batch operation
    print(f"Encoding {len(processed_texts)} text segments...")
    all_segment_embeddings = model.encode(
        processed_texts,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=True
    )
    
    # Recombine chunked embeddings
    result_idx = 0
    for num_chunks in tqdm(doc_lengths, desc="Assembling document embeddings"):
        if num_chunks == 1:
            # Single document, just use the embedding
            all_embeddings.append(all_segment_embeddings[result_idx])
            result_idx += 1
        else:
            # Multiple chunks, average them
            chunk_embeddings = all_segment_embeddings[result_idx:result_idx+num_chunks]
            avg_embedding = np.mean(chunk_embeddings, axis=0)
            # Re-normalize
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            all_embeddings.append(avg_embedding)
            result_idx += num_chunks
    
    return np.array(all_embeddings)

def split_text_to_chunks(text, max_tokens=512, overlap=50, tokenizer=None):
    """Split text into chunks of approximately max_tokens with overlap
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of text chunks
    """
    if tokenizer is None:
        # Create a default tokenizer if none provided
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    
    # Add a safety margin to avoid exceeding limits
    effective_max_tokens = max_tokens - 10
    
    # Simple approach: split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Get token count for this sentence
        sentence_tokens = len(tokenizer.encode(sentence))
        
        if sentence_tokens > effective_max_tokens:
            # Sentence itself is too long, needs its own special handling
            # Split by whitespace and add words until reaching max_tokens
            words = sentence.split()
            word_chunk = []
            word_chunk_length = 0
            
            for word in words:
                word_tokens = len(tokenizer.encode(word))
                if word_chunk_length + word_tokens <= effective_max_tokens:
                    word_chunk.append(word)
                    word_chunk_length += word_tokens
                else:
                    # Save current word chunk and start a new one
                    chunks.append(' '.join(word_chunk))
                    word_chunk = [word]
                    word_chunk_length = word_tokens
            
            # Add any remaining words
            if word_chunk:
                chunks.append(' '.join(word_chunk))
                
            # Skip to next sentence
            continue
        
        # Check if adding this sentence exceeds our chunk token limit
        if current_length + sentence_tokens > effective_max_tokens:
            # Save current chunk and start a new one
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_tokens = 0
            overlap_chunk = []
            
            # Add sentences from the end of the previous chunk for overlap
            for prev_sent in reversed(current_chunk):
                prev_tokens = len(tokenizer.encode(prev_sent))
                if overlap_tokens + prev_tokens <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_tokens += prev_tokens
                else:
                    break
            
            current_chunk = overlap_chunk + [sentence]
            current_length = overlap_tokens + sentence_tokens
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Verify chunk lengths and truncate if necessary
    verified_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        if len(tokens) > max_tokens:
            # Truncate chunk to fit within max_tokens
            truncated_tokens = tokens[:max_tokens-1] + [tokens[-1]]  # Keep the EOS token
            verified_chunks.append(tokenizer.decode(truncated_tokens))
            print(f"Warning: Truncated chunk from {len(tokens)} to {len(truncated_tokens)} tokens")
        else:
            verified_chunks.append(chunk)
    
    return verified_chunks

def embed_dataset(df, model, text_column='combined_text'):
    """Generate embeddings for an entire dataset"""
    texts = df[text_column].tolist()
    # Use chunking for long documents
    return generate_embeddings_with_chunking(texts, model)

def save_embeddings(embeddings, file_path):
    """Save embeddings to a file"""
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """Load embeddings from a file"""
    return np.load(file_path)
