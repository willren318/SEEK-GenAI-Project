import faiss
import numpy as np

def create_index(embeddings):
    """Create a FAISS index from embeddings"""
    # Get embedding dimension
    d = embeddings.shape[1]
    
    # Create index using L2 distance
    index = faiss.IndexFlatL2(d)
    
    # Add vectors to the index
    index.add(np.array(embeddings).astype('float32'))
    
    return index

def save_index(index, file_path):
    """Save FAISS index to disk"""
    faiss.write_index(index, file_path)

def load_index(file_path):
    """Load FAISS index from disk"""
    return faiss.read_index(file_path)

def build_and_save_index(embeddings, index_path):
    """Build FAISS index and save it"""
    index = create_index(embeddings)
    save_index(index, index_path)
    return index
