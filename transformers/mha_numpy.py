import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q=X @ W_q
    K=X @ W_k
    V=X @ W_v
    return Q,K,V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k=Q.shape[-1]
    scores=Q @ K.T
    scores=scores/np.sqrt(d_k)
    # weights=np.exp(scores)/np.sum(np.exp(scores),axis=-1,keepdims=True) # horizontal,feature-wise
    # some values are so large that they cause overflow (become inf), making the softmax calculation produce nan values
    max_scores=np.max(scores,axis=-1,keepdims=True)
    shifted_scores=scores-max_scores
    weights=np.exp(shifted_scores)/np.sum(np.exp(shifted_scores),axis=-1,keepdims=True) # horizontal,feature-wise
    A=weights @ V
    return A

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len,d_model=Q.shape
    # d_k=d_model//n_heads
    if d_model%n_heads!=0:
        raise ValueError(f"{d_model}%{n_heads}!=0")
    Q_mha=Q.reshape(seq_len,n_heads,-1).transpose(1,0,2)
    K_mha=K.reshape(seq_len,n_heads,-1).transpose(1,0,2)
    V_mha=V.reshape(seq_len,n_heads,-1).transpose(1,0,2)
    A_sa=[]
    for i in range(n_heads):
        A_sa.append(self_attention(Q_mha[i],K_mha[i],V_mha[i]))
    A_mha=np.concatenate(A_sa,axis=-1) # horizontal,feature-wise
    return A_mha

# Test Case 1
np.random.seed(42)
X = np.random.permutation(np.arange(16)).reshape(4, 4)
W_q = np.random.randint(0, 4, size=(4, 4))
W_k = np.random.randint(0, 5, size=(4, 4))
W_v = np.random.randint(0, 6, size=(4, 4))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
result = multi_head_attention(Q, K, V, n_heads=2)
print(np.round(result).astype(int).tolist())

# Expected Output:
# [[103, 109, 46, 99], [103, 109, 46, 99], [103, 109, 46, 99], [103, 109, 46, 99]]



# Test Case 2
np.random.seed(42)
X = np.random.permutation(np.arange(48)).reshape(6, 8)
W_q = np.random.randint(0, 4, size=(8, 8))
W_k = np.random.randint(0, 5, size=(8, 8))
W_v = np.random.randint(0, 6, size=(8, 8))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
result = multi_head_attention(Q, K, V, n_heads=4)
print(np.round(result[0]).astype(int).tolist())

# Expected Output:
# [500, 463, 399, 495, 377, 450, 531, 362]