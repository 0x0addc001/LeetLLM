import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, query, key, value, mask=None):
        # Calculate attention scores
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Multiply by values to get context vector
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # input: 
        # (batch_size, seq_len, d_model) 
        # → linear projection → (batch_size, seq_len, d_model)
        # → view → (batch_size, seq_len, num_heads, d_k)
        # → transpose → (batch_size, num_heads, seq_len, d_k)
        
        # Apply mask to all heads if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        # Apply attention
        attn_output, attention_weights = self.attention(query, key, value, mask)
        # (batch_size, num_heads, seq_len, d_k)
        
        # Concatenate heads and project back to d_model
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attention_weights

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-attention layer with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward layer with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        # Masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-head cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention layer with residual connection
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention layer with residual connection
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed forward layer with residual connection
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, max_len=5000):
        super(Encoder, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Add positional encoding and apply dropout
        x = self.dropout(self.pos_encoding(x))
        
        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout, max_len=5000):
        super(Decoder, self).__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Add positional encoding and apply dropout
        x = self.dropout(self.pos_encoding(x))
        
        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder and Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # Output projection layer
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding lookups
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)
        
        # Encode source sequence
        enc_output = self.encoder(src_embedded, src_mask)
        
        # Decode target sequence
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # Project to target vocabulary
        output = self.output_proj(dec_output)
        
        return output

# Helper function to create masks
def create_pad_mask(sequence, pad_idx):
    return (sequence != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
    
    # Create sample input
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 49  # Decoder input is shifted right
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create masks
    src_mask = create_pad_mask(src, pad_idx=0)
    tgt_mask = create_subsequent_mask(tgt_seq_len)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    
    print("Input shapes:")
    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")
    print(f"Source mask: {src_mask.shape}")
    print(f"Target mask: {tgt_mask.shape}")
    print(f"Output shape: {output.shape}")