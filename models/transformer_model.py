# models/transformer_model.py
import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    """
    Transformer model for music generation.
    Takes pitch, duration, velocity sequences as input and predicts next note.
    """
    def __init__(self, 
                 pitch_vocab=128, 
                 dur_vocab=32, 
                 vel_vocab=128, 
                 d_model=256, 
                 n_heads=4, 
                 n_layers=3, 
                 dim_feedforward=512, 
                 dropout=0.1,
                 max_seq_len=512):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.pitch_vocab = pitch_vocab
        self.dur_vocab = dur_vocab
        self.vel_vocab = vel_vocab
        self.max_seq_len = max_seq_len

        # Embeddings
        self.pitch_embed = nn.Embedding(pitch_vocab, d_model)
        self.dur_embed   = nn.Embedding(dur_vocab, d_model)
        self.vel_embed   = nn.Embedding(vel_vocab, d_model)

        # Positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.pitch_out = nn.Linear(d_model, pitch_vocab)
        self.dur_out   = nn.Linear(d_model, dur_vocab)
        self.vel_out   = nn.Linear(d_model, vel_vocab)

    def forward(self, x_pitch, x_dur, x_vel):
        """
        Forward pass
        x_pitch, x_dur, x_vel: (batch_size, seq_len)
        Returns: logits for pitch, duration, velocity
        """
        seq_len = x_pitch.size(1)
        positions = torch.arange(seq_len, device=x_pitch.device).unsqueeze(0)

        pitch_emb = self.pitch_embed(x_pitch)
        dur_emb   = self.dur_embed(x_dur)
        vel_emb   = self.vel_embed(x_vel)
        pos_emb   = self.pos_embed(positions)

        x = pitch_emb + dur_emb + vel_emb + pos_emb
        x = x.permute(1,0,2)  # seq_len, batch_size, d_model
        x = self.transformer(x)
        x = x.permute(1,0,2)  # batch_size, seq_len, d_model

        pitch_logits = self.pitch_out(x)
        dur_logits   = self.dur_out(x)
        vel_logits   = self.vel_out(x)

        return pitch_logits, dur_logits, vel_logits

# -------------------------
# Sampling helper
# -------------------------
def sample_from_logits(logits, temperature=1.0, top_k=None):
    """
    Sample from logits using temperature scaling and optional top_k filtering.
    """
    logits = logits / temperature
    if top_k is not None:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        probs = torch.zeros_like(logits)
        probs[topk_idx] = torch.softmax(topk_vals, dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
    sample = torch.multinomial(probs, num_samples=1).item()
    return sample

# -------------------------
# Melodic constraint helper
# -------------------------
def constrain_pitch_to_scale(pitch, scale_notes, prev_pitch=None, max_jump=12):
    """
    Constrain pitch to a given scale and limit large jumps.
    """
    pitch = min(scale_notes, key=lambda x: abs(x - pitch))
    if prev_pitch is not None:
        if abs(pitch - prev_pitch) > max_jump:
            pitch = prev_pitch + max_jump * ((pitch - prev_pitch)//abs(pitch - prev_pitch))
            pitch = min(scale_notes, key=lambda x: abs(x - pitch))
    return pitch
