import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_loader import MIDIDataset

# --------------------------------------------------
# Auto-detect MIDI folder
# --------------------------------------------------
def auto_detect_midi_folder(start_path="data"):
    for root, dirs, files in os.walk(start_path):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                print(f"🎵 MIDI files detected in: {root}")
                return root
    raise FileNotFoundError("❌ No MIDI files found in the project.")

# --------------------------------------------------
# Tiny Transformer model
# --------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, pitch_vocab=128, dur_vocab=32, vel_vocab=128,
                 d_model=128, n_heads=4, n_layers=2, dim_feedforward=256):
        super().__init__()
        self.pitch_embed = nn.Embedding(pitch_vocab, d_model)
        self.dur_embed   = nn.Embedding(dur_vocab, d_model)
        self.vel_embed   = nn.Embedding(vel_vocab, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pitch_out = nn.Linear(d_model, pitch_vocab)
        self.dur_out   = nn.Linear(d_model, dur_vocab)
        self.vel_out   = nn.Linear(d_model, vel_vocab)

    def forward(self, p, d, v):
        x = self.pitch_embed(p) + self.dur_embed(d) + self.vel_embed(v)
        x = self.transformer(x)
        return self.pitch_out(x), self.dur_out(x), self.vel_out(x)

# --------------------------------------------------
# Training function
# --------------------------------------------------
def train_model(epochs=5, batch_size=16, seq_len=50, lr=1e-3, step=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    midi_folder = auto_detect_midi_folder()
    dataset = MIDIDataset(midi_folder, seq_len=seq_len, step=step)

    if len(dataset) == 0:
        raise ValueError("Dataset empty. Add MIDI files!")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = TinyTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")

        for x_pitch, x_dur, x_vel, y_pitch, y_dur, y_vel in pbar:
            x_pitch, x_dur, x_vel = x_pitch.to(device), x_dur.to(device), x_vel.to(device)
            y_pitch, y_dur, y_vel = y_pitch.to(device), y_dur.to(device), y_vel.to(device)

            optimizer.zero_grad()
            pitch_logits, dur_logits, vel_logits = model(x_pitch, x_dur, x_vel)

            loss_pitch = F.cross_entropy(pitch_logits.reshape(-1, pitch_logits.size(-1)), y_pitch.reshape(-1))
            loss_dur   = F.cross_entropy(dur_logits.reshape(-1, dur_logits.size(-1)), y_dur.reshape(-1))
            loss_vel   = F.cross_entropy(vel_logits.reshape(-1, vel_logits.size(-1)), y_vel.reshape(-1))

            loss = loss_pitch + loss_dur + loss_vel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_pitch, x_dur, x_vel, y_pitch, y_dur, y_vel in val_loader:
                x_pitch, x_dur, x_vel = x_pitch.to(device), x_dur.to(device), x_vel.to(device)
                y_pitch, y_dur, y_vel = y_pitch.to(device), y_dur.to(device), y_vel.to(device)

                pitch_logits, dur_logits, vel_logits = model(x_pitch, x_dur, x_vel)
                val_loss += (
                    F.cross_entropy(pitch_logits.reshape(-1, pitch_logits.size(-1)), y_pitch.reshape(-1)) +
                    F.cross_entropy(dur_logits.reshape(-1, dur_logits.size(-1)), y_dur.reshape(-1)) +
                    F.cross_entropy(vel_logits.reshape(-1, vel_logits.size(-1)), y_vel.reshape(-1))
                ).item()

        val_loss /= len(val_loader)
        print(f"Epoch {ep}: Train Loss={total_loss/len(train_loader):.4f} | Val Loss={val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{ep}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"📦 Saved checkpoint: {checkpoint_path}")

    return checkpoint_path

# --------------------------------------------------
# Run script
# --------------------------------------------------
if __name__ == "__main__":
    train_model()
