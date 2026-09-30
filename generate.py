import os
import datetime
import torch
import torch.nn as nn
import numpy as np
import pretty_midi
import soundfile as sf
import random

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated_music")
MIDI_DIR = os.path.join(GENERATED_DIR, "midi")
WAV_DIR = os.path.join(GENERATED_DIR, "wav")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = max(
    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)],
    key=os.path.getctime
)
os.makedirs(MIDI_DIR, exist_ok=True)
os.makedirs(WAV_DIR, exist_ok=True)

SAMPLE_RATE = 44100

# -------------------------
# MIDI note → frequency
# -------------------------
def midi_to_freq(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12)

# -------------------------
# Temperature Sampling
# -------------------------
def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# -------------------------
# Tiny Transformer
# -------------------------
class TinyTransformer(nn.Module):
    def __init__(self, pitch_vocab=128, dur_vocab=32, vel_vocab=128, d_model=128, n_heads=4, n_layers=2, dim_feedforward=256):
        super().__init__()
        
        self.pitch_embed = nn.Embedding(pitch_vocab, d_model)
        self.dur_embed   = nn.Embedding(dur_vocab, d_model)
        self.vel_embed   = nn.Embedding(vel_vocab, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pitch_out = nn.Linear(d_model, pitch_vocab)
        self.dur_out   = nn.Linear(d_model, dur_vocab)
        self.vel_out   = nn.Linear(d_model, vel_vocab)

    def forward(self, x_pitch, x_dur, x_vel):
        x = self.pitch_embed(x_pitch) + self.dur_embed(x_dur) + self.vel_embed(x_vel)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.pitch_out(x), self.dur_out(x), self.vel_out(x)

# -------------------------
# Load Trained Model
# -------------------------
def load_model():
    model = TinyTransformer()
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model

# -------------------------
# Generate Notes
# -------------------------
def generate_notes(model, seed_note=60, seq_len=50, temperature=1.0):
    
    x_pitch = torch.tensor([[seed_note]], dtype=torch.long)
    x_dur   = torch.tensor([[random.randint(1, 8)]], dtype=torch.long)
    x_vel   = torch.tensor([[random.randint(10, 30)]], dtype=torch.long)

    notes = [seed_note]

    for _ in range(seq_len - 1):
        with torch.no_grad():
            pitch_logits, dur_logits, vel_logits = model(x_pitch, x_dur, x_vel)

        pitch_id = sample_with_temperature(pitch_logits[0, -1], temperature)
        dur_id   = sample_with_temperature(dur_logits[0, -1], temperature)
        vel_id   = sample_with_temperature(vel_logits[0, -1], temperature)

        notes.append(pitch_id)

        x_pitch = torch.tensor([[pitch_id]], dtype=torch.long)
        x_dur   = torch.tensor([[dur_id]], dtype=torch.long)
        x_vel   = torch.tensor([[vel_id]], dtype=torch.long)

    return notes

# -------------------------
# Notes → MIDI
# -------------------------
def notes_to_midi(notes, midi_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    start = 0
    for note_number in notes:
        duration = 0.3 + (random.random() * 0.4)  # random duration 0.3–0.7
        end = start + duration
        
        midi_note = pretty_midi.Note(
            velocity=random.randint(60, 110),
            pitch=int(note_number),
            start=start,
            end=end
        )
        instrument.notes.append(midi_note)
        start = end

    midi.instruments.append(instrument)
    midi.write(midi_path)
    return midi_path

# -------------------------
# Notes → WAV
# -------------------------
def notes_to_wav(notes, wav_path):
    audio = np.array([], dtype=np.float32)

    for note in notes:
        duration = 0.3 + (random.random() * 0.4)
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        wave = 0.2 * np.sin(2 * np.pi * midi_to_freq(note) * t)
        audio = np.concatenate([audio, wave])

    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path

# -------------------------
# Full Pipeline
# -------------------------
def generate_music(seed_note=60, seq_len=50, temperature=1.0):
    model = load_model()
    notes = generate_notes(model, seed_note=seed_note, seq_len=seq_len, temperature=temperature)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    midi_path = os.path.join(MIDI_DIR, f"music_{timestamp}.mid")
    wav_path = os.path.join(WAV_DIR, f"music_{timestamp}.wav")

    notes_to_midi(notes, midi_path)
    notes_to_wav(notes, wav_path)

    print(f"🎵 MIDI saved: {midi_path}")
    print(f"🔊 WAV saved: {wav_path}")

    return midi_path, wav_path

# -------------------------
# Run Example
# -------------------------
if __name__ == "__main__":
    generate_music(seed_note=60, seq_len=80, temperature=1.0)
