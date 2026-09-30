import os
import pretty_midi
import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------------------
# Load all MIDI files from folder
# -------------------------------
def load_midi_files(folder_path):
    midi_files = []
    folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"MIDI folder not found: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".mid", ".midi")):
                midi_path = os.path.join(root, filename)
                try:
                    midi = pretty_midi.PrettyMIDI(midi_path)
                    midi_files.append(midi)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to load {filename}: {e}")
    print(f"✅ Loaded {len(midi_files)} MIDI files from {folder_path}")
    return midi_files

# -------------------------------
# Convert PrettyMIDI → sequences
# -------------------------------
def midi_to_note_sequences(midi_obj):
    notes = []
    for inst in midi_obj.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            dur = note.end - note.start
            if dur > 0.05:
                notes.append((note.start, note.pitch, dur, note.velocity))

    notes.sort(key=lambda x: x[0])
    if not notes:
        return [], [], []

    pitches = [n[1] for n in notes]
    durs = [max(1, min(int(round(n[2] / 0.05)), 31)) for n in notes]
    vels = [max(30, min(int(round(n[3])), 120)) for n in notes]
    return pitches, durs, vels

# -------------------------------
# PyTorch Dataset
# -------------------------------
class MIDIDataset(Dataset):
    def __init__(self, folder_path, seq_len=50, step=5):
        self.seq_len = seq_len
        self.step = step
        self.samples = []

        midi_files = load_midi_files(folder_path)
        for midi in midi_files:
            p_seq, d_seq, v_seq = midi_to_note_sequences(midi)
            if len(p_seq) < seq_len:
                continue
            for i in range(0, len(p_seq) - seq_len, step):
                self.samples.append((
                    np.array(p_seq[i:i+seq_len]),
                    np.array(d_seq[i:i+seq_len]),
                    np.array(v_seq[i:i+seq_len])
                ))

        print(f"✅ Prepared {len(self.samples)} sequences from MIDI files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pitch, dur, vel = self.samples[idx]
        x_pitch, y_pitch = pitch[:-1], pitch[1:]
        x_dur, y_dur = dur[:-1], dur[1:]
        x_vel, y_vel = vel[:-1], vel[1:]
        return (
            torch.tensor(x_pitch, dtype=torch.long),
            torch.tensor(x_dur, dtype=torch.long),
            torch.tensor(x_vel, dtype=torch.long),
            torch.tensor(y_pitch, dtype=torch.long),
            torch.tensor(y_dur, dtype=torch.long),
            torch.tensor(y_vel, dtype=torch.long)
        )
