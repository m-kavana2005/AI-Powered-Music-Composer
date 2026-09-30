import os
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import librosa
import librosa.display

# ---------------------------------------------------------
# Get latest file from a folder
# ---------------------------------------------------------
def get_latest_file(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]
    if not files:
        raise FileNotFoundError(f"No files found in {folder}")
    return max(files, key=os.path.getctime)

# ---------------------------------------------------------
# Extract pitch contour from MIDI
# ---------------------------------------------------------
def extract_midi_pitch_sequence(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    pitches, times = [], []

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitches.append(note.pitch)
            times.append(note.start)

    if not pitches:
        raise ValueError("No notes found in MIDI!")

    return np.array(times), np.array(pitches)

# ---------------------------------------------------------
# Plot pitch contour
# ---------------------------------------------------------
def plot_midi_pitch(times, pitches, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(times, pitches, marker='o')
    plt.title("Pitch Contour (MIDI)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Pitch (MIDI number)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------------------------------------------------
# Analyze WAV spectral centroid
# ---------------------------------------------------------
def analyze_audio(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.frames_to_time(range(len(centroid)), sr=sr)
    return times, centroid

def plot_spectral_centroid(times, centroid, save_path):
    plt.figure(figsize=(12, 5))
    plt.plot(times, centroid)
    plt.title("Spectral Centroid Over Time (WAV Brightness)")
    plt.xlabel("Time (sec)")
    plt.ylabel("Centroid Frequency (Hz)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------------------------------------------------
# Evaluate quality of a MIDI file (simple heuristic)
# ---------------------------------------------------------
def evaluate_quality(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)
    note_counts = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
    score = min(10, max(1, note_counts // 10))  # Scale to 1-10
    return score

# ---------------------------------------------------------
# Full quality test (auto-detect latest MIDI/WAV)
# ---------------------------------------------------------
def quality_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    midi_dir = os.path.join(base_dir, "generated_music", "midi")
    wav_dir = os.path.join(base_dir, "generated_music", "wav")

    latest_midi = get_latest_file(midi_dir)
    latest_wav = get_latest_file(wav_dir)

    print("Using MIDI:", latest_midi)
    print("Using WAV:", latest_wav)

    output_dir = os.path.join(base_dir, "quality_tests")
    os.makedirs(output_dir, exist_ok=True)

    # MIDI analysis
    times, pitches = extract_midi_pitch_sequence(latest_midi)
    pitch_graph_path = os.path.join(output_dir, "pitch_contour.png")
    plot_midi_pitch(times, pitches, pitch_graph_path)

    # WAV analysis
    spec_times, centroid = analyze_audio(latest_wav)
    centroid_graph_path = os.path.join(output_dir, "spectral_centroid.png")
    plot_spectral_centroid(spec_times, centroid, centroid_graph_path)

    # MIDI quality score
    score = evaluate_quality(latest_midi)

    print("\n🎵 Quality Test Completed!")
    print("Pitch Contour Graph:", pitch_graph_path)
    print("Spectral Brightness Graph:", centroid_graph_path)
    print("MIDI Quality Score:", score, "/ 10")
    return score
