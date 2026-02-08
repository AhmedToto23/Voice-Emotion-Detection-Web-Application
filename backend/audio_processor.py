"""
Audio processing utilities for emotion detection
Matches the training pipeline exactly
"""

import numpy as np
import librosa

# Configuration (must match training)
SAMPLE_RATE = 16000
DURATION = 3.5
N_MFCC = 40
MIN_AUDIO_ENERGY = 0.001


def preprocess_audio(file_path):
    """
    Load and preprocess audio file to fixed format with quality checks.

    Args:
        file_path: Path to .wav file

    Returns:
        Preprocessed audio array (mono, 16kHz, fixed duration) or None if invalid
    """
    try:
        # Load audio at target sample rate
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Validate audio energy (reject near-silent files)
        audio_energy = np.sqrt(np.mean(audio ** 2))
        if audio_energy < MIN_AUDIO_ENERGY:
            return None

        # Calculate target length
        target_length = int(SAMPLE_RATE * DURATION)

        # Pad or trim to fixed length
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]

        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        return audio

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def extract_features(file_path):
    """
    Extract enhanced MFCC features with deltas from audio file.

    Args:
        file_path: Path to .wav file

    Returns:
        Feature vector (MFCCs + delta + delta-delta, aggregated) or None
    """
    # Preprocess audio
    audio = preprocess_audio(file_path)

    if audio is None:
        return None

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    # Extract delta (first derivative) features
    mfcc_delta = librosa.feature.delta(mfccs)

    # Extract delta-delta (second derivative) features
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # Aggregate across time using mean and std
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)

    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    # Concatenate all features (40*6 = 240 features)
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std
    ])

    return features