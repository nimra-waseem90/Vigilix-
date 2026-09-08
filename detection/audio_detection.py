"""
Vigilix Audio Detection

Uses a pretrained Audio Spectrogram Transformer (AST) to detect
security-related sounds from audio files or video audio tracks.

Detects:
- Gunshots
- Explosions
- Screams
- Shouts
- Glass breaking
- Sirens
- Alarms
- Smoke alarms

Usage:

    python detection/audio_detection.py --source data/raw/audio/test2.wav

    python detection/audio_detection.py --source data/raw/audio/normal.mp3

    python detection/audio_detection.py --source data/raw/videos/test.mp4
"""

import argparse
import os
import sys
import tempfile
import shutil

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from alert.alert_manager import audio_alert, ALERT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

TARGET_SR = 16000

# Analyze 2-second windows
WINDOW_SECONDS = 2.0

# Move forward 1 second
# 50% overlap
HOP_SECONDS = 1.0

# Temporary testing threshold.
#
# Your pistol-shot sample produced approximately:
# Gunshot, gunfire = 0.089
#
# Therefore 0.08 is being used for testing.
CONFIDENCE_THRESHOLD = 0.08


# ============================================================
# SECURITY AUDIO WATCHLIST
# ============================================================

WATCHLIST = {
    "Gunshot, gunfire": "Gunshot",
    "Explosion": "Explosion",
    "Screaming": "Scream",
    "Shout": "Shout",
    "Glass": "Glass breaking",
    "Siren": "Siren",
    "Alarm": "Alarm",
    "Smoke detector": "Smoke alarm",
}


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None
_feature_extractor = None


# ============================================================
# LOAD AST MODEL
# ============================================================

def _load_model():

    global _model
    global _feature_extractor

    if _model is None:

        from transformers import (
            ASTForAudioClassification,
            ASTFeatureExtractor
        )

        print("\nLoading AST audio model...")
        print(f"Model: {MODEL_ID}")

        _feature_extractor = (
            ASTFeatureExtractor.from_pretrained(
                MODEL_ID
            )
        )

        _model = (
            ASTForAudioClassification.from_pretrained(
                MODEL_ID
            )
        )

        _model.eval()

        print("AST model loaded successfully!")

    return _model, _feature_extractor


# ============================================================
# EXTRACT AUDIO FROM VIDEO
# ============================================================

def _extract_audio(video_path, out_wav_path):

    try:

        from moviepy import VideoFileClip

    except ImportError:

        try:

            from moviepy.editor import VideoFileClip

        except ImportError:

            print(
                "\nERROR: MoviePy is not installed."
            )

            print(
                "Install it with:"
            )

            print(
                "python -m pip install moviepy"
            )

            return False

    print("\nExtracting audio from video...")
    print(f"Video: {video_path}")

    try:

        clip = VideoFileClip(video_path)

    except Exception as e:

        print(
            "\nERROR: Could not open video."
        )

        print(
            f"Reason: {e}"
        )

        return False

    if clip.audio is None:

        print(
            "This video has no audio track."
        )

        clip.close()

        return False

    try:

        try:

            clip.audio.write_audiofile(
                out_wav_path,
                fps=TARGET_SR,
                nbytes=2,
                codec="pcm_s16le",
                logger=None
            )

        except TypeError:

            clip.audio.write_audiofile(
                out_wav_path,
                fps=TARGET_SR,
                nbytes=2,
                codec="pcm_s16le",
                verbose=False,
                logger=None
            )

    except Exception as e:

        print(
            "\nERROR: Audio extraction failed."
        )

        print(
            f"Reason: {e}"
        )

        clip.close()

        return False

    finally:

        clip.close()

    print(
        "Audio extracted successfully!"
    )

    return True


# ============================================================
# WATCHLIST MATCHING
# ============================================================

def _match_watchlist(label_text):

    label_lower = label_text.strip().lower()

    for keyword, friendly_name in WATCHLIST.items():

        if keyword.lower() == label_lower:

            return friendly_name

    return None


# ============================================================
# ANALYZE AUDIO
# ============================================================

def analyze_audio_file(
    audio_path,
    required_hits=1,
    on_event=None
):

    import soundfile as sf
    import librosa
    import torch

    model, feature_extractor = _load_model()

    print("\nLoading audio...")

    # --------------------------------------------------------
    # LOAD AUDIO
    # --------------------------------------------------------

    try:

        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False
        )

    except Exception:

        print(
            "\nSoundFile could not read this file."
        )

        print(
            "Trying librosa..."
        )

        audio, sr = librosa.load(
            audio_path,
            sr=None,
            mono=False
        )

    # --------------------------------------------------------
    # STEREO -> MONO
    # --------------------------------------------------------

    if audio.ndim > 1:

        # librosa may return channels x samples
        if audio.shape[0] <= 8:

            audio = np.mean(
                audio,
                axis=0
            )

        else:

            audio = np.mean(
                audio,
                axis=1
            )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    if sr != TARGET_SR:

        print(
            f"Resampling audio: "
            f"{sr} Hz → {TARGET_SR} Hz"
        )

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

        sr = TARGET_SR

    # --------------------------------------------------------
    # CHECK AUDIO
    # --------------------------------------------------------

    if len(audio) == 0:

        print(
            "\nERROR: Audio file is empty."
        )

        return []

    duration = len(audio) / sr

    print(
        f"Audio duration: {duration:.2f} seconds"
    )

    print(
        f"Sample rate: {sr} Hz"
    )

    # --------------------------------------------------------
    # CHUNK SETTINGS
    # --------------------------------------------------------

    window_len = int(
        WINDOW_SECONDS * sr
    )

    hop_len = int(
        HOP_SECONDS * sr
    )

    total_len = len(audio)

    # --------------------------------------------------------
    # PAD SHORT AUDIO
    # --------------------------------------------------------

    if total_len < window_len:

        audio = np.pad(
            audio,
            (
                0,
                window_len - total_len
            )
        )

        total_len = len(audio)

    # --------------------------------------------------------
    # DETECTION STATE
    # --------------------------------------------------------

    streak = {}

    events_found = []

    alerted_events = set()

    print("\n========================================")
    print("       AST AUDIO ANALYSIS")
    print("========================================")

    # ========================================================
    # PROCESS WINDOWS
    # ========================================================

    for start in range(
        0,
        max(total_len - window_len, 1) + 1,
        hop_len
    ):

        chunk = audio[
            start:start + window_len
        ]

        # ----------------------------------------------------
        # PAD FINAL CHUNK
        # ----------------------------------------------------

        if len(chunk) < window_len:

            chunk = np.pad(
                chunk,
                (
                    0,
                    window_len - len(chunk)
                )
            )

        chunk_start_seconds = (
            start / sr
        )

        # ====================================================
        # AST FEATURE EXTRACTION
        # ====================================================

        inputs = feature_extractor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            logits = model(
                **inputs
            ).logits

        # AudioSet is multi-label.
        probabilities = torch.sigmoid(
            logits
        )[0]

        # ====================================================
        # DEBUG: TOP 10 PREDICTIONS
        # ====================================================

        top_indices = torch.topk(
            probabilities,
            k=10
        ).indices.tolist()

        print(
            f"\n[Window {chunk_start_seconds:.1f}s]"
        )

        for idx in top_indices:

            label = model.config.id2label[
                idx
            ]

            confidence = probabilities[
                idx
            ].item()

            print(
                f"  {label}: "
                f"{confidence:.3f}"
            )

        # ====================================================
        # CHECK WATCHLIST
        # ====================================================

        best_match = None

        for idx in range(
            len(probabilities)
        ):

            confidence = probabilities[
                idx
            ].item()

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            label = model.config.id2label[
                idx
            ]

            match = _match_watchlist(
                label
            )

            if match:

                if (
                    best_match is None
                    or confidence > best_match[1]
                ):

                    best_match = (
                        match,
                        confidence,
                        label
                    )

        # ====================================================
        # DETECTION FOUND
        # ====================================================

        if best_match:

            name, confidence, label = (
                best_match
            )

            streak[name] = (
                streak.get(name, 0) + 1
            )

            print(
                f"\n[audio] {name} detected"
            )

            print(
                f"        AST label: {label}"
            )

            print(
                f"        Confidence: "
                f"{confidence:.3f}"
            )

            print(
                f"        Time: "
                f"{chunk_start_seconds:.1f}s"
            )

            print(
                f"        Hit: "
                f"{streak[name]}/{required_hits}"
            )

            # ------------------------------------------------
            # EVENT KEY
            # ------------------------------------------------

            event_key = (
                name,
                round(
                    chunk_start_seconds,
                    1
                )
            )

            # ------------------------------------------------
            # TRIGGER ALERT
            # ------------------------------------------------

            if (
                streak[name] >= required_hits
                and event_key not in alerted_events
            ):

                alerted_events.add(
                    event_key
                )

                event = (
                    name,
                    confidence,
                    chunk_start_seconds
                )

                events_found.append(
                    event
                )

                print(
                    "\n========================================"
                )

                print(
                    "          AUDIO ALERT"
                )

                print(
                    "========================================"
                )

                print(
                    f"Event: {name}"
                )

                print(
                    f"Confidence: "
                    f"{confidence:.3f}"
                )

                print(
                    f"Time: "
                    f"{chunk_start_seconds:.1f}s"
                )

                # --------------------------------------------
                # CALLBACK
                # --------------------------------------------

                if on_event:

                    on_event(
                        name,
                        confidence,
                        chunk_start_seconds
                    )

                # Reset streak

                streak[name] = 0

        else:

            # No security sound in this window
            streak = {}

    return events_found


# ============================================================
# SAVE AUDIO EVIDENCE
# ============================================================

def _save_audio_evidence(
    source_path,
    name,
    timestamp
):

    evidence_dir = os.path.join(
        ALERT_DIR,
        "audio"
    )

    os.makedirs(
        evidence_dir,
        exist_ok=True
    )

    safe_name = (
        name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    evidence_name = (
        f"audio_{safe_name}_"
        f"{int(timestamp)}s.wav"
    )

    evidence_path = os.path.join(
        evidence_dir,
        evidence_name
    )

    # --------------------------------------------------------
    # CURRENT IMPLEMENTATION
    # --------------------------------------------------------
    #
    # Copies the complete source file.
    #
    # Later we can improve this to save only a short
    # event clip around the detection timestamp.
    # --------------------------------------------------------

    try:

        shutil.copy(
            source_path,
            evidence_path
        )

        print(
            "\nAudio evidence saved:"
        )

        print(
            evidence_path
        )

        return evidence_path

    except Exception as e:

        print(
            f"Could not save audio evidence: "
            f"{e}"
        )

        return ""


# ============================================================
# RUN DIRECT AUDIO
# ============================================================

def run_on_audio(
    audio_path
):

    print("\n========================================")
    print("       VIGILIX AUDIO DETECTION")
    print("========================================")

    print("\nSource audio:")
    print(audio_path)

    print(
        "\nUsing audio file directly."
    )

    print(
        "No video/audio extraction required."
    )

    # --------------------------------------------------------
    # CALLBACK
    # --------------------------------------------------------

    def _on_event(
        name,
        confidence,
        timestamp
    ):

        evidence_path = (
            _save_audio_evidence(
                audio_path,
                name,
                timestamp
            )
        )

        audio_alert(
            name,
            confidence,
            evidence_path
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    events = analyze_audio_file(
        audio_path,
        required_hits=1,
        on_event=_on_event
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if not events:

        print(
            "\nNo watchlisted sounds detected."
        )

    else:

        print(
            f"\nTotal audio events detected: "
            f"{len(events)}"
        )

    print(
        "\n========================================"
    )

    print(
        "       AUDIO TEST COMPLETE"
    )

    print(
        "========================================"
    )

    return events


# ============================================================
# RUN VIDEO
# ============================================================

def run_on_video(
    video_path,
    keep_wav=False
):

    tmp_dir = tempfile.mkdtemp(
        prefix="vigilix_audio_"
    )

    wav_path = os.path.join(
        tmp_dir,
        "audio.wav"
    )

    print("\n========================================")
    print("       VIGILIX AUDIO DETECTION")
    print("========================================")

    print("\nSource video:")
    print(video_path)

    # --------------------------------------------------------
    # EXTRACT AUDIO
    # --------------------------------------------------------

    has_audio = _extract_audio(
        video_path,
        wav_path
    )

    if not has_audio:

        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        return []

    # --------------------------------------------------------
    # CALLBACK
    # --------------------------------------------------------

    def _on_event(
        name,
        confidence,
        timestamp
    ):

        evidence_path = (
            _save_audio_evidence(
                wav_path,
                name,
                timestamp
            )
        )

        audio_alert(
            name,
            confidence,
            evidence_path
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    events = analyze_audio_file(
        wav_path,
        required_hits=1,
        on_event=_on_event
    )

    # --------------------------------------------------------
    # CLEAN TEMP AUDIO
    # --------------------------------------------------------

    if not keep_wav:

        try:

            os.remove(
                wav_path
            )

            os.rmdir(
                tmp_dir
            )

        except OSError:

            pass

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if not events:

        print(
            "\nNo watchlisted sounds detected."
        )

    else:

        print(
            f"\nTotal audio events detected: "
            f"{len(events)}"
        )

    print(
        "\n========================================"
    )

    print(
        "       AUDIO TEST COMPLETE"
    )

    print(
        "========================================"
    )

    return events


# ============================================================
# RUN ON ANY SOURCE
# ============================================================

def run_on_source(
    source_path,
    keep_wav=False
):

    source_path = os.path.abspath(
        source_path
    )

    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not os.path.isfile(
        source_path
    ):

        print(
            "\nERROR: File not found:"
        )

        print(
            source_path
        )

        return []

    extension = os.path.splitext(
        source_path
    )[1].lower()

    # --------------------------------------------------------
    # DIRECT AUDIO
    # --------------------------------------------------------

    if extension in AUDIO_EXTENSIONS:

        return run_on_audio(
            source_path
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if extension in VIDEO_EXTENSIONS:

        return run_on_video(
            source_path,
            keep_wav=keep_wav
        )

    # --------------------------------------------------------
    # UNKNOWN FORMAT
    # --------------------------------------------------------

    print(
        f"\nERROR: Unsupported file type: "
        f"{extension}"
    )

    print(
        "\nSupported audio:"
    )

    print(
        "  WAV, MP3, FLAC, M4A, OGG"
    )

    print(
        "\nSupported video:"
    )

    print(
        "  MP4, AVI, MOV, MKV, WEBM"
    )

    return []


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Vigilix audio event detection"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to an audio or video file"
        )
    )

    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help=(
            "Keep extracted WAV when "
            "processing a video"
        )
    )

    args = parser.parse_args()

    run_on_source(
        args.source,
        keep_wav=args.keep_wav
    )

