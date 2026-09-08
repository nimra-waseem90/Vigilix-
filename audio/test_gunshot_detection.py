from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch
import librosa

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIO_FILE = "data/raw/audio/normal.mp3"

# Initial threshold based on our clean gunshot test
CONFIDENCE_THRESHOLD = 0.10

print("========================================")
print("       VIGILIX GUNSHOT DETECTOR")
print("========================================")

print("\nLoading AST model...")

feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
model = ASTForAudioClassification.from_pretrained(MODEL_ID)
model.eval()

print("AST loaded successfully!")

print(f"\nLoading audio: {AUDIO_FILE}")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

print(f"Audio loaded successfully")
print(f"Sample rate: {sr}")
print(f"Duration: {len(audio) / sr:.2f} seconds")

# Prepare input
inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

# Run AST
with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.sigmoid(outputs.logits)[0]

# Find Gunshot, gunfire specifically
gunshot_index = None

for index, label in model.config.id2label.items():
    if label.lower() == "gunshot, gunfire":
        gunshot_index = int(index)
        break

if gunshot_index is None:
    print("\nERROR: Gunshot label not found in AST model.")
    exit()

gunshot_confidence = probabilities[gunshot_index].item()

print("\n========================================")
print("          GUNSHOT ANALYSIS")
print("========================================")

print(f"Gunshot confidence: {gunshot_confidence:.3f}")
print(f"Threshold:          {CONFIDENCE_THRESHOLD:.3f}")

if gunshot_confidence >= CONFIDENCE_THRESHOLD:
    print("\n🔫 GUNSHOT DETECTED!")
else:
    print("\nNo gunshot detected.")

print("\n========================================")
print("              TEST COMPLETE")
print("========================================")