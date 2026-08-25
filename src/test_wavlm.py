import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "microsoft/wavlm-base-plus"

print("Loading WavLM...")

# Audio feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Pretrained WavLM model
model = AutoModel.from_pretrained(MODEL_NAME)

print()
print("WavLM loaded successfully!")
print("Model:", MODEL_NAME)
print("PyTorch:", torch.__version__)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print("Expected sample rate:", feature_extractor.sampling_rate)