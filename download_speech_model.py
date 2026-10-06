"""
download_speech_model.py
========================
One-time download of the Whisper speech-to-text model used for voice input
(see `SPEECH_MODEL_NAME` in rag_core/config.py). About 970 MB for the default
`openai/whisper-small.en`. After this, voice input works fully offline.

Only the single weights file and the small config/tokenizer files are fetched,
not the duplicate TensorFlow/Flax/PyTorch copies in the same repository.

Usage:
    python download_speech_model.py
"""

from huggingface_hub import snapshot_download

from rag_core.config import SPEECH_MODEL_NAME

FILES = [
    "config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json",
    "added_tokens.json", "special_tokens_map.json",
]

if __name__ == "__main__":
    print(f"Downloading {SPEECH_MODEL_NAME} (about 970 MB for whisper-small.en) ...")
    path = snapshot_download(SPEECH_MODEL_NAME, allow_patterns=FILES)
    print(f"Done: {path}")
