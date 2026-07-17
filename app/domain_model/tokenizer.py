from pathlib import Path

import sentencepiece as spm

from config import domain_settings


def train_tokenizer(corpus_path: Path, model_prefix: Path, vocab_size: int | None = None) -> Path:
    vocab_size = vocab_size or domain_settings.MODEL_VOCAB
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
    )
    return Path(f"{model_prefix}.model")


def load_tokenizer(model_path: Path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode(sp, text: str) -> list[int]:
    return sp.encode(text, out_type=int)


def decode(sp, ids: list[int]) -> str:
    return sp.decode(ids)
