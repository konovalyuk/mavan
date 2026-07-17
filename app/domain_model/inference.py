from pathlib import Path

import torch
import torch.nn.functional as F

from config import domain_settings
from app.domain_model.model import DomainTransitionModel
from app.domain_model.tokenizer import load_tokenizer, encode


def predict(domain_id: str, state: str, action: str) -> dict[str, float]:
    base = domain_settings.model_path(domain_id)
    ckpt_path = base / "model.pt"
    meta_path = base / "meta.pt"
    tok_path = base / "sp.model"
    if not ckpt_path.exists() or not meta_path.exists() or not tok_path.exists():
        return {}

    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    state2id = meta["state2id"]
    id2state = {int(k): v for k, v in meta["id2state"].items()}

    sp = load_tokenizer(tok_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = DomainTransitionModel(
        vocab_size=sp.get_piece_size(), num_states=saved["num_states"],
        d_model=domain_settings.MODEL_D_MODEL, n_layers=domain_settings.MODEL_N_LAYERS,
        n_heads=domain_settings.MODEL_N_HEADS, d_ff=domain_settings.MODEL_FFN,
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    ids = encode(sp, f"[S] {state} [A] {action} [OUT]")
    if not ids:
        return {}
    x = torch.tensor([ids[:256] + [0] * max(0, 256 - len(ids))], dtype=torch.long, device=device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=-1)[0]
    return {id2state[i]: float(probs[i]) for i in range(len(id2state)) if probs[i] > 0.001}
