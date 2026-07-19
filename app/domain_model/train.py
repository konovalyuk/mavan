"""Online / continual training for domain transition model (deterministic, no LLM)."""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from app.domain import store
from app.domain_model.model import DomainTransitionModel
from app.domain_model.tokenizer import encode, load_tokenizer, train_tokenizer
from config import domain_settings


def _outcomes(s: dict) -> list[dict]:
    return s.get("outcome_states") or []


def _build_vocab(samples: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    states = sorted({o["state"] for s in samples for o in _outcomes(s) if o.get("state")})
    return {s: i for i, s in enumerate(states)}, {i: s for i, s in enumerate(states)}


def soft_trust_ok(samples: list[dict]) -> bool:
    """Whether Stage-2 should treat the model as well-calibrated (does not stop conveyor)."""
    s = domain_settings
    if len(samples) < s.TRAIN_MIN_SAMPLES:
        return False
    if len({x["state"] for x in samples}) < s.TRAIN_MIN_UNIQUE_STATES:
        return False
    if len({x["action"] for x in samples}) < s.TRAIN_MIN_UNIQUE_ACTIONS:
        return False
    avg_q = sum(x.get("quality_avg", 0) for x in samples) / max(len(samples), 1)
    return avg_q >= s.TRAIN_MIN_AVG_QUALITY


def _mix_samples(fresh: list[dict], core: list[dict]) -> list[dict]:
    ratio = domain_settings.TRAIN_REPLAY_RATIO
    n_core = int(len(fresh) * ratio / max(1 - ratio, 0.01))
    return fresh + (random.sample(core, min(n_core, len(core))) if core else [])


def _write_corpus(samples: list[dict], path: Path) -> None:
    lines = [
        f"[S] {s['state']} [A] {s['action']} [OUT] {o['state']}"
        for s in samples for o in _outcomes(s) if o.get("state")
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_batch(samples: list[dict], state2id: dict[str, int], sp, device):
    xs, ys = [], []
    for s in samples:
        for o in _outcomes(s):
            st = o.get("state", "")
            if st not in state2id:
                continue
            ids = encode(sp, f"[S] {s['state']} [A] {s['action']} [OUT]")
            if ids:
                xs.append(ids[:256])
                ys.append(state2id[st])
    if not xs:
        return torch.zeros(0, 1, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    max_t = max(len(x) for x in xs)
    padded = [x + [0] * (max_t - len(x)) for x in xs]
    return torch.tensor(padded, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def _ensure_tokenizer(samples: list[dict], base: Path):
    tok_prefix = base / "sp"
    model_path = tok_prefix.with_suffix(".model")
    if model_path.exists():
        return load_tokenizer(model_path)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        _write_corpus(samples, Path(f.name))
        tmp = Path(f.name)
    try:
        train_tokenizer(tmp, tok_prefix)
    finally:
        tmp.unlink(missing_ok=True)
    return load_tokenizer(model_path)


async def train_step(domain_id: str, micro_batch: list[dict]) -> dict:
    """One online update on an ephemeral micro-batch (+ capped replay mix).

    Does not persist raw web text. Writes weights checkpoint and S-A-O to capped replay.
    """
    if not micro_batch:
        return {"ok": False, "reason": "empty micro_batch"}

    replay = await store.get_core_memory(domain_id)
    batch_samples = _mix_samples(micro_batch, replay)
    pool_for_vocab = micro_batch + replay
    state2id, id2state = _build_vocab(pool_for_vocab)
    if not state2id:
        return {"ok": False, "reason": "no outcome states"}

    base = domain_settings.model_path(domain_id)
    base.mkdir(parents=True, exist_ok=True)
    sp = _ensure_tokenizer(pool_for_vocab, base)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = base / "model.pt"
    meta_path = base / "meta.pt"

    model = DomainTransitionModel(
        vocab_size=sp.get_piece_size(),
        num_states=len(state2id),
        d_model=domain_settings.MODEL_D_MODEL,
        n_layers=domain_settings.MODEL_N_LAYERS,
        n_heads=domain_settings.MODEL_N_HEADS,
        d_ff=domain_settings.MODEL_FFN,
    ).to(device)

    if ckpt_path.exists() and meta_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=True)
        old_meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        if saved.get("num_states") == len(state2id) and set(old_meta.get("state2id", {})) == set(state2id):
            model.load_state_dict(saved["state_dict"])

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    xs, ys = _make_batch(batch_samples, state2id, sp, device)
    if xs.size(0) == 0:
        return {"ok": False, "reason": "empty tensor batch"}

    bs = domain_settings.MODEL_BATCH
    steps = max(1, domain_settings.TRAIN_STEPS_PER_MICRO)
    last_loss = 0.0
    for _ in range(steps):
        perm = torch.randperm(xs.size(0))
        for i in range(0, xs.size(0), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    torch.save({"state_dict": model.state_dict(), "num_states": len(state2id)}, ckpt_path)
    torch.save(
        {"state2id": state2id, "id2state": {str(k): v for k, v in id2state.items()}},
        meta_path,
    )

    await store.add_replay(domain_id, micro_batch)
    domain = await store.get_domain(domain_id) or {}
    train_steps = int(domain.get("train_steps") or 0) + steps
    replay_after = await store.get_core_memory(domain_id)
    trusted = soft_trust_ok(replay_after)
    await store.update_domain(
        domain_id,
        model_checkpoint=str(ckpt_path),
        train_steps=train_steps,
        train_cycle=train_steps,
        model_trusted=trusted,
    )
    return {
        "ok": True,
        "loss": last_loss,
        "train_steps": train_steps,
        "micro_size": len(micro_batch),
        "replay_size": len(replay_after),
        "model_trusted": trusted,
        "checkpoint": str(ckpt_path),
    }


# Legacy name used by older scripts — online step without requiring TRAIN_MIN gate
async def train_cycle(domain_id: str, fresh: list[dict]) -> str:
    result = await train_step(domain_id, fresh)
    if result.get("ok"):
        return "ready" if result.get("model_trusted") else "updated"
    return f"insufficient: {result.get('reason', 'train failed')}"
