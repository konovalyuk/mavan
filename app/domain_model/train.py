import random
from pathlib import Path

import torch
import torch.nn.functional as F

from app.domain import store
from app.domain.settings import domain_settings
from app.domain_model.model import DomainTransitionModel
from app.domain_model.tokenizer import train_tokenizer, load_tokenizer, encode


def _outcomes(s: dict) -> list[dict]:
    return s.get("outcome_states") or []


def _build_vocab(samples: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    states = sorted({o["state"] for s in samples for o in _outcomes(s) if o.get("state")})
    return {s: i for i, s in enumerate(states)}, {i: s for i, s in enumerate(states)}


def _enough(samples: list[dict]) -> tuple[bool, str]:
    s = domain_settings
    if len(samples) < s.TRAIN_MIN_SAMPLES:
        return False, f"samples {len(samples)} < {s.TRAIN_MIN_SAMPLES}"
    if len({x["state"] for x in samples}) < s.TRAIN_MIN_UNIQUE_STATES:
        return False, "unique states"
    if len({x["action"] for x in samples}) < s.TRAIN_MIN_UNIQUE_ACTIONS:
        return False, "unique actions"
    avg_q = sum(x.get("quality_avg", 0) for x in samples) / max(len(samples), 1)
    if avg_q < s.TRAIN_MIN_AVG_QUALITY:
        return False, f"avg quality {avg_q:.1f}"
    return True, "ok"


def _mix_samples(fresh: list[dict], core: list[dict]) -> list[dict]:
    ratio = domain_settings.TRAIN_REPLAY_RATIO
    n_core = int(len(fresh) * ratio / max(1 - ratio, 0.01))
    return fresh + (random.sample(core, min(n_core, len(core))) if core else [])


def _write_corpus(samples: list[dict], path: Path) -> None:
    lines = [f"[S] {s['state']} [A] {s['action']} [OUT] {o['state']}"
             for s in samples for o in _outcomes(s) if o.get("state")]
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


async def train_cycle(domain_id: str, fresh: list[dict]) -> str:
    core = await store.get_core_memory(domain_id)
    pool = fresh + core
    ok, reason = _enough(pool)
    if not ok:
        return f"insufficient: {reason}"

    batch_samples = _mix_samples(fresh, core)
    base = domain_settings.model_path(domain_id)
    base.mkdir(parents=True, exist_ok=True)
    corpus = base / "corpus.txt"
    _write_corpus(batch_samples, corpus)
    tok_prefix = base / "sp"
    model_path = tok_prefix.with_suffix(".model")
    if not model_path.exists():
        train_tokenizer(corpus, tok_prefix)
    sp = load_tokenizer(model_path)

    state2id, id2state = _build_vocab(batch_samples)
    if not state2id:
        return "insufficient: no next states"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = base / "model.pt"
    meta_path = base / "meta.pt"

    model = DomainTransitionModel(
        vocab_size=sp.get_piece_size(), num_states=len(state2id),
        d_model=domain_settings.MODEL_D_MODEL, n_layers=domain_settings.MODEL_N_LAYERS,
        n_heads=domain_settings.MODEL_N_HEADS, d_ff=domain_settings.MODEL_FFN,
    ).to(device)

    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=True)
        if saved.get("num_states") == len(state2id):
            model.load_state_dict(saved["state_dict"])

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    xs, ys = _make_batch(batch_samples, state2id, sp, device)
    if xs.size(0) == 0:
        return "insufficient: empty batch"

    bs = domain_settings.MODEL_BATCH
    for _ in range(domain_settings.MODEL_EPOCHS):
        perm = torch.randperm(xs.size(0))
        for i in range(0, xs.size(0), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({"state_dict": model.state_dict(), "num_states": len(state2id)}, ckpt_path)
    torch.save({"state2id": state2id, "id2state": {str(k): v for k, v in id2state.items()}}, meta_path)
    await store.update_domain(domain_id, status="ready", model_checkpoint=str(ckpt_path))
    return "ready"
