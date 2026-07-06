import torch
import torch.nn as nn

from app.domain_model.blocks import TransformerBlock


class DomainTransitionModel(nn.Module):
    """(state, action) -> logits over terminal outcome state ids."""

    def __init__(self, vocab_size: int, num_states: int, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4, d_ff: int = 512, max_len: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.embed(input_ids) + self.pos(pos)
        mask = torch.tril(torch.ones(t, t, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        return self.head(x[:, -1, :])
