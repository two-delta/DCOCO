# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash backbone with the DCOCO pairwise candidate corrector.

DCOCO keeps the DFlash backbone unchanged. For every predicting draft slot it
takes the backbone's top-K candidates and scores each transition "candidate i
at slot s -> candidate j at slot s+1" with a small MLP over a per-edge pair
state and per-candidate context features. The shared candidate walk then picks
the draft path (greedy prefix-first, or temperature-sampled).
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix


@support_torch_compile
class DCOCOHead(nn.Module):
    """Pairwise transition scorer over the top-K candidates of each slot.

    With ``h`` the draft hidden state per slot, ``c`` the raw candidate logits
    and ``E`` the target embeddings of the candidates:

        pe[s, i]      = LN(cand_ctx_proj([LN0(cand_proj(E[s, i])),
                                          LN0(ctx_h_proj(h[s]))]))
        st[s]         = pair_proj([h[s], h[s+1]])
        delta[s, i, j] = trans_mlp([st[s], pe[s, i], pe[s+1, j], c[s, i], c[s+1, j]])
        score[s, i, j] = c[s+1, j] + delta[s, i, j]

    ``LN0`` is a non-affine LayerNorm. The last ``trans_mlp`` layer is trained
    from zero, so an untrained corrector reproduces the backbone's argmax path.
    The head has no per-slot parameters: it serves any number of slots.
    """

    def __init__(
        self,
        *,
        model_hidden_size: int,
        config: dict[str, Any],
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.candidate_topk = int(config["dcoco_candidate_topk"])
        if self.candidate_topk < 1:
            raise ValueError("DCOCO candidate_topk must be positive.")
        self.emb_dim = int(config["dcoco_emb_dim"])
        self.state_dim = int(config["dcoco_state_dim"])
        mlp_dim = int(config["dcoco_mlp_dim"])
        hidden, emb = model_hidden_size, self.emb_dim
        # The corrector is small and its trans_mlp weight is split after
        # loading, so every layer stays unquantized in the model dtype.
        self.cand_proj = ReplicatedLinear(
            hidden,
            emb,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "cand_proj"),
        )
        self.ctx_h_proj = ReplicatedLinear(
            hidden,
            emb,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "ctx_h_proj"),
        )
        self.cand_ctx_proj = ReplicatedLinear(
            2 * emb,
            emb,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "cand_ctx_proj"),
        )
        self.ctx_out_norm = nn.LayerNorm(emb)
        self.pair_proj = ReplicatedLinear(
            2 * hidden,
            self.state_dim,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "pair_proj"),
        )
        # Input layout of trans_mlp[0]: [state, pe_prev, pe_cur, c_prev, c_cur].
        self.trans_mlp = nn.Sequential(
            nn.Linear(self.state_dim + 2 * emb + 2, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
        )
        self._state_weight: torch.Tensor | None = None
        self._prev_weight: torch.Tensor
        self._cur_weight: torch.Tensor

    @torch.no_grad()
    def materialize_inference_buffers(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Split trans_mlp[0] into state / predecessor / successor blocks.

        The trained first layer consumes concat(state, pe_prev, pe_cur, c_prev,
        c_cur) for every (i, j) pair. Evaluating it as
            W_state @ state + W_prev @ [pe_prev, c_prev] + W_cur @ [pe_cur, c_cur]
        computes each candidate's contribution once instead of materializing
        the K*K concatenation.
        """
        weight = self.trans_mlp[0].weight
        state, emb = self.state_dim, self.emb_dim
        prev_logit = state + 2 * emb
        self._state_weight = weight[:, :state].to(device, dtype).contiguous()
        self._prev_weight = (
            torch.cat(
                (
                    weight[:, state : state + emb],
                    weight[:, prev_logit : prev_logit + 1],
                ),
                dim=1,
            )
            .to(device, dtype)
            .contiguous()
        )
        self._cur_weight = (
            torch.cat(
                (
                    weight[:, state + emb : state + 2 * emb],
                    weight[:, prev_logit + 1 : prev_logit + 2],
                ),
                dim=1,
            )
            .to(device, dtype)
            .contiguous()
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        candidate_logits: torch.Tensor,
        pass_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate-walk scores ``[num_reqs, num_slots, K, K]``.

        Row 0 holds the start scores (the raw logits of slot 0, broadcast over
        predecessors); rows s >= 1 score slot s given the slot s-1 candidate.
        """
        if self._state_weight is None:
            raise RuntimeError(
                "Call materialize_inference_buffers() after loading the DCOCO head."
            )
        _, n_slots, topk = candidate_logits.shape
        if topk != self.candidate_topk:
            raise ValueError(
                f"DCOCO expects candidate_topk={self.candidate_topk}, got {topk}."
            )
        logits = candidate_logits.float()
        first = logits[:, :1, None, :].expand(-1, 1, topk, -1)
        if n_slots == 1:
            return first.contiguous()

        dtype = self._state_weight.dtype
        token_embeddings = token_embeddings.to(dtype)
        pass_hidden = pass_hidden.to(dtype)
        emb = self.emb_dim
        base = F.layer_norm(self.cand_proj(token_embeddings), (emb,))
        context = F.layer_norm(self.ctx_h_proj(pass_hidden), (emb,))
        context = context.unsqueeze(-2).expand_as(base)
        pe = self.ctx_out_norm(self.cand_ctx_proj(torch.cat((base, context), dim=-1)))
        # Per-candidate features for the split first layer: [pe, c].
        features = torch.cat((pe, logits.to(dtype).unsqueeze(-1)), dim=-1)
        prev = F.linear(features[:, :-1], self._prev_weight)  # [B, S-1, K, M]
        cur = F.linear(features[:, 1:], self._cur_weight)  # [B, S-1, K, M]
        state = self.pair_proj(
            torch.cat((pass_hidden[:, :-1], pass_hidden[:, 1:]), dim=-1)
        )
        state = F.linear(state, self._state_weight, self.trans_mlp[0].bias)
        pre = prev + state.unsqueeze(-2)  # [B, S-1, K, M]
        hidden = self.trans_mlp[1](pre.unsqueeze(-2) + cur.unsqueeze(-3))
        delta = self.trans_mlp[2](hidden).squeeze(-1)  # [B, S-1, K, K]
        pairs = logits[:, 1:, None, :] + delta.float()
        return torch.cat((first, pairs), dim=1)


class DCOCO(DFlashQwen3Model):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        draft_config = config.dflash_config
        block_size = draft_config.get("block_size", getattr(config, "block_size", 0))
        # With sample_from_anchor every block position predicts; otherwise the
        # anchor is the bonus token and only the block_size - 1 masks predict.
        anchor = bool(draft_config.get("sample_from_anchor", False))
        max_tokens = block_size if anchor else block_size - 1
        if not 1 <= spec.num_speculative_tokens <= max_tokens:
            raise ValueError(
                f"DCOCO requires 1 <= num_speculative_tokens <= {max_tokens} "
                f"(block_size={block_size}, sample_from_anchor={anchor})."
            )
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        with set_model_tag("dcoco_head"):
            self.dcoco = DCOCOHead(
                model_hidden_size=config.hidden_size,
                config=draft_config,
                prefix=maybe_prefix(prefix, "dcoco"),
            )


class DCOCOForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DCOCO

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.draft_id_to_target_id is not None:
            raise ValueError("DCOCO candidates require the full target vocabulary.")
        self.has_own_lm_head = bool(getattr(self.config, "has_own_lm_head", False))
        # DCOCO is trained on raw lm_head logits: no scale, softcap or softmax.
        self.candidate_logits_processor = LogitsProcessor(self.config.vocab_size)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.dcoco.candidate_topk
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        expected = {
            name.removeprefix("model.")
            for name, _ in self.named_parameters()
            if name.startswith("model.dcoco.")
            or (self.has_own_lm_head and name == "lm_head.weight")
        }
        seen: set[str] = set()
        head_weights: list[tuple[str, torch.Tensor]] = []

        def normalized_weights():
            for name, value in weights:
                name = name.removeprefix("model.")
                seen.add(name)
                # Load the corrector separately so the backbone's stacked
                # parameter mapping never touches it.
                if name.startswith("dcoco."):
                    head_weights.append((name.removeprefix("dcoco."), value))
                else:
                    yield name, value

        super().load_weights(normalized_weights())
        AutoWeightsLoader(self.model.dcoco).load_weights(head_weights)
        missing = expected - seen
        if missing:
            raise ValueError(
                f"DCOCO checkpoint coverage mismatch: missing={sorted(missing)}"
            )
        head = self.model.dcoco
        parameter = head.trans_mlp[0].weight
        head.materialize_inference_buffers(parameter.device, parameter.dtype)
