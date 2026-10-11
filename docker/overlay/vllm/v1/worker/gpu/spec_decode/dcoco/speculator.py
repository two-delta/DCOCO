# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator


class DCOCOSpeculator(LiLiCorrSpeculator):
    """DFlash drafting with the DCOCO pairwise candidate corrector.

    Reuses LiLiCorr's target embedding / LM head binding and the shared
    candidate walk. DCOCO reads no context anchor, and its checkpoints may be
    trained with the anchor position as a prediction slot.
    """

    _speculator_name = "DCOCO"
    _candidate_top_k_key = "dcoco_candidate_topk"
    _supports_sample_from_anchor = True

    def prepare_context_anchor(
        self, input_batch: InputBatch, num_rejected: torch.Tensor
    ) -> None:
        return

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.model.dcoco(
            self.target_embeddings(candidate_ids), unary_logits, hidden_states
        )
