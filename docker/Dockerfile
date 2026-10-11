# vLLM release image + the DCOCO drafter.
#
#   docker build -t dcoco:vllm-v0.31.0 docker/
#
# overlay/vllm/ holds the DCOCO drafter's runtime files (two-delta vllm, branch DCOCO,
# 79694b45c) applied to the v0.31.0 tag: 3 new files and 5 stock files with DCOCO wired in.
# The change is Python-only, so the image's compiled kernels are untouched.
#
# The build refuses a base whose 5 stock files differ from v0.31.0 (overlay/BASE_SHA256SUMS):
# copying whole files over a different vLLM version would silently revert unrelated upstream
# code. Bump VLLM_VERSION only together with an overlay regenerated for that tag.
ARG VLLM_VERSION=v0.31.0
FROM vllm/vllm-openai:${VLLM_VERSION}

COPY overlay/ /tmp/dcoco-overlay/
RUN set -eu; \
    SITE=$(python3 -c "import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))"); \
    cd "$SITE" && sha256sum --quiet -c /tmp/dcoco-overlay/BASE_SHA256SUMS; \
    cp -r /tmp/dcoco-overlay/vllm/. "$SITE/vllm/"; \
    python3 -m compileall -q "$SITE/vllm/model_executor/models/dcoco.py" \
        "$SITE/vllm/v1/worker/gpu/spec_decode"; \
    python3 -c "from vllm.model_executor.models.registry import ModelRegistry; \
assert 'DCOCODraftModel' in ModelRegistry.get_supported_archs(), 'DCOCO not registered'"; \
    rm -rf /tmp/dcoco-overlay

# `vllm bench serve` (scripts/benchmark.py runs it from this image) needs vllm's bench extras,
# which the release image does not ship.
RUN python3 -m pip install --no-cache-dir pandas matplotlib seaborn datasets scipy plotly

# Entrypoint unchanged from the base image: `vllm serve <model> ...`.
