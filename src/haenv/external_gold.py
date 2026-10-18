"""Supported public interface; implementation lives in haenv.contracts.external_gold."""
from haenv.contracts.external_gold import (
    BLOCKS,
    FRAMINGS,
    FieldClass,
    GATES,
    LATENT_CLASSES,
    PROBES,
    SLOT,
    blocks_for,
    external_framings,
    external_probes,
    gates_for,
    latent_classes,
    manifest,
    manifest_sha,
    passthrough_keys,
    probe_blocks_for,
    register_framing,
    register_gate,
    register_gold_block,
    register_probe,
)

__all__ = ['BLOCKS', 'FRAMINGS', 'FieldClass', 'GATES', 'LATENT_CLASSES', 'PROBES', 'SLOT', 'blocks_for', 'external_framings', 'external_probes', 'gates_for', 'latent_classes', 'manifest', 'manifest_sha', 'passthrough_keys', 'probe_blocks_for', 'register_framing', 'register_gate', 'register_gold_block', 'register_probe']
