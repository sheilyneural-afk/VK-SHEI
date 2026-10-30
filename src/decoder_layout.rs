//! Backend-neutral resolution of the conventional decoder tensor contract.
//!
//! This module knows tensor roles, not model names, vendors or accelerators.
//! Concrete adapters may add stricter operation checks after resolution.

use anyhow::{Result, ensure};

use crate::model_bundle::ModelBundle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderLayoutKind {
    DecoderOnlyGated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderLayout {
    pub kind: DecoderLayoutKind,
    pub layers: usize,
    pub tied_embeddings: bool,
}

pub fn resolve(bundle: &ModelBundle) -> Result<DecoderLayout> {
    for name in ["model.embed_tokens.weight", "model.norm.weight"] {
        ensure!(
            bundle.has_tensor(name),
            "decoder bundle missing tensor {name}"
        );
    }
    let tied_embeddings = bundle.spec.tie_word_embeddings == Some(true);
    if !tied_embeddings {
        ensure!(
            bundle.has_tensor("lm_head.weight"),
            "decoder bundle missing tensor lm_head.weight"
        );
    }
    for layer in 0..bundle.spec.num_hidden_layers {
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let name = format!("model.layers.{layer}.{suffix}");
            ensure!(
                bundle.has_tensor(&name),
                "decoder bundle missing tensor {name}"
            );
        }
    }
    Ok(DecoderLayout {
        kind: DecoderLayoutKind::DecoderOnlyGated,
        layers: bundle.spec.num_hidden_layers,
        tied_embeddings,
    })
}
