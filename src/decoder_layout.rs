//! Backend-neutral resolution of the conventional decoder tensor contract.
//!
//! This module knows tensor roles, not model names, vendors or accelerators.
//! Concrete adapters may add stricter operation checks after resolution.

use anyhow::{Context, Result, ensure};

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
    let hidden = bundle.spec.hidden_size;
    let intermediate = bundle
        .spec
        .intermediate_size
        .context("missing intermediate_size")?;
    let vocab = bundle.spec.vocab_size.context("missing vocab_size")?;
    let q_out = bundle.spec.attention.query_heads * bundle.spec.attention.head_dim;
    let kv_out = bundle.spec.attention.kv_heads * bundle.spec.attention.head_dim;
    let shape = |name: &str, expected: &[usize]| -> Result<()> {
        let actual = bundle
            .tensor_shape(name)
            .ok_or_else(|| anyhow::anyhow!("decoder bundle missing tensor {name}"))?;
        ensure!(
            actual == expected,
            "tensor {name} has shape {actual:?}, expected {expected:?}"
        );
        Ok(())
    };
    for name in ["model.embed_tokens.weight", "model.norm.weight"] {
        ensure!(
            bundle.has_tensor(name),
            "decoder bundle missing tensor {name}"
        );
    }
    shape("model.embed_tokens.weight", &[vocab, hidden])?;
    shape("model.norm.weight", &[hidden])?;
    let tied_embeddings = bundle.spec.tie_word_embeddings == Some(true);
    if !tied_embeddings {
        ensure!(
            bundle.has_tensor("lm_head.weight"),
            "decoder bundle missing tensor lm_head.weight"
        );
        shape("lm_head.weight", &[vocab, hidden])?;
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
        shape(
            &format!("model.layers.{layer}.input_layernorm.weight"),
            &[hidden],
        )?;
        shape(
            &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            &[hidden],
        )?;
        shape(
            &format!("model.layers.{layer}.self_attn.q_proj.weight"),
            &[q_out, hidden],
        )?;
        shape(
            &format!("model.layers.{layer}.self_attn.k_proj.weight"),
            &[kv_out, hidden],
        )?;
        shape(
            &format!("model.layers.{layer}.self_attn.v_proj.weight"),
            &[kv_out, hidden],
        )?;
        shape(
            &format!("model.layers.{layer}.self_attn.o_proj.weight"),
            &[hidden, q_out],
        )?;
        for suffix in ["gate_proj", "up_proj"] {
            shape(
                &format!("model.layers.{layer}.mlp.{suffix}.weight"),
                &[intermediate, hidden],
            )?;
        }
        shape(
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            &[hidden, intermediate],
        )?;
    }
    Ok(DecoderLayout {
        kind: DecoderLayoutKind::DecoderOnlyGated,
        layers: bundle.spec.num_hidden_layers,
        tied_embeddings,
    })
}
