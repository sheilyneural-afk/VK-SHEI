use anyhow::{Context, Result};
use mslim::torch_runtime::{TorchChatMessage, TorchModelRuntime};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

fn with_reference_document(document: &str, question: &str) -> Vec<TorchChatMessage> {
    vec![
        TorchChatMessage {
            role: "user".to_owned(),
            content: format!(
                "Texto de referencia (única fuente para las siguientes preguntas):\n\n{}",
                document.trim()
            ),
        },
        TorchChatMessage {
            role: "assistant".to_owned(),
            content: "He leído el texto. Responderé solo con datos que aparezcan en él.".to_owned(),
        },
        TorchChatMessage {
            role: "user".to_owned(),
            content: question.to_owned(),
        },
    ]
}

fn main() -> Result<()> {
    let mut model = PathBuf::from("deploy/local-production/runtime/models/active");
    let mut document_path = None::<PathBuf>;
    let mut max_tokens = 16_usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(it.next().context("falta --model")?),
            "--document" => {
                document_path = Some(PathBuf::from(it.next().context("falta --document")?))
            }
            "--max-tokens" => {
                max_tokens = it.next().context("falta --max-tokens")?.parse()?;
            }
            other => anyhow::bail!("argumento desconocido: {other}"),
        }
    }
    let runtime = TorchModelRuntime::load(&model)?;
    let mut cases: Vec<(String, Vec<i64>)> = Vec::new();
    cases.push((
        "raw_hola".to_owned(),
        runtime.encode_raw_chat_prompt_token_ids(
            &[TorchChatMessage {
                role: "user".to_owned(),
                content: "Hola".to_owned(),
            }],
            false,
        )?,
    ));
    if let Some(path) = document_path {
        let document = fs::read_to_string(&path)?;
        let question = "¿Cuáles son los tres códigos TEST-CODE, TEST-CODE-2 y TEST-DOCUMENT y qué ocurrió a las 3:17?";
        cases.push((
            "book_test_code_raw_template".to_owned(),
            runtime.encode_raw_chat_prompt_token_ids(
                &with_reference_document(&document, question),
                false,
            )?,
        ));
    }
    let mut rows = Vec::new();
    for (name, prompt_ids) in cases {
        let report = runtime.compare_native_inference(&prompt_ids, max_tokens)?;
        rows.push(json!({
            "case": name,
            "report": report,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "same_weights_tokenizer_greedy": true,
            "authority": "dense_bf16_vs_mslim_same_prompt_ids",
            "comparisons": rows,
        }))?
    );
    Ok(())
}
