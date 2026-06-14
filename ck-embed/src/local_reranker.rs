//! Local cross-encoder reranker.
//!
//! Jina-Reranker-Tiny (~33 MB ONNX) scoring query/document pairs jointly.
//! Ported from `brightworker/org-store`. Model + tokenizer are bundled into the
//! binary (`bundled-models` feature) so reranking works fully offline.
//!
//! ONNX interface: `input_ids` + `attention_mask`, shape `[batch, MAX_LEN]`,
//! producing `logits` of shape `[batch, 1]` — one relevance score per pair.

use anyhow::{Context, Result};
use half::f16;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::reranker::{RerankResult, Reranker};

/// Fixed sequence length for query/document pairs (longer inputs truncated).
const MAX_SEQUENCE_LENGTH: usize = 128;

#[cfg(feature = "bundled-models")]
const RERANKER_MODEL: &[u8] = include_bytes!("../models/reranker.onnx");
#[cfg(feature = "bundled-models")]
const RERANKER_TOKENIZER: &[u8] = include_bytes!("../models/reranker-tokenizer.json");

pub struct LocalReranker {
    session: Session,
    tokenizer: Tokenizer,
}

impl LocalReranker {
    /// Construct from the bundled model + tokenizer bytes. Offline.
    #[cfg(feature = "bundled-models")]
    pub fn new_bundled() -> Result<Self> {
        let session = Session::builder()
            .context("create reranker session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("set optimization level")?
            .commit_from_memory(RERANKER_MODEL)
            .context("load bundled reranker ONNX model")?;

        let tokenizer = Tokenizer::from_bytes(RERANKER_TOKENIZER)
            .map_err(|e| anyhow::anyhow!("load bundled reranker tokenizer: {e}"))?;

        Ok(Self { session, tokenizer })
    }

    /// Tokenize a (query, document) pair to padded ids + attention mask.
    fn tokenize_pair(&self, query: &str, doc: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode((query, doc), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let ids = encoding.get_ids();
        let len = ids.len().min(MAX_SEQUENCE_LENGTH);

        let mut attention_mask = vec![0i64; MAX_SEQUENCE_LENGTH];
        let mut padded_ids = vec![1i64; MAX_SEQUENCE_LENGTH]; // pad id = 1 (Jina)
        for i in 0..len {
            attention_mask[i] = 1;
            padded_ids[i] = ids[i] as i64;
        }
        Ok((padded_ids, attention_mask))
    }
}

impl Reranker for LocalReranker {
    fn id(&self) -> &'static str {
        "local_reranker"
    }

    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = documents.len();
        let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * MAX_SEQUENCE_LENGTH);
        let mut all_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * MAX_SEQUENCE_LENGTH);
        for doc in documents {
            let (ids, mask) = self.tokenize_pair(query, doc)?;
            all_input_ids.extend(ids);
            all_attention_mask.extend(mask);
        }

        let shape = [batch_size, MAX_SEQUENCE_LENGTH];
        let input_ids_tensor =
            Tensor::from_array((shape, all_input_ids)).context("build input_ids tensor")?;
        let attention_mask_tensor =
            Tensor::from_array((shape, all_attention_mask)).context("build attention_mask tensor")?;

        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            })
            .context("reranker ONNX inference")?;

        let output = outputs
            .get("logits")
            .or_else(|| Some(&outputs[0]))
            .context("no output tensor from reranker")?;

        let scores: Vec<f32> = if let Ok((_, s)) = output.try_extract_tensor::<f32>() {
            s.iter().copied().collect()
        } else {
            let (_, s) = output
                .try_extract_tensor::<f16>()
                .context("extract reranker logits")?;
            s.iter().map(|x| x.to_f32()).collect()
        };

        // One score per document, returned in input order (caller sorts).
        Ok(documents
            .iter()
            .zip(scores)
            .map(|(doc, score)| RerankResult {
                query: query.to_string(),
                document: doc.clone(),
                score,
            })
            .collect())
    }
}

#[cfg(all(test, feature = "bundled-models"))]
mod tests {
    use super::*;

    #[test]
    fn ranks_relevant_document_highest() {
        let mut r = LocalReranker::new_bundled().expect("bundled reranker");
        let docs = vec![
            "completely unrelated topic about cars".to_string(),
            "buy groceries at the store".to_string(),
            "random meeting notes".to_string(),
        ];
        let out = r.rerank("grocery shopping list", &docs).expect("rerank");
        assert_eq!(out.len(), 3);
        let best = out
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(best.document, "buy groceries at the store");
    }
}
