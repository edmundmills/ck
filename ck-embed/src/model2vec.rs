//! Local model2vec embedder.
//!
//! Loads the `futur/Qwen3-Embedding-0.6B-model2vec-onnx` model (256-dim static
//! embeddings) via ONNX Runtime. Ported from `brightworker/org-store`. The model
//! and tokenizer are bundled into the binary (`bundled-models` feature) so this
//! provider is fully offline — no HuggingFace download, no network.
//!
//! The model2vec ONNX graph takes an `EmbeddingBag`-style interface:
//! - `input_ids`: 1-D tensor of all token ids (flattened across the batch)
//! - `offsets`:   1-D tensor of per-sequence start indices into `input_ids`
//! and returns `embeddings` (f16), one row per sequence.

use anyhow::{Context, Result, ensure};
use half::f16;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::Embedder;

/// Output dimensionality of the bundled model2vec model.
pub const MODEL2VEC_DIM: usize = 256;

#[cfg(feature = "bundled-models")]
const EMBEDDING_MODEL: &[u8] = include_bytes!("../models/embedding.onnx");
#[cfg(feature = "bundled-models")]
const EMBEDDING_TOKENIZER: &[u8] = include_bytes!("../models/embedding-tokenizer.json");

pub struct Model2VecEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    model_name: String,
}

impl Model2VecEmbedder {
    /// Construct from the bundled model + tokenizer bytes. Offline.
    #[cfg(feature = "bundled-models")]
    pub fn new_bundled(model_name: &str) -> Result<Self> {
        let session = Session::builder()
            .context("create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("set optimization level")?
            .commit_from_memory(EMBEDDING_MODEL)
            .context("load bundled model2vec ONNX model")?;

        let tokenizer = Tokenizer::from_bytes(EMBEDDING_TOKENIZER)
            .map_err(|e| anyhow::anyhow!("load bundled model2vec tokenizer: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            model_name: model_name.to_string(),
        })
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let num_tokens = input_ids.len();

        // Single sequence: offsets is just [0].
        let input_ids_tensor =
            Tensor::from_array(([num_tokens], input_ids)).context("build input_ids tensor")?;
        let offsets_tensor =
            Tensor::from_array(([1usize], vec![0i64])).context("build offsets tensor")?;

        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "offsets" => offsets_tensor,
            })
            .context("model2vec ONNX inference")?;

        let output = outputs
            .get("embeddings")
            .or_else(|| Some(&outputs[0]))
            .context("no output tensor from model2vec")?;

        let (_, slice) = output
            .try_extract_tensor::<f16>()
            .context("extract model2vec output tensor (f16)")?;
        let embedding: Vec<f32> = slice.iter().map(|x| x.to_f32()).collect();

        ensure!(
            embedding.len() == MODEL2VEC_DIM,
            "unexpected embedding dimension: expected {MODEL2VEC_DIM}, got {}",
            embedding.len()
        );
        Ok(embedding)
    }
}

impl Embedder for Model2VecEmbedder {
    fn id(&self) -> &'static str {
        "model2vec"
    }

    fn dim(&self) -> usize {
        MODEL2VEC_DIM
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }
}

#[cfg(all(test, feature = "bundled-models"))]
mod tests {
    use super::*;

    #[test]
    fn embeds_to_expected_dimension() {
        let mut e = Model2VecEmbedder::new_bundled("qwen3-model2vec").expect("bundled embedder");
        let out = e.embed(&["hello world".to_string()]).expect("embed");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), MODEL2VEC_DIM);
    }

    #[test]
    fn similar_text_is_closer_than_unrelated() {
        fn cos(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }
        let mut e = Model2VecEmbedder::new_bundled("qwen3-model2vec").expect("bundled embedder");
        let v = e
            .embed(&[
                "error handling and retry logic".to_string(),
                "exception recovery after failures".to_string(),
                "a cat sat on the warm mat".to_string(),
            ])
            .expect("embed");
        assert!(cos(&v[0], &v[1]) > cos(&v[0], &v[2]));
    }
}
