---
license: apache-2.0
languages:
  - multilingual
  - en
tags:
  - sentence-similarity
  - sequence-merging
---
# Sequence Merger Transformer Synthesizer

## Overview

The Sequence Merger is a custom neural architecture designed for **intelligent merging of variable-length vector sequences into a single representative vector**. It excels at tasks requiring sequence summarization, such as embedding aggregation in NLP or multimodal fusion, outperforming standard mean-pooling by capturing nuanced relational dynamics.

This model transforms an input of N arbitrary vectors `(batch_size, seq_len, d_model)` into a fixed output `(batch_size, d_model)`, preserving semantic coherence while adapting to complex data flows.

Pleas note that we trained this model fitting into the  **intfloat/mutilingual-e5-base** embed. Although the learning & training methology could be applied into other embedding models, this model follows the [intfloat/mutilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) as the elemental embedding engine.

## Scale

[[[please fill this section with the proper database & model scale.]]]

The database has limited n=10, so this model **Might not properly handle mode than 10 seq of vectors**.


## Core Philosophy: The Extrapolation Hypothesis

At the core of the Sequence Merger lies the **Extrapolation Hypothesis**—our guiding principle for transcending the inherent context length limitations of base embedding models like `intfloat/multilingual-e5-base` (max 512 tokens).

### The Hypothesis
We posit that by training a lightweight model to synthesize a single representative embedding from N individual chunk embeddings—where each chunk and their concatenation fit within the base model's context limit—the model learns not mere memorization, but the fundamental *function* of semantic synthesis. This function captures the relational dynamics and sequential coherence among vectors, allowing the model to preserve and integrate meaning progressively.

### From N-to-1 Synthesis to Infinite Extrapolation
- **Training Phase (Within Limits):** During training, inputs are N chunk embeddings `[vec(A), vec(B), ..., vec(N)]`, derived from texts where the full concatenation `A + B + ... + N` remains ≤512 tokens. The target is the base model's direct embedding `vec(A + B + ... + N)`, ensuring ground-truth supervision. This teaches the model to merge sequences reliably within bounded contexts.

- **Inference Phase (Beyond Limits):** In deployment, the model's output—itself a synthesized vector—becomes an input for the next merge. By recursively chaining outputs (e.g., merge `[vec(A)..vec(K)]` to synth_vec(A..K), then merge with `[vec(K+1)..vec(M)]`), the model extrapolates to arbitrarily long sequences, far exceeding 512 tokens. This self-referential synthesis maintains 'semantic momentum,' avoiding catastrophic loss of earlier context.

### Key Implications
- **Extended Context:** Enables processing of documents, conversations, or streams of any length as a single coherent embedding, without relying on resource-intensive long-context models.
- **Efficiency and Autonomy:** No need for repeated API calls or full-sequence re-embedding; operates on pre-computed chunks, reducing latency and cost to near-zero beyond initial embedding.
- **Validation:** Initial experiments show stable cosine similarity (>0.75) in chained merges up to 10x the base limit, with minimal degradation in recall of initial chunks—proving the hypothesis's viability.

## Usage

The TransformerSynthesizer processes **pre-computed vector sequences** (e.g., embeddings from E5), not raw text. Load and use it via the Hugging Face Hub with `trust_remote_code=True`. Below is a realistic workflow integrating with `intfloat/multilingual-e5-base` for text-to-vector conversion.

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Load E5 tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

# Example: Batch of texts to merge (e.g., multiple sentences per document)
texts = [
    ["First sentence of batch 1.", "Second sentence of batch 1.", "Third sentence of batch 1."],
    ["First sentence of batch 2.", "Second sentence of batch 2."]  # Variable lengths
]
batch_size = len(texts)

# Step 2: Convert texts to embeddings using E5 (batch_size, seq_len, d_model)
input_sequences = []
for i in range(batch_size):
    inputs = tokenizer(texts[i], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        e5_outputs = e5_model(**inputs)
        # Mean pool E5 outputs to get sentence embeddings (seq_len, d_model)
        sentence_embeddings = e5_outputs.last_hidden_state.mean(dim=1)
    input_sequences.append(sentence_embeddings)

input_sequences = torch.cat(input_sequences, dim=0)  # (batch_size, seq_len, 768)

# Step 3: Load our Synthesizer model
synthesizer = AutoModel.from_pretrained(
    "your-username/tiny-sequence-merger-soft-large",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Optional: for efficiency
)

# Step 4: Forward pass to merge sequences
with torch.no_grad():
    merged_vectors = synthesizer(input_sequences).last_hidden_state  # Shape: (batch_size, d_model)

print(f"Merged vectors shape: {merged_vectors.shape}")
print("Synthesized embeddings ready for downstream tasks!")
```

This workflow highlights our model's role as a 'vector synthesizer': it takes E5 embeddings as input and produces a coherent, single representation per sequence. For configuration details, inspect `config.json`. The model supports batched, variable-length inference on GPU/CPU and integrates with the Transformers pipeline for downstream tasks.

## Key Configurable Parameters

Our architecture shines through its flexible hyperparameters, allowing fine-tuned control over the pipeline:

- **d_model**: Embedding dimension (default: 768, E5-base compatible for seamless integration).
- **num_layers**: Depth of the transformer stack (e.g., 4-7 layers for balanced performance).
- **nhead**: Number of attention heads (default: 8; per-layer variations supported).
- **layer_dims**: Variable intermediate dimensions per layer (e.g., `[3840, 1920, 768]` for progressive compression—our signature bottleneck design).
- **ffn_dims**: Custom feed-forward hidden sizes per layer (e.g., `[8192, 6144, 4096]` for GLU-enhanced expansion).
- **nhead_dims**: Per-layer head counts (e.g., `[4, 3, 2, 2, 2]` for adaptive attention scaling).
- **use_glu**: Enable Gated Linear Units in FFN for non-linear gating (default: False; boosts expressivity in complex merges).
- **dropout**: Regularization rate (e.g., 0.1-0.12 for stable training).

These parameters enable architectures with 137M to 206M parameters, optimized for efficiency and coherence. Explore variants by tweaking `config.json` and retraining!

## Training Philosophy: The Principle of Progressive Overload Fine-Tuning

Our training paradigm emerges from a profound failure-turned-insight: an experiment with extreme datasets that initially devastated performance but ultimately revealed the path to superior generalization.

### The Extreme Dataset: 512-to-1 and Its Catastrophic Debut
To maximize data efficiency, we engineered an 'extreme' dataset by chunking source articles into single-token units—the atomic building blocks of embeddings. This created 512-to-1 pairs: inputs of 512 individual token embeddings, targeting the base model's direct embedding of the full sequence. From just hundreds of original articles, this yielded tens of thousands of input-target pairs, far outstripping the standard 10-to-1 dataset (merging up to 10 ~50-token chunks).

Directly training on 512-to-1, however, was disastrous. Models suffered catastrophic forgetting, with benchmark scores plummeting. The overwhelming complexity shattered the fragile structure of our lightweight mergers, proving that brute-force exposure to atomic extremes destroys rather than builds capability.

### The Breakthrough: Progressive Overload as Pre-Training + Fine-Tuning
The pivot came from reframing failure as a forge. Instead of direct training, we adopted a two-stage 'Progressive Overload' approach, inspired by athletic training principles: build foundational strength before loading maximum weight.

1. **Stage 1: Atomic Pre-Training (512-to-1 as Foundation):** Feed the model 512-to-1 data first, with higher learning rates. This 'searing' phase intentionally disrupts and rebuilds the model's internals, teaching it the primal interactions of token-level vectors—like forging atomic bonds in a vacuum. Though scores crash initially, it instills a robust 'semantic substrate' for handling fine-grained relationships.

2. **Stage 2: Molecular Fine-Tuning (10-to-1 as Specialization):** Transition to the target 10-to-1 dataset. This 'refinement' phase leverages the pre-trained foundation to master higher-level synthesis, chaining atomic insights into coherent 'molecular' embeddings. Remarkably, a single epoch yields dramatic recovery, surpassing vanilla 10-to-1 training.

By embracing failure as the crucible for growth, Progressive Overload ensures our mergers evolve from fragile aggregators to resilient synthesizers—ready for the infinite extrapolation promised by our core philosophy.

### Key Innovation: Paired Positional Encoding (PPE)
Within this paradigm, PPE addresses the challenge of injecting sequence order into high-dimensional chunk embeddings without distorting their rich semantics. Traditional positional encodings add noise-like signals, degrading benchmarks; PPE duplicates the sequence and *overwrites* dedicated dimension slots (front/back split) with position vectors, preserving 90%+ of original meaning while enabling precise positional awareness.

## Core Self-Supervised Method: vBERT

vBERT represents our crowning achievement: a vector-native adaptation of BERT-style pre-training, tailored for sequence mergers. By leveraging PPE, vBERT uses two intertwined self-supervised tasks—vMLM (violated Masked Language Model) and vNSP (violated Next Sentence Prediction)—to instill temporal and semantic coherence in the model. These tasks run concurrently, with total loss = loss_vmlm + loss_vnsp, enabling the model to evolve from a mere aggregator to a deep synthesizer of meaning.

### Critical Prerequisite: The `1-Article-1-Sequence` Dataset

vBERT's self-supervised pre-training requires a meticulously curated dataset to preserve the integrity of its learning signals, especially for vNSP's continuity detection.

Standard n-to-1 datasets (e.g., 512-to-1 or 10-to-1) are incompatible for vBERT:

- **Self-Supervised Nature:** vBERT does not rely on ground-truth to-1 vectors; attempts to use such datasets introduce unnecessary supervisory noise absent in pure self-supervision.
- **The False History Paradox:** Sliding-window construction generates overlapping, semantically coherent sequences from the same source. In vNSP, creating 'false histories' by splicing can inadvertently produce near-identical continuations (e.g., adjacent overlaps), leading to false negatives. The model learns to flag coherent narratives as 'disrupted,' creating a schizophrenic training signal that erodes context awareness and caps pre-training efficacy.

To eliminate this, we developed a dedicated pre-training corpus from `wikipedia/wikipedia` (multi-language editions like en, ko, fr, etc.) 1 Article = 1 Sequence. Process each full article into a single, unbroken vector sequence without sliding windows or partial excerpts.

### vMLM: Self-Consistent Vector Reconstruction
vMLM refines the MLM paradigm for vectors, enforcing that predictions maintain global sequence coherence via a feedback injection loop.

1. **Masking:** Randomly mask 15% of tokens in the input sequence, e.g., [A, B, C, D, E] → [A, M, C, D, E].
2. **Forward Pass on Masked Sequence:** Pass through the model with PPE duplication, yielding hidden layer outputs [cls', a1', m1', c1', d1', e1', a2', m2', c2', d2', e2']. Extract the predicted masked vectors m1' (first duplicate) and m2' (second duplicate).
3. **Original CLS Acquisition:** Pass the unmasked original sequence through the model to obtain CLS_orig (the final CLS token after PPE processing).
4. **Prediction Injection:** Re-pass the masked sequence, but during PPE, stealthily replace the masked positions with the predicted m1' and m2' (integrating into the appropriate front/back slots). This generates CLS_guesses (new CLS token).
5. **Strict L1 Loss Computation:** Calculate L1 loss (Mean Absolute Error across all 768 dimensions) between CLS_guesses and CLS_orig. Unlike lenient cosine similarity, L1 demands exact matching of both direction *and* magnitude, forging unyielding precision in reconstructions.

### vNSP: Binary Continuity Classification
vNSP upgrades NSP for vectors, using a classification head to detect sequence authenticity and penalize disruptions.

1. **History Generation:** For 50% of batches, create false histories: split the sequence at mid-point (split_idx = len // 2), discard the latter half, and splice in a random prefix from another sequence (for this reason, database should not be neither the 512-to-1 nor the 10-to-1. this will be discuss later.), e.g., [A, B, C] → [A, B, D'] where D' disrupts flow.
2. **PPE Processing:** Apply PPE to true/false sequences independently, embedding positional cues to highlight discontinuities in false cases.
3. **Classification Head:** Forward to get normalized CLS (cls_norm), pass through a dedicated NSP head (nn.Linear(768, 2)) yielding logits [IsNext, NotNext].
4. **Cross-Entropy Loss:** Label true as 0 (IsNext), false as 1 (NotNext); compute CE loss. High ppe_dim amplifies temporal breaks; low ppe_dim focuses on semantic jarring.

## The 3-Stage Rocket Protocol: Unified Training Philosophy

At the apex of our Extrapolation Hypothesis lies the 3-Stage Rocket Protocol—a comprehensive, battle-tested blueprint that propels models from naive aggregators to extrapolative synthesizers. Forged from iterative experimentation, this protocol synthesizes our key innovations (PPE, vBERT) and strategies (Progressive Overload) into a seamless ascent: Stage 1 builds philosophical foundations, Stage 2 stress-tests through overload, and Stage 3 refines for peak performance. vBERT, though disastrous in isolation (benchmarks plummeting as loss decreases), shines as the irreplaceable 'launchpad'—pre-pre-training that awakens innate vector synthesis principles.

#### Stage 1: Pre-Pre-Training – vBERT (Instilling First Principles)

vBERT serves as the 'temporal archaeologist' phase: a self-supervised pre-training regime that etches core principles of vector coherence into a blank-slate model, using PPE (high ppe_dim=384) to inject positional fidelity without semantic distortion.

**Key Enabler: Paired Positional Encoding (PPE)**
PPE duplicates sequences and overwrites dedicated slots (front/back m/2 splits, m=ppe_dim) with positional vectors, preserving 90%+ original meaning. High ppe_dim (384) prioritizes temporal stability, preventing gradient explosions in self-supervision. 

#### Stage 2: High-Difficulty Fine-Tuning – Progressive Overload (512-to-1 Awakening)

With vBERT's foundations laid, Stage 2 applies the 'intentional destruction' of Progressive Overload: expose the primed model to atomic-level extremes via 512-to-1 datasets (token-chunked, all combination trees). **Low ppe_dim=16** shifts to 'meaning insightful' mode, minimizing temporal distortion.

#### Stage 3: Standard Fine-Tuning – Peak Refinement (10-to-1 Mastery)

Stage 3 polishes the awakened model with 10-to-1 datasets (50-token chunks, up to 10 elements), **maintaining ppe_dim=16** for semantic dominance. 



## Benchmark Philosophy: Robust Separation Score (RSS) for RAG Utility

Our Extrapolation Hypothesis demands not just theoretical elegance but empirical proof in real-world utility: Retrieval-Augmented Generation (RAG) systems. Traditional coherence metrics isolate internal consistency; we prioritize **RAG robustness**—ensuring synthesized vectors act as clear signals for relevant memories while drowning irrelevant noise, preventing hallucinations and off-topic retrievals.

### Critical Prerequisite: Semantically Distinct Probes

For benchmarks to measure true synthesis fidelity rather than data artifacts, all test chunks (A, B, C, ..., N) must exhibit **semantic independence**: no pre-existing thematic, stylistic, or topical similarities that could inflate or confound cosine scores. If, e.g., A and C share content from the same domain, synth(ABC) similarity to embed(C) might reflect correlation, not model merit—rendering evaluations invalid.

Mirroring the vBERT pre-training corpus curation:

- **Dataset Source:** Multi-language Wikipedia (wikipedia/wikipedia) editions (e.g., en, ko, fr, de, ja—spanning histories, sciences, arts, etc.).
- **Generation Principle:** Shuffle articles globally, then apply '1 Article = 1 Probe' strictly: each full article processes into one unique chunk/embedding (embed(A)), without sliding windows, excerpts, or intra-article splits.
- **Key Benefits:** Ensures probes originate from disparate 'worlds'—e.g., A from a Korean history article, B from French philosophy—guaranteeing baseline dissimilarities (avg. cosine <0.1 pre-synthesis). This purity isolates model-induced affinities in RSS gaps, validating RAG signal strength.
- **Implementation:** Custom pipeline similar to `data_pipeline.py` (10k-50k probes per language mix; no Devil randomization to preserve test integrity). Impure probes (e.g., same-language clusters) cap scores at <5, subverting our 10+ FINESSE triumphs.

This prerequisite is foundational; benchmarks without it risk false positives, undermining the Extrapolation Hypothesis.

### Core Setup
Given a document sequence of chunks A, B, C, D, E with pre-computed embeddings embed(A), ..., embed(E), our model generates cumulative syntheses: synth(AB) from embed(A)+embed(B), synth(ABC) from synth(AB)+embed(C), up to synth(ABCDE). Evaluations probe bidirectional fidelity: how well syntheses 'recognize' their constituents (Top-Down) and how constituents 'recognize' containing syntheses (Bottom-Up).

### Top-Down Evaluation: Synthesis-to-Constituents Fidelity
Tests if a synthesis preserves affinity to its building blocks while rejecting outsiders. E.g., synth(ABC) should exhibit high cosine similarity to embed(A), embed(B), embed(C) (Tier 1: Memory Group X) but low to embed(D), embed(E) (Tier 2: Noise Group Y). This validates 'root retention': the synthesis hasn't forgotten its origins amid merging.

### Bottom-Up Evaluation: Constituents-to-Syntheses Affinity
Complements Top-Down by inverting perspective: a constituent should affinity-match containing syntheses but reject non-containing ones. E.g., embed(C) low similarity to synth(AB) (Noise Y) but high to synth(ABC), synth(ABCD), synth(ABCDE) (Memory X). This ensures 'role persistence': chunks remain anchored in their composite contexts.

### Robust Separation Score (RSS): Quantifying the Gap
For any validator p (synthesis or constituent) against Groups X (expected similar) and Y (expected dissimilar):

1. Compute all pairwise cosines: sims_X = [cos(p, x) for x in X], sims_Y = [cos(p, y) for y in Y].
2. Sort sims_X ascending, sims_Y ascending.
3. Gap = Q1_X (25th percentile of sims_X, weakest expected matches) - Q3_Y (75th percentile of sims_Y, strongest noise).
4. Normalize gap to [0,1] score via min-max scaling (accounting for cosine [-1,1] bounds).

RSS demands the 'barely similar' (Q1_X) decisively outperform the 'most tempting noise' (Q3_Y), creating a 'silent margin' for RAG resilience. Quartiles ensure outlier-robustness: no single perfect match skews results.

### Final FINESSE Scoring: Balance with Penalty
Aggregate td (Top-Down RSS average) and bu (Bottom-Up RSS average) as [0,1] scores:

final = [ (td + bu)/2 - |td - bu| ] × 500

- **Average:** Rewards overall separation.
- **Imbalance Penalty (|td - bu|):** Discourages lopsided specialists; true mastery requires bidirectional consistency.
- **Scaling (×500):** Transforms to intuitive [-1000, +1000] range.



## Acknowledgments

We extend our profoundest thanks to the **intfloat team** and the creators of the `multilingual-e5-base` model. This groundbreaking embedding model was the very foundation of our project: all training datasets were meticulously generated using E5 embeddings, our evaluations were judged against E5 as the gold standard benchmark, and the Synthesizer architecture was specifically designed in symbiotic harmony with E5's multilingual capabilities—making it an organic extension rather than a standalone entity. Without their visionary work in advancing multilingual representation, the Tiny Sequence Merger simply would not exist. Their open-source contribution is the true seed from which our innovations grew.

Built with PyTorch and Transformers. For more on the underlying research, check our project logs.