# LLM for Materials Discovery

This repository is a hands-on learning project for understanding the end-to-end lifecycle of modern LLM systems through a small scientific/materials-discovery themed project.

The goal is not to build a production-grade materials discovery model. The goal is to build a small, interpretable, end-to-end “LLM lab” that helps me understand:

* tiny LLM pretraining from scratch
* continued pretraining / domain adaptation
* supervised fine-tuning
* preference alignment with DPO / RLHF-style methods
* scientific data preparation and evaluation
* inference internals such as KV cache
* model acceleration techniques
* deployment and serving benchmarks

The project theme is **materials discovery**, inspired by scientific and industrial use cases such as materials literature understanding, material-property reasoning, semiconductor/manufacturing workflows, and trustworthy scientific hypothesis generation.

The core philosophy is:

> Do not teach the model to invent science.
> Teach it to reason from provided scientific evidence.

---

## Course Roadmap

### Module 0 — Project Setup and Learning Contract

Purpose: define the project scope, learning goals, and experiment style.

In this module, I set up the repository gradually instead of over-engineering the structure upfront.

Learning goals:

* Understand the full LLM lifecycle at a high level.
* Define what this project is and is not.
* Keep the project cost-conscious and learning-oriented.
* Use the repo as both codebase and lab notebook.

Deliverables:

* README roadmap
* experiment log template
* cost log template
* first minimal environment setup

---

### Module 1 — Tiny GPT from Scratch

Purpose: understand pretraining mechanics from first principles.

In this module, I will build or study a very small decoder-only Transformer and train it on a tiny text corpus.

Learning goals:

* tokenization
* embeddings
* positional encoding
* causal self-attention
* Transformer blocks
* next-token prediction loss
* training loop
* sampling and generation
* loss curves and checkpoints

Expected output:

* a tiny GPT-like model trained from random initialization
* sample generations over time
* notes on what pretraining actually teaches

Important limitation:

This model is not expected to be useful. It is only for understanding the mechanics of pretraining.

---

### Module 2 — Scientific and Materials Data Pipeline

Purpose: learn how scientific data becomes model-training data.

The project will use a hybrid materials-science dataset:

1. unstructured scientific text, such as materials or semiconductor-related abstracts
2. structured material records, such as formula, crystal system, band gap, formation energy, and stability
3. generated instruction data for supervised fine-tuning
4. preference pairs for alignment

Learning goals:

* scientific data cleaning
* JSONL dataset design
* train/validation/test splits
* converting structured records into natural-language examples
* creating instruction-following data
* creating grounded vs hallucinated preference pairs

Expected datasets:

* raw scientific text dataset
* structured material record dataset
* SFT instruction dataset
* DPO preference dataset
* small held-out evaluation set

---

### Module 3 — Continued Pretraining / Domain Adaptation

Purpose: replace expensive real pretraining with a budget-friendly pretraining-like experiment.

Instead of pretraining a useful LLM from scratch, I will take a small pretrained model and continue training it on materials/scientific text.

Learning goals:

* causal language modeling on domain text
* full-weight fine-tuning
* domain adaptation
* validation perplexity
* checkpointing
* learning-rate sensitivity
* catastrophic forgetting
* GPU memory usage during training

Expected output:

* base model vs domain-adapted model comparison
* training and validation loss curves
* generated samples before and after domain adaptation

Key idea:

> Pretraining from scratch teaches a model from random weights.
> Continued pretraining teaches an existing model to adapt toward a domain.

---

### Module 4 — Supervised Fine-Tuning

Purpose: teach the model to follow scientific/materials-related instructions.

The model will be fine-tuned on tasks such as:

* summarize a scientific abstract
* extract material / method / property / application
* answer questions from a structured material record
* explain whether a material is likely metallic, semiconducting, or insulating
* generate cautious, evidence-based hypotheses

Learning goals:

* instruction formatting
* chat templates
* response-only loss masking
* LoRA vs full fine-tuning
* overfitting on small instruction datasets
* behavior change after SFT

Expected output:

* SFT model
* before/after examples
* small task-level evaluation

Key idea:

> Continued pretraining changes domain familiarity.
> SFT changes interaction behavior.

---

### Module 5 — Preference Alignment with DPO

Purpose: understand post-training alignment in a small, practical setting.

The model will be trained on preference pairs:

* chosen answer: grounded, cautious, evidence-based
* rejected answer: hallucinated, overconfident, vague, or unsupported

Learning goals:

* preference data format
* chosen vs rejected responses
* reference model
* DPO objective
* alignment as behavior shaping
* tradeoff between caution and usefulness

Expected output:

* DPO-aligned model
* comparison between SFT and DPO outputs
* examples where DPO reduces hallucination
* examples where alignment may overcorrect

Key idea:

> Alignment is not mainly about adding knowledge.
> It changes which answers the model prefers to produce.

---

### Module 6 — Scientific Evaluation

Purpose: evaluate whether the model is useful and trustworthy in a scientific setting.

Evaluation should focus on grounding, factual consistency, and uncertainty.

Possible evaluations:

* factual QA from structured material records
* material-property consistency checks
* extraction accuracy for material/property/method triples
* summarization faithfulness
* hallucination detection
* uncertainty behavior when evidence is missing

Learning goals:

* task-specific LLM evaluation
* why generic text quality is not enough
* how to use structured records as ground truth
* how to compare base, continued-pretrained, SFT, and DPO models

Expected output:

* evaluation script
* small benchmark report
* model comparison table

Key idea:

> Scientific LLMs should be evaluated by whether they stay faithful to evidence, not just whether they sound fluent.

---

### Module 7 — Inference Internals and KV Cache

Purpose: understand why LLM inference is different from ordinary neural network inference.

This module studies generation in two ways:

1. naive generation: recompute the full sequence every step
2. KV-cache generation: reuse previous attention keys and values

Learning goals:

* autoregressive decoding
* prefill vs decode
* KV cache
* sequence length effects
* batch size effects
* tokens per second
* time per output token
* GPU memory usage

Expected output:

* naive vs KV-cache benchmark
* explanation of prefill/decode behavior
* notes on why long-context inference is memory-heavy

Key idea:

> KV cache trades memory for speed.
> It avoids recomputing the past during generation.

---

### Module 8 — Acceleration Lab

Purpose: build practical intuition for GPU acceleration and model efficiency.

Experiments may include:

* CPU vs GPU matrix multiplication
* FP32 vs FP16/BF16
* naive attention vs optimized scaled-dot-product attention
* quantization: FP16 vs 8-bit vs 4-bit
* PyTorch profiler
* torch.compile
* one small Triton kernel
* optional raw CUDA vector-add kernel

Learning goals:

* CUDA mental model
* GPU memory hierarchy
* kernel launch overhead
* compute-bound vs memory-bound operations
* attention bottlenecks
* quantization tradeoffs
* profiling training and inference

Expected output:

* benchmark tables
* profiler notes
* explanation of what made the model faster or slower

Key idea:

> Acceleration is not one trick.
> It is the process of finding the bottleneck and choosing the right optimization.

---

### Module 9 — Serving and Deployment

Purpose: turn the model into a small service and measure serving behavior.

The first version should use a simple API server. Later, the project can compare that with an LLM serving engine such as vLLM.

Learning goals:

* FastAPI model serving
* request/response format
* streaming generation
* batching
* concurrent request testing
* latency vs throughput
* TTFT: time to first token
* TPOT: time per output token
* cost-aware EC2 deployment

Expected output:

* local API endpoint
* optional EC2 deployment
* simple load test
* latency/throughput benchmark
* cost log

Key idea:

> Model serving is not just calling `model.generate()`.
> It is a systems problem involving latency, throughput, memory, batching, and cost.

---

## Suggested Build Order

The project should be built step by step:

1. README and learning contract
2. tiny GPT from scratch
3. scientific/materials data pipeline
4. continued pretraining
5. supervised fine-tuning
6. evaluation suite
7. DPO alignment
8. KV-cache inference benchmark
9. acceleration experiments
10. FastAPI serving
11. vLLM serving benchmark
12. optional Triton/CUDA toy kernel

---

## Final Expected Outcome

By the end of this project, I should have:

* a tiny GPT trained from scratch
* a small domain-adapted scientific language model
* an SFT scientific assistant
* a DPO-aligned cautious assistant
* a small scientific evaluation suite
* inference benchmarks with and without KV cache
* acceleration benchmarks for precision, attention, quantization, and profiling
* a simple deployed model API
* a written experiment report explaining what worked, what failed, and what each stage taught

The most important artifact is not the model itself.

The most important artifact is the learning report:
what changed, why it changed, what became faster, what became more grounded, what failed, and what the toy setup does not prove.
