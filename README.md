# Soft Thinking on Qwen3: a small ablation

A training-free experiment on **soft thinking**: during the reasoning phase, the model is fed a probability-weighted mixture of token embeddings instead of a single discrete token. This repo implements the method for Qwen3 and tests on GSM8K whether the mixture changes accuracy.

The idea follows [Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space](https://arxiv.org/abs/2505.15778) (Zhang et al., 2025), including its entropy-based "Cold Stop".

## Results

**Soft thinking made no measurable difference to accuracy.** Qwen3-0.6B, 600 GSM8K test questions (shuffled with seed 0), greedy decoding.

The model is asked to put its final answer in `\boxed{}`. The answer is read from the last `\boxed{...}` if present, otherwise from `#### N` or "answer is N", otherwise from the last number in the output, and counts as correct if it matches the gold answer.

| Condition    | What is fed back during thinking                                | Accuracy | Correct |
| ------------ | --------------------------------------------------------------- | -------- | ------- |
| `hard`     | Top-1 token embedding (ordinary discrete thinking)              | 0.723    | 434/600 |
| `soft`     | Probability-weighted mix of the top-5 embeddings                | 0.723    | 434/600 |
| `norm`     | Same as`soft`, rescaled to the top-1 embedding's norm         | 0.733    | 440/600 |
| `shuffled` | Top-1 kept, the other 4 replaced by random tokens, same weights | 0.702    | 421/600 |

### Paired comparison against `hard`

All conditions answer the same questions, so each variant is compared with `hard` question by question using McNemar's exact test. Only questions where the two conditions disagree count.

| Comparison               | Only`hard` right | Only other right | p-value |
| ------------------------ | ------------------ | ---------------- | ------- |
| `hard` vs `soft`     | 51                 | 51               | 1.00    |
| `hard` vs `norm`     | 44                 | 50               | 0.61    |
| `hard` vs `shuffled` | 55                 | 42               | 0.22    |

### Conclusion

- **No variant is distinguishable from discrete thinking.** None of the comparisons comes near the 0.05 threshold.
- **Soft thinking is not a no-op.** It changed the outcome on 102 of 600 questions (17%), but fixed exactly as many answers as it broke.
- **Random tokens do about as well as the real mixture.** `shuffled` is 2.2 points below `hard`, which is within noise. One possible reading is that the model mostly relies on the top-1 token and ignores the rest of the mixture; this experiment does not test that directly.
- **This is not proof of equality.** With 600 questions the test can reliably detect gaps of about 4.5 points or more, so smaller true differences cannot be ruled out.

## Setup

- **Install:** `pip install torch transformers datasets scipy` (transformers 4.51 or later).
- **Run:** set `name` (hard, soft, shuffled, norm) and `START, END` in the script, then run it once per condition.
- **Settings:** top-k 5 (1 for `hard`), temperature 1.0, batch size 10, up to 1024 thinking and 512 answer tokens.
- **Stopping rule:** thinking ends on `</think>`, at the token limit, or after 64 consecutive steps with entropy at or below 0.1.

## Limitations

- One small model (0.6B) and one dataset. The result may not carry over to larger models or harder tasks.
- Each condition was run once; `shuffled` depends on the random seed.
- Lenient answer extraction (falling back to the last number) can credit outputs that ignored the requested format.
