# NIAH + StreamingLLM Eviction — 16K Context

**Model:** Llama-3.1-8B-Instruct Q4_K_M
**Context:** 16384, ngl=20, sink=128
**Test:** Needle="The secret code for Project Alpha is 7492-BLUE-DELTA."

## Baseline (F16, no eviction)

| Position | Result |
|----------|--------|
| 10%      | FOUND  |
| 25%      | FOUND  |
| 50%      | FOUND  |
| 75%      | FOUND  |
| 90%      | FOUND  |
| **Accuracy** | **100%** |

## 85% StreamingLLM Eviction

| Position | Result | Model Answer |
|----------|--------|-------------|
| 10%      | MISS   | "There is no mention of a secret code for Project Alpha" |
| 25%      | MISS   | "There is no mention of a secret code for Project Alpha" |
| 50%      | MISS   | "There is no mention of a secret code for Project Alpha" |
| 75%      | MISS   | "There is no mention of a secret code for Project Alpha" |
| 90%      | FOUND  | "The secret code for Project Alpha is 7492-BLUE-DELTA." |
| **Accuracy** | **20%** |

## Analysis

StreamingLLM eviction removes tokens by **position** (keep first `sink` + most recent).
At 85% eviction with 16K context:
- Effective cache = 2457 tokens (128 sink + ~2329 recent)
- Only needle at 90% position survives (within recent window)
- All earlier positions (10-75%) are evicted regardless of attention importance

**Conclusion:** StreamingLLM position-based eviction destroys retrieval for non-recent information.
For high-eviction NIAH to work, need **attention-aware eviction** (H2O, Scissorhands, etc.)
that keeps high-attention tokens regardless of position.

## Bug Fix Note

Original NIAH script had a false-positive bug: detection checked `full_output` (including
prompt echo containing the needle). Fixed to check `answer` text only.
