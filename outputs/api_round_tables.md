# Results tables

Generated 2026-11-10 14:50 from `outputs/`. Target: at least 85% agreement with the human label on the pairs a method answers, with probability at least 90% (α = 0.15, δ = 0.1). Every number is a percentage averaged over 1000 random calibration/test splits.

- **Accuracy**: agreement with the human label on the answered pairs, averaged over splits that answered anything. **Macro** averages the five data sources equally.
- **Coverage**: share of pairs answered. Splits that answered nothing count as 0.
- **Answered**: share of splits that answered at least one pair.
- **Success**: share of splits that answered something *and* met the accuracy target. Splits that abstained on everything count as failures here.
- **Kept**: share of splits where the guarantee held = Success + (100 − Answered). Abstaining is safe, so this is the number to compare with 90%.
- **Calls**: mean number of judges queried per pair. **Certified**: share of splits where a routing threshold was certified.

## Overview

| Run | Trust or Escalate: cover | acc | kept | Router (raw): cover | acc | kept | Router (learned): cover | acc | kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1: Kimi K2.5 alone | 68.4 | 89.3 | 90.5 | 70.0 | 88.9 | 89.1 | 69.4 | 88.9 | 88.9 |
| E2: Gemini Flash alone | 52.2 | 86.8 | 92.5 | 53.8 | 86.8 | 90.8 | 53.5 | 86.9 | 90.1 |
| E3: Kimi → Gemini | 58.0 | 91.1 | 94.8 | 53.7 | 86.7 | 90.4 | 77.0 | 88.6 | 89.5 |
| E3b: Gemini → Kimi | 66.8 | 90.4 | 93.2 | 54.6 | 86.5 | 87.6 | 72.3 | 90.0 | 88.6 |

Trust or Escalate is `cascaded_selective` (one threshold per judge, δ split across judges). The routers are the end-to-end method (one threshold for the whole cascade, certified at the full δ).

## E1: Kimi K2.5 alone

`outputs/fast_api_kimi/`

**Trust or Escalate and baselines** (`baseline_comparison.json`)

| Policy | Accuracy | Macro | Coverage | Answered | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct:kimi-k2.5` | 81.4 | 82.4 | 100.0 | 100.0 | 0.0 | 0.0 |
| `heuristic:kimi-k2.5` | 91.6 | 91.8 | 61.4 | 100.0 | 100.0 | 100.0 |
| `cascaded_heuristic` | 91.6 | 91.8 | 61.4 | 100.0 | 100.0 | 100.0 |
| `point_estimate:kimi-k2.5` | 85.2 | 85.8 | 86.4 | 100.0 | 55.3 | 55.3 |
| `selective:kimi-k2.5` | 89.3 | 89.5 | 68.4 | 99.8 | 90.3 | 90.5 |
| **cascaded_selective** | 89.3 | 89.5 | 68.4 | 99.8 | 90.3 | 90.5 |
| `setup:vanilla_single:kimi-k2.5` | 77.5 | 78.8 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:position_swap:kimi-k2.5` | 81.8 | 82.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:prompt_order_vote:kimi-k2.5` | 78.4 | 79.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:swap_consistency:kimi-k2.5` | 87.1 | 87.5 | 75.6 | 100.0 | 99.6 | 99.6 |

**End-to-end router** (`global_router_comparison.json`)

| Method | Accuracy | Macro | Coverage | Calls | Answered | Certified | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **global_raw_confidence** | 88.9 | 89.2 | 70.0 | 1.00 | 99.8 | 99.8 | 88.9 | 89.1 |
| **global_learned_router** | 88.9 | 89.2 | 69.4 | 1.00 | 99.5 | 99.5 | 88.4 | 88.9 |

## E2: Gemini Flash alone

`outputs/fast_api_gemini/`

**Trust or Escalate and baselines** (`baseline_comparison.json`)

| Policy | Accuracy | Macro | Coverage | Answered | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `heuristic:gemini-flash` | 88.4 | 88.4 | 88.3 | 100.0 | 100.0 | 100.0 |
| `cascaded_heuristic` | 88.4 | 88.4 | 88.3 | 100.0 | 100.0 | 100.0 |
| `point_estimate:gemini-flash` | 85.9 | 85.9 | 94.4 | 97.3 | 54.7 | 57.4 |
| `selective:gemini-flash` | 86.8 | 87.0 | 52.2 | 56.5 | 49.0 | 92.5 |
| **cascaded_selective** | 86.8 | 87.0 | 52.2 | 56.5 | 49.0 | 92.5 |
| `setup:vanilla_single:gemini-flash` | 84.4 | 84.5 | 99.9 | 100.0 | 22.3 | 22.3 |
| `setup:position_swap:gemini-flash` | 84.3 | 84.3 | 99.8 | 100.0 | 17.4 | 17.4 |
| `setup:prompt_order_vote:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `setup:swap_consistency:gemini-flash` | 86.7 | 86.8 | 94.5 | 100.0 | 99.1 | 99.1 |

**End-to-end router** (`global_router_comparison.json`)

| Method | Accuracy | Macro | Coverage | Calls | Answered | Certified | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **global_raw_confidence** | 86.8 | 86.9 | 53.8 | 1.00 | 58.1 | 58.1 | 48.9 | 90.8 |
| **global_learned_router** | 86.9 | 87.0 | 53.5 | 1.00 | 59.5 | 59.5 | 49.6 | 90.1 |

## E3: Kimi → Gemini

`outputs/fast_api_kimi_gemini/`

**Trust or Escalate and baselines** (`baseline_comparison.json`)

| Policy | Accuracy | Macro | Coverage | Answered | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct:kimi-k2.5` | 81.4 | 82.4 | 100.0 | 100.0 | 0.0 | 0.0 |
| `direct:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `heuristic:gemini-flash` | 88.4 | 88.4 | 88.3 | 100.0 | 100.0 | 100.0 |
| `cascaded_heuristic` | 87.6 | 87.5 | 92.0 | 100.0 | 100.0 | 100.0 |
| `point_estimate:kimi-k2.5` | 85.2 | 85.8 | 86.4 | 100.0 | 55.3 | 55.3 |
| `point_estimate:gemini-flash` | 85.9 | 85.9 | 94.4 | 97.3 | 54.7 | 57.4 |
| `selective:kimi-k2.5` | 89.3 | 89.5 | 68.4 | 99.8 | 90.3 | 90.5 |
| `selective:gemini-flash` | 86.8 | 87.0 | 52.2 | 56.5 | 49.0 | 92.5 |
| **cascaded_selective** | 91.1 | 91.3 | 58.0 | 99.8 | 94.6 | 94.8 |
| `setup:vanilla_single:kimi-k2.5` | 77.5 | 78.8 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:position_swap:kimi-k2.5` | 81.8 | 82.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:prompt_order_vote:kimi-k2.5` | 78.4 | 79.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:swap_consistency:kimi-k2.5` | 87.1 | 87.5 | 75.6 | 100.0 | 99.6 | 99.6 |
| `setup:vanilla_single:gemini-flash` | 84.4 | 84.5 | 99.9 | 100.0 | 22.3 | 22.3 |
| `setup:position_swap:gemini-flash` | 84.3 | 84.3 | 99.8 | 100.0 | 17.4 | 17.4 |
| `setup:prompt_order_vote:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `setup:swap_consistency:gemini-flash` | 86.7 | 86.8 | 94.5 | 100.0 | 99.1 | 99.1 |
| `setup:consistency_escalation` | 85.0 | 85.1 | 100.0 | 100.0 | 50.8 | 50.8 |

**End-to-end router** (`global_router_comparison.json`)

| Method | Accuracy | Macro | Coverage | Calls | Answered | Certified | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **global_raw_confidence** | 86.7 | 86.7 | 53.7 | 1.69 | 58.1 | 58.1 | 48.5 | 90.4 |
| **global_learned_router** | 88.6 | 88.9 | 77.0 | 1.41 | 94.6 | 94.6 | 84.1 | 89.5 |

## E3b: Gemini → Kimi

`outputs/fast_api_gemini_kimi/`

**Trust or Escalate and baselines** (`baseline_comparison.json`)

| Policy | Accuracy | Macro | Coverage | Answered | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `direct:kimi-k2.5` | 81.4 | 82.4 | 100.0 | 100.0 | 0.0 | 0.0 |
| `heuristic:kimi-k2.5` | 91.6 | 91.8 | 61.4 | 100.0 | 100.0 | 100.0 |
| `cascaded_heuristic` | 87.6 | 87.7 | 92.0 | 100.0 | 100.0 | 100.0 |
| `point_estimate:gemini-flash` | 85.9 | 85.9 | 94.4 | 97.3 | 54.7 | 57.4 |
| `point_estimate:kimi-k2.5` | 85.2 | 85.8 | 86.4 | 100.0 | 55.3 | 55.3 |
| `selective:gemini-flash` | 86.8 | 87.0 | 52.2 | 56.5 | 49.0 | 92.5 |
| `selective:kimi-k2.5` | 89.3 | 89.5 | 68.4 | 99.8 | 90.3 | 90.5 |
| **cascaded_selective** | 90.4 | 90.5 | 66.8 | 99.8 | 93.0 | 93.2 |
| `setup:vanilla_single:gemini-flash` | 84.4 | 84.5 | 99.9 | 100.0 | 22.3 | 22.3 |
| `setup:position_swap:gemini-flash` | 84.3 | 84.3 | 99.8 | 100.0 | 17.4 | 17.4 |
| `setup:prompt_order_vote:gemini-flash` | 84.9 | 84.9 | 100.0 | 100.0 | 49.4 | 49.4 |
| `setup:swap_consistency:gemini-flash` | 86.7 | 86.8 | 94.5 | 100.0 | 99.1 | 99.1 |
| `setup:vanilla_single:kimi-k2.5` | 77.5 | 78.8 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:position_swap:kimi-k2.5` | 81.8 | 82.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:prompt_order_vote:kimi-k2.5` | 78.4 | 79.6 | 100.0 | 100.0 | 0.0 | 0.0 |
| `setup:swap_consistency:kimi-k2.5` | 87.1 | 87.5 | 75.6 | 100.0 | 99.6 | 99.6 |
| `setup:consistency_escalation` | 85.0 | 85.4 | 100.0 | 100.0 | 50.1 | 50.1 |

**End-to-end router** (`global_router_comparison.json`)

| Method | Accuracy | Macro | Coverage | Calls | Answered | Certified | Success | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **global_raw_confidence** | 86.5 | 86.7 | 54.6 | 1.47 | 58.1 | 58.1 | 45.7 | 87.6 |
| **global_learned_router** | 90.0 | 90.1 | 72.3 | 1.52 | 95.0 | 95.0 | 83.6 | 88.6 |
