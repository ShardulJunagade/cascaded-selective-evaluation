# Phase 1: raw source inspection
Statistics of each source **as published**, before any filtering, pairing or randomization by this pipeline. Produced by `python -m prepare_data.inspect_raw`.

## Overview
| Source | Raw rows | Download | Size | Time | Annotator ids |
|---|---|---|---|---|---|
| rlhf_v | 5733 | ok | 838.7 MB | 160s | no |
| visit_bench | 5011 | ok | 5.5 MB | 4s | no |
| visionarena | 29849 | ok | 14.3 GB | 10698s | yes |
| mm_rlhf | 16342 | ok | 27.9 GB | 10972s | no |

## rlhf_v
`openbmb/RLHF-V-Dataset`

```json
{
  "rows": 5733,
  "fields": [
    "ds_name",
    "image",
    "text",
    "origin_dataset",
    "origin_split",
    "idx",
    "image_path"
  ],
  "label_values": {
    "always chosen (correction)": 5733
  },
  "winner_position_balance": "n/a - chosen side is not positioned in the raw data",
  "annotator_ids": false,
  "annotations_per_item": 1,
  "length_chosen": {
    "min": 0.0,
    "p25": 148.0,
    "median": 301.0,
    "p75": 525.0,
    "max": 2156.0
  },
  "length_rejected": {
    "min": 3.0,
    "p25": 116.0,
    "median": 260.0,
    "p75": 512.0,
    "max": 2125.0
  },
  "p_longer_is_preferred": 0.7017,
  "origin_datasets": [
    [
      "coco",
      2616
    ],
    [
      "vqav2",
      831
    ],
    [
      "LCS-558K",
      326
    ],
    [
      "sharegpt4v-textvqa",
      68
    ],
    [
      "sharegpt4v-wikiart",
      61
    ],
    [
      "sharegpt4v-web-celebrity",
      53
    ],
    [
      "sharegpt4v-web-landmark",
      45
    ]
  ],
  "scanned": 4000,
  "sampling": "seeded random sample"
}
```

## visit_bench
`https://raw.githubusercontent.com/mlfoundations/VisIT-Bench/main/visit_bench_human_preferences.csv`

```json
{
  "rows": 5011,
  "fields": [
    "image_url",
    "instruction",
    "A",
    "B",
    "A_model",
    "B_model",
    "sel_a",
    "sel_b"
  ],
  "unique_tuples": 1010,
  "judgments_per_tuple": {
    "2": 1,
    "4": 36,
    "5": 973
  },
  "unique_images": 466,
  "unique_instructions": 443,
  "label_values": {
    "A selected": 2471,
    "B selected": 2540
  },
  "winner_position_balance": 0.4931,
  "annotator_ids": false,
  "models": {
    "LlamaAdapter-v2 prediction": 796,
    "panda_gpt_13b_output": 773,
    "llava13b_output": 750,
    "human_verified_reference": 698,
    "mPLUG-Owl prediction": 683,
    "MiniGPT-4 prediction": 680,
    "instruct_blip_output": 631
  },
  "length_a": {
    "min": 2.0,
    "p25": 201.0,
    "median": 414.0,
    "p75": 607.0,
    "max": 2326.0
  },
  "length_b": {
    "min": 2.0,
    "p25": 176.0,
    "median": 407.0,
    "p75": 602.0,
    "max": 1963.0,
    "missing": 5
  },
  "p_longer_is_preferred": 0.5158,
  "images_are_urls": true
}
```

## visionarena
`lmarena-ai/VisionArena-Battle`

```json
{
  "rows": 29849,
  "fields": [
    "judge",
    "winner",
    "num_turns",
    "language",
    "categories",
    "images",
    "question_id",
    "model_a",
    "model_b",
    "conversation_a",
    "conversation_b",
    "tstamp",
    "conv_metadata"
  ],
  "scanned": 29849,
  "label_values": {
    "model_b": 9354,
    "tie": 5216,
    "model_a": 9374,
    "tie (bothbad)": 5905
  },
  "winner_position_balance": 0.5005,
  "tie_fraction": 0.3726,
  "annotator_ids": true,
  "unique_voters": 13655,
  "mean_votes_per_voter": 2.186,
  "votes_per_voter": {
    "1": 8447,
    "2": 2474,
    "3": 1030,
    "4": 542,
    "5": 315,
    "6": 186,
    "7": 146,
    "8": 105,
    "9": 82,
    "10": 56
  },
  "voters_with_at_least_k_plus_1": {
    "1": 5208,
    "2": 2734,
    "3": 1704,
    "5": 847
  },
  "turns_per_conversation": {
    "1": 25713,
    "2": 2624,
    "3": 813,
    "4": 314,
    "5": 165,
    "6": 79,
    "7": 39,
    "8": 30
  },
  "single_turn": 25713,
  "languages": [
    [
      "English",
      18483
    ],
    [
      "Russian",
      3211
    ],
    [
      "Chinese",
      2326
    ],
    [
      "Vietnamese",
      1320
    ],
    [
      "unknown",
      633
    ],
    [
      "Spanish",
      615
    ],
    [
      "German",
      475
    ],
    [
      "Japanese",
      438
    ],
    [
      "Portuguese",
      399
    ],
    [
      "Korean",
      312
    ],
    [
      "French",
      303
    ],
    [
      "Italian",
      142
    ]
  ],
  "categories": [
    [
      "ocr",
      4758
    ],
    [
      "none",
      4164
    ],
    [
      "diagram,ocr",
      2578
    ],
    [
      "homework,ocr",
      2485
    ],
    [
      "captioning",
      2148
    ],
    [
      "diagram,homework,ocr",
      1341
    ],
    [
      "humor,ocr",
      1060
    ],
    [
      "captioning,ocr",
      1016
    ],
    [
      "entity_recognition",
      952
    ],
    [
      "is_code,ocr",
      857
    ],
    [
      "creative_writing",
      744
    ],
    [
      "diagram",
      722
    ]
  ]
}
```

## mm_rlhf
`yifanzhang114/MM-RLHF`

```json
{
  "rows": 16342,
  "fields": [
    "id",
    "question",
    "answer",
    "image",
    "video",
    "models_output",
    "faithfulness",
    "helpfulness",
    "ethical",
    "final_ranking",
    "score_reasons",
    "ranking_reason"
  ],
  "scanned": 4000,
  "sampling": "seeded random sample",
  "with_image": 3333,
  "with_video": 667,
  "subset_distribution": [
    [
      "short",
      1602
    ],
    [
      "long",
      1148
    ],
    [
      "<no image>",
      667
    ],
    [
      "mcq",
      335
    ],
    [
      "safety",
      248
    ]
  ],
  "outputs_per_prompt": {
    "3": 616,
    "4": 3352,
    "5": 32
  },
  "label_values": {
    "ranking, no pairwise label": 16342
  },
  "annotator_ids": false,
  "image_archives": [
    [
      "long.zip",
      "14.0 GB"
    ],
    [
      "mcq.zip",
      "998.3 MB"
    ],
    [
      "safety.zip",
      "300.4 MB"
    ],
    [
      "short.zip",
      "4.1 GB"
    ],
    [
      "video.zip",
      "8.3 GB"
    ]
  ]
}
```
