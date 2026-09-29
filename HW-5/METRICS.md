# METRICS — DATA 266 HW5

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. Two trained models: LoRA r=4 and LoRA r=16.

From `lora_dialogsum.ipynb` on kernel Python (HW1).

## Setup

| Field | Value |
|-------|-------|
| Dataset | `neil-code/dialogsum-test` |
| Train subset | 256 rows, shuffle seed 670 |
| Held-out | validation `dev_56`, `dev_357` |
| Model | `google/flan-t5-small` |
| Prompt | `summarize the dialogue:` + dialogue |
| Target | summary |
| Epochs | 2 |
| Batch | 4 |
| lr | 1e-3 |
| Max input / target tokens | 256 / 64 |
| LoRA targets | q, v |
| alpha | 16 |
| dropout | 0.05 |
| Device | mps |

## Parameters

| Model | Total | Trainable | Trainable % |
|-------|------:|----------:|------------:|
| Baseline (no LoRA) | 76,961,152 | 76,961,152 | 100 |
| LoRA r=4 | 77,133,184 | 172,032 | 0.223 |
| LoRA r=16 | 77,649,280 | 688,128 | 0.886 |

## Loss

Trainer mean is over all steps. Last number is step 128.

| Rank | Mean train loss | Step 8 | Step 128 |
|------|----------------:|-------:|---------:|
| r=4 | 1.796 | 2.464 | 1.511 |
| r=16 | 1.793 | 2.459 | 1.498 |

Plot: `figures/train_loss.png`

## Same two dialogues

Overlap is unigram F1 against the dataset summary. I counted it in the notebook. Not the rouge package.

| | dev_56 | dev_357 | Mean |
|--|-------:|--------:|-----:|
| Baseline | 0.194 | 0.000 | 0.097 |
| r=4 | 0.222 | 0.286 | 0.254 |
| r=16 | 0.143 | 0.375 | 0.259 |

### dev_56

Reference: #Person1# and #Person2# talk about Li Na who is pressed by her mother for marriage.

| | Output |
|--|--------|
| Baseline | Li Na's mother has been building a fire under her since her neighbour's daughter got married. |
| r=4 | Li Na's mother has been building a fire under her neighbour's daughter. |
| r=16 | Li Na's mother has been building a fire under Li Na's neighbour's daughter. |

### dev_357

Reference: Alice is over an hour late for the appointment with Adam. She explains the reason for her lateness and apologizes.

| | Output |
|--|--------|
| Baseline | Adam, I'm sorry. I'm sorry. |
| r=4 | Alice's late getting off work for a start and then she misses the bus. Alice's boss asks Alice to do urgent letters. |
| r=16 | Adam is over an hour late. Alice is late getting off work for a start and then misses the bus. #Person1#'s boss asks Alice to do urgent letters. |
