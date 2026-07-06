# Echo-50M Table 4 Reproduction Attempt

Checkpoint:
/labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B/final/model.pt

Training:
- Model: Echo-50M / Koopman
- Data: FineWeb-Edu sample-10BT
- Tokens: 3B
- Seq len: 2048
- Effective batch: 96

Results:
| Metric | Value |
|---|---:|
| FW ppl ↓ | 19.31 |
| WikiText ppl ↓ | 46.26 |
| HS acc_norm ↑ | 28.30 |
| PIQA acc ↑ | 58.05 |
| ARC-E acc ↑ | 43.18 |
| ARC-C acc_norm ↑ | 23.04 |
| WG acc ↑ | 51.62 |
| LMB acc ↑ | 16.32 |

Raw outputs:
- /labs/mpsnyder/cody1212/results/echo50m_table4/fineweb_ppl.json
- /labs/mpsnyder/cody1212/results/echo50m_table4/wikitext_ppl.json
- /labs/mpsnyder/cody1212/results/echo50m_table4/zeroshot.json
