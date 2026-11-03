# Confidence review: full-data model at 1280

CPU analysis of 423,242 saved predictions on all 548 validation images. The model
SHA-256 and validation-manifest SHA-256 match the supplied comparison report.
No new inference or training was performed. The checkpoint is
`9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840`.

This is a fixed-operating-point diagnostic: score-ordered, same-class greedy
matching at IoU >= 0.5, with the existing ignored-region overlap approximation.
These numbers are not COCO AP or official VisDrone scores. Thresholds 0.05 through
0.80 were evaluated at increments of 0.01. Test-dev remains untouched.

| Confidence | Precision | Recall | Micro F1 | Macro F1 |
|---|---:|---:|---:|---:|
| 0.20 | 59.43% | 58.05% | 58.73% | 43.14% |
| 0.25 | 66.17% | 54.62% | 59.84% | 42.73% |
| 0.27 | 68.65% | 53.34% | 60.03% | 42.56% |
| 0.30 | 71.92% | 51.10% | 59.75% | 41.66% |
| 0.40 | 80.99% | 44.09% | 57.10% | 37.07% |

The aggregate micro-F1 maximum is 0.27; the equal-weight per-class macro-F1
maximum is 0.20. Micro F1 gives more influence to common classes, particularly
cars. Differences around the aggregate optimum are small and have no measured
uncertainty; do not present 0.27 as a universal optimal value.

A provisional demo default of 0.30 trades a little recall for precision while
staying close to the observed micro-F1 maximum. Confidence 0.20 is a useful
recall-oriented alternative but introduces substantially more false positives.
Per-class performance and use-case requirements must guide any deployment choice.

The AP comparison still uses confidence 0.001 to retain the low-confidence tail
for precision-recall integration. A demo threshold is a separate setting.

Next: verify exported-model prediction parity, freeze the chosen configuration,
and evaluate the final candidate on held-out test-dev with clearly documented
scoring. The present validation precision/recall should not be promised on new data.
