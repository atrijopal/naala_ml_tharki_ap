# Confidence calibration (R2)

Data: 4107 predicted fields from 198 validation bills (CORD + synthetic); 97.4% of fields are correct.

| Aggregator | AUROC | ECE raw | ECE calibrated (2-fold CV) | flag if score below | fields flagged | errors caught | precision of unflagged |
|---|---|---|---|---|---|---|---|
| p_first | 0.8709 | 0.0149 | 0.0053 | 0.992 | 8.0% | 60.0% | 98.9% |
| p_mean | 0.8818 | 0.0178 | 0.0059 | 0.9916 | 8.0% | 63.8% | 99.0% |
| p_min **(best)** | 0.8879 | 0.0123 | 0.0054 | 0.9884 | 8.0% | 67.6% | 99.1% |

Operating point: flag the lowest-scoring 8% of fields.

Trade-off with the best aggregator (flag more fields = catch more errors, more taps for the user):

| fields flagged | flag if score below | real errors caught | unflagged fields correct | flags on a 20-field bill |
|---|---|---|---|---|
| 1% | 0.5466 | 26% | 98.1% | 0.2 |
| 2% | 0.7372 | 38% | 98.4% | 0.4 |
| 5% | 0.9596 | 59% | 98.9% | 1.0 |
| 8% | 0.9884 | 68% | 99.1% | 1.6 |
| 10% | 0.993 | 69% | 99.1% | 2.0 |
| 15% | 0.9976 | 74% | 99.2% | 3.0 |
| 20% | 0.9991 | 79% | 99.3% | 4.0 |

Reading: AUROC 0.5 would mean confidence says nothing about errors; 1.0 means every wrong field scores below every right one. With `p_min` and the flag threshold 0.9884, the app highlights 8.0% of fields and thereby catches 67.6% of the real errors; the fields it does not flag are 99.1% correct.

By field type: amount (n=293, acc 99%, AUROC 0.9853), grand_total (n=197, acc 99%, AUROC 0.9949), name (n=782, acc 96%, AUROC 0.9003), qty (n=782, acc 99%, AUROC 0.7), subtotal (n=198, acc 98%, AUROC 0.9214), total (n=782, acc 99%, AUROC 0.8606), type (n=292, acc 99%, AUROC 0.872), unit_price (n=781, acc 94%, AUROC 0.9218)
