# Processed sample bill 3

A generated Indian-style bill that was never used for training, validation or testing (`final_017.jpg` in `examples/final_run`). The photo went through the real app and model.

| Step | File |
|---|---|
| 1. Raw photo | `1_photo.jpg` |
| 2. What the model extracted, with the fields the app highlights for review | `2_extracted.json` |
| 3. Corrections a user makes on the check screen | `3_corrections.json` |
| 4. The corrected bill in the problem statement's JSON schema | `4_corrected_bill_problem_statement_schema.json` |
| 5. The split (who had what, every charge divided per person) | `5_split.json` |
| 6. The plain-language explanation | `6_explanation.txt` |

## Corrections

- none needed: the reading was correct.

## Final split

Three people share the bill: the first item three ways, the second as two portions for Riya and one for Aman, the last between Aman and Sara, the rest one person each. Amounts are in rupees; every non-item line is divided in proportion to each person's pre-tax subtotal.

| | Riya | Aman | Sara | Bill |
|---|---:|---:|---:|---:|
| Poha | 20.00 | 20.00 | 20.00 | 60.00 |
| Mutton Curry | 426.67 | 213.33 | 0.00 | 640.00 |
| Paneer Butter Masala | 0.00 | 0.00 | 260.00 | 260.00 |
| Paratha | 110.00 | 0.00 | 0.00 | 110.00 |
| Chicken Biryani | 0.00 | 200.50 | 0.00 | 200.50 |
| Idli Sambar | 0.00 | 67.50 | 67.50 | 135.00 |
| Service Charge | 27.83 | 25.07 | 17.38 | 70.28 |
| GST | 27.83 | 25.07 | 17.38 | 70.28 |
| **Total (Rs)** | **612.33** | **551.47** | **382.26** | **1546.06** |

## Explanation for Sara

> Sara: Poha (1/3 share, Rs 20.00) + Paneer Butter Masala (full, Rs 260.00) + Idli Sambar (1/2 share, Rs 67.50) = Rs 347.50 pre-tax subtotal. Sara's pre-tax subtotal is 24.7% of the Rs 1405.50 of items on the bill, so Sara carries 24.7% of each tax, service charge, discount and rounding line. Charges and discounts are shared in proportion to each person's pre-tax subtotal, not equally: Service Charge Rs 17.38, GST Rs 17.38; net Rs 34.76 (+10.0% of Sara's subtotal). Final: Rs 382.26.
