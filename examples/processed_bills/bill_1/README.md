# Processed sample bill 1

A generated Indian-style bill that was never used for training, validation or testing (`final_001.jpg` in `examples/final_run`). The photo went through the real app and model.

| Step | File |
|---|---|
| 1. Raw photo | `1_photo.jpg` |
| 2. What the model extracted, with the fields the app highlights for review | `2_extracted.json` |
| 3. Corrections a user makes on the check screen | `3_corrections.json` |
| 4. The corrected bill in the problem statement's JSON schema | `4_corrected_bill_problem_statement_schema.json` |
| 5. The split (who had what, every charge divided per person) | `5_split.json` |
| 6. The plain-language explanation | `6_explanation.txt` |

## Corrections

- `items[5].unit_price`: read `310`, corrected to `305`

## Final split

Three people share the bill: the first item three ways, the second as two portions for Riya and one for Aman, the last between Aman and Sara, the rest one person each. Amounts are in rupees; every non-item line is divided in proportion to each person's pre-tax subtotal.

| | Riya | Aman | Sara | Bill |
|---|---:|---:|---:|---:|
| Dal Tadka | 110.34 | 110.33 | 110.33 | 331.00 |
| Egg Curry | 76.67 | 38.33 | 0.00 | 115.00 |
| Veg Sandwich | 0.00 | 0.00 | 70.00 | 70.00 |
| Filter Coffee | 45.00 | 0.00 | 0.00 | 45.00 |
| Samosa | 0.00 | 40.00 | 0.00 | 40.00 |
| Mutton Curry | 0.00 | 305.00 | 305.00 | 610.00 |
| Discount | -34.80 | -74.05 | -72.80 | -181.65 |
| Service Charge | 11.60 | 24.68 | 24.27 | 60.55 |
| CGST | 11.83 | 25.18 | 24.75 | 61.76 |
| SGST | 11.83 | 25.18 | 24.75 | 61.76 |
| **Total (Rs)** | **232.47** | **494.65** | **486.30** | **1213.42** |

## Explanation for Sara

> Sara: Dal Tadka (1/3 share, Rs 110.33) + Veg Sandwich (full, Rs 70.00) + Mutton Curry (1/2 share, Rs 305.00) = Rs 485.33 pre-tax subtotal. Sara's pre-tax subtotal is 40.1% of the Rs 1211.00 of items on the bill, so Sara carries 40.1% of each tax, service charge, discount and rounding line. Charges and discounts are shared in proportion to each person's pre-tax subtotal, not equally: Discount -Rs 72.80, Service Charge Rs 24.27, CGST Rs 24.75, SGST Rs 24.75; net Rs 0.97 (+0.2% of Sara's subtotal). Final: Rs 486.30.
