# Processed sample bill 2

A generated Indian-style bill that was never used for training, validation or testing (`final_016.jpg` in `examples/final_run`). The photo went through the real app and model.

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
| Fresh Lime Soda | 28.50 | 28.50 | 28.50 | 85.50 |
| Dal Tadka | 173.33 | 86.67 | 0.00 | 260.00 |
| Veg Biryani | 0.00 | 102.75 | 102.75 | 205.50 |
| Discount | -30.27 | -32.69 | -19.69 | -82.65 |
| Service Charge | 20.18 | 21.79 | 13.13 | 55.10 |
| CGST | 4.29 | 4.63 | 2.79 | 11.71 |
| SGST | 4.29 | 4.63 | 2.79 | 11.71 |
| Round Off | 0.05 | 0.05 | 0.03 | 0.13 |
| **Total (Rs)** | **200.37** | **216.33** | **130.30** | **547.00** |

## Explanation for Sara

> Sara: Fresh Lime Soda (1/3 share, Rs 28.50) + Veg Biryani (1/2 share, Rs 102.75) = Rs 131.25 pre-tax subtotal. Sara's pre-tax subtotal is 23.8% of the Rs 551.00 of items on the bill, so Sara carries 23.8% of each tax, service charge, discount and rounding line. Charges and discounts are shared in proportion to each person's pre-tax subtotal, not equally: Discount -Rs 19.69, Service Charge Rs 13.13, CGST Rs 2.79, SGST Rs 2.79, Round Off Rs 0.03; net -Rs 0.95 (-0.7% of Sara's subtotal). Final: Rs 130.30.
