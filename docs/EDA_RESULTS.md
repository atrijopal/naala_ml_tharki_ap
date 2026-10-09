# EDA: CORD and SROIE (before fine-tuning)

Run: CPU kernel `atrijopal/splitsnap-eda` (4.8 min, no GPU). Raw output: `out/splitsnap-eda/eda/` (`eda.json`, `EDA_REPORT.md`, `eda_hist.png`, `eda_samples.png`). All numbers measured on the full datasets.

## CORD v2 (1,000 receipts: 800 / 100 / 100, Indonesian, phone photos)
| Fact | Value |
|---|---|
| Items per receipt | median 2, p95 7, max 22 (short bills) |
| Qty > 1 | 17% of items (max 202: an outlier) |
| Item price (rupiah) | median 25,000, p95 150,000, max 3.6M; min is negative (-34,363, a discount stored as an item) |
| Grand total | median 45,500, p95 447,488, **max 5.58 billion (label noise)**, min 57 |
| Charges present | TAX in 42% of receipts, service in 12%, discount in 3%, other about 0%. **No CGST/SGST anywhere.** |
| `sub_total` block present | 68% of receipts; `total` missing in 2 |
| Converts to our schema | 97% (train), 98% (val), 95% (test) |
| Arithmetic-consistent after converting | 85% / 89% / 85% (15% of labels do not add up) |
| Receipts with nested sub-items (we ignore them) | 16% |
| Image size | median 864 x 1296 (h/w 1.5, tall), 5th-95th pct 0.42-9.4 MP, max 12 MP |
| Look | phone photos on tables/cloth backgrounds, soft focus, mean brightness 144 (spread 75-200), contrast std about 40 |
| Sharpness (Laplacian var) | median 72 (5th-95th: 14-395) |
| Tilt | 5th/95th pct -1.0 / +5.5 degrees; 10.6% tilted more than 2 degrees (estimator limited to +-10) |
| Possible near-duplicates across splits | **28 pairs** (e.g. train_15 / validation_35). Not visually verified; could be the same restaurant template. |

## SROIE (626 train / 347 test, Malaysian, scans)
| Fact | Value |
|---|---|
| Layout in both Kaggle copies | `SROIE2019/{train,test}/{img,entities,box}`; **test has labels** (347 of 347) |
| Label completeness | company 100%, date 100%, address 99.8-100%, total 99.8-100% |
| The two Kaggle copies | identical label statistics (`urbikn/sroie-datasetv2`, `ryanznie/sroie-datasetv2-with-labels`); images not compared byte for byte |
| Train/test id overlap | 0 |
| Date formats | dd/mm/yyyy 54%, then dd-mm-yy, dd-mm-yyyy, dd/mm/yy, dd Mon yyyy; 3-5% other |
| Total (RM) | median 27, p5 4, p95 209 (train) / 270 (test) |
| Text length | company median 24 chars, address median 70 chars; 52 OCR box lines per receipt |
| Words in the OCR text | GST 95%, TAX 93%, ROUNDING 58-62%, DISCOUNT 21%, SERVICE 7-8%, CASH 93%, CHANGE 71-74% |
| Image size | median 825 x 1697 (h/w 2.0, long and narrow), **5th-95th pct 0.66-34.8 MP; the largest are 34.8 MP** |
| Look | flat white scans (brightness 239), sharp (sharpness median 290, up to 4,491), tilt about 0 (0.7% over 2 degrees) |
| Visible in thumbnails | **handwritten blue totals circled on many receipts**, name overlays at the top of images (e.g. "tan woon yann") |
| Possible near-duplicates across train/test | 329 pairs; likely same-store templates, not verified |
| What the labels do NOT have | **no line items, no charge lines.** Only company/date/address/total. SROIE helps with totals and reading robustness, not with charge classification. |

## What this means
1. **Cap image size at load.** 35 MP SROIE scans (and 12 MP CORD) would make augmentation slow and memory-hungry (tens of MB per colour channel as float arrays, several loader workers at once). Downscale to about 2,500 px long side before augmenting.
2. **Augmentation must differ per source.** CORD is already soft and dark; strong blur could make digits unreadable (label noise). SROIE is flat, sharp, bright scans, nothing like the phone photos we must handle, so it needs photo-style augmentation (background/margin, shadow, lower brightness).
3. **The retake gate misfires.** The current thresholds flag **55% of CORD** (43.5% "blurry", 28% "low resolution", 13% low contrast) and **19% of clean SROIE scans as "washed out"**. Must be recalibrated before R2 prompts are shown to users.
4. **CORD label noise:** a few extreme outliers (5.58 billion total, qty 202, negative prices) and 15% inconsistent bills. Filter the extreme ones; keep the rest.
5. Receipts are short (median 2 items), so long Indian bills (5-10 items) are under-represented in real data.
6. The "ink darkness" column is unreliable on huge sparse scans (5th percentile gray is 255 when text covers under 5% of pixels). Do not tune on it.
