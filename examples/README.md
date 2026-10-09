# Examples

Everything here is a generated Indian-style bill or a screenshot of the app. No real customer bills are included.

| Folder | Contents |
|---|---|
| `demo_bills/` | Ten bills shown as thumbnails on the app's first screen ("Or try an example photo"); each goes through the real model like any upload. Labels in `bill_NN.json`. Seeds 7001-7010. |
| `final_run/` | 100 bills made with seeds 8001-8100 after training had finished: never used for training, validation, test or checkpoint selection. Labels in `final_NNN.json`. Used for the before/after test, the degradation study, the error analysis and the sample bills. |
| `processed_bills/` | The three processed sample bills required by the problem statement: raw photo, extracted JSON, corrections, corrected bill in the problem statement's schema, split, explanation. See the README in each `bill_N/`. |
| `synthetic_flat/`, `synthetic_augmented/` | Bills as the generator draws them, and the same bills after the training augmentation. |
| `augmentation_by_profile/` | The generator-profile augmentation examples (the SROIE and CORD profile examples use dataset images and are not distributed). |
| `app_screens/` | Screenshots of the mobile web app. |

**Not included on purpose.** Images from CORD and SROIE are not redistributed in this repository. The report's figures that show SROIE scans (and the `examples/sroie/`, `examples/overview.jpg` files they come from) can be rebuilt with `python scripts/make_examples.py`, which downloads single files from the public SROIE copy on Kaggle (needs a Kaggle account).

The generated bills come from `splitsnap/synth.py`. They look like Indian restaurant and mess bills (CGST, SGST, service charge, packaging, discounts, round-off, whole-rupee and decimal prices, several fonts, thermal-print fading) but they are renders, not photographs of real bills. Scores on them show how well the model reads this generator's style, not how it reads real bills.
