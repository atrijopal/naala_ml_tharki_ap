# Common tasks. All of them run on the local machine; training is done on Kaggle (see README).
PY ?= python
CKPT ?= ckpt/best_fp16

.PHONY: app demo test e2e report studies notebooks
app:            ## run the web app with the real model on http://localhost:8000
	CKPT=$(CKPT) PRELOAD=1 $(PY) -m uvicorn splitsnap.app:app --host 127.0.0.1 --port 8000
demo:           ## run the web app without a model (two built-in sample bills)
	DEMO=1 $(PY) -m uvicorn splitsnap.app:app --host 127.0.0.1 --port 8000
test:           ## unit and endpoint tests (no GPU needed)
	$(PY) tests/test_core.py
e2e:            ## browser test of the whole flow (needs headless Firefox and the app on :8765)
	$(PY) tests/e2e_ui.py
report:         ## rebuild the PDF report from the logs and results
	$(PY) scripts/report_data.py && $(PY) scripts/walkthroughs.py && cd report && tectonic report.tex && cp report.pdf SplitSnap_Model_Report.pdf
studies:        ## local measurements behind the report (need the checkpoint and the app on :8765)
	$(PY) scripts/before_after.py --ckpt $(CKPT) --n 30
	$(PY) scripts/degradation_study.py --ckpt $(CKPT)
	$(PY) scripts/error_analysis.py --ckpt $(CKPT)
	$(PY) scripts/sroie_scan_check.py --ckpt $(CKPT)
	$(PY) scripts/load_test.py
notebooks:      ## regenerate the three notebooks in notebooks/
	$(PY) scripts/make_notebooks.py
