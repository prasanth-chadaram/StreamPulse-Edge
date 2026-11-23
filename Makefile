PY ?= python3
SRC := PYTHONPATH=src $(PY)

.PHONY: install run sim media test soak clean

install:
	$(PY) -m pip install -r requirements.txt

run:
	$(SRC) -m streampulse --config config/default.yaml

sim:
	tools/make_test_media.sh
	NUM_STREAMS=4 tools/start_rtsp_sim.sh

media:
	tools/make_test_media.sh

test:
	$(SRC) -m pytest tests/ -q

soak:
	$(SRC) tests/soak_test.py

soak-quick:
	$(SRC) tests/soak_test.py --quick

clean:
	rm -rf reports/soak_report_*.md .pytest_cache src/*.egg-info
