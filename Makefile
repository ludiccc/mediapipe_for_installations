# Top-level shortcuts.  The app itself is plain Python (python mediapipe_osc.py);
# this Makefile only gathers the checks of the receiver examples.
#
#   make test            the app's test-suite (pytest)
#   make test-examples   every example folder's own `make test` (no camera, no Processing /
#                        openFrameworks / TouchDesigner needed: stand-ins + osc_simulator.py)
#   make sim             run the OSC simulator on port 5005

PYTHON ?= python3

.PHONY: test test-examples sim

test:
	$(PYTHON) -m pytest tests/ -q

test-examples:
	$(MAKE) -C python-examples test PYTHON=$(PYTHON)
	$(MAKE) -C processing-examples test PYTHON=$(PYTHON)
	$(MAKE) -C openFrameworks-examples test PYTHON=$(PYTHON)
	$(MAKE) -C touchdesigner-examples test PYTHON=$(PYTHON)

sim:
	$(PYTHON) osc_simulator.py
