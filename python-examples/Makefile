# Python examples for mediapipe_osc.py — one receiver per detector.
#
#   make all            check the modules, then run every example one after the other
#                       (close a window with q / Esc to go on to the next one)
#   make run-blobs      run one example (run-body_landmarks, run-hand_landmarks,
#                       run-hand_gestures, run-face_landmarks, run-blobs)
#   make test           automatic check without camera or window: the simulator
#                       (../osc_simulator.py) feeds every example for a few seconds
#   make sim            run the simulator alone (then start an example in another terminal)
#   make deps           pip install the required modules
#
# Variables:  PYTHON=python   (Windows)   PORT=9000   (another OSC port)
#
# Every example can also be started by hand:  python blobs/blobs.py --port 5005

PYTHON ?= python3
PORT   ?= 5005
EXAMPLES = body_landmarks hand_landmarks hand_gestures face_landmarks blobs
SIM = ../osc_simulator.py

.PHONY: all build check compile deps run run-% test sim clean help

all: build run

build: check compile

check:
	@$(PYTHON) -c "import cv2, numpy, pythonosc" 2>/dev/null \
	  || { echo "missing modules: run 'make deps' (pip install -r requirements.txt)"; exit 1; }
	@echo "python modules OK ($(PYTHON))"

compile:
	@for e in $(EXAMPLES); do $(PYTHON) -m py_compile $$e/$$e.py && echo "compiled $$e/$$e.py"; done

deps:
	$(PYTHON) -m pip install -r requirements.txt

run:
	@for e in $(EXAMPLES); do \
	  echo ""; echo "=== $$e  (q or Esc closes the window and starts the next example) ==="; \
	  $(PYTHON) $$e/$$e.py --port $(PORT) || true; \
	done

run-%:
	$(PYTHON) $*/$*.py --port $(PORT)

sim:
	$(PYTHON) $(SIM) --port $(PORT)

# Headless self-test: simulator + every example for 3 s each; fails if nothing arrives.
test: check
	@$(PYTHON) $(SIM) --port $(PORT) --seconds 40 --quiet & pid=$$!; sleep 1; status=0; \
	for e in $(EXAMPLES); do \
	  echo "--- $$e"; \
	  $(PYTHON) $$e/$$e.py --port $(PORT) --headless --seconds 3 || status=1; \
	done; \
	kill $$pid 2>/dev/null; wait $$pid 2>/dev/null; \
	if [ $$status -eq 0 ]; then echo "ALL PYTHON EXAMPLES OK"; else echo "SOME EXAMPLE FAILED"; fi; exit $$status

clean:
	rm -rf */__pycache__ __pycache__

help:
	@sed -n '2,16p' Makefile
