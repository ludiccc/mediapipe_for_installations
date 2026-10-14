# TouchDesigner examples for mediapipe_osc.py — one build script per detector.
#
# TouchDesigner networks are binary .toe files, so each example is a Python script that,
# run INSIDE TouchDesigner, builds its network (OSC In DAT + callbacks + tables + Script
# CHOPs/SOPs + a render).  There is no command-line way to run a script in TouchDesigner,
# so:
#
#   make all            = make test + make run
#   make test           check the five scripts without TouchDesigner: they are run in a small
#                       stand-in of the TouchDesigner Python API (../tests/stubs/touchdesigner)
#                       and fed with the messages of ../osc_simulator.py
#   make run            open TouchDesigner (if it is installed here) and print, for each example,
#                       the exact line to paste in its Textport
#   make run-blobs      the same for one example (run-body_landmarks, run-hand_landmarks,
#                       run-hand_gestures, run-face_landmarks, run-blobs)
#   make sim            run the simulator (then build an example in TouchDesigner)
#
# Windows: run make from Git Bash / MSYS2, with PYTHON=python.  Without make: open
# TouchDesigner, drag <example>/<example>.py into the network, right-click > Run Script.

PYTHON ?= python3
PORT ?= 5005
EXAMPLES = body_landmarks hand_landmarks hand_gestures face_landmarks blobs
STUBS = ../tests/stubs/touchdesigner

# TouchDesigner executable, best effort (override: make TOUCHDESIGNER=/path/to/TouchDesigner)
ifeq ($(OS),Windows_NT)
  TOUCHDESIGNER ?= $(firstword $(wildcard /c/Program\ Files/Derivative/TouchDesigner*/bin/TouchDesigner.exe) TouchDesigner.exe)
  OPEN_TD = start "" "$(TOUCHDESIGNER)"
else ifeq ($(shell uname -s),Darwin)
  TOUCHDESIGNER ?= TouchDesigner
  OPEN_TD = open -a "$(TOUCHDESIGNER)"
else
  TOUCHDESIGNER ?=
  OPEN_TD = echo "TouchDesigner runs on Windows and macOS only; open it there."
endif

.PHONY: all test run run-% sim compile clean help

all: test run

compile:
	@for e in $(EXAMPLES); do $(PYTHON) -m py_compile $$e/$$e.py && echo "syntax OK $$e/$$e.py"; done

# Headless check: every script builds its network in the stand-in and receives 3 s of
# simulated messages; fails if a callback raises or no data reaches the tables.
test: compile
	@status=0; for e in $(EXAMPLES); do \
	  echo "--- $$e"; \
	  $(PYTHON) $(STUBS)/run_example.py $$e/$$e.py 3 | grep '###' || status=1; \
	done; \
	if [ $$status -eq 0 ]; then echo "ALL TOUCHDESIGNER SCRIPTS OK (stub harness)"; else echo "SOME SCRIPT FAILED"; fi; exit $$status

run:
	@for e in $(EXAMPLES); do $(MAKE) --no-print-directory instructions-$$e; done
	@-$(OPEN_TD)

run-%:
	@$(MAKE) --no-print-directory instructions-$*
	@-$(OPEN_TD)

instructions-%:
	@echo ""
	@echo "=== $* ==="
	@echo "In TouchDesigner, either drag  $(CURDIR)/$*/$*.py  into a network and right-click > Run Script,"
	@echo "or paste this line in the Textport (Alt+T):"
	@echo ""
	@echo "    exec(open(r'$(CURDIR)/$*/$*.py').read())"
	@echo ""
	@echo "Then start the camera side:  python ../mediapipe_osc.py $(shell sed -n 's/^# Camera side: *python mediapipe_osc.py \([^ ]*\( --[a-z-]*\)*\).*/\1/p' $*/$*.py | head -1)   (or python ../osc_simulator.py)"

sim:
	$(PYTHON) ../osc_simulator.py --port $(PORT)

clean:
	rm -rf */__pycache__ __pycache__

help:
	@sed -n '2,22p' Makefile
