# Processing examples for mediapipe_osc.py — one sketch per detector.
#
#   make all            build every sketch with processing-java, then run them one after
#                       the other (close a sketch window to go on to the next one)
#   make run-blobs      run one sketch (run-body_landmarks, run-hand_landmarks,
#                       run-hand_gestures, run-face_landmarks, run-blobs)
#   make build          only compile (into build/<sketch>/)
#   make test           check without Processing: compiles the sketches against small
#                       stand-ins of PApplet/oscP5 (needs javac + python3) and feeds them
#                       with ../osc_simulator.py for a few seconds
#   make sim            run the simulator alone (then run a sketch from the Processing IDE)
#
# Needs processing-java on the PATH (see README.md):  make PROCESSING_JAVA=/path/to/processing-java
# Every sketch can also be opened in the Processing IDE and started with the Run button.

PROCESSING_JAVA ?= processing-java
PYTHON ?= python3
PORT ?= 5005
EXAMPLES = body_landmarks hand_landmarks hand_gestures face_landmarks blobs
STUBS = ../tests/stubs/processing

.PHONY: all build run run-% test sim check clean help

all: build run

check:
	@command -v $(PROCESSING_JAVA) >/dev/null 2>&1 \
	  || { echo "processing-java not found: install Processing and put processing-java on the PATH"; \
	       echo "(macOS: Tools > Install \"processing-java\";  or: make PROCESSING_JAVA=/path/to/processing-java)"; exit 1; }
	@echo "using $$(command -v $(PROCESSING_JAVA))"

build: check
	@for e in $(EXAMPLES); do \
	  echo "=== building $$e"; \
	  $(PROCESSING_JAVA) --sketch="$(CURDIR)/$$e" --output="$(CURDIR)/build/$$e" --force --build || exit 1; \
	done
	@echo "all sketches compiled"

run: check
	@for e in $(EXAMPLES); do \
	  echo ""; echo "=== $$e  (close the sketch window to start the next example) ==="; \
	  $(PROCESSING_JAVA) --sketch="$(CURDIR)/$$e" --output="$(CURDIR)/build/$$e" --force --run || true; \
	done

run-%: check
	$(PROCESSING_JAVA) --sketch="$(CURDIR)/$*" --output="$(CURDIR)/build/$*" --force --run

sim:
	$(PYTHON) ../osc_simulator.py --port $(PORT)

# No Processing needed: compile each .pde against the stand-in classes in
# ../tests/stubs/processing and run it headless for 3 s while the simulator sends data.
test:
	@command -v javac >/dev/null 2>&1 || { echo "javac (a JDK) is needed for make test"; exit 1; }
	@status=0; $(PYTHON) ../osc_simulator.py --port $(PORT) --seconds 60 --quiet & pid=$$!; sleep 1; \
	for e in $(EXAMPLES); do \
	  echo "--- $$e"; \
	  sh $(STUBS)/build.sh $$e/$$e.pde build/stub/$$e || { status=1; continue; }; \
	  out=$$(java -cp build/stub/$$e Runner $$e 3 2>&1); echo "$$out"; \
	  echo "$$out" | grep -q "oscEvent failures 0" || status=1; \
	  echo "$$out" | grep -q "received [1-9]" || status=1; \
	done; \
	kill $$pid 2>/dev/null; wait $$pid 2>/dev/null; \
	if [ $$status -eq 0 ]; then echo "ALL PROCESSING SKETCHES OK (stub harness)"; else echo "SOME SKETCH FAILED"; fi; exit $$status

clean:
	rm -rf build

help:
	@sed -n '2,15p' Makefile
