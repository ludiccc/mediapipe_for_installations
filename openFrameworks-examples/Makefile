# openFrameworks examples for mediapipe_osc.py — one app per detector.
#
#   make all              compile every example (Release) with openFrameworks' own makefiles,
#                         then run them one after the other (close a window, or press q, to
#                         go on to the next one)
#   make run-blobs        run one example (run-body_landmarks, run-hand_landmarks,
#                         run-hand_gestures, run-face_landmarks, run-blobs)
#   make build            only compile;  make clean  removes the binaries and obj/ folders
#   make projects         write the IDE projects (Xcode / Visual Studio / ...) with the
#                         projectGenerator (see generate_projects.sh / .bat)
#   make test             check without openFrameworks: compiles the apps against small
#                         stand-ins of ofMain.h / ofxOsc.h (g++) and feeds them with
#                         ../osc_simulator.py for a few seconds
#   make sim              run the simulator alone
#
# openFrameworks location: put this folder inside <openFrameworks>/apps/ (e.g.
# <openFrameworks>/apps/openFrameworks-examples/) and nothing needs configuring;
# anywhere else, say where it is:   make OF_ROOT=/path/to/openFrameworks all
#
# Linux: works as is (after <openFrameworks>/scripts/linux/<distro>/install_dependencies.sh
#        and compileOF.sh).   macOS: with Xcode's command line tools (or open the Xcode
#        projects).   Windows: from an MSYS2 shell of the openFrameworks msys2 release,
#        or generate and open the Visual Studio solutions (generate_projects.bat).

EXAMPLES = body_landmarks hand_landmarks hand_gestures face_landmarks blobs
PYTHON ?= python3
PORT ?= 5005
STUBS = ../tests/stubs/openframeworks

# --- where is openFrameworks? -------------------------------------------------
ifndef OF_ROOT
  ifneq ($(wildcard ../../libs/openFrameworksCompiled),)
    OF_ROOT := $(abspath ../..)
  endif
endif
OF_ROOT := $(abspath $(OF_ROOT))

.PHONY: all build run run-% clean projects test sim check help

all: build run

check:
	@test -d "$(OF_ROOT)/libs/openFrameworksCompiled" || { \
	  echo "openFrameworks not found."; \
	  echo "Copy this folder into <openFrameworks>/apps/  or run:  make OF_ROOT=/path/to/openFrameworks $(MAKECMDGOALS)"; \
	  exit 1; }
	@echo "openFrameworks: $(OF_ROOT)"

build: check
	@for e in $(EXAMPLES); do \
	  echo ""; echo "=== building $$e"; \
	  $(MAKE) -C $$e OF_ROOT="$(OF_ROOT)" Release || exit 1; \
	done
	@echo "all examples compiled"

run: check
	@for e in $(EXAMPLES); do \
	  echo ""; echo "=== $$e  (close the window or press q to start the next example) ==="; \
	  $(MAKE) -C $$e OF_ROOT="$(OF_ROOT)" RunRelease || true; \
	done

run-%: check
	$(MAKE) -C $* OF_ROOT="$(OF_ROOT)" Release
	$(MAKE) -C $* OF_ROOT="$(OF_ROOT)" RunRelease

clean:
	@for e in $(EXAMPLES); do \
	  if test -d "$(OF_ROOT)/libs/openFrameworksCompiled"; then $(MAKE) -C $$e OF_ROOT="$(OF_ROOT)" clean || true; fi; \
	  rm -rf $$e/obj $$e/bin/$$e $$e/bin/$$e.app $$e/bin/$$e_debug $$e/bin/$$e.exe $$e/bin/*.dll $$e/bin/libs; \
	done
	rm -rf build

projects:
	OF_ROOT="$(OF_ROOT)" sh ./generate_projects.sh

sim:
	$(PYTHON) ../osc_simulator.py --port $(PORT)

# No openFrameworks needed: compile each app against the stand-in headers in
# ../tests/stubs/openframeworks and run it headless for 3 s while the simulator sends data.
test:
	@command -v g++ >/dev/null 2>&1 || { echo "g++ is needed for make test"; exit 1; }
	@status=0; $(PYTHON) ../osc_simulator.py --port $(PORT) --seconds 60 --quiet & pid=$$!; sleep 1; \
	for e in $(EXAMPLES); do \
	  echo "--- $$e"; \
	  sh $(STUBS)/build.sh $$e build/stub/$$e || { status=1; continue; }; \
	  out=$$(OF_STUB_SECONDS=3 ./build/stub/$$e 2>&1); echo "$$out"; \
	  echo "$$out" | grep -q "received [1-9]" || status=1; \
	  echo "$$out" | grep -q "done: [1-9]" || status=1; \
	done; \
	kill $$pid 2>/dev/null; wait $$pid 2>/dev/null; \
	if [ $$status -eq 0 ]; then echo "ALL OPENFRAMEWORKS EXAMPLES OK (stub harness)"; else echo "SOME EXAMPLE FAILED"; fi; exit $$status

help:
	@sed -n '2,24p' Makefile
