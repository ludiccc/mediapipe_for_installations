#!/bin/sh
# Write the IDE projects (Xcode on macOS, Visual Studio on Windows, Code::Blocks/Makefile
# on Linux) for every example with openFrameworks' projectGenerator command-line tool.
#
#   OF_ROOT=/path/to/openFrameworks ./generate_projects.sh            # platform = this computer
#   OF_ROOT=/path/to/openFrameworks ./generate_projects.sh osx        # or: vs, linux64, msys2 ...
#   PG=/path/to/projectGenerator ./generate_projects.sh               # if it is not found
#
# Without the command-line tool: open the projectGenerator app, press "import", choose an
# example folder, check that "ofxOsc" is listed, press "Generate" — one example at a time.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
EXAMPLES="body_landmarks hand_landmarks hand_gestures face_landmarks blobs"

if [ -z "$OF_ROOT" ]; then
	if [ -d "$HERE/../../libs/openFrameworksCompiled" ]; then OF_ROOT=$(cd "$HERE/../.." && pwd); fi
fi
if [ ! -d "$OF_ROOT/libs/openFrameworksCompiled" ]; then
	echo "openFrameworks not found: OF_ROOT=/path/to/openFrameworks ./generate_projects.sh"; exit 1
fi

PLATFORM=$1
if [ -z "$PLATFORM" ]; then
	case "$(uname -s)" in
		Darwin) PLATFORM=osx ;;
		MINGW*|MSYS*|CYGWIN*) PLATFORM=vs ;;
		*) PLATFORM=linux64 ;;
	esac
fi

# the command-line projectGenerator lives in different places depending on the release
if [ -z "$PG" ]; then
	for c in \
		"$OF_ROOT/projectGenerator/projectGenerator.app/Contents/Resources/app/app/projectGenerator" \
		"$OF_ROOT/projectGenerator-osx/projectGenerator.app/Contents/Resources/app/app/projectGenerator" \
		"$OF_ROOT/projectGenerator/resources/app/app/projectGenerator" \
		"$OF_ROOT/projectGenerator/resources/app/app/projectGenerator.exe" \
		"$OF_ROOT/projectGenerator-vs/resources/app/app/projectGenerator.exe" \
		"$OF_ROOT/projectGenerator/projectGenerator" \
		"$OF_ROOT/apps/projectGenerator/commandLine/bin/projectGenerator" \
		"$(command -v projectGenerator 2>/dev/null || true)"; do
		if [ -n "$c" ] && [ -x "$c" ]; then PG=$c; break; fi
	done
fi
if [ -z "$PG" ]; then
	echo "projectGenerator command-line tool not found under $OF_ROOT."
	echo "Use the projectGenerator app instead (import each example folder, Generate),"
	echo "or point to the tool:  PG=/path/to/projectGenerator ./generate_projects.sh"
	exit 1
fi

echo "projectGenerator: $PG   platform: $PLATFORM   openFrameworks: $OF_ROOT"
for e in $EXAMPLES; do
	echo "=== $e"
	"$PG" -o"$OF_ROOT" -p"$PLATFORM" -a"ofxOsc" "$HERE/$e"
done
echo "done: open $HERE/<example>/<example>.xcodeproj (macOS) or <example>.sln (Windows)"
