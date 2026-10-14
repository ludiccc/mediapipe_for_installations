@echo off
rem Write the Visual Studio solutions for every example with openFrameworks' projectGenerator
rem command-line tool (Windows).  Run it from this folder:
rem
rem   generate_projects.bat                           (this folder inside <openFrameworks>\apps\)
rem   generate_projects.bat C:\path\to\openFrameworks
rem
rem Without the command-line tool: open projectGenerator.exe, press "import", choose an
rem example folder, check that "ofxOsc" is listed, press "Generate" - one example at a time.
setlocal
set HERE=%~dp0
set OF_ROOT=%~1
if "%OF_ROOT%"=="" set OF_ROOT=%HERE%..\..
if not exist "%OF_ROOT%\libs\openFrameworksCompiled" (
	echo openFrameworks not found: generate_projects.bat C:\path\to\openFrameworks
	exit /b 1
)
set PG=
if exist "%OF_ROOT%\projectGenerator\resources\app\app\projectGenerator.exe" set PG=%OF_ROOT%\projectGenerator\resources\app\app\projectGenerator.exe
if exist "%OF_ROOT%\projectGenerator-vs\resources\app\app\projectGenerator.exe" set PG=%OF_ROOT%\projectGenerator-vs\resources\app\app\projectGenerator.exe
if "%PG%"=="" (
	echo projectGenerator.exe command-line tool not found under %OF_ROOT%.
	echo Use the projectGenerator app instead: import each example folder and press Generate.
	exit /b 1
)
echo projectGenerator: %PG%   openFrameworks: %OF_ROOT%
for %%e in (body_landmarks hand_landmarks hand_gestures face_landmarks blobs) do (
	echo === %%e
	"%PG%" -o"%OF_ROOT%" -p"vs" -a"ofxOsc" "%HERE%%%e"
)
echo done: open ^<example^>\^<example^>.sln in Visual Studio
endlocal
