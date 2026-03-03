@echo off
::
:: build_ext.bat — build autoite._core in-place for development (Windows)
::
:: Usage:
::   build_ext.bat            — Release build
::   build_ext.bat Debug      — Debug build
::   build_ext.bat clean      — remove _build_ext/ directory
::
:: After a successful build, autoite\_core.cp313-win_amd64.pyd is written
:: into autoite/ and importable immediately without reinstalling the package.
::

setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release
if "%BUILD_TYPE%"=="clean" (
    echo Cleaning _build_ext...
    if exist _build_ext rmdir /s /q _build_ext
    goto :eof
)

:: ── Locate VS 2019 Build Tools ──────────────────────────────────────────── ::
set VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
if not exist "%VS_ROOT%" (
    echo ERROR: VS 2019 Build Tools not found at "%VS_ROOT%"
    exit /b 1
)

set CMAKE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe

:: ── Activate 64-bit MSVC environment ────────────────────────────────────── ::
call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

:: ── Locate Python executable ─────────────────────────────────────────────:: ::
for /f "tokens=*" %%i in ('where python') do (
    set PYTHON_EXE=%%i
    goto :found_python
)
:found_python

:: ── CMake configure ──────────────────────────────────────────────────────:: ::
echo.
echo [1/2] Configuring (%BUILD_TYPE%)...
"%CMAKE%" -B _build_ext -S . ^
    -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DPython3_EXECUTABLE="%PYTHON_EXE%"

if errorlevel 1 (
    echo ERROR: CMake configure failed.
    exit /b 1
)

:: ── CMake build ──────────────────────────────────────────────────────────:: ::
echo.
echo [2/2] Building...
"%CMAKE%" --build _build_ext --config %BUILD_TYPE% --parallel

if errorlevel 1 (
    echo ERROR: Build failed.
    exit /b 1
)

:: ── Install in-place ─────────────────────────────────────────────────────:: ::
"%CMAKE%" --install _build_ext --prefix .

if errorlevel 1 (
    echo ERROR: Install step failed.
    exit /b 1
)

echo.
echo Build complete. Extension installed to autoite\_core*.pyd
python -c "from autoite._core import loo_objective; print('  Import test: OK')"

endlocal
