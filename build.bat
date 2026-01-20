@echo off
setlocal

:: setting Grafana dir
set "grafana_dir=C:\Program Files\grafana-v11.5.3"
set "plugin_dir=%grafana_dir%\data\plugins\ibumblebee-alert4ml-datasource"

:: show help info
if "%1"=="" (
    echo Available commands:
    echo   all               - build all
    echo   install-frontend  - install frontend
    echo   install-go-deps   - install go deps
    echo   build-rs          - build Rust code
    echo   build-go          - build Go code
    echo   build-ts          - build TypeScript code
    echo   copy-so           - copy .dll
    echo   prepare-plugin    - prepare plugin dirs
    echo   copy-dist         - copy dist dir
    echo   start-grafana     - start Grafana
    echo   clean             - clean build target files
    exit /b
)

:: run all steps
if "%1"=="all" (
    call :install-frontend
    call :install-go-deps
    call :build-rs
    call :build-go
    call :build-ts
    call :copy-so
    call :prepare-plugin
    call :copy-dist
    call :start-grafana
    exit /b
)

:install-frontend
echo Installing frontend dependencies...
call npm install
exit /b

:install-go-deps
echo Installing Go dependencies...
go mod download
exit /b

:build-rs
echo Building Rust components...
cd pkg\rsod
cargo build --release
cd ..\..
exit /b

:build-go
echo Building Go components...
mage -v build:windows
exit /b

:build-ts
echo Building TypeScript components...
call npm run build
exit /b

:copy-so
echo Copying DLL file...
copy /Y "pkg\rsod\target\release\lib*.dll" "dist\"
exit /b

:prepare-plugin
echo Preparing plugin directory...
if not exist "%plugin_dir%" mkdir "%plugin_dir%"
if exist "%plugin_dir%" rd /s /q "%plugin_dir%"
exit /b

:copy-dist
echo Copying distribution files...
xcopy /E /I /Y dist "%plugin_dir%"
exit /b

:start-grafana
@echo off
IF not exist "%grafana_dir%" (
    echo "Error: %grafana_dir% not found"
    exit \b 1
)

copy /Y ".\.config\data\grafana\defaults.ini" "%grafana_dir%\conf\"
@echo off
if not exist "%grafana_dir%\data\grafana.db" (
    copy /Y ".\.config\data\grafana\grafana.db" "%grafana_dir%\data\"
)
echo Starting Grafana...
cd "%grafana_dir%"
bin\grafana.exe server
exit /b

:clean
echo Cleaning up...
cd pkg\rsod
cargo clean
cd ..\..
if exist "%plugin_dir%" rd /s /q "%plugin_dir%"
if exist dist rd /s /q dist
call npm run clean
exit /b

:: call input command
call :%1
exit /b 