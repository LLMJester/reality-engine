@echo off
rem Discovery Programme v3, run 5 (V3-042): the designed experiment, restarted automatically after a crash (not after
rem a normal stop). Until 2026-10-29 this started the run 1-4 evolution loop (python -m v3.run loop).
rem   exit 0 = run complete, paused (state\PAUSE) or halted for the human -> stop
rem   exit 2 = refused (pre-registration lock missing or broken) -> stop
rem   other  = crash -> wait 60 s and restart (resumes from stored records)
cd /d "%~dp0\.."
:loop
echo [%date% %time%] run 5 loop start >> runs\run5_loop.log
python -m v3.run5 loop >> runs\run5_loop.log 2>&1
set rc=%errorlevel%
echo [%date% %time%] run 5 loop exit %rc% >> runs\run5_loop.log
if %rc%==0 goto end
if %rc%==2 goto end
if exist state\PAUSE goto end
ping -n 61 127.0.0.1 > nul
goto loop
:end
