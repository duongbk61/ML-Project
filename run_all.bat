@echo off
:: run_all.bat — Train every model for 200 episodes across all seeds and feedback weights.
::
:: Coverage:
::   Seeds          : 0, 1, 2
::   Feedback weights (HCRL/VI-TAMER): 5, 10, 20, 50
::   Credit variant  : uniform cw=3
::   Models         : Baseline Q-Learning, HCRL, VI-TAMER, RLHF, RLHF Ensemble
::   Timing exp     : 4 conditions x 3 seeds (internal), per feedback weight
::
:: Usage:
::   run_all.bat

setlocal

set EPISODES=200

echo ========================================================
echo  Full training run
echo  episodes=%EPISODES%  seeds=[0 1 2]  fw=[5 10 20 50]
echo ========================================================

:: ---------------------------------------------------------------------------
:: 1. Baseline Q-Learning
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   BASELINE Q-LEARNING
echo ================================================================
for %%S in (0 1 2) do (
    echo.
    echo   [Baseline] seed=%%S
    uv run python run.py --episodes %EPISODES% --seed %%S
    if errorlevel 1 goto :error
)

:: ---------------------------------------------------------------------------
:: 2. HCRL (TAMER)
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   HCRL (TAMER)
echo ================================================================
for %%W in (5 10 20 50) do (
    for %%S in (0 1 2) do (
        echo.
        echo   [HCRL] fw=%%W  seed=%%S  credit=uniform cw=3
        uv run python train_hcrl.py --episodes %EPISODES% --seed %%S --feedback-weight %%W --credit-window 3 --credit-fn uniform --skip-charts
        if errorlevel 1 goto :error
    )
)

:: ---------------------------------------------------------------------------
:: 3. VI-TAMER
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   VI-TAMER
echo ================================================================
for %%W in (5 10 20 50) do (
    for %%S in (0 1 2) do (
        echo.
        echo   [VI-TAMER] fw=%%W  seed=%%S  credit=uniform cw=3
        uv run python train_vi_tamer.py --episodes %EPISODES% --seed %%S --feedback-weight %%W --credit-window 3 --credit-fn uniform --skip-charts
        if errorlevel 1 goto :error
    )
)

:: ---------------------------------------------------------------------------
:: 4. RLHF (standard)
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   RLHF (standard)
echo ================================================================
for %%S in (0 1 2) do (
    echo.
    echo   [RLHF] seed=%%S
    uv run python train_rlhf.py --episodes %EPISODES% --seed %%S --skip-charts
    if errorlevel 1 goto :error
)

:: ---------------------------------------------------------------------------
:: 5. RLHF Ensemble
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   RLHF Ensemble
echo ================================================================
for %%S in (0 1 2) do (
    echo.
    echo   [RLHF Ensemble n=3] seed=%%S
    uv run python train_rlhf_ensemble.py --episodes %EPISODES% --seed %%S --n-models 3 --skip-charts
    if errorlevel 1 goto :error

    echo.
    echo   [RLHF Ensemble n=5] seed=%%S
    uv run python train_rlhf_ensemble.py --episodes %EPISODES% --seed %%S --n-models 5 --skip-charts
    if errorlevel 1 goto :error
)

:: ---------------------------------------------------------------------------
:: 6. Feedback Timing Experiment
:: ---------------------------------------------------------------------------
echo.
echo ================================================================
echo   FEEDBACK TIMING EXPERIMENT
echo ================================================================
for %%W in (5 10 20 50) do (
    echo.
    echo   [Timing] fw=%%W  (seeds 0,1,2 run internally)
    uv run python feedback_timing_experiment.py --episodes %EPISODES% --auto --skip-charts --feedback-weight %%W
    if errorlevel 1 goto :error
)

echo.
echo ========================================================
echo  All done. Results in experiment-results\ep%EPISODES%\
echo ========================================================
goto :end

:error
echo.
echo ERROR: A training job failed. Check output above.
exit /b 1

:end
endlocal
