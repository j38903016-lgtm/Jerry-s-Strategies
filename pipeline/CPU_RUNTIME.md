# CPU runtime repair (2026-11-03)

Production remains at 16 processes and 4 XGBoost threads per model. Features,
labels, search space, random seed, thresholds, state machine and database
destinations are unchanged.

The CUDA-enabled XGBoost 3.2.0 build accessed NVIDIA devices during CPU
training and workers stalled in driver locks. BLAS also created approximately
128 helper threads per process. The repair caps BLAS helpers at 1 and OpenMP
at XGB_N_JOBS before importing numerical libraries.

For REQUIRE_GPU=0, pipeline.py prefers the isolated official CPU build of the
same XGBoost version if installed at the path below. GPU mode retains the
original GPU-capable installation. No other application's runtime is changed.

```bash
/home/jerry/us_stock_pipeline/.venv/bin/python -m pip install --no-deps \
  --target /data/us_stock_pipeline/runtime_cpu_xgb_3_2_0 xgboost-cpu==3.2.0
```

PIPELINE_CPU_XGB_RUNTIME can override this directory. The selected module path
and version are logged at startup. Keep this directory when migrating the
server; without it CPU mode falls back to the installed XGBoost build.

Validation: AA was retrained on identical production data with the same seed
and search settings. All model record fields except creation time, and all
250 test/live prediction rows, matched the pre-repair run exactly. Today's
incomplete run was resumed using its original run ID; completed symbols were
preserved. All 203 models and live predictions completed successfully.

中文：仅修复CPU运行时与辅助线程池；因子、模型参数、阈值和信号规则未变。
