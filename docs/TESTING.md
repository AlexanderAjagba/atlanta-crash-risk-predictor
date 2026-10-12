# Test setup and results

Completed 2026-09-25.

- Python: **46 passed, 2 skipped**. The skipped tests read the reserved holdout or its saved metrics and now require an explicit `--run-holdout` option. This option was not used.
- Frontend: **5 passed** across two test files.
- Production TypeScript/Vite build: passed.

## Coverage added

Frontend tests check the original map as default, pointer navigation between maps, removal of inactive map components, Arrow/Home/End keyboard focus and selection, accessible tab/panel associations, verified historical totals, year filtering and removal of stale cell details when the chosen area has no data in the new year.

Inference tests check time-feature boundaries (rush hour/night), missing/invalid values, timezone rejection, batching and order, input immutability, saved-model loading/probabilities, schema mismatch rejection and mocked Hugging Face download arguments. Existing tests cover chronological separation, feature leakage, geographic buffers and imbalance comparisons.

The frontend uses Vitest, Testing Library and jsdom. Tab tests mock both map components, while research-map tests render the actual SVG research component. No real-browser/WebGL, external basemap service, or live Hugging Face repository integration was tested. These tests establish software behavior, not new model predictive accuracy.

## Run

From the project root, in the configured Python environment:

```bash
python -m pytest tests -q
```

The completed run used the isolated task environment with its bundled OpenMP library for XGBoost. It did not modify the project's Python environment. If the local interpreter lacks dependencies, use the same analysis environment or install the project requirements before rerunning.

From frontend:

```bash
npm test
npm run test:watch
npm run build
```

Do not add `--run-holdout` to routine or CI commands. The holdout option enables checks of existing artifacts; it does not itself perform the planned final-model evaluation. Two pre-existing scripts also have direct Python entry points; use pytest rather than invoking those test files as scripts, since pytest owns the holdout opt-in behavior.

## Remaining limitations

Older pre-existing model artifacts emit scikit-learn version warnings (saved with 1.9.0; test environment 1.9.1). Their smoke tests passed, but this is not a compatibility guarantee. Final-model tests passed using its matching 1.9.1 environment. No models were resaved or retrained to silence warnings.

The frontend build retains its existing large-MapLibre-chunk warning. Dependency installation reported 13 npm audit findings; no force-upgrade or unrelated security remediation was performed. No warnings were suppressed.

Existing data, final models, original-map design and research-map implementation were preserved. Tests/configuration and frontend development-test dependencies were added. The 2025 holdout and holdout metrics were not read by this test run.
