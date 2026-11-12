# tidyHEBO reproduction repository

- Keep the code small, direct and integrated into the existing controller.
- Public entry: `python run_experiments.py ACTION EXPERIMENT` from the root.
- README.md describes the paper experiments, data and public commands.
- Scientific formulas, GP settings, budgets, seeds and initial designs must not
  change under the name of cleanup. Never bypass checkpoint compatibility guards.
- Define quality metrics only in src/bo_benchmark/analysis/metrics.py. Other
  modules load/validate data, call those functions, serialize or plot.
- Preserve historical plotting data, hashes and source provenance in the repo.
  A figure recipe must reject incomplete coverage and never synthesize results.
- Keep model, baseline and oracle environments isolated. Use Mamba first; report
  a concrete reason for any pip fallback. Keep the pinned numerical versions.
- Never delete or overwrite the user's original results or environments during
  publication preparation. Historical source snapshots are kept separately.
- Validate the relevant numerical/seed contracts and real oracle smokes. Report
  actual checks and limits; a portable/mock test is not an oracle smoke.
