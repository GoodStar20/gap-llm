# Gap Classification with Hallucination Guard

This implementation uses **Path A (classical ML + rules engine)**.

Why Path A:
- deterministic behavior with no API/network dependency;
- easier enterprise audit trail from explicit rule references;
- hard hallucination guard enforced after model prediction.

## Structure

- `model/`: classifier and recommendation engine
- `rules/`: explicit rules store (`rules.yaml`)
- `eval/`: synthetic eval set builder and report generator
- `tests/`: pytest coverage for guardrails and determinism
- `examples/`: worked JSON input/output examples for each action
- `logs/`: runtime logs (low-confidence fallback / guard drops)

## Setup

```bash
python -m pip install -U pytest
```

## Windows PowerShell Notes

- PowerShell on some environments does not support `&&` command chaining.
- Use `;` to run sequential commands.
- Use `$LASTEXITCODE` when you want conditional execution.

Examples:

```bash
python -m eval.run_eval; python -m pytest -q
python -m eval.run_eval; if ($LASTEXITCODE -eq 0) { python -m pytest -q }
```

## Quick Start (Run + Test)

From the repository root:

```bash
python -m eval.run_eval
python -m pytest -q
```

One-command check:

```bash
python -m eval.run_eval; if ($LASTEXITCODE -eq 0) { python -m pytest -q }
```

PowerShell sequential (always runs both):

```bash
python -m eval.run_eval; python -m pytest -q
```

Key outputs:
- `eval/report.md`: accuracy, hallucination rate, calibration error, confusion matrix
- `logs/recommendation.log`: hallucination drops and low-confidence fallback events

## Run Evaluation

```bash
python -m eval.run_eval
```

This generates `eval/report.md` including:
- accuracy
- hallucination rate (post-processor drops)
- mean confidence calibration error
- confusion matrix

## Run Tests

```bash
python -m pytest -q
```

The test suite includes:
- hallucination guard rejection (`INVENTED_ACTION` + fake evidence)
- deterministic same-input same-output behavior
- minimum rules coverage check (>=10 rules)
- low-confidence fallback to `MANUAL_REVIEW`

## Run a Single Prediction

```bash
python -m model.predict examples\reduce_custom_fields.json
```

## Recommendation Contract

Engine output has:
- `recommended_action`: one of  
  `USE_STANDARD_SCOPE_ITEM`, `REDUCE_CUSTOM_FIELDS`, `MIGRATE_USER_EXIT_TO_BADI`, `ACTIVATE_SCOPE_ITEM`, `MANUAL_REVIEW`
- `confidence`: float in `[0.0, 1.0]`
- `rationale`: <= 280 chars and must match allowed rationale templates
- `evidence_refs`: non-empty, every value must exist in `rules/rules.yaml`

If any output field violates the contract, post-processor returns:
- `recommended_action = MANUAL_REVIEW`
- `confidence <= 0.5`
- `evidence_refs = ["RULE_HALLUCINATION_GUARD_v1"]`


### Screenshot
<img width="886" height="266" alt="image" src="https://github.com/user-attachments/assets/ce282b01-3952-4595-b6ac-2752d8b47915" />

