# Labeling Protocol (Week 2)

## Goal
Create consistent labels for short-horizon recovery action prediction from:
- instruction/subgoal
- attempted action
- before/after observations

## Per-Sample Fields
- `sample_id`
- `instruction`
- `attempted_action`
- `before_image`
- `after_image`
- `source_type` (`natural` or `synthetic`)
- `failure_type` (`F1`-`F5`)
- `gold_recovery_action`
- `candidate_vocab`
- `state_before`
- `goal_state`

## Labeling Rules
1. Pick exactly one primary failure label using precedence in `failure_taxonomy.md`.
2. Choose exactly one executable `gold_recovery_action` from the closed action space.
3. Ensure action arguments appear in `candidate_vocab`.
4. For ambiguous cases, prefer action that maximizes short-horizon solvability under symbolic rules.

## Quality Control
1. Format check: valid JSONL and required fields.
2. Vocabulary check: all action arguments legal.
3. Executability check: run executor and verify at least one-step consistency.
4. Spot check: sample 10% manually for label coherence.

## Split Policy
- Train/Val/Test at sample level: `70/15/15`.
- Preserve ratio by `source_type`.
- Preserve approximate distribution across `failure_type`.

## Exclusion Criteria
- Missing instruction or attempted action.
- Missing before/after image path.
- Ambiguous gold action with no dominant executable option.
- Out-of-vocabulary objects/locations not mapped in `candidate_vocab`.
