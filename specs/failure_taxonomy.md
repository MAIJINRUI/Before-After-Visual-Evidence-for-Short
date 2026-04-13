# Failure Taxonomy (`F1`-`F5`)

## Labels
- `F1`: target not visible / wrong view
- `F2`: wrong object
- `F3`: precondition not met
- `F4`: no state change / ineffective action
- `F5`: wrong location

## Primary-Label Precedence
When multiple labels may apply, assign a single primary label using:

`F3 > F2 > F5 > F1 > F4`

Rationale:
- Precondition violations (`F3`) directly block execution.
- Object identity errors (`F2`) are prioritized over spatial/view errors.
- Wrong location (`F5`) is prioritized over visibility (`F1`) and ineffectiveness (`F4`).

## Examples
- Attempt `Pick(mug)` when not near mug: `F3`.
- Picked plate instead of mug: `F2`.
- Placed mug on table instead of sink: `F5`.
- Target out of frame, action likely needs camera adjustment: `F1`.
- Action executed but state unchanged: `F4`.
