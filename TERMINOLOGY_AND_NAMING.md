# Locked terminology and naming for the computational study

This vocabulary is to be used consistently in the Python package, orchestration notebook, saved metadata, verification reports, figures, and manuscript-facing documentation.

| Retired term | Required replacement |
|---|---|
| campaign | computational study; full study |
| full campaign | full computational study |
| run_full_campaign | run_full_study |
| campaign.py | study.py |
| campaign report | study summary |
| production run | main calculation; full-resolution calculation |
| production mode | FULL_STUDY |
| production figures | publication figures |
| pilot | parameter-selection and convergence study |
| PILOT | PARAMETER_SELECTION |
| smoke test | quick numerical check |
| SMOKE | QUICK_CHECK |
| verification gate | verification criterion; verification status |
| gate | acceptance criterion / status, according to context |
| manifest | case metadata |
| production registry | validated-results index |
| campaign directory | study directory |
| campaign configuration | study configuration |
| mechanism cases | mechanistic-analysis cases |
| width campaign | axial-scale study |
| disease campaign | disease-parameter study |

## Retained technical terms

The following are technically precise and should be retained when applicable:

- checkpoint / restart
- verification
- validation against an exact or independent numerical reference
- admissibility
- matched-mean control
- parent-reference audit
- ETDRK4
- pseudospectral discretization
- de-aliasing
- resonance classification
- modal-energy budget
- parameter study
- convergence study

## Run-mode vocabulary

The user-facing orchestration notebook shall use:

- `QUICK_CHECK`
- `VERIFICATION`
- `PARAMETER_SELECTION`
- `FULL_STUDY`
- `FIGURES`

`FULL_STUDY` is the default run mode.

## Scientific naming principle

Software-development language must not substitute for physical or numerical terminology. File names, function names, notebook text, console messages, JSON keys, and figure labels must describe the actual computational or physical operation being performed.
