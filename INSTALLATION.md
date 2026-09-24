# Installation and engineering validation

**INSTALLATION_TEST_STATUS = BLOCKED_BY_BUILD_DEPENDENCY_ACQUISITION.**
The local record is Python 3.12.3 with pip 24.0 in three separate environments
without system/user-site packages. Official PyPI and an official setuptools
wheel URL timed out. No build dependency or NumPy wheel was acquired; the build
requirement was not lowered. No new offline wheels were found in the explicitly
checked local release locations. This is not a global disk/backup search or a
software functionality verdict.

| Check | Recorded local result |
| --- | --- |
| Wheel build | Attempted offline after acquisition failed; exit 2, missing `setuptools.build_meta` backend. No wheel generated. |
| Wheel install and installed-wheel imports | NOT_RUN: no wheel. A separate Python -I import from outside the source directory correctly failed because rail3 was not installed. |
| Editable install | Attempted; exit 2, missing setuptools backend. |
| pip check of an installed release | NOT_RUN: no installed release/environment closure. |
| Original 15 tests in fresh environment | Four gateway fixtures passed; 11 NumPy tests did not run. A loader error is not a passed test. |
| Existing NumPy source environment | 22 unique artificial tests passed: 11 core, four gateway, seven adapter. Python 3.12.3 / NumPy 2.5.2. No wheel-install success inferred. |
| Fresh standard-library environment | 11 fixtures passed: four gateway, seven adapter. |
| Safe help/list and saved-summary hashes | Checked without scientific execution. |

The package-build attempts above were recorded on the preceding engineering
snapshot (package version 0.0.0). The v0.1.0 packaging metadata keeps the same
Python/build/runtime requirements; its own installation remains unverified
while the offline wheelhouse is unavailable. No old build result is relabelled
as a new version install.

These engineering outcomes are separate from all frozen scientific records.
A user-reported Python 3.13.5/setuptools 82.0.1 check elsewhere was not relabelled
as local Python 3.12 success. GitHub CI, if later run, is another dated environment,
not retroactive verification of the original experiments.

## Requirements and editable installation

`pyproject.toml` requires Python >=3.12, `setuptools>=68`, and
`numpy>=1.26,<3`. Optional plotting requires `matplotlib>=3.6`.
Use a new environment; do not alter a historical scientific environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/release_smoke.py
.venv/bin/python -m unittest tests.test_release_gateway tests.test_release_entrypoints -v
```

The 11 core plus four gateway plus seven adapter fixtures are artificial tests,
not scientific reproduction. No model, SAM, labels, checkpoint or bootstrap is
loaded. Full historical research modules need additional dependencies/inputs;
minimal installation does not imply that all research modules are importable.

## Wheel installation in a separate environment

The following are recipes, not a claim that the blocked local attempts passed:

```bash
python3.12 -m venv .build-venv
.build-venv/bin/python -m pip install 'setuptools>=68' build wheel 'numpy>=1.26,<3'
.build-venv/bin/python -m build --wheel --no-isolation --outdir dist
python3.12 -m venv .wheel-venv
.wheel-venv/bin/python -m pip install dist/rail3_research_snapshot-0.1.0-py3-none-any.whl
.wheel-venv/bin/python -m pip check
```

From a directory outside the checkout, call the installed environment's absolute
Python path with `-I -c 'import rail3; print(rail3.__file__)'`. Do not insert
`src/` or set PYTHONPATH for this installed-wheel check. For an authorized local
wheelhouse, use pip's `--no-index --find-links <wheelhouse>` with the same build
and runtime requirements. No CUDA/PyTorch/SAM/data download is part of these
minimal commands. Rendering saved figures adds the presentation extra and keeps
outputs separate from the supplied evidence.
