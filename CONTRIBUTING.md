# Contributing to gzl

Thanks for your interest in `gzl`! Contributions of all sizes are
welcome, from typo fixes through bug reports and benchmark scripts to entire
new graph constructors.

## Before you start

If your change is non-trivial (anything beyond, say, a typo or a small bug
fix), please open a GitHub issue first to discuss the design. This avoids
wasted work on patches that don't fit the project's direction.

## Licensing of contributions

`gzl` is licensed under **AGPL-3.0-or-later** (see [`LICENSE`](./LICENSE)).
By submitting a contribution (a pull request, patch, or any other form of
change), you agree that:

1. **AGPL and MIT.** Your contribution is licensed to the project under
   AGPL-3.0-or-later, like the rest of the codebase, and in addition under
   the [MIT License](https://opensource.org/license/mit). Opening a pull
   request is the agreement. The MIT grant keeps the project free to change
   its licence without having to contact every contributor.

2. **You have the right to make this grant.** You represent that the
   contribution is your original work, or that you have the authority from
   any other rights-holder (employer, university, etc.) to make this grant.
   If you contribute work created during paid employment, please make sure
   you have your employer's permission first. In Germany, § 69b UrhG gives
   the employer the rights to software written as part of the job.

### Third-party reference data

These terms cover code and this project's own data. They cannot cover
third-party research data, such as the Monte Carlo series under
`gzl/data/MC_patched/`, which stay on their sources' terms. A reference
file enters the repository only with:

* a row in `gzl/data/PROVENANCE.csv` recording the source, the evidence
  and the SHA-256.
* a citable published source or recorded permission, and a `NOTICE`
  entry for that source.

`tests/test_reference_data.py` fails otherwise, and also when a pinned file
changes without its manifest row.

## Sign your commits (Developer Certificate of Origin)

In addition, every commit must be signed off using the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/),
a one-line confirmation that you have the right to make the contribution.
Use:

```bash
git commit -s -m "Your commit message"
```

This appends a `Signed-off-by: Your Name <you@example.com>` line to the
commit message. The full DCO text is reproduced at the bottom of this file
for reference.

## Workflow

1. Fork the repository and create a feature branch
   (`git checkout -b feature/my-change`).
2. Make your changes. Keep commits focused and reasonably small.
3. Add or update tests so that `pytest` covers the new behaviour.
4. Add or update the relevant docstrings and (if user-visible) a line in the
   `README.md`.
5. Run the full suite locally:

   ```bash
   pytest -v
   ```

6. Make sure all existing examples still run:

   ```bash
   for f in examples/*.py; do python "$f" || break; done
   ```

7. Open a pull request against `main` with a clear description of what your
   change does, why, and any benchmarks or accuracy numbers if relevant.

## Code style

- Python ≥ 3.11, type hints encouraged on new public functions.
- Docstrings in NumPy or Google style — match what's already in the file you
  are editing.
- Keep public API surface small. Anything that's an implementation detail
  should start with `_` or live in a private module.
- New `.py` files must carry the standard SPDX header:

  ```python
  # SPDX-License-Identifier: AGPL-3.0-or-later
  # Copyright (C) <year> <your name>
  ```

## Bug reports

A useful bug report contains: the failing input (smallest reproducer), the
expected output, the actual output, and the output of

```bash
gzl info
```

which gives the `gzl` version, the directory it was imported from, and the
`numpy`, `scipy`, `epsteinlib`, `networkx` and `h5py` versions. Numerical-accuracy
bugs especially benefit from a reference value, e.g. via `zeta_circle` or
`epsteinlib.epstein_zeta`.

If the library misbehaves on an installation you did not build yourself,
`gzl selftest` is worth running first: it checks the shipped data against
its SHA-256 manifest and reproduces values whose answers are known
independently of the code that produces them.

---

## Developer Certificate of Origin v1.1

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
