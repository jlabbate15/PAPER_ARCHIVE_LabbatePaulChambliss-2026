# DESC provenance for the Delta_res (orange) curve

The `TrappedResonance` objective is **not in upstream DESC**. It exists only in
a fork, and the data in `../desc_data/` was generated with one specific commit
of that fork.

## Pin

| | |
|---|---|
| repository | https://github.com/jlabbate15/DESC_TrappedRes |
| **commit** | **`4b77719dfa44ef26df29ba80b9c620ecb19faaf4`** |
| commit date | 2026-07-06 10:59:32 -0700 |
| commit subject | "Remove duplicate CHANGELOG Performance Improvements section" |
| author | ejpaul \<ejp2170@columbia.edu\> |
| branch containing it | `ejp-merge-master` (also `claude/analytic-omega-prime`, `ejp-local-wip-backup`) |
| objective file | `desc/objectives/_trapped_resonance.py` |
| that file's blob SHA | `a832223ba01a55e0541e15965939b79ea19c4ff6` |
| upstream of the fork | https://github.com/PlasmaControl/DESC |

## Check out the COMMIT, not the branch

As of 2026-08-29 `origin/ejp-merge-master` is **70 commits ahead** of the pinned
commit. Checking out the branch will NOT give you the code that produced this
data. Use the SHA:

```bash
git clone https://github.com/jlabbate15/DESC_TrappedRes
cd DESC_TrappedRes
git checkout 4b77719dfa44ef26df29ba80b9c620ecb19faaf4
```

`generate_delta_res.py` verifies this automatically: it reads the HEAD SHA of
the `--desc-path` checkout and refuses to run on a mismatch unless you pass
`--allow-version-mismatch`.

## Reference checkout

The checkout used for the original generation is on Perlmutter at
`/pscratch/sd/e/epaul/research/20260606_rerunning_John_paper_calcs/DESC_TrappedRes_clean`
(detached HEAD at the pinned commit, working tree clean as of 2026-08-29).

## Verification

Re-running `generate_delta_res.py` at this commit reproduces the packaged npz to
1e-7 relative: exact agreement on the beta scan at Bcrit = 6.24 (all five
points), and max absolute difference 8.9e-7 on the beta = 0 Bcrit curve, which
is GPU floating-point noise rather than a real difference.

Note the packaged npz are the ORIGINAL March 2026 files. This generator is a
faithful reproduction of them, verified as above, not their literal provenance.
