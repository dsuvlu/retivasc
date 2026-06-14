# retivasc Audit Report

- **Date:** 2026-06-14
- **Auditor:** Lead audit (adversarially verified findings)
- **Scope:** `retivasc` compact interview-demo, judged against `retivasc_development_handoff.md` and the project's seven non-negotiables. Production-feature gaps (CI, model cards, U-Net, packaging) are explicitly out of scope and are not flagged as defects.

## Executive summary

The demo is scientifically honest and substantively demo-ready. None of the four scientific non-negotiables is violated: there is no predictive AD modeling or AUROC/calibration overclaiming on ROSE, no synthetic biomarker numbers presented as results, no unsupported AD diagnostic language, and no raw medical images are committed (tests run data-free, 17 passing). The subject-level-split machinery is correct and tested at the primitive level. The most material risks are (1) a latent leakage-adjacent correctness bug in the ROSE auto-discovery loader, which can silently collapse distinct subjects into one group when a `manifest.csv` is absent, and (2) a coverage gap: the entire ROSE manifest loader — including the `split_group = subject_id` guarantee that backs the #1 ROSE non-negotiable — has zero direct tests. Both are gated behind the non-preferred discovery fallback / are mitigated by tested split primitives, so they are medium, not critical. A completeness gap also exists in the primary HTML deliverable: the cross-species roadmap figure (a named non-negotiable for the minute 6-7 visual) is generated on disk but never embedded in `reports/retivasc_pi_demo.html`. The remaining findings are low/info test-tightening and scope-discipline notes. Overall verdict: ship-able after closing the two medium loader/test items and embedding the roadmap figure.

## Git repository status (CRITICAL — infrastructure, not a code finding)

The git repository is currently **DESTROYED**:

- `.git/` exists but the branch `main` **has no commits** (`git log` / `git reflog` both report "your current branch 'main' does not have any commits yet").
- `.agents/` and `.codex/` are **empty directories**.
- There is **no reflog and history is unrecoverable**.
- The **working tree is intact** (all source, notebooks, figures, and docs present and staged) and **all 17 tests pass** (`pixi run test` → `17 passed`).

**Recommendation:** Re-initialize cleanly and create a single clean initial commit of the intact working tree, then commit this audit as a checkpoint. The working tree is the sole source of truth right now; commit it before making any further changes so subsequent edits are recoverable.

## Findings table

| # | Severity | Dimension | File:location | Title |
|---|----------|-----------|---------------|-------|
| 1 | medium | leakage | `src/retivasc/io.py:232-240` (used at 272, 286) | ROSE discovery loader infers `subject_id` from filename number only, ignoring parent directory (silent cross-fold group collapse) |
| 2 | medium | tests | `tests/test_io.py` (whole file); `src/retivasc/io.py:243-300` | ROSE manifest loader is entirely untested, including the subject-level `split_group` guarantee |
| 3 | medium | repro | `src/retivasc/report_html.py:~1041-1069` | Static HTML report omits the required cross-species roadmap figure |
| 4 | low | tests | `tests/test_splits.py:22-37`; `src/retivasc/splits.py:63,70` | `grouped_train_test_split` overlap test is redundant with the function's internal guard |
| 5 | low | tests | `tests/test_features.py:48-56` | Fractal-dimension test checks only finiteness/positivity, not meaningful box-counting math |
| 6 | low | tests | `tests/test_features.py:59-71` | `connected_component_count` counting behavior is never value-tested |
| 7 | low | tests | `tests/test_metrics.py` (whole file); `src/retivasc/metrics.py:17-54` | Documented empty-mask metric conventions are untested |
| 8 | low | repro | `src/retivasc/report_html.py:1113-1142`; `README.md:83-91` | Committed `docs/` GitHub Pages site adds publishing machinery the spec deferred |
| 9 | info | codecorrect | `src/retivasc/segment.py:17` | `frangi(black_ridges=False)` correct for OCTA but would silently invert on dark-vessel fundus if reused |
| 10 | info | repro | `pixi.toml:35` | `report` pixi task deviates from spec command but is consistent and documented |

## Detailed findings

### Finding 1 (medium, leakage) — ROSE discovery loader collapses distinct subjects into one group

- **Claim:** When no ROSE `manifest.csv` is present, `subject_id` is derived purely from the image stem via `_infer_subject_id`, which only inspects `path.stem` and ignores the parent directory. ROSE-1 stores layers and train/test partitions using repeating numeric filenames (e.g. `train/01.png` and `test/01.png`, or `AD/01.png` and `control/01.png`). Two physically distinct images sharing a stem number receive the **same** `subject_id`, silently collapsing distinct subjects into one group — exactly the silently-wrong group-column failure that non-negotiable #3 (subject-level splits) forbids, and it happens without the loud failure the spec mandates for uncertain parsing.
- **Evidence:** `_infer_subject_id` (io.py:232-240) reads only `stem = path.stem` then `re.search(r"\d+", stem)` and returns the first number; it never consults `path.parts`/parent dirs. By contrast `_infer_label` (io.py:143-153) **does** inspect `path.parts`, so `AD/01.png` and `control/01.png` resolve to `subject_id='01'` with different labels. The loader raises only when layer/label/subject_id is `None` (io.py:273), never when an ambiguous repeated number is confidently-but-wrongly parsed. The downstream `grouped_train_test_split` guard (splits.py:43-49) catches only *conflicting-label* groups, so it would surface `AD/01` vs `control/01` as a confusing error but would **not** catch a same-label cross-partition collision (e.g. `control/train/01` + `control/test/01`), which silently merges two distinct subjects.
- **Recommendation:** In the discovery path, incorporate the disambiguating parent/split/label folder into `subject_id`, or — per spec line 253 ("If filename parsing is uncertain, fail loudly and ask the user to supply a small metadata CSV") — refuse to auto-infer `subject_id` from a bare leading number and require a `manifest.csv` with an explicit `subject_id`/`split_group` column.
- **Confidence:** high. (Severity held at medium: the bug is gated behind the non-preferred auto-discovery fallback; the canonical demo path supplies a manifest, which the loader prefers at io.py:249-255. Not high because the realistic demo path avoids it; not low because it silently violates a non-negotiable when triggered.)

### Finding 2 (medium, tests) — ROSE manifest loader, including the subject-level split guarantee, is entirely untested

- **Claim:** The single most important ROSE non-negotiable — subject-level splits with `split_group = subject_id` and no leakage — has zero direct test coverage. `test_io.py` exercises only `load_fives_manifest`. A regression that set `split_group` to `image_id` (or per-row instead of per-subject) would pass the entire suite while silently introducing the exact subject-level leakage the handoff forbids.
- **Evidence:** Grep across `tests/` for `load_rose_manifest` and `split_group` returns nothing; `test_io.py` imports only `load_fives_manifest`. The ROSE loader sets `split_group = subject_id` at io.py:254 and io.py:286, but no test asserts this. The leakage primitive `assert_group_split_safe` *is* well tested (`test_splits.py`), but it only prevents leakage if `split_group` is genuinely the subject id — so the tested guard's effectiveness depends on this untested assignment.
- **Recommendation:** Add a data-free test that builds a tiny synthetic ROSE tree (or a `manifest.csv`) with two images from the same subject and asserts (a) every row's `split_group == subject_id`, (b) two same-subject rows share one `split_group`, and (c) ambiguous filenames raise `ValueError` rather than inventing labels. This also directly exercises Finding 1.
- **Confidence:** high. (Medium not high: the leakage-prevention primitive is already tested and the loader assignment is a one-liner; but it touches the #1 ROSE non-negotiable, so above low.)

### Finding 3 (medium, repro) — Static HTML report omits the required cross-species roadmap figure

- **Claim:** The pixi `report` task is `python -m retivasc.report_html`, so `report_html.py` produces the real PI deliverable (`reports/retivasc_pi_demo.html` + `docs/`). That deliverable contains roadmap **text** but never embeds `figures/cross_species_roadmap.png`, even though the figure is generated and present on disk. The cross-species hook to Howell/Reagan mouse retinal data is a named non-negotiable, and spec section 7 requires the report to "include figure and text connecting human vascular features to Howell/Reagan mouse retinal phenotypes."
- **Evidence:** `grep cross_species_roadmap reports/retivasc_pi_demo.html` → 0 matches; `grep -c cross_species_roadmap src/retivasc/report_html.py` → 0. The only `<img>` in the committed HTML is `../figures/fives_calibration_demo.png`. `figures/cross_species_roadmap.png` (124938 bytes) exists on disk and is embedded by notebook 03 (lines 157-159), but not by the report builder. The builder already has the exact embed+copy pattern for the calibration figure (lines 856, 1104-1107).
- **Recommendation:** Embed `figures/cross_species_roadmap.png` in `render_report()` and copy it into `docs/assets/` (mirroring the calibration figure), so the central cross-species narrative shows the schematic the spec calls for. Without it the 8-minute demo's minute 6-7 visual is missing from the HTML artifact.
- **Confidence:** high. (Medium: the translational narrative *text* is present and the figure exists in the retained notebook, so the story is not absent — only the schematic visual is missing from the primary deliverable; a one-line embed closes it.)

### Finding 4 (low, tests) — `grouped_train_test_split` overlap test is redundant with the function's own guard

- **Claim:** `grouped_train_test_split` already calls `assert_group_split_safe` internally before returning (splits.py:63, 70). The test re-calls it on the returned frames, so a broken splitter would raise inside the function before the test's assertion ran — the disjointness check is structurally incapable of catching what the function does not already prevent. It verifies the wrapper, not an independent property.
- **Evidence:** `assert_group_split_safe(train, test, group_col); return train, test` appears at splits.py:62-64 and 69-71 — the same call the test repeats at test_splits.py:37. The test also never exercises the `StratifiedShuffleSplit` branch's correctness (both classes survive, same-subject rows stay together) independently.
- **Recommendation:** Verify an independent property: assert `set(train.subject_id)` is disjoint from `set(test.subject_id)` directly, assert per-subject row counts are preserved, and assert both label classes appear in train and test for the stratified branch.
- **Confidence:** high. (Low: the existing `len(train)>0`/`len(test)>0` checks do catch a degenerate single-fold splitter, and the leakage property is independently covered by `test_assert_group_split_safe_raises_on_overlap`. This is a test-quality nit, not a correctness/leakage defect.)

### Finding 5 (low, tests) — Fractal-dimension test does not guard the box-counting math

- **Claim:** The test asserts only `math.isfinite(value)` and `value > 0`. A solid mask returns 2.0 and the grid mask ~1.92 — both pass; so would a near-constant or trivial area-style stub. There is no discriminating bound, so the feature math is not genuinely guarded.
- **Evidence:** Running the function: grid mask → 1.9170, full mask → 2.0000, single line → 1.0000. The test body is only `assert math.isfinite(value); assert value > 0`.
- **Recommendation:** Tighten to `1.0 < value < 2.0` for the grid mask and add a contrasting straight-line case expected near 1.0, so the test fails if the slope math degrades.
- **Confidence:** high. (Low: implementation is correct box-counting and unrelated to non-negotiables; merely a loose assertion.)

### Finding 6 (low, tests) — `connected_component_count` counting behavior is never value-tested

- **Claim:** `connected_component_count` appears in the required-keys assertion but no test verifies it counts disjoint components. A wrong-connectivity, return-0/1, or off-by-one bug would pass; the only check is key existence.
- **Evidence:** The function returns `int(labels.max())` with `connectivity=2`; two disjoint vertical lines correctly yield 2, but `test_features.py` only checks the key set on a single-line mask.
- **Recommendation:** Add a mask with two clearly separated blobs and assert `connected_component_count == 2` (and `== 0` for an empty mask), pinning the contract.
- **Confidence:** high. (Low: small correct implementation, missing assertion, unrelated to non-negotiables.)

### Finding 7 (low, tests) — Documented empty-mask metric conventions are untested

- **Claim:** The handoff (section 6.6) says "Handle empty masks explicitly and document behavior." `metrics.py` defines specific conventions — dice/iou of two empty masks return 1.0; sensitivity/specificity return 1.0 when there are no positives/negatives — and none of these branches is tested.
- **Evidence:** metrics.py lines 21-22, 30-31, 41-42, 52-53 implement the empty-case conventions. `test_metrics.py` covers only `dice(mask,mask)==1`, `dice(mask,empty)==0`, `iou(mask,mask)==1`, and one mixed sensitivity/specificity case — no two-empty-mask or all-negative inputs.
- **Recommendation:** Add `dice_score(empty,empty)==1.0`, `iou_score(empty,empty)==1.0`, and sensitivity/specificity returning 1.0 on no-positive / no-negative inputs.
- **Confidence:** high. (Low: the handoff's own `test_metrics.py` spec prescribes only the 4 tests already present, so this is unspecced hardening, not a spec violation.)

### Finding 8 (low, repro) — Committed `docs/` Pages site is publishing machinery the spec deferred

- **Claim:** The handoff (section 4) says "Do not add a docs site, CI matrix, model cards, or package-publishing machinery before the interview," and section 5 defers CI. `report_html.py` builds a self-contained GitHub Pages site (`docs/` with `.nojekyll`) and the README adds a "Publish With GitHub Pages" section — exactly the docs-site/publishing machinery the spec asked to defer, expanding the maintained surface (every `pixi run report` rewrites four HTML files plus copied assets).
- **Evidence:** `build_report` (report_html.py:1113-1142) writes `docs/index.html`, `docs/data_audit_flow.html`, copies `docs/assets/fives_calibration_demo.png`, and writes `docs/.nojekyll`; all four `docs/` files are tracked. README.md:83-91 documents Pages deployment. The spec MVP deliverable list is only `reports/retivasc_pi_demo.html` + four figures.
- **Recommendation:** Optional. If trimming, drop the `docs/` generation and Pages instructions to match the compact directive; otherwise leave as-is — the site is honest and does no scientific harm.
- **Confidence:** high. (Low: scope-discipline creep against explicit prohibition language, but no scientific harm.)

### Finding 9 (info, codecorrect) — `frangi(black_ridges=False)` would silently invert on dark-vessel fundus if reused

- **Claim:** `vesselness = filters.frangi(gray, black_ridges=False)` (segment.py:17) hardcodes bright-ridge detection. Correct for the only call site (ROSE OCTA, bright vessels), but FIVES fundus vessels are dark and would need `black_ridges=True`. If pointed at fundus images it would produce a near-inverted/mostly-wrong mask with no error.
- **Evidence:** segment.py:17 hardcodes `black_ridges=False` with no override; the function is invoked only in `notebooks/01_rose_octa_feature_demo.py:159` on OCTA, and FIVES uses manual masks for features. A bright bar segments correctly; a dark vessel on bright background would not.
- **Recommendation:** Expose `black_ridges` as a parameter (default `False` for OCTA) or add a docstring caveat. No change required for the current demo.
- **Confidence:** high. (Info: latent-reuse maintainability note; current usage is correct.)

### Finding 10 (info, repro) — `report` pixi task deviates from the spec command but is consistent and documented

- **Claim:** The spec (section 5) defines `report = "marimo export html notebooks/03_pi_demo_report.py -o reports/retivasc_pi_demo.html"`; the implementation uses a hand-built static HTML module instead. This is a deliberate, defensible design change (curated PI-facing report vs raw marimo export), the README matches it, and it runs cleanly. Recorded for awareness only; the substantive figure-embedding gap is captured in Finding 3.
- **Evidence:** pixi.toml:35 vs handoff section 5. `pixi run report` writes `reports/retivasc_pi_demo.html`; tests (17 passed) and lint pass cleanly. README documents the new command.
- **Recommendation:** No action required. If strict spec parity matters, restore the marimo-export task or add a one-line README note (largely already covered).
- **Confidence:** high. (Info: documented, working deviation.)

## Deliverable status vs the handoff MVP

The handoff MVP names **5 required artifacts**: `reports/retivasc_pi_demo.html` plus four figures.

| Artifact | Status | Notes |
|----------|--------|-------|
| `reports/retivasc_pi_demo.html` | Present | Generated by `report_html.py`; honest content. Missing the cross-species figure embed (Finding 3). |
| `figures/fives_calibration_demo.png` | Present | Embedded in the report and Pages site. |
| `figures/data_audit_flow.png` | Present | On disk. |
| `figures/cross_species_roadmap.png` | Present | On disk; embedded by notebook 03 but NOT by the report (Finding 3). |
| `figures/rose_pipeline_panel.png` | **Missing** | Blocked on ROSE raw data not being present locally. |
| `figures/rose_feature_distributions.png` | **Missing** | Blocked on ROSE raw data not being present locally. |

The two missing ROSE figures (`rose_pipeline_panel.png`, `rose_feature_distributions.png`) are **blocked on the ROSE raw dataset not being present locally** — only FIVES is available in this environment. They are not a code defect: the pipeline to produce them exists, but the input data is absent. This should be stated plainly in the demo if asked about live ROSE figures.

## Prioritized next actions

1. **Re-initialize git and create a clean initial commit** of the intact, test-passing working tree (17 tests green), then commit this audit (`reports/AUDIT.md`) as a checkpoint. Do this first — the working tree is currently the only copy of the project.
2. **Fix Finding 1 (ROSE discovery `subject_id`):** disambiguate `subject_id` with the parent folder, or fail loudly per spec line 253 and require an explicit `manifest.csv` `subject_id`/`split_group` column. This closes the only leakage-adjacent correctness bug.
3. **Add the ROSE loader test (Finding 2):** a data-free synthetic-manifest test asserting `split_group == subject_id`, same-subject rows share one group, and ambiguous filenames raise. Guards the #1 ROSE non-negotiable and covers Finding 1.
4. **Embed the cross-species roadmap figure in the HTML report (Finding 3)** and copy it into `docs/assets/`, restoring the minute 6-7 visual.
5. **Tighten the low-severity tests (Findings 4-7):** discriminating fractal-dimension bounds, a two-blob component-count assertion, and empty-mask metric-convention assertions.
6. **Decide on scope for the `docs/` Pages site (Finding 8)** — trim to match the compact directive or keep and note the intentional deviation.
7. **Add a docstring caveat to `classical_vesselness_mask` (Finding 9)** noting the bright-vessel assumption, to prevent future fundus misuse.
