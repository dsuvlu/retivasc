# Feature Roadmap

This roadmap separates features that retivasc can compute now from features that
need artery/vein labels, optic-disc context, calibrated scale, or external
training data.

## Implemented Now

- Tortuosity burden: early paper-aligned proxy available on FIVES and ROSE.
- Caliber dispersion: FIVES-only label-free proxy for diverging large-vessel
  widths. It cannot say whether arterioles narrowed or venules widened.
- Candidate crossing density: FIVES-only 4-way skeleton topology proxy. It is
  not a validated arteriovenous crossing detector.
- Major branch count: FIVES-only late large-vessel skeleton proxy.
- ROSE macula OCTA context: density, fractal dimension, dropout heterogeneity,
  endpoints, and component summaries. These are exploratory or late/context
  features, not Reagan-style optic-disc caliber or crossing measurements.

## Deferred Features

- CRAE, CRVE, and AVR end to end: `retivasc.vascular_avr.compute_avr` computes
  the Knudtson equivalent widths once arteriole and venule widths are available,
  but the package does not yet classify arteries and veins or select the
  optic-disc annulus.
- Artery/vein classification: fundus images need color, caliber, central-reflex,
  topology, and disc-position cues. OCTA needs layer and flow context. A labeled
  artery/vein dataset such as RITE or AV-DRIVE is the right substrate for an
  end-to-end demo.
- A/V-resolved crossings: the current crossing proxy should be upgraded only
  after vessels are labeled as arteriole or venule.
- Nicking grading: this requires resolved artery-over-vein crossings plus a
  calibrated venular caliber profile through each crossing.

## Why These Stay Roadmap Items

Reagan et al. 2025 reports early vascular changes in caliber asymmetry and
arteriovenous crossings before density becomes a useful signal. Those are
precisely the features that need artery/vein identity and optic-disc context.
Until that dependency lands, retivasc should present label-free caliber
dispersion and 4-way crossing candidates as exploratory proxies, not as validated
CRAE/CRVE/AVR, AVC, or nicking measurements.

ROSE is a macula OCTA dataset. It does not contain the optic-disc-centered
large-vessel view needed for Reagan-style caliber or crossing measurements, so
ROSE analyses should emphasize tortuosity burden and clearly label density,
dropout, and fractal summaries as exploratory context.
