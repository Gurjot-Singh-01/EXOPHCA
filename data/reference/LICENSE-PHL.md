# PHL Habitable Worlds Catalog — licence and required credit

This file records the terms under which the PHL catalogue in this repository is
redistributed. It covers three files:

| File | What it is |
| --- | --- |
| `data/reference/hwc.csv` | The PHL Habitable Worlds Catalog, as downloaded (5,599 rows) |
| `data/reference/hwc_potentially_habitable.csv` | The same catalogue's potentially-habitable subset |
| `data/processed/phl_comparison.csv` | A derived table written by `scripts/validate_phl.py`: matched rows, this project's score beside the published PHL ESI |

## Licence

**Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported**
(CC BY-NC-SA 3.0), as stated by the PHL for its website content.

Source page: <https://phl.upr.edu/phl-website-content-license>

The licence page states that the PHL website content — *"which includes text,
images, animations, videos, and data"* — is licensed under CC BY-NC-SA 3.0
Unported, and that it "can be copied, distributed, transmitted, and adapted"
under three conditions.

## The three conditions

1. **Attribution.** Credit the catalogue as
   `Credit: The Planetary Habitability Laboratory @ UPR Arecibo (phl.upra.edu)`
   — or simply `PHL @ UPR Arecibo` — in a way that does not imply the PHL
   endorses this project.
2. **Non-commercial.** The data may not be used for commercial purposes.
3. **Share-alike.** If you alter, transform or build upon the data, the result
   must be distributed under the same or a similar licence.

Any of these conditions can be waived by the PHL; the licence page names
Prof. Abel Méndez (`abel.mendez@upr.edu`) as the contact for a waiver. This
project has not requested one and relies on the licence as published.

## What follows for this repository

- The root `LICENSE` (MIT) covers this project's own code and documentation
  **only**. It does not cover the three files above.
- `data/processed/phl_comparison.csv` is a derivative of the catalogue, so it
  carries the same CC BY-NC-SA 3.0 terms: attribution, non-commercial use, and
  share-alike on redistribution.
- The catalogue is used **only** for external validation. No PHL value enters
  the score, the habitable-zone class, the tier or any probability;
  `tests/test_reference_cases.py` asserts that the science modules contain no
  PHL reference strings.
- The files are unchanged from the download. Nothing in this repository
  relicenses them.

The catalogue page records "Last Update March 21, 2024" at the time these terms
were read (2026-11-19).
