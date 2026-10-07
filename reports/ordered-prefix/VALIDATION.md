# Report validation

Completed 7 October 2026. Checks examine the handover report and stored exports; they do not constitute new model inference or a fresh notebook execution.

- `python3 validation/check_report.py` passes. It checks balanced HTML tag nesting, unique anchor IDs, fragment targets, local file targets, image alternative text, absence of external rendering dependencies and possible IPv4 addresses.
- All four assets are byte-identical to the four PNG images embedded in the September 15 server HTML. No scientific figure was redrawn or altered.
- Four primary model/domain combinations each contain 10,000 images, 200 classes and 90,000 nine-budget rows according to the saved scope table.
- Across all nine budgets and three metrics, saved absolute means reproduce the direct domain-gap CSV and endpoint-adjusted difference CSV. Relative-progress means reproduce the ratio and paired progress-difference CSVs. All bootstrap denominator minima are positive. Both reported primary LPIPS endpoint estimates and the presented absolute LPIPS values are checked against the CSVs.
- All six historical adjusted LPIPS estimates and their interval endpoints match the surviving historical JSON summaries after rounding. Earlier HTML transcription discrepancies are documented in `SOURCES.md`.
- Browser CLI visual inspection used an isolated `supervisor-handover` session on a `file://` URL. Desktop screenshots at 1440 × 1000 and a mobile screenshot at 390 × 844 are retained here. The inspected layouts preserve the warm editorial style and have no page-wide horizontal overflow. Tables have their own horizontal scrolling when needed.
- An offline browser reload reports document state `complete`, all four images loaded, zero remote resource requests, and page width equal to the 1440-pixel viewport. No browser page errors were reported.
- Figures link to their local original-resolution files. Citation links require connectivity; report rendering does not.
- Existing package notebook and runner paths were inspected read-only. Package README and `docs/notebooks.md` were still being assembled at the time of report validation, so their guidance remains explicitly planned. Package runner execution and Git publication belong to the packaging task.

Ship `ordered-prefix-ood.html`, `SOURCES.md`, and all four files in `assets/` together. `sources/` and `validation/` are working evidence/inspection artifacts and are not required to read the public report.
