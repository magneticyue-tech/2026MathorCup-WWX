# Result interpretation

What the solver outputs mean and whether they are plausible. Writing owns this
document and fills it in before drafting the corresponding paper section;
modeling reviews the interpretation column and raises a `review/` thread when a
reading is wrong or a result is implausible (see `COLLABORATION.md`).

Keep the measured value separate from its interpretation: the value column is
objective, the interpretation column is a reading of it, and the review column
records modeling's check. The paper fuses the two, so this file is the only place
they stay separable — `sensitivity_analysis.md` and `strengths_and_weaknesses.md`
cite it rather than the paper text.

| Result (value + source) | Interpretation | Modeling review | Status |
|-------------------------|----------------|-----------------|--------|
|                         |                |                 |        |

## Notes
- "Result" cites the figure/table or `code/data/processed/` file and the seed.
- "Interpretation" states meaning and implications for
  `sensitivity_analysis.md` and `strengths_and_weaknesses.md`.
- "Modeling review": `ok`, or a link to the `review/` thread that flags it.
- Report effects as numbers or figure references, not adjectives.
- Mark an unwritten interpretation `[PLACEHOLDER]` and one awaiting a re-run
  `[PENDING]`, each naming what clears it. A row whose value is delivered but
  uninterpreted stays visible that way instead of reading as final.
