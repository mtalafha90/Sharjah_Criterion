CSV extraction from 2018Obs...138..267A.pdf (Alrefay et al., 2018, The Observatory, Vol. 138).

Included files:
- Table_I_positive_observations.csv
- Table_II_negative_observations.csv
- Table_III_examples_observations_parameters.csv
- Table_IV_ARCV_W_equation_5.csv
- Table_V_ARCV_W_equation_6.csv
- Table_VI_not_in_agreement_with_Yallop.csv

Notes:
- The PDF is scanned/image-only. Tables I, II and III were OCR-extracted from 600-dpi page renders and include a raw_ocr_line column for checking against the source.
- OCR sometimes confuses the printed decimal point with degree signs, hyphens, or letters. Parsed columns were normalised to decimal points where possible.
- Tables IV, V and VI were manually checked/transcribed from the page images because they are short.
- Please verify any critical numeric values against the original page images/PDF before publication-grade use.

Corrections (November 2026):
- Eight values that the OCR had misread, but that still parsed as numbers, were
  corrected in Tables I and II. Each ARCV or ARCL correction is confirmed by the
  relation cos(ARCL) = cos(ARCV) cos(DAZ) with the other two values of the same row,
  and the lag correction (078 -> 07) by HilalPy's copy of the row (LT 6.96 min); the
  raw_ocr_line column keeps the OCR text.
    Table I  1991/04/15 28.40 36.73  ARCV 11102  -> 11.02
    Table I  2001/04/24 21.42 39.83  ARCV 111.31 -> 11.31
    Table I  2009/12/17 25.62 45.62  ARCL 111.50 -> 11.50
    Table I  2011/11/26 24.60 46.45  ARCV 4.12   -> 14.12
    Table II 2005/11/02 27.52 41.70  ARCV 011.62 -> 01.62
    Table II 2010/08/10 26.37 49.82  Lag  078    -> 07
    Table II 2011/07/01 28.40 36.73  ARCV 011.34 -> 01.34
    Table II 2015/01/20 26.37 49.82  ARCV 011.00 -> 01.00
  Two misread DAZ values were also corrected, each confirmed by the same relation
  (DAZ is not an input of the two-stage model, so no result changes):
    Table I  1992/07/01 26.45 36.38  DAZ  01.14  -> 11.14
    Table II 1998/10/20 24.60 46.45  DAZ  011.62 -> 01.62
- 41 of the 466 rows still have a lag, ARCV or ARCL value that does not parse as a
  number; compare_twostep_with_alrefay_rowwise.py leaves them out (425 rows used).
