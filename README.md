# GY-302 (BH1750FVI) calibration and GUM uncertainty budget, 1–43 lx

Code and data for the manuscript:

> J. A. Carvalho Neto, R. Y. A. Fontana, A. L. V. Gimenes, M. M. Frigo and F. L. Caneppele,
> *Calibration and uncertainty evaluation of a low-cost digital illuminance sensor (BH1750FVI) in the 1–43 lx
> range for poultry-house lighting control*, submitted to MAPAN – Journal of the Metrology Society of India.

This repository reproduces every number in Tables 2–4 and Figs. 1–3 of the manuscript. The analysis covers:
- ordinary least-squares calibration of the raw sensor indication (Stage 1, n = 26) and validation of the
  firmware-implemented correction on an independent data set (Stage 2, n = 22);
- measurement errors and the joint F-test of the identity line;
- an uncertainty budget following the GUM (JCGM 100:2008), with expanded uncertainty (k = 2) at 10, 25 and 33 lx.

## Contents

| Path | Description |
|---|---|
| `gum_uncertainty_budget.py` | Regression, measurement errors, identity tests and GUM budget. It prints the results and writes `gum_results.json`. |
| `make_figures.py` | Figures 1–3 (600 dpi PNG and TIFF) in `figures/`, built from `gum_results.json`. |
| `data/stage1_calibration.csv` | Stage 1: raw GY-302 indication and reference illuminance (n = 26). |
| `data/stage2_validation.csv` | Stage 2: corrected GY-302 indication and reference illuminance (n = 22). |
| `results/expected_output.txt` | Reference console output of `gum_uncertainty_budget.py`. |

The same 48 paired observations are embedded in `gum_uncertainty_budget.py`. The CSV files are a convenience copy.

## Usage

```bash
pip install -r requirements.txt
python gum_uncertainty_budget.py      # prints the results and writes gum_results.json
python make_figures.py                # writes figures/Fig1–3
```

The script asserts that the regression coefficients and error measures match the values reported for the same data.
Its output should be identical to `results/expected_output.txt`.

Tested with Python 3.15, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## Data provenance

Indications came from a GY-302 module (ROHM BH1750FVI, continuous high-resolution mode, 1 lx resolution). Reference
values came from an ITR-3000 digital luxmeter. The source was a dimmable white LED lamp, measured at night in the laboratory.
The data are transcribed verbatim from Tables 5 and 6 of:

> J. A. Carvalho Neto, *Sistema de iluminação de granjas controladas por lógica fuzzy* [Fuzzy-logic controlled lighting
> system for poultry houses], M.Sc. dissertation, Faculdade de Ciências Agronômicas, Universidade Estadual Paulista
> (Unesp), Botucatu, 2021.

The reference luxmeter has no calibration certificate. The uncertainties are therefore conditional on the luxmeter
having conformed to its specification (±6 % of reading for LED sources). See the manuscript for details.

## How to cite

Cite the manuscript and this archive. The archive DOI is given by the Zenodo badge and in `CITATION.cff`.

## Licence

- Code: MIT (see `LICENSE`).
- Data (`data/`, `results/`): CC BY 4.0 (see `LICENSE-DATA`).
