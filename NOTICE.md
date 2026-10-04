# NOTICE — third-party material

This file records everything in this repository that came from someone else, what its
licence is, and how it is used. It complements — it does not replace — [`LICENSE`](LICENSE)
(code, MIT) and [`LICENSE-DATA`](LICENSE-DATA) (data, CC BY 4.0).

> Scope note: those two files say *what grant applies to which path*. This file says
> *what is not originally ours*. A path can appear in both.

---

## 1. Bundled — code that ships inside this repository

### `haenv_kernel/`

The L0 kernel this project builds on. It carries its own licence file at
[`haenv_kernel/LICENSE`](haenv_kernel/LICENSE) — **MIT,
Copyright (c) 2026 Theta Health** — and is therefore *not* covered by the repository's
own MIT grant; read that file instead.

---

## 2. Measured on — reference data that is *not* redistributed here

### LifeSnaps

The wearable streams are shaped by empirical constants measured on the **LifeSnaps** dataset:

> Yfantidou, S., Karagianni, C., Efstathiou, S., Vakali, A., Palotti, J., Giakatos, D. P.,
> Marchioro, T., Kazlouski, A., Ferrari, E., & Girdzijauskas, Š. (2022).
> *LifeSnaps: a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the
> wild.* Zenodo. [doi:10.5281/zenodo.7229547](https://doi.org/10.5281/zenodo.7229547) —
> **CC BY 4.0**. Described in *Scientific Data* 9:663 (2022),
> [doi:10.1038/s41597-022-01764-x](https://doi.org/10.1038/s41597-022-01764-x).

No LifeSnaps record is redistributed here. What this repository carries is a handful of aggregate
summary statistics per stream — a distribution family, a spread, a lag-1 coefficient, an
availability rate and two transition probabilities — listed with their provenance in
[`haenv/wearable.py`](haenv/wearable.py) and in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
CC BY 4.0 is the same grant this repository's own data carries ([`LICENSE-DATA`](LICENSE-DATA)).

---

## 3. Runtime and build dependencies

| | licence |
|---|---|
| [PyYAML](https://pyyaml.org/) — the only runtime dependency | MIT |
| `actions/checkout`, `actions/setup-python` — CI only | MIT |

No GPL, AGPL, non-commercial or unlicensed dependency is present. Re-check with
`pip-licenses` or `uv tree` after any dependency change.

---

## 4. Everything else

Original work of Theta Health, under [`LICENSE`](LICENSE) (code) or
[`LICENSE-DATA`](LICENSE-DATA) (data). All patients, courses and clinical events in the
shipped case packs are **generated** — see [`docs/DATA_CARD.md`](docs/DATA_CARD.md) for
how, and [`docs/ETHICS.md`](docs/ETHICS.md) for what that does and does not entitle you to
conclude.

If you believe something in this repository is yours and is not listed above, please tell
us — see [`SECURITY.md`](SECURITY.md), which routes exactly that report privately.
