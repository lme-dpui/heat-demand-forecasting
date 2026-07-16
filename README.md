# Heat Demand Forecasting via Wavelet Scalograms

[![Dataset on Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17398331.svg)](https://doi.org/10.5281/zenodo.17398331)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.egyai.2026.100704-blue)](https://doi.org/10.1016/j.egyai.2026.100704)

Day-ahead (24h-ahead), hourly district heat demand forecasting using continuous wavelet transform (CWT) time–frequency representations and a convolutional neural network.

## Overview

District heating systems need accurate demand forecasts to balance diverse energy sources, minimize carbon emissions, and reduce operating temperatures without risking supply reliability. This repository implements the forecasting approach described in:

> Ramachandran, A., Chatterjee, S., Neergaard, T. F. B., Oberndoerfer, M., Maier, A., & Bayer, S. (2026). *A deep learning framework for heat demand forecasting using time–frequency representations of decomposed features.* Energy and AI, 24, 100704. https://doi.org/10.1016/j.egyai.2026.100704

The method decomposes historical heat-consumption and weather time series, encodes them as wavelet scalogram images (via CWT), and feeds the resulting multi-channel images through a CNN (optionally augmented with self- or cross-attention) to predict the next 24 hours of hourly demand. This repo covers the three Bronderslev, Denmark district heating zones evaluated in the paper: DMA A, DMA B, and DMA C.

The paper is open access (CC BY). This repository does not bundle a copy of the PDF — use the DOI above.

## Repository structure

```
.
├── conf/                       # Hydra configuration
│   ├── README.md               # field-by-field config reference
│   ├── config.yaml             # top-level run config (paths, time window, variables)
│   └── setup/wavelet/vgg.yaml  # architecture + wavelet encoding config
├── config.py                   # structured config dataclasses
├── definitions.py              # data directory resolution
├── main.py                     # training/evaluation entry point
├── src/
│   ├── dataset.py              # data loading, wavelet encoding, train/val/test split
│   ├── features.py, feature_names.py  # feature engineering and the feature-subset spec
│   ├── image_representation.py # CWT scalogram generation
│   ├── trainer.py / tester.py  # training loop, early stopping, evaluation
│   ├── metrics.py, utils.py
│   └── model/                  # network architecture
│       ├── model.py                     # training/evaluation orchestration
│       └── WaveletScalogram.py          # VGG-style CNN over scalograms (the paper's model)
└── requirements.txt
```

## Getting started

### Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- For GPU training, install the PyTorch build matching your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)) before installing the rest of `requirements.txt` — recent GPUs (e.g. Blackwell-generation cards) need a CUDA 12.8+ wheel.

### Data

The Bronderslev heat-meter and weather data used in the paper is published as an open dataset on [Zenodo](https://zenodo.org/records/17398331) (CC BY 4.0):

> Ramachandran, A., Bayer, S., Maier, A., & Chatterjee, S. (2025). *A Real World Multi-Year Hourly District Heating Demand Data for Denmark* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17398331

It covers the three district metered areas (DMAs) evaluated in the paper, hourly from 2016-01-01 to 2019-12-31. Download and unpack it, then point `HEAT_DATA_DIR` (or `definitions.py`) at the directory holding the files. The pipeline reads them in the dataset's flat layout:

```
<HEAT_DATA_DIR>/
  <dma>_norm_aggregated.csv      # summed_averaged: True (default) — average consumption per active meter
  <dma>_aggregated.csv           # summed_averaged: False — total consumption summed across meters
  <dma>_contributing_meters.csv  # number of active meters (not read by the pipeline)
  weather.csv
```

- `<dma>` is one of `dma_a`, `dma_b`, `dma_c` (matches `paths.dma` in the config, lowercased).
- Each demand CSV is an hourly, datetime-indexed single-column series in kWh. The pipeline reads the first column as the timestamp index and the second as consumption, regardless of its header name (so the `norm_aggregated` file's `0` column is fine).
- The weather CSV needs at least: `dt` (Unix timestamp, seconds), `feels_like`, `temp`, `temp_min`, `temp_max`, `humidity`, `dew_point`.

To run on your own data instead, provide files matching the same names, layout, and columns.

### Configuration

Runs are configured with [Hydra](https://hydra.cc/). The defaults live in `conf/config.yaml` (data paths, time window, feature list) and `conf/setup/wavelet/vgg.yaml` (network architecture and wavelet encoding parameters — mother wavelet, number of scales, etc.). Any field can be overridden from the command line. See `conf/README.md` for a field-by-field reference, including the feature-variable naming convention.

## Running

Train and evaluate on a single zone with the default (VGG) architecture:

```bash
python main.py paths.dma=dma_a
```

Sweep across the three zones with a Hydra multirun:

```bash
python main.py -m paths.dma=dma_a,dma_b,dma_c
```

Outputs (checkpoints, metrics, prediction plots) are written under `outputs/` (single run) or `multirun/` (sweeps).

## Citation

If you use this code, please cite:

```bibtex
@article{ramachandran2026heat,
  title   = {A deep learning framework for heat demand forecasting using time--frequency representations of decomposed features},
  author  = {Ramachandran, Adithya and Chatterjee, Satyaki and Neergaard, Thorkil Flensmark B. and Oberndoerfer, Maximilian and Maier, Andreas and Bayer, Siming},
  journal = {Energy and AI},
  volume  = {24},
  pages   = {100704},
  year    = {2026},
  doi     = {10.1016/j.egyai.2026.100704}
}
```

If you use the dataset, please also cite it:

```bibtex
@dataset{ramachandran2025heatdata,
  title     = {A Real World Multi-Year Hourly District Heating Demand Data for Denmark},
  author    = {Ramachandran, Adithya and Bayer, Siming and Maier, Andreas and Chatterjee, Satyaki},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17398331}
}
```

## License

MIT — see `LICENSE`.
