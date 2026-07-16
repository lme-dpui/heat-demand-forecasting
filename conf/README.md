# Config reference

This project is configured with [Hydra](https://hydra.cc/). `config.yaml` holds
the top-level run settings; `setup/wavelet/vgg.yaml` holds the architecture and
wavelet-encoding parameters composed in via the `defaults` list. Every field
below can be overridden from the command line (`key.path=value`) or swept over
with `-m` — see the main `README.md` for run examples.

## `info`

| Field | Meaning |
|---|---|
| `comments` | Free-text note describing the run; written to the log, has no effect on behavior. |
| `exp_no` | Names the run's output folder (`outputs/<exp_no>/<date>/<time>/`, via the `hydra.run.dir` interpolation above). `main.py` also reassigns it in-memory to a timestamp-based name mid-run, but nothing in the active pipeline reads that reassigned value. |

## `paths`

| Field | Meaning |
|---|---|
| `dma` | DMA to load: `dma_a`, `dma_b`, or `dma_c` (lowercased to match the data layout in the main README). |
| `summed_averaged` | Selects which demand CSV variant to read: `True` → `<dma>_norm_aggregated.csv` (average consumption per active meter), `False` → `<dma>_aggregated.csv` (total consumption summed across meters). |
| `weather_data_path` | Path to the weather CSV, resolved relative to `HEAT_DATA_DIR` (see `definitions.py`); `weather.csv` in the published dataset layout. |

Run artifacts are written to the Hydra run directory (`hydra.run.dir` above) under fixed subdirectory names — `Model`, `Metrics`, `Predictions`, `TruthPrediction`, `TestData` — defined in `definitions.py` rather than configured here.

## `variables`

| Field | Meaning |
|---|---|
| `features` | The feature-subset spec (which inputs to feed the model). Not set here directly — it comes from the selected `features` set (see "Selecting features" below). Holiday handling lives inside this spec's `calendar` block (`calendar.use_previous_holiday`), so it can differ per DMA. |
| `plot_forecast_individually` | Also save one plot per individual test sample, in addition to the aggregate forecast-vs-truth plot. |
| `plot_sample_size` | Number of individual samples to plot when the above is enabled. |

### Selecting features

Which inputs the model sees is expressed as a small **subset spec** — you toggle,
per source, the raw signal, which decomposition parts, and which lags/horizons —
rather than a hand-written list of columns. `src/feature_names.py` expands the
spec into the model's input channels (one per selected feature).

The spec has `demand`, `weather` and `calendar` blocks:

```yaml
demand:
  raw: true                            # include the raw demand series (cons)
  decompose: [seasonal, trend, resid]  # any of seasonal / trend / resid
  lags_days: [6]                       # 1..6; day-lags of the raw + decomposed demand
weather:
  # Base keys are NEXT-day; *_current_day keys add the current-day horizon.
  # (This is the reverse of the calendar block, where the base is current-day.)
  decompose:                           # next-day decomposition, variable -> parts
    temp: [trend, resid]               # variable -> any of seasonal / trend / resid;
    feels_like: [trend, resid]         #   variable is temp / feels_like / temp_max / temp_min
  raw: [feels_like, temp]              # raw next-day weather (temp, feels_like, temp_min,
                                       #   temp_max, humidity, dew_point)
  future_days: [1]                     # only 1 (next-day); horizon for the next-day raw weather
  decompose_current_day:               # current-day decomposition (same variable/part options)
    temp: [trend, resid]
  raw_current_day: [feels_like, temp]  # raw current-day weather (same variable options)
calendar:
  holiday: false                       # weekend/public-holiday flag (and holiday_next_day)
  cyclical: false                      # hour/day/month sine-cosine (and cyclical_next_day)
  use_previous_holiday: false          # not a channel: on holidays, source *_past_6 lags from the previous holiday
```

**Selection order does not matter** — the expander always emits channels in a
fixed order (demand decomposition → weather decomposition [current-day, then
next-day] → raw demand → raw weather [current-day, then next-day] → calendar), so
the same subset always yields the same channels and
the same model behavior. Omit a block or key to exclude it; an unknown part or an
out-of-range lag/horizon raises an error listing the valid options.

#### Feature reference

Everything selectable is listed below and validated on expansion by
`src/feature_names.py`, so a typo, unknown variable, or out-of-range
lag/horizon fails fast with the list of valid options.

**Available variables**

| Source | Variable | Meaning | Decomposable? | Available raw? |
|---|---|---|:---:|:---:|
| Demand | `cons` | Hourly heat demand for the DMA — the target signal (the `_norm_aggregated` / `_aggregated` CSV). | ✅ | ✅ |
| Weather | `temp` | Ambient air temperature. | ✅ | ✅ |
| Weather | `feels_like` | Apparent ("feels-like") temperature. | ✅ | ✅ |
| Weather | `temp_min` | Minimum temperature. | ✅ | ✅ |
| Weather | `temp_max` | Maximum temperature. | ✅ | ✅ |
| Weather | `humidity` | Relative humidity. | ❌ | ✅ |
| Weather | `dew_point` | Dew-point temperature. | ❌ | ✅ |
| Calendar | `holiday` | Weekend / Danish-public-holiday flag (binary 0/1). | — | — |
| Calendar | `cyclical` | Sine & cosine of hour-of-day, day-of-week, and month-of-year. | — | — |

*Decomposing* a signal splits it into three additive parts you select by name:
`seasonal`, `trend`, `resid` (residual).

**Spec keys — what each one adds**

| Key | Value | Adds |
|---|---|---|
| `demand.raw` | bool | Raw demand series (`cons`) over the input window. |
| `demand.decompose` | list of `seasonal` / `trend` / `resid` | Those decomposition parts of demand. |
| `demand.lags_days` | list of ints `1`–`6` | Day-lagged copies of the raw **and** decomposed demand (`…_past_N`). |
| `weather.raw` | list of variables | Raw **next-day** weather. |
| `weather.raw_current_day` | list of variables | Raw **current-day** weather. |
| `weather.decompose` | map `variable → [parts]` | **Next-day** weather decomposition. |
| `weather.decompose_current_day` | map `variable → [parts]` | **Current-day** weather decomposition. |
| `weather.future_days` | `[1]` | Horizon for `weather.raw` — only next-day (`1`) is available. |
| `calendar.holiday` | bool | Current-day holiday flag (paper `h_categorical`). |
| `calendar.holiday_next_day` | bool | Next-day holiday flag (paper `h_categorical`). |
| `calendar.cyclical` | bool | Current-day cyclical calendar features. |
| `calendar.cyclical_next_day` | bool | Next-day cyclical calendar features. |
| `calendar.use_previous_holiday` | bool | *Not a channel.* On a holiday, source the `*_past_6` lag features from the previous occurrence of that holiday instead of a fixed 6-day lag (paper `h_lagged`). |

Omit a key (or set it to `false` / leave it empty) to exclude it.

**Horizon convention — mind the asymmetry**

- **Weather** — the base keys (`raw`, `decompose`) are the **next day** (the forecast day); the `*_current_day` keys add the current day.
- **Calendar** — the base keys (`holiday`, `cyclical`) are the **current day**; the `*_next_day` keys add the next day.
- **Demand** — the base is the current input window; `lags_days` adds past days.

**Built-column names**

Each selected feature becomes one CNN input channel. Reading a channel name or
overriding a nested key uses this naming:

| Piece | Code |
|---|---|
| Raw demand | `cons` |
| Demand decomposition | `seasonal`, `trend`, `resid` (full names) |
| Day-lag *N* | `…_past_N` |
| Weather variable base | `temp`→`t`, `feels_like`→`fl`, `temp_max`→`tmax`, `temp_min`→`tmin` |
| Weather decomposition part | `seasonal`→`s`, `trend`→`t`, `resid`→`r` |
| Weather horizon (decomposed) | current day `_f0`, next day `_f1` |
| Weather horizon (raw) | current day `_f_0`, next day `_f_1` |
| Calendar next-day copy | `…_f` |

Worked examples: `t_f1_r` = temperature · next day · residual; `fl_f0_t` =
feels-like · current day · trend; `cons_past_6` = demand lagged 6 days;
`temp_f_1` = raw next-day temperature; `week_sine_f` = next-day day-of-week sine.

Feature sets are named files under `conf/features/`, selected as a Hydra config
group rather than typed out at the call site:

```bash
python main.py features=dma_a paths.dma=dma_a   # DMA-specific set
python main.py                                                 # features=default (baseline)
```

Shipped sets: `default` (baseline), `dma_a`, `dma_b`, `dma_c` (the
paper's per-zone "Proposed" sets). To add your own, drop a
`conf/features/<name>.yaml` (starting with `# @package variables.features`)
alongside them and run `python main.py features=<name>`. To tweak a single knob
for one run, override a nested key, e.g.
`variables.features.weather.decompose.temp=[trend,resid]`.

## `model_params`

| Field | Meaning |
|---|---|
| `epoch` | Maximum training epochs. |
| `restore_epoch` | `-1` trains from scratch; any other value resumes from that epoch's saved checkpoint. |
| `lr` | Learning rate. |
| `lr_scheduler_factor` | `ReduceLROnPlateau` decay factor applied to the LR when the validation loss plateaus (paper Section 2.6.2). |
| `batch_size` | Training batch size. |
| `patience` | Early-stopping patience, in epochs. |
| `loss_function` | Training loss, e.g. `MSE`. |

## `time`

| Field | Meaning |
|---|---|
| `start_time_train` | Inclusive lower bound of the full dataset (`>=`). |
| `end_time_test` | Exclusive upper bound of the full dataset (`<`). |
| `percentage_of_train_data` | Fraction of the pre-test period actually used for training. |
| `data_length_in_weeks_test` | Length of the held-out test window, in weeks. |
| `historical_values` | Lookback window fed to the model, in hours. |
| `prediction_horizon` | Forecast horizon, in hours (24 for day-ahead). |
| `validation_period` | Fraction of the training data held out for validation. |

## Architecture / encoding (`setup/wavelet/vgg.yaml`)

Composed in via `defaults: - setup: wavelet/vgg`.

- `architecture` — VGG-style CNN topology: `convolution` (layer widths, kernel/stride/padding, pooling, batch norm) and `dense` (layer widths, dropout, activation).
- `encoding_schema` — CWT wavelet encoding: `wavelet_function` (mother wavelet, e.g. `gaus4`, `morl`, `mexh`) and `num_frequencies` (number of scales).
