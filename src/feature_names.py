"""Turn a feature *subset* spec into the model's ordered input channels.

Feature selection is expressed as a small structured subset (see
``conf/features/<name>.yaml``) rather than a hand-written list of column names:
per source you toggle the raw signal, which decomposition parts to include, and
which lags / forecast horizons. :func:`expand_feature_spec` expands that spec
into the ordered list of built-column names that :class:`~src.dataset.HeatDataset`
slices out.

The expansion order is fixed and independent of how the spec is written, so the
same subset always maps to the same channels in the same order (channel order
matters — each channel is a CNN input plane). The order is:

1. demand decomposition — current window, then each lag
2. weather decomposition — current-day, then next-day
3. raw demand — current window, then each lag
4. raw weather — current-day, then next-day
5. calendar / holiday features

The dicts below record *which built column each selectable feature maps to*.
Column names are produced by :mod:`src.features`; keep the two in sync.
"""

# Demand decomposition part -> built column name.
_DEMAND_DECOMPOSITION = {"seasonal": "seasonal", "trend": "trend", "resid": "resid"}
# Raw demand column.
_DEMAND_RAW = "cons"

# Weather variables whose signal is decomposed -> built column prefix base.
# The horizon index is appended by the expander: {base}_f0 (current day) /
# {base}_f1 (next day), matching the columns src.features builds.
_WEATHER_DECOMPOSITION_BASE = {
    "temp": "t",
    "feels_like": "fl",
    "temp_max": "tmax",
    "temp_min": "tmin",
}
# Decomposition part -> built column suffix.
_DECOMPOSITION_SUFFIX = {"seasonal": "s", "trend": "t", "resid": "r"}

# Weather variables available as raw future features.
_WEATHER_RAW = ["temp", "feels_like", "temp_min", "temp_max", "humidity", "dew_point"]

# Cyclical calendar columns, in their fixed order.
_CYCLICAL = ["hour_sine", "hour_cos", "week_sine", "week_cos", "month_sine", "month_cos"]

_MAX_LAG_DAYS = 6      # src.features builds demand lags 1..6
_MAX_FUTURE_DAYS = 1   # ceiling for weather.future_days (next-day raw); src.features builds horizons _f_0 (current) and _f_1 (next)


def _as_list(value):
    """Normalise a config value (list, OmegaConf list, or None) to a plain list."""
    if value is None:
        return []
    return list(value)


def _validate(name, values, allowed):
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {name}: {unknown}. Allowed values are: {sorted(allowed)}."
        )


def _validate_ints(name, values, lo, hi):
    bad = [v for v in values if not (isinstance(v, int) and lo <= v <= hi)]
    if bad:
        raise ValueError(f"{name} must be integers in [{lo}, {hi}]; got {bad}.")


def expand_feature_spec(features) -> list:
    """Expand a feature-subset spec into ordered built-column names.

    :param features: mapping with optional ``demand``, ``weather`` and
        ``calendar`` blocks (a plain dict or an OmegaConf ``DictConfig``):

        * ``demand.raw`` (bool) — include the raw demand series (``cons``).
        * ``demand.decompose`` (list of ``seasonal``/``trend``/``resid``).
        * ``demand.lags_days`` (list of 1..6) — day-lags added to the raw and
          decomposed demand signals.
        * ``weather.decompose`` / ``weather.decompose_current_day`` (mapping
          ``variable -> [parts]``) — decomposition of the next-day / current-day
          weather signal, for ``temp``/``feels_like``/``temp_max``/``temp_min``.
        * ``weather.raw`` / ``weather.raw_current_day`` (lists of weather
          variables) — raw next-day / current-day weather.
        * ``weather.future_days`` (list, default ``[1]``) — horizon for the
          next-day raw weather; only the next-day horizon (``1``) is available
          (current-day raw is selected via ``weather.raw_current_day``).
        * ``calendar.holiday`` / ``calendar.holiday_next_day`` (bool) and
          ``calendar.cyclical`` / ``calendar.cyclical_next_day`` (bool).

        The ``calendar`` block may also carry ``use_previous_holiday`` (bool); that
        is a data-processing switch consumed by :class:`~src.dataset.HeatDataset`,
        not an input channel, so it is ignored here.
    :return: ordered list of built-column names to slice out of the dataframe
    :raises ValueError: on an unknown part/variable or an out-of-range lag/horizon
    """
    features = features or {}
    demand = features.get("demand", {}) or {}
    weather = features.get("weather", {}) or {}
    calendar = features.get("calendar", {}) or {}

    components = _as_list(demand.get("decompose"))
    lags = _as_list(demand.get("lags_days"))
    include_raw_demand = bool(demand.get("raw", False))

    weather_decompose = weather.get("decompose", {}) or {}
    weather_decompose_current = weather.get("decompose_current_day", {}) or {}
    weather_raw = _as_list(weather.get("raw"))
    weather_raw_current = _as_list(weather.get("raw_current_day"))
    future_days = _as_list(weather.get("future_days")) or [1]

    # Validate up front so mistakes surface with a helpful message.
    _validate("demand.decompose part", components, _DEMAND_DECOMPOSITION)
    _validate_ints("demand.lags_days", lags, 1, _MAX_LAG_DAYS)
    _validate("weather.decompose variable", weather_decompose.keys(), _WEATHER_DECOMPOSITION_BASE)
    for var in weather_decompose:
        _validate(f"weather.decompose.{var} part", _as_list(weather_decompose[var]), _DECOMPOSITION_SUFFIX)
    _validate("weather.decompose_current_day variable", weather_decompose_current.keys(), _WEATHER_DECOMPOSITION_BASE)
    for var in weather_decompose_current:
        _validate(f"weather.decompose_current_day.{var} part", _as_list(weather_decompose_current[var]), _DECOMPOSITION_SUFFIX)
    _validate("weather.raw variable", weather_raw, _WEATHER_RAW)
    _validate("weather.raw_current_day variable", weather_raw_current, _WEATHER_RAW)
    _validate_ints("weather.future_days", future_days, 1, _MAX_FUTURE_DAYS)

    columns = []

    # 1. demand decomposition — current window, then each lag.
    for part in components:
        columns.append(_DEMAND_DECOMPOSITION[part])
    for lag in lags:
        for part in components:
            columns.append(f"{_DEMAND_DECOMPOSITION[part]}_past_{lag}")

    # 2. weather decomposition — current-day (_f0), then next-day (_f1).
    for var, parts in weather_decompose_current.items():
        base = _WEATHER_DECOMPOSITION_BASE[var]
        for part in _as_list(parts):
            columns.append(f"{base}_f0_{_DECOMPOSITION_SUFFIX[part]}")
    for var, parts in weather_decompose.items():
        base = _WEATHER_DECOMPOSITION_BASE[var]
        for part in _as_list(parts):
            columns.append(f"{base}_f1_{_DECOMPOSITION_SUFFIX[part]}")

    # 3. raw demand — current window, then each lag.
    if include_raw_demand:
        columns.append(_DEMAND_RAW)
        for lag in lags:
            columns.append(f"{_DEMAND_RAW}_past_{lag}")

    # 4. raw weather — current-day (_f_0), then next-day (_f_1 via future_days).
    for var in weather_raw_current:
        columns.append(f"{var}_f_0")
    for var in weather_raw:
        for day in future_days:
            columns.append(f"{var}_f_{day}")

    # 5. calendar / holiday.
    if calendar.get("holiday"):
        columns.append("holiday")
    if calendar.get("holiday_next_day"):
        columns.append("holiday_f")
    if calendar.get("cyclical"):
        columns.extend(_CYCLICAL)
    if calendar.get("cyclical_next_day"):
        columns.extend(f"{col}_f" for col in _CYCLICAL)

    return columns
