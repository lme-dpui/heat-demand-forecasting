"""Feature engineering for the heat-demand pipeline.

Pure dataframe transforms that turn the raw demand and weather CSVs into the
fully featured hourly dataframe consumed by :class:`~src.dataset.HeatDataset`:
seasonal-trend-residual decomposition, lagged/future weather, holiday and
weekend encoding, and cyclical (sine/cosine) calendar features. Each function
takes the dataframe (and any config values it needs) explicitly and returns a
new dataframe, so nothing here depends on pipeline state.
"""
import numpy as np
import pandas as pd
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from definitions import DATA_DIR
from pathlib import Path


def read_demand_data(path_to_data: str) -> pd.DataFrame:
    """
    Read consumption data from file and derive its base features.

    Loads the hourly ``cons`` series, adds its additive seasonal/trend/residual
    decomposition, and appends 1-to-6-day lagged copies of each of those signals
    (``*_past_1`` .. ``*_past_6``).

    :param path_to_data: path to the demand CSV (datetime index, single column)
    :return: dataframe of consumption plus decomposition and lag features
    """

    df_demand_en = pd.read_csv(path_to_data, index_col=0)  # read the demand data as a dataframe
    df_demand_en.index = pd.to_datetime(df_demand_en.index, utc=True)
    df_demand_en.columns = ['cons']

    df_demand_en = df_demand_en.dropna()

    sd = seasonal_decompose(df_demand_en['cons'], model='additive')
    df_demand_en['seasonal'] = sd.seasonal
    df_demand_en['trend'] = sd.trend
    df_demand_en['resid'] = sd.resid

    for i in range(1, 7):
        df_demand_en['cons_past_' + str(i)] = df_demand_en['cons'].shift(i * 24)
        df_demand_en['seasonal_past_' + str(i)] = df_demand_en['seasonal'].shift(i * 24)
        df_demand_en['trend_past_' + str(i)] = df_demand_en['trend'].shift(i * 24)
        df_demand_en['resid_past_' + str(i)] = df_demand_en['resid'].shift(i * 24)

    df_demand_en = df_demand_en.dropna()

    return df_demand_en


def weather_day_merge(df_demand_en: pd.DataFrame, weather_data_path: str) -> pd.DataFrame:
    '''
    Merge the demand series with current-day and next-day weather features.

    Reads the weather CSV, attaches the weather at both horizons — current
    day (``*_f_0``, no shift) and next day (``*_f_1``, 24h ahead) — and adds
    the seasonal/trend/residual decomposition of the temperature and
    feels-like signals at each horizon (``*_f0_*`` / ``*_f1_*``).

    :param df_demand_en: demand dataframe (datetime index)
    :param weather_data_path: weather CSV path relative to ``DATA_DIR``
    :return: the demand dataframe merged with the weather features
    '''

    weather_data_path = Path(DATA_DIR, weather_data_path)  # get the path of weather data

    df_weather = pd.read_csv(weather_data_path)  # read weather data
    df_weather.index = pd.to_datetime(df_weather['dt'], unit='s', utc=True)
    df_weather.drop_duplicates(keep=False, inplace=True)  # drop duplicate entries if any, to avoid problems with merging

    df_weather = df_weather[['feels_like', 'temp', 'temp_min', 'temp_max', 'humidity', 'dew_point']]  # keep only the weather features used downstream

    # Attach the weather at each forecast horizon, tagged with the horizon index
    # (_f_0 = current day / no shift, _f_1 = next day / 24h ahead). Both horizons
    # are always built; feature_names.expand_feature_spec selects the subset.
    horizon_frames = []
    for horizon in (0, 1):
        shifted = df_weather.shift(-24 * horizon)  # weather `horizon` days ahead (0 = current day)
        shifted.columns = df_weather.columns + f'_f_{horizon}'
        horizon_frames.append(shifted)
    df_demand_en = pd.merge(df_demand_en, pd.concat(horizon_frames, axis=1), left_index=True, right_index=True)

    # Seasonal-trend-residual decomposition of the temperature/feels-like signals
    # at each horizon (current day -> *_f0_*, next day -> *_f1_*).
    decompose_bases = {'temp_max': 'tmax', 'temp_min': 'tmin', 'temp': 't', 'feels_like': 'fl'}
    for horizon in (0, 1):
        for source, base in decompose_bases.items():
            source_column = f'{source}_f_{horizon}'
            prefix = f'{base}_f{horizon}'
            decomposed = seasonal_decompose(df_demand_en[source_column], model='additive')
            df_demand_en[f'{prefix}_s'] = decomposed.seasonal
            df_demand_en[f'{prefix}_t'] = decomposed.trend
            df_demand_en[f'{prefix}_r'] = decomposed.resid

    df_demand_en = df_demand_en.dropna()  # droping the first row with NaN generated for differencing

    return df_demand_en


def encode_holiday_information(df_demand_en: pd.DataFrame, use_previous_holiday: bool) -> pd.DataFrame:
    """
    Binary encoding of holiday information. Weekends are also considered as holidays.

    Sets ``holiday`` to 1 on Danish public holidays (OR-ed with the incoming
    weekend flag). When ``use_previous_holiday`` is set, the ``*_past_6`` lag
    features on weekday holidays are additionally overwritten with the matching
    hour from the previous occurrence of the same holiday.

    :param df_demand_en: dataframe with a 'holiday' weekend flag already populated
    :param use_previous_holiday: source *_past_6 lag features from the previous holiday
    :return: the dataframe with ``holiday`` binarised (and lag features adjusted)
    """
    dk_holidays = holidays.DK()  # get the list of holidays

    holiday_df = pd.DataFrame(df_demand_en.index.date, columns=['date'], index=df_demand_en.index)

    unique_dates = sorted(holiday_df['date'].unique())
    holiday_map = {date_obj: dk_holidays.get(date_obj) for date_obj in unique_dates}

    holiday_df['holiday_name'] = holiday_df['date'].map(holiday_map)

    holiday_dates_df = holiday_df[holiday_df['holiday_name'].notna()][
        ['date', 'holiday_name']
    ].drop_duplicates().sort_values(by=['holiday_name', 'date']).reset_index(drop=True)

    # For each holiday name, shift the date to get the previous occurrence
    holiday_dates_df['previous_holiday_date_specific'] = holiday_dates_df.groupby('holiday_name')['date'].shift(1)


    holiday_df = pd.merge(
        holiday_df,
        holiday_dates_df[['date', 'holiday_name', 'previous_holiday_date_specific']],
        on=['date', 'holiday_name'],  # Merge on both to be precise
        how='left'
    )

    # For each holiday hour, build the matching hour on the previous occurrence
    # of the same holiday: previous-holiday date (midnight) + this row's hour,
    # localised to UTC. Rows without a previous holiday stay NaT.
    holiday_df.index = df_demand_en.index
    previous_holiday_midnight = pd.to_datetime(holiday_df['previous_holiday_date_specific'])
    hour_offsets = pd.Series(pd.to_timedelta(holiday_df.index.hour, unit='h'), index=holiday_df.index)
    holiday_df['target_previous_holiday_timestamp'] = pd.to_datetime(
        previous_holiday_midnight + hour_offsets, utc=True)

    columns_to_fetch = ['cons', 'seasonal', 'trend', 'resid']
    for col_name in columns_to_fetch:
        mapper_series = df_demand_en[col_name]
        new_column_name_in_holiday_df = f'{col_name}_prev_holiday'
        holiday_df[new_column_name_in_holiday_df] = holiday_df['target_previous_holiday_timestamp'].map(
            mapper_series)



    if use_previous_holiday:
        # On weekday holidays only, replace the fixed 6-day lag features with the
        # matching hour from the previous occurrence of the same holiday.
        base_cols_for_past = ['cons', 'seasonal', 'trend', 'resid']
        weekday_holiday_mask = holiday_df['target_previous_holiday_timestamp'].notna() & (holiday_df.index.dayofweek < 5)

        for col_base in base_cols_for_past:
            df_demand_en.loc[weekday_holiday_mask, f'{col_base}_past_6'] = holiday_df.loc[weekday_holiday_mask, f'{col_base}_prev_holiday']

    # A row is a named holiday when holiday_map assigned it a name; combine that
    # with the weekend flag already in `holiday`, then binarise to 0/1.
    is_holiday = holiday_df['holiday_name'].notna().to_numpy()
    df_demand_en['holiday'] = df_demand_en['holiday'].values + is_holiday  # combining weekends with holidays
    df_demand_en["holiday"] = np.where(df_demand_en["holiday"] > 0, 1, 0)  # binary encoding of dates to 0 and 1

    return df_demand_en


def generate_cylicly_encoded_data(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclicly encode hour, day, and month information via sine and cosine.

    Adds ``{hour,week,month}_{sine,cos}`` columns and their 24h-ahead copies
    (``*_f``) so the network sees calendar phase for both the input window and
    the forecast day.

    :param base_data: hourly-indexed input dataframe
    :return: ``base_data`` merged with the cyclical calendar features
    """

    # Create hourly index for our hourly sampled data
    date_index_hour = base_data.index

    encoded_list = list()
    encoded_columns = list()

    # encoding hour of day
    hourly_data = date_index_hour.hour.values
    hourly_encoded_sin = np.sin((2 * np.pi * hourly_data) / (max(hourly_data) + 1))
    hourly_encoded_cos = np.cos((2 * np.pi * hourly_data) / (max(hourly_data) + 1))
    encoded_list.append(hourly_encoded_sin)
    encoded_list.append(hourly_encoded_cos)
    encoded_columns.extend(['hour_sine', 'hour_cos'])

    # dow - day of week
    dow_data = date_index_hour.dayofweek.values
    dow_encoded_sin = np.sin((2 * np.pi * dow_data) / (max(dow_data) + 1))
    dow_encoded_cos = np.cos((2 * np.pi * dow_data) / (max(dow_data) + 1))
    encoded_list.append(dow_encoded_sin)
    encoded_list.append(dow_encoded_cos)
    encoded_columns.extend(['week_sine', 'week_cos'])

    # enconding month of year
    monthly_data = date_index_hour.month.values
    monthly_encoded_sin = np.sin((2 * np.pi * monthly_data) / (max(monthly_data) + 1))
    monthly_encoded_cos = np.cos((2 * np.pi * monthly_data) / (max(monthly_data) + 1))
    encoded_list.append(monthly_encoded_sin)
    encoded_list.append(monthly_encoded_cos)
    encoded_columns.extend(['month_sine', 'month_cos'])

    cyclic_df = pd.DataFrame(np.vstack(encoded_list).T, columns=encoded_columns)
    cyclic_df.index = date_index_hour

    future_time_data = cyclic_df.shift(-24)  # create arrays for future day
    future_time_data.columns = cyclic_df.columns + '_f'

    dataset = pd.merge(base_data, cyclic_df, left_index=True, right_index=True)
    dataset = pd.merge(dataset, future_time_data, left_index=True, right_index=True)

    return dataset


def encode_day_of_week_and_holidays(df_demand_en: pd.DataFrame, use_previous_holiday: bool) -> pd.DataFrame:
    """
    Add weekend, holiday and cyclical calendar features.

    Marks weekends in the ``holiday`` flag, folds in public holidays via
    :func:`encode_holiday_information`, adds the 24h-ahead holiday flag
    (``holiday_f``), and appends the cyclical calendar encodings.

    :param df_demand_en: consumption with weather data
    :param use_previous_holiday: forwarded to :func:`encode_holiday_information`
    :return: the dataframe with calendar/holiday features added
    """

    # day of week is from 0 to 6 for Monday to Sunday. For saturday or sunday, day of week is binarized as 1, and 0 for others below
    dayofweek = df_demand_en.index.weekday.values  # extract day of week from the dates
    dayofweek = (dayofweek > 4).astype(int)
    df_demand_en['holiday'] = dayofweek  # create a separate column for combining holiday info with weekend as we are considering holidays to have same identity as weekends
    df_demand_en = encode_holiday_information(df_demand_en, use_previous_holiday)  # binary encoding of holiday information

    df_demand_en['holiday_f'] = df_demand_en['holiday'].shift(
        -24)  # create a separate column for combining holiday info with weekend as we are considering holidays to have same identity as weekends

    df_demand_en = generate_cylicly_encoded_data(df_demand_en)  # cyclic encoding of hour, day and month information

    return df_demand_en
