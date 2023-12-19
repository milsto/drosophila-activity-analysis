import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2
from scipy.signal import resample
from scipy.signal import butter, sosfilt

NUM_ROWS_FOR_IMAGE = 6
NUM_COLS_FOR_IMAGE = 4
SAMPLING_PERIOD = 900  # in seconds - 30 for original rate, 900 for vanessa equivalence (min 30)
SKIP_DAYS = 2  # If 2 => skip frist two days, start from third day
LAST_FULL_DAY = 10  # if 10 => ends after full 10 days
results_path = "results.xlsx"
folder_path = "data/"


def chi_sq_periodogram(x, period_range = (16*60*60, 32*60*60), sampling_rate = 1/60, alpha = 0.05, time_resolution = 0.1*60*60):
    out = pd.DataFrame({'period': np.arange(period_range[0], period_range[1]+time_resolution, time_resolution)})
    N = len(x)
    corrected_alpha = 1 - (1 - alpha) ** (1 / N)
    out['power'] = out['period'].apply(lambda p: calc_Qp(p, x, sampling_rate))
    out['signif_threshold'] = out['period'].apply(lambda p: chi2.ppf(1-corrected_alpha, round(p*sampling_rate), loc=0, scale=1))
    out['p_value'] = out.apply(lambda row: chi2.sf(row['power'], round(row['period']*sampling_rate), loc=0, scale=1), axis=1)
    return out

def calc_Qp(target_period, values, sampling_rate):
    col_num = round(target_period * sampling_rate)
    row_num = len(values) // col_num  # K
    dt = pd.DataFrame({'col': (np.arange(len(values)) % col_num) + 1, 'values': values})
    avg_P = dt.groupby('col').mean()['values']
    avg_all = np.mean(values)
    numerator = np.sum((avg_P - avg_all) ** 2) * (dt.shape[0] * row_num)
    denom = np.sum((values - avg_all) ** 2)
    return numerator / denom


results = {}

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    print("Processing:", file_path, end=" ")
    df = pd.read_excel(file_path, skiprows=3)
    df = df.iloc[:-1]

    fig, axs = plt.subplots(nrows=NUM_ROWS_FOR_IMAGE, ncols=NUM_COLS_FOR_IMAGE, figsize=(20, 20))
    fig2, axs2 = plt.subplots(nrows=NUM_ROWS_FOR_IMAGE, ncols=NUM_COLS_FOR_IMAGE, figsize=(4*20, 20))
    fig3, axs3 = plt.subplots(nrows=NUM_ROWS_FOR_IMAGE, ncols=NUM_COLS_FOR_IMAGE, figsize=(4*20, 20))

    plt.figure(figsize=(80, 20))
    plt.plot(df["TEMPERATURE"])
    plt.savefig(f"{os.path.basename(file_name)}_temperature.png")
    plt.close()

    # replace above 99.5 percentil with the 99.5 percentil
    for i, col in enumerate(list(df.columns[7:])):
        print(col, end=" ")

        limit = df[col].quantile(0.995)
        df[col][df[col] > limit] = limit

        # remove first 2 days
        data = df[col].copy()

        # Add one measurements after very 59
        index = np.arange(59, len(data) + 1, 59)
        values = data[index - 1]
        data = np.insert(data, index, values)

        # Skip first 2 days
        data = data[2880*SKIP_DAYS:2880*LAST_FULL_DAY]

        # Filtering the data
        sos = butter(2, 1/(4*60*68), "low", output="sos", fs=1/30)
        data_filtered = sosfilt(sos, data)
        # data = data_filtered

        # calculate period and power
        autocorr = np.fft.ifft(np.abs(np.fft.fft(data))**2).real

        # data = resample(data, len(data) // 30)
        if SAMPLING_PERIOD > 30:
            data = data.reshape(-1, SAMPLING_PERIOD // 30).mean(axis=-1)

        chip = chi_sq_periodogram(data, sampling_rate=1.0/SAMPLING_PERIOD, alpha=0.05)
        amax_idx = np.argmax(chip["power"])
        chip_period = chip["period"][amax_idx] / 60. / 60. if amax_idx > 0 else -1.0
        chip_power = np.max(chip["power"]) - chip["signif_threshold"][amax_idx] if amax_idx > 0 else -1.0

        # print(chip["period"][np.argmax(chip["power"])] / 60. / 60.)

        # look only from 20 to 28h for peak
        start = int(20 * 60 * 60 / 30)
        end = int(28 * 60 * 60 / 30)
        peak_lag = np.argmax(autocorr[start:end]) + start
        peak_power = autocorr[peak_lag]
        peak_power_relative = peak_power / autocorr[0]

        peak_lag_hours = peak_lag * 30 / 60 / 60

        file_name = os.path.basename(file_path)
        if not col in results:
            results[col] = {}
        results[col][file_name + "_peak_lag"] = peak_lag_hours
        results[col][file_name + "_chip_period"] = chip_period
        results[col][file_name + "_peak_power"] = peak_power
        results[col][file_name + "_peak_power_relative"] = peak_power_relative
        results[col][file_name + "chip_power"] = chip_power

        axs[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].plot(chip["period"] / 60 / 60, chip["power"])
        axs[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].plot(chip["period"] / 60 / 60, chip["signif_threshold"])
        axs[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_xlabel("Period")
        axs[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_ylabel("Power")
        axs[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_title(col)

        axs2[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].plot(data)
        axs2[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_xlabel("Period")
        axs2[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_ylabel("Power")
        axs2[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_title(col)

        axs3[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].plot(data_filtered)
        axs3[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_xlabel("Period")
        axs3[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_ylabel("Power")
        axs3[i // NUM_COLS_FOR_IMAGE, i % NUM_COLS_FOR_IMAGE].set_title(col)
    
    fig.savefig(f"{os.path.basename(file_name)}.png")
    fig2.savefig(f"{os.path.basename(file_name)}_data.png")
    fig3.savefig(f"{os.path.basename(file_name)}_data_filtered.png")

    print()

df_results = pd.DataFrame(results)
df_results['mean'] = df_results[df_results > 0].mean(axis=1)
df_results['std'] = df_results[df_results > 0].std(axis=1)
df_results['median'] = df_results[df_results > 0].median(axis=1)
df_results.to_excel(results_path)
