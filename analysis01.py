import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.signal import detrend, butter, filtfilt, welch
from scipy.stats import f
import pywt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Specify the folder path containing the CSV files
folder_path = "C:/Users/anapb/Documents/LIFUdata/9088/5mark/"

# Function to extract timestamp from filename and convert to datetime object
def extract_timestamp(filename):
    basename = os.path.basename(filename)
    timestamp_str = basename.split('.')[0]
    return datetime.strptime(timestamp_str, "%y%m%d_%H%M%S")

# List of file names and their timestamps
files = [(os.path.join(folder_path, f), extract_timestamp(f)) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Sort files by timestamp
sorted_files = sorted(files, key=lambda x: x[1])

# Define a function to preprocess the data
def preprocess_data(data):
    # Convert time from milliseconds to seconds
    data['time'] = data['time'] / 1000
    # Remove the first second of data
    data = data[data['time'] > 1].reset_index(drop=True)
    # Calculate the norm of the acceleration, subtracting the acceleration due to gravity (1 g)
    data['acc_norm'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2) - 1
    data['acc_norm'] = detrend(data['acc_norm'], type='linear')
    return data

# Remove the average offset of the norm of the acceleration for each dataset individually
def remove_dataset_offset(data):
    offset = data['acc_norm'].mean()
    data['acc_norm'] -= offset
    return data

# Apply a Butterworth filter to the data for frequencies below 100 Hz
def apply_butterworth_filter(data, cutoff=50, start=1, fs=1000):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    normal_start = start / nyquist
    b, a = butter(6, normal_cutoff, btype='low', analog=False)
    data['acc_norm'] = filtfilt(b, a, data['acc_norm'])
    b, a = butter(3, normal_start, btype='high', analog=False)
    data['acc_norm'] = filtfilt(b, a, data['acc_norm'])

    return data

# Load datasets with session names
session_data = {}
for i, (file, timestamp) in enumerate(sorted_files):
    session_name = f"Session {i+1}"
    data = pd.read_csv(file, header=None, names=['time', 'acc_x', 'acc_y', 'acc_z'])
    data = preprocess_data(data)  # Preprocess data (convert time, remove first second, calculate norm)
    data = remove_dataset_offset(data)  # Remove offset
    data = apply_butterworth_filter(data)  # Apply Butterworth filter
    session_data[session_name] = data

def nextpow2(x):
    return np.ceil(np.log2(x))

# Calculate the frequency spectrum of the norm of the acceleration for each dataset
def calculate_normalized_frequency_spectrum(data, fs=None, nperseg=None, noverlap=None):
    # Ensure 'acc_norm' and 'time' are numpy arrays
    acc_norm = np.asarray(data['acc_norm'])
    time = np.asarray(data['time'])
# Calculate the sampling frequency if not provided
    if fs is None:
        fs = 1.0 / (time[1] - time[0])
# Calculate the Power Spectral Density using Welch's method
    freq, psd = welch(acc_norm, fs=fs, nperseg=round(3*fs), noverlap=round(1.5*fs), nfft=nperseg)
    return freq, psd

# def calculate_normalized_frequency_spectrum(data, fs=None):
#     # Ensure 'acc_norm' and 'time' are numpy arrays
#     acc_norm = np.asarray(data['acc_norm'])
#     time = np.asarray(data['time'])
#
#     # Calculate the sampling frequency if not provided
#     if fs is None:
#         fs = 1.0 / (time[1] - time[0])
#
#     # Compute the FFT of the normalized acceleration
#     n = len(acc_norm)
#     fft_result = np.fft.fft(acc_norm)
#     fft_freq = np.fft.fftfreq(n, d=1/fs)
#
#     # Only keep the positive frequencies and corresponding FFT results
#     positive_freqs = fft_freq[:n // 2]
#     positive_fft_result = fft_result[:n // 2]
#
#     # Calculate the Power Spectral Density (PSD)
#     psd = (np.abs(positive_fft_result) ** 2) / (fs * n)
#
#     return positive_freqs, psd

# Calculate the wavelet frequency density plot of the norm of the acceleration for each dataset
def calculate_wavelet_frequency_density(data, fs=None, scales=None):
    # Ensure 'acc_norm' and 'time' are numpy arrays
    acc_norm = np.asarray(data['acc_norm'])
    time = np.asarray(data['time'])

    # Calculate the sampling frequency if not provided
    if fs is None:
        fs = 1.0 / (time[1] - time[0])

    if scales is None:
        scales = np.arange(1, 128)

    # Compute Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(acc_norm, scales, 'cmor', sampling_period=1/fs)

    return time, frequencies, np.abs(coefficients)


def add_wavelet_and_power_density_plot(fig, time, frequencies, coefficients, session_name, row, col, nsessions):
    # Calculate power density
    power_density = np.mean(np.abs(coefficients) ** 2, axis=1)

    # Create subplots:
    fig_wav = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=True, horizontal_spacing=0.005)

    # Add wavelet heatmap
    fig_wav.add_trace(go.Heatmap(
        z=coefficients,
        x=time,
        y=frequencies,
        colorscale='Viridis',
        name=session_name,
        legendgroup=session_name,
        showlegend=False,
        showscale=(row==1)
    ), row=1, col=1)
    fig_wav.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig_wav.update_yaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    # set interval between 0 and 20 Hz on y axis
    fig_wav.update_yaxes(range=[0, np.log10(20)], row=1, col=1)

    # Add power density line plot
    fig_wav.add_trace(go.Scatter(
        x=power_density,
        y=frequencies,
        mode='lines',
        name="Power Density",
        line=dict(color='orange'),
        showlegend=False
    ), row=1, col=2)
    fig_wav.update_xaxes(title_text="Power Density", row=1, col=2)
    fig_wav.update_yaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig_wav.update_yaxes(range=[0, np.log10(20)], row=1, col=2)

    fig.add_trace(fig_wav.data[0], row=row, col=1)
    fig.add_trace(fig_wav.data[1], row=row, col=2)

    return fig

# Recalculate the normalized frequency spectrum for each session
freq_spectra = {}
for session_name, data in session_data.items():
    freq, norm_spectrum = calculate_normalized_frequency_spectrum(data,1000)
    freq_spectra[session_name] = (freq, norm_spectrum)

colors = [
    'blue', 'red', 'green', 'purple', 'orange', 'brown',
    'pink', 'gray', 'yellow', 'teal', 'cyan', 'magenta',
    'lime', 'maroon', 'navy', 'olive', 'aqua', 'fuchsia',
    'gold', 'silver'
]
ext_colors = colors * ((100 // len(colors)) + 1)

# Generate the colormap
color_map = {f"Session {i + 1}": ext_colors[i] for i in range(100)}

# Determine the number of rows needed for the subplots
num_sessions = len(session_data)
fig_wavelet = make_subplots(rows=num_sessions, cols=2, column_widths=[0.8, 0.2], shared_yaxes=True, horizontal_spacing=0.005, shared_xaxes=False, subplot_titles = [
    f"Session {i//2 + 1}" if i % 2 == 0 else ""
    for i in range(len(session_data.keys()) * 2)
], vertical_spacing=0.1)

# 1) Histogram of the norm of the acceleration
fig = make_subplots(rows=1, cols=1, subplot_titles=["Histogram of Acceleration Norm"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Histogram(x=data['acc_norm'], opacity=0.5, name=session_name, marker_color=color, legendgroup="g1",
        showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Acceleration Norm (m/s^2)", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_hist = fig

# 2) Linear plot of the acceleration in the X axis for each session
fig = make_subplots(rows=1, cols=1, subplot_titles=["Acceleration X"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Scatter(x=data['time'], y=data['acc_x'], mode='lines', name=session_name, marker_color=color, legendgroup="g2",
        showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration X (m/s^2)", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_acc_x = fig

# 3) Linear plot of the acceleration in the Y axis for each session
fig = make_subplots(rows=1, cols=1, subplot_titles=["Acceleration Y"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Scatter(x=data['time'], y=data['acc_y'], mode='lines', name=session_name, marker_color=color, legendgroup="g3",
        showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration Y (m/s^2)", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_acc_y = fig

# 4) Linear plot of the acceleration in the Z axis for each session
fig = make_subplots(rows=1, cols=1, subplot_titles=["Acceleration Z"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Scatter(x=data['time'], y=data['acc_z'], mode='lines', name=session_name, marker_color=color, legendgroup="g4", showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration Z (m/s^2)", row=4, col=1)
fig.update_layout(font=dict(size=15))
fig_acc_z = fig

# 5) Time series of the norm of the acceleration for each session
fig = make_subplots(rows=1, cols=1, subplot_titles=["Acceleration Norm"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Scatter(x=data['time'], y=data['acc_norm'], mode='lines', name=session_name, marker_color=color, legendgroup="g5", showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration Norm (m/s^2)", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_acc_norm = fig

# 6) Frequency spectrum of the norm of the acceleration with normalization
fig = make_subplots(rows=1, cols=1, subplot_titles=["Normalized Frequency Spectrum of Acceleration Norm"])
for session_name, (freq, norm_spectrum) in freq_spectra.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Scatter(x=freq, y=norm_spectrum, mode='lines', name=session_name, marker_color=color, legendgroup="g6", showlegend=True), row=1, col=1)
fig.update_xaxes(range=[0, 20], row=1, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Normalized Power (m^2/s^3)", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_freq_spectrum = fig

# 7) Boxplot of the norm of the acceleration for each session
fig = make_subplots(rows=1, cols=1, subplot_titles=["Boxplot of Acceleration Norm"])
for session_name, data in session_data.items():
    color = color_map.get(session_name, 'black')
    fig.add_trace(go.Box(y=data['acc_norm'], name=session_name, marker_color=color, legendgroup="g7", showlegend=True), row=1, col=1)
fig.update_xaxes(title_text="Session", row=1, col=1)
fig.update_yaxes(title_text="Acceleration Norm (m/s^2)", row=1, col=1)
fig.update_layout(font=dict(size=15))
fig_boxplot = fig

# Add wavelet frequency density plots for each session
for i, (session_name, data) in enumerate(session_data.items(), start=1):
    time, frequencies, coefficients = calculate_wavelet_frequency_density(data)
    index = frequencies <= 20
    frequencies = frequencies[index]
    coefficients = coefficients[index, :]
    fig_wavelet = add_wavelet_and_power_density_plot(fig_wavelet, time, frequencies, coefficients, session_name, row=i, col=1, nsessions=len(session_data))
    fig_wavelet.update_xaxes(title_text="Time (s)", row=i, col=1)
    fig_wavelet.update_yaxes(title_text="Frequency (Hz)", row=i, col=1)


# Update layout with larger height and legends on the side
fig_wavelet.update_layout(title_text="Wavelet Frequency Density Plots", showlegend=False, height=1000)

# Increase font size
fig_wavelet.update_layout(font=dict(size=18))

# Create a summary table with statistical analysis for each session
summary_stats = []

for session_name, data in session_data.items():
    stats = {
        "Session": session_name,
        "Mean Acceleration Norm": data['acc_norm'].mean(),
        "Standard Deviation Acceleration Norm": data['acc_norm'].std(),
        "Median Acceleration Norm": data['acc_norm'].median()
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats).round(4)

# Perform statistical tests to see if there is a significant difference between the sessions
session_names = list(session_data.keys())
ftest_results = []

for i in range(len(session_names) - 1):
    for j in range(i + 1, len(session_names)):
        session1 = session_names[i]
        session2 = session_names[j]

        # Calculate F-Test
        var1 = np.var(session_data[session1]['acc_norm'], ddof=1)
        var2 = np.var(session_data[session2]['acc_norm'], ddof=1)

        F = var1 / var2
        df1 = len(session_data[session1]['acc_norm']) - 1
        df2 = len(session_data[session2]['acc_norm']) - 1

        p_value = 1 - f.cdf(F, df1, df2)

        if F > 1:
            p_value = 2 * (1 - f.cdf(F, df1, df2))
        else:
            p_value = 2 * f.cdf(F, df1, df2)

        ftest_results.append({
            "Comparison": f"{session1} vs {session2}",
            "F-Statistic": round(F, 7),
            "P-Value": "{:.2e}".format(p_value)
        })

ftest_df = pd.DataFrame(ftest_results)

# Create rounded summary statistics table
summary_table_rounded = go.Figure(data=[go.Table(
    header=dict(values=list(summary_df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[summary_df[col] for col in summary_df.columns],
               fill_color='lavender',
               align='left'))
])

# Create rounded F-Test results table
ftest_table_rounded = go.Figure(data=[go.Table(
    header=dict(values=list(ftest_df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[ftest_df[col] for col in ftest_df.columns],
               fill_color='lavender',
               align='left'))
])

# Convert rounded tables to HTML
summary_table_rounded_html = summary_table_rounded.to_html(full_html=False, include_plotlyjs='cdn')
ftest_table_rounded_html = ftest_table_rounded.to_html(full_html=False, include_plotlyjs='cdn')

# Create the updated HTML content with interactive tables and rounded values
html_content_rounded = f"""
<h1>Comparison of Sessions</h1>
<h2> Histogram of Acceleration Norm</h2>
{fig_hist.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Frequency Spectrum</h2>
{fig_freq_spectrum.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Boxplot of Acceleration Norm</h2>
{fig_boxplot.to_html(full_html=False, include_plotlyjs='cdn')}
<h2> Acceleration norm vs Time</h2>
{fig_acc_norm.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Wavelet Analysis</h2>
{fig_wavelet.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Summary Statistics</h2>
{summary_table_rounded_html}
<h2>F-Test Results</h2>
{ftest_table_rounded_html}
"""

with open("./analysis.html", "w") as f:
    f.write(html_content_rounded)
