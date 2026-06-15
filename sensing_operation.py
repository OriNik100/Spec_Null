import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import helper_functions as hlp

# ============================================================
# Parameters
# ============================================================
c = 3e8
B = 2e6
fs = 5 * B
T = 60e-6
t = np.arange(0, T, 1/fs)

SNR_dB = -35
N_trials = 50000
true_distance = 1500
filename = 'simulation_results.npz'


# ============================================================
# Filtering function: brick-wall bandpass in frequency domain
# ============================================================
def bandpass_filter(x, fs, f_low, f_high):
    """
    Brick-wall bandpass filter via FFT.
    Keeps frequency bins in [f_low, f_high] only.
    Output is complex (same as input).
    """
    N = len(x)
    X = fft(x)
    freqs = fftfreq(N, 1/fs)
    mask = (freqs >= f_low) & (freqs <= f_high)
    X_filt = X * mask
    return ifft(X_filt)


# ============================================================
# Diagnostic function
# ============================================================
def analyze_signal(name, x, fs):
    N = len(x)
    energy = np.sum(np.abs(x)**2)
    avg_power = energy / N

    Nfft = 2**14
    X = np.fft.fftshift(fft(x, Nfft))
    f = np.fft.fftshift(fftfreq(Nfft, 1/fs))
    P = np.abs(X)**2
    P_norm = P / np.sum(P)
    f_center = np.sum(f * P_norm)
    beta_rms = np.sqrt(np.sum((f - f_center)**2 * P_norm))

    P_cumsum = np.cumsum(P) / np.sum(P)
    f_99_low  = f[np.searchsorted(P_cumsum, 0.005)]
    f_99_high = f[np.searchsorted(P_cumsum, 0.995)]
    occupied_bw = f_99_high - f_99_low

    autocorr = np.correlate(x, x, mode='full')
    ac_mag = np.abs(autocorr) / np.max(np.abs(autocorr))
    above_half = np.where(ac_mag > 1/np.sqrt(2))[0]
    width_3dB_samples = above_half[-1] - above_half[0]
    width_3dB_us = width_3dB_samples / fs * 1e6

    peak_idx = np.argmax(ac_mag)
    mask = np.ones_like(ac_mag, dtype=bool)
    lo = max(0, peak_idx - 3*width_3dB_samples)
    hi = min(len(ac_mag), peak_idx + 3*width_3dB_samples + 1)
    mask[lo:hi] = False
    pslr_db = 20*np.log10(np.max(ac_mag[mask]) + 1e-12)

    print(f"\n  {name}")
    print(f"  {'-'*45}")
    print(f"  RMS bandwidth (Gabor)  : {beta_rms/1e6:.3f} MHz")
    print(f"  Occupied BW (99%)      : {occupied_bw/1e6:.3f} MHz")
    print(f"  3-dB autocorr width    : {width_3dB_us:.3f} μs")
    print(f"  Peak-to-Sidelobe Ratio : {pslr_db:.2f} dB")


# ============================================================
# Plot correlation comparison
# ============================================================
def plot_correlation_comparison(signals_dict, SNR_dB, true_distance, fs, c, title_suffix=""):
    plt.figure(figsize=(12, 6))
    delay = (2 * true_distance) / c
    colors = ['blue', 'green', 'red']
    styles = ['-', '-', '-.']

    for idx, (name, tx_signal) in enumerate(signals_dict.items()):
        Ps = np.mean(np.abs(tx_signal)**2)
        noise_power = Ps * 10**(-SNR_dB / 10)

        total_samples = len(tx_signal) + int(2 * 3000 / c * fs)
        tx_padded = np.pad(tx_signal, (0, total_samples - len(tx_signal)))

        X_f = fft(tx_padded)
        freqs = fftfreq(total_samples, 1/fs)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
        X_f_delayed = X_f * phase_shift
        rx_signal_clean = ifft(X_f_delayed)

        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        rx_signal_single = rx_signal_clean + (noise_i + 1j * noise_q)

        correlation_single = np.correlate(rx_signal_single, tx_signal, mode='valid')

        time_corr = np.arange(len(correlation_single)) / fs
        plt.plot(time_corr * 1000, np.abs(correlation_single),
                 color=colors[idx % len(colors)],
                 linestyle=styles[idx % len(styles)],
                 linewidth=2,
                 label=name)

    plt.axvline(x=delay * 1000, color='black', linestyle=':', linewidth=2,
                label=f'True Delay ({delay*1000:.4f} ms)')
    plt.title(f'Cross-Correlation Comparison{title_suffix}', fontsize=14)
    plt.xlabel('Delay Time (ms)', fontsize=12)
    plt.ylabel('Correlation Magnitude', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

# ============================================================
# ROC + RMSE calculation 
# ============================================================
import numpy as np
from scipy.fft import fft, ifft, fftfreq

def calculate_roc_for_signal___(tx_signal, SNR_dB, N_trials, true_distance, fs, name):

    estimation_errors = np.zeros(N_trials)
    # c must be defined globally or passed as an argument (e.g., c = 3e8)
    delay = (2 * true_distance) / c 
    
    # Calculate the exact index corresponding to the true distance
    true_delay_idx = int(np.round(delay * fs))
    
    Ps = np.mean(np.abs(tx_signal)**2)
    noise_power = Ps * 10**(-SNR_dB / 10)

    total_samples = len(tx_signal) + int(2 * 3000 / c * fs)
    tx_padded = np.pad(tx_signal, (0, total_samples - len(tx_signal)))
    X_f = fft(tx_padded)
    freqs = fftfreq(total_samples, 1/fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    X_f_delayed = X_f * phase_shift
    rx_signal_clean = ifft(X_f_delayed)

    # Arrays to store the correlation values at the true delay index
    vals_noise_only = np.zeros(N_trials)
    vals_signal_plus_noise = np.zeros(N_trials)

    for i in range(N_trials):
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        complex_noise = noise_i + 1j * noise_q

        rx_noise_only = complex_noise
        rx_sig_noise  = rx_signal_clean + complex_noise

        corr_noise_only = np.correlate(rx_noise_only, tx_signal, mode='valid')
        corr_sig_noise  = np.correlate(rx_sig_noise,  tx_signal, mode='valid')

        # Sample the correlation at the exact known location (true distance)
        vals_noise_only[i] = np.abs(corr_noise_only[true_delay_idx])
        vals_signal_plus_noise[i] = np.abs(corr_sig_noise[true_delay_idx])

        # Estimate the delay for RMSE calculation (taking the max as before)
        est_delay_idx  = np.argmax(np.abs(corr_sig_noise))
        est_delay_time = est_delay_idx / fs
        estimation_errors[i] = est_delay_time - delay

    # Calculate dynamic thresholds based on the sampled values
    min_thresh = min(np.min(vals_noise_only), np.min(vals_signal_plus_noise))
    max_thresh = max(np.max(vals_noise_only), np.max(vals_signal_plus_noise))
    
    # Use the min and max thresholds for the linspace instead of hardcoded values
    thresholds = np.linspace(min_thresh, max_thresh, 500000) 
    P_fa = np.zeros(len(thresholds))
    P_d  = np.zeros(len(thresholds))
    
    for idx, thresh in enumerate(thresholds):
        P_fa[idx] = np.sum(vals_noise_only > thresh) / N_trials
        P_d[idx]  = np.sum(vals_signal_plus_noise > thresh) / N_trials

    rmse_time = np.sqrt(np.mean(estimation_errors**2))
    rmse_distance = rmse_time * c / 2
    print(f"{name}: RMSE = {rmse_distance:.3f} m")
    
    return P_fa, P_d, rmse_distance

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import correlate
from sklearn.metrics import roc_curve

def calculate_roc_for_signal(tx_signal, SNR_dB, N_trials, true_distance, fs, name, c=3e8, max_distance=3000):
    
    estimation_errors = np.zeros(N_trials)
    delay = (2 * true_distance) / c 
    
    # Calculate the exact index corresponding to the true distance
    true_delay_idx = int(np.round(delay * fs))
    
    Ps = np.mean(np.abs(tx_signal)**2)
    noise_power = Ps * 10**(-SNR_dB / 10)

    # Dynamic padding based on max expected distance
    max_delay = (2 * max_distance) / c
    total_samples = len(tx_signal) + int(np.ceil(max_delay * fs))
    tx_padded = np.pad(tx_signal, (0, total_samples - len(tx_signal)))
    
    X_f = fft(tx_padded)
    freqs = fftfreq(total_samples, 1/fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    X_f_delayed = X_f * phase_shift
    rx_signal_clean = ifft(X_f_delayed)

    vals_noise_only = np.zeros(N_trials)
    vals_signal_plus_noise = np.zeros(N_trials)

    # Pre-calculate standard deviation for noise
    noise_std = np.sqrt(noise_power / 2)

    for i in range(N_trials):
        # Generate complex noise
        complex_noise = np.random.normal(0, noise_std, len(rx_signal_clean)) + \
                   1j * np.random.normal(0, noise_std, len(rx_signal_clean))

        rx_noise_only = complex_noise
        rx_sig_noise  = rx_signal_clean + complex_noise

        # Using scipy's correlate with FFT is MUCH faster for long signals
        corr_noise_only = correlate(rx_noise_only, tx_signal, mode='valid', method='fft')
        corr_sig_noise  = correlate(rx_sig_noise,  tx_signal, mode='valid', method='fft')

        vals_noise_only[i] = np.abs(corr_noise_only[true_delay_idx])
        vals_signal_plus_noise[i] = np.abs(corr_sig_noise[true_delay_idx])

        # --- Sub-sample peak estimation (Parabolic Interpolation) ---
        abs_corr = np.abs(corr_sig_noise)
        est_delay_idx = np.argmax(abs_corr)
        
        # Parabolic interpolation for fine tuning the peak (if not at the very edges)
        if 0 < est_delay_idx < len(abs_corr) - 1:
            alpha = abs_corr[est_delay_idx - 1]
            beta  = abs_corr[est_delay_idx]
            gamma = abs_corr[est_delay_idx + 1]
            # Offset from the discrete peak (-0.5 to 0.5)
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            fine_est_delay_idx = est_delay_idx + p
        else:
            fine_est_delay_idx = est_delay_idx
            
        est_delay_time = fine_est_delay_idx / fs
        estimation_errors[i] = est_delay_time - delay

    # --- Fast and accurate ROC calculation using sklearn ---
    # Create labels: 0 for noise only, 1 for signal+noise
    y_true = np.concatenate([np.zeros(N_trials), np.ones(N_trials)])
    y_scores = np.concatenate([vals_noise_only, vals_signal_plus_noise])
    
    # Calculate ROC curve. fpr = P_fa, tpr = P_d
    P_fa, P_d, thresholds = roc_curve(y_true, y_scores)

    # RMSE Calculation
    rmse_time = np.sqrt(np.mean(estimation_errors**2))
    rmse_distance = rmse_time * c / 2
    print(f"{name}: RMSE = {rmse_distance:.3f} m")
    
    return P_fa, P_d, rmse_distance


# ============================================================
# Ambiguity Function Calculation and Plotting
# ============================================================
def calculate_ambiguity_function(signal, fs, max_doppler, num_doppler_bins=101):
    """
    Calculates the ambiguity function for a given signal.
    
    signal: 1D array of the transmitted signal.
    fs: Sampling frequency (Hz).
    max_doppler: Maximum Doppler frequency shift to calculate (Hz) +/-.
    num_doppler_bins: Number of samples along the Doppler axis.
    
    Returns: delays (time axis), dopplers (frequency axis), and ambiguity_matrix (normalized).
    """
    N = len(signal)
    t = np.arange(N) / fs
    
    # Create the Doppler axis
    dopplers = np.linspace(-max_doppler, max_doppler, num_doppler_bins)
    
    # Length of the delay axis (full cross-correlation length is 2*N - 1)
    delay_len = 2 * N - 1
    ambiguity_matrix = np.zeros((num_doppler_bins, delay_len), dtype=complex)
    
    # Calculate correlation for each Doppler shift
    for i, fd in enumerate(dopplers):
        # Apply Doppler shift to the signal
        doppler_shifted_signal = signal * np.exp(1j * 2 * np.pi * fd * t)
        
        # Cross-correlation with the original signal
        ambiguity_matrix[i, :] = np.correlate(doppler_shifted_signal, signal, mode='full')
        
    # Normalize the matrix and take the absolute value (magnitude)
    ambiguity_matrix = np.abs(ambiguity_matrix)
    #ambiguity_matrix /= np.max(ambiguity_matrix)
    
    # Create the delay axis centered around zero
    delays = (np.arange(delay_len) - (N - 1)) / fs
    
    return delays, dopplers, ambiguity_matrix


def plot_ambiguity_function(delays, dopplers, ambiguity_matrix, title="Ambiguity Function"):
    """
    Plots the ambiguity function as a 2D heatmap contour.
    """
    plt.figure(figsize=(10, 8))
    
    # Convert the delay axis to microseconds for better visual scaling
    X, Y = np.meshgrid(delays * 1e6, dopplers)
    
    # Create the filled contour plot (heatmap)
    cp = plt.contourf(X, Y, ambiguity_matrix, levels=50, cmap='jet')
    plt.colorbar(cp, label='Normalized Magnitude')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Delay [$\mu s$]', fontsize=12)
    plt.ylabel('Doppler Shift [Hz]', fontsize=12)
    plt.xlim(-15,15)
    plt.ylim(-20000,20000)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
# ============================================================
# 3D Ambiguity Function Plotting
# ============================================================
# Note: You might need to add this import at the top of your file 
# depending on your matplotlib version, though usually it's built-in now:
# from mpl_toolkits.mplot3d import Axes3D

def plot_ambiguity_function_3d(delays, dopplers, ambiguity_matrix, title="3D Ambiguity Function"):
    """
    Plots the ambiguity function as a 3D surface plot.
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Create a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert the delay axis to microseconds for better visual scaling
    X, Y = np.meshgrid(delays * 1e6, dopplers)
    
    # Plot the 3D surface
    # rstride and cstride control the resolution of the grid (lower is finer)
    # alpha adds a slight transparency so we can see the shape better
    surf = ax.plot_surface(X, Y, ambiguity_matrix, cmap='viridis', 
                           rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.9)
    
    # Add a color bar mapping values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Normalized Magnitude')
    
    # Set labels and title
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Delay [$\mu s$]', fontsize=12, labelpad=10)
    ax.set_ylabel('Doppler Shift [Hz]', fontsize=12, labelpad=10)
    ax.set_zlabel('Magnitude', fontsize=12, labelpad=10)
    
    
    # Adjust viewing angle for a better initial perspective (Elevation, Azimuth)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()


# ============================================================
# 1. Load signals + create chirp
# ============================================================
k = B / T
tx_chirp = np.exp(1j * np.pi * k * t**2)

with np.load(filename) as data:
    tx_isac   = data['isac_signal']
    tx_vouras = data['vouras_signal']
print("Signals loaded.")

# ============================================================
# 2. APPLY BANDPASS FILTER [0, B] = [0, 2 MHz] to ALL signals
# ============================================================
print("\n" + "="*55)
print("Applying bandpass filter [0, 2 MHz] to all signals...")
print("="*55)

tx_chirp_f  = bandpass_filter(tx_chirp,  fs, 0, B)
tx_isac_f   = bandpass_filter(tx_isac,   fs, 0, B)
tx_vouras_f = bandpass_filter(tx_vouras, fs, 0, B)

# Re-normalize to unit average power 
tx_chirp_f  = tx_chirp_f  / np.sqrt(np.mean(np.abs(tx_chirp_f)**2))
tx_isac_f   = tx_isac_f   / np.sqrt(np.mean(np.abs(tx_isac_f)**2))
tx_vouras_f = tx_vouras_f / np.sqrt(np.mean(np.abs(tx_vouras_f)**2))

# ============================================================
# 3. Verify filtering worked
# ============================================================
print("\n--- After filtering ---")
analyze_signal("Chirp  (filtered)",  tx_chirp_f,  fs)
analyze_signal("ISAC   (filtered)",  tx_isac_f,   fs)
analyze_signal("Vouras (filtered)",  tx_vouras_f, fs)

# ============================================================
# 4. Plot spectrum AFTER filtering
# ============================================================
freqs_c, S_c = hlp.spectrum(tx_chirp_f,  fs, 2**14)
freqs_i, S_i = hlp.spectrum(tx_isac_f,   fs, 2**14)
freqs_v, S_v = hlp.spectrum(tx_vouras_f, fs, 2**14)

plt.figure(figsize=(10, 6))
plt.plot(freqs_c/1e6, 20*np.log10(np.abs(S_c)+1e-80), label='Chirp',  color='blue')
plt.plot(freqs_i/1e6, 20*np.log10(np.abs(S_i)+1e-80), label='ISAC',   color='green')
plt.plot(freqs_v/1e6, 20*np.log10(np.abs(S_v)+1e-80), label='Vouras', color='red')
plt.axvline(x=0,   color='black', linestyle=':', linewidth=1)
plt.axvline(x=B/1e6, color='black', linestyle=':', linewidth=1,
            label=f'Filter band [0, {B/1e6:.0f} MHz]')
plt.title("Power Spectrum AFTER bandpass filter [0, B]", fontsize=14)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power [dB]")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.5)
plt.tight_layout()

# ============================================================
# 5. ROC at SNR=-8dB
# ============================================================
# ============================================================
# 5. ROC at SNR=-8dB
# ============================================================
print(f"\nRunning ROC at SNR = {SNR_dB} dB ({N_trials} trials each)...")
P_fa_chirp,  P_d_chirp,  _ = calculate_roc_for_signal(tx_chirp_f,  SNR_dB, N_trials, true_distance, fs, "Chirp")
P_fa_isac,   P_d_isac,   _ = calculate_roc_for_signal(tx_isac_f,   SNR_dB, N_trials, true_distance, fs, "ISAC")
P_fa_vouras, P_d_vouras, _ = calculate_roc_for_signal(tx_vouras_f, SNR_dB, N_trials, true_distance, fs, "Vouras")

plt.figure(figsize=(10, 8))
plt.loglog(P_fa_chirp,  P_d_chirp,  linewidth=2.5, color='blue',  label='Ideal LFM Chirp')
plt.loglog(P_fa_isac,   P_d_isac,   linewidth=2.5, color='green', linestyle='-',  label='ISAC Signal')
plt.loglog(P_fa_vouras, P_d_vouras, linewidth=2.5, color='red',   linestyle='-.', label='Vouras Signal')

plt.title(f'ROC (filtered to [0, B], SNR = {SNR_dB} dB)', fontsize=14)
plt.xlabel('Probability of False Alarm ($P_{FA}$)', fontsize=12)
plt.ylabel('Probability of Detection ($P_D$)', fontsize=12)

# קביעת גבולות הגיוניים לסקאלה לוגריתמית
min_pfa = 1 / N_trials # ההסתברות המינימלית שאינה אפס בהינתן מספר הניסויים
plt.xlim([5e-4, 0.1]) # מעניין אותנו P_fa נמוך ועד חצי
plt.ylim([1e-3, 1.05])   # חובה להתחיל ממספר חיובי, עד טיפה מעל 1 כדי שלא ייחתך

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()

plt.show()

# ============================================================
# 6. Correlation comparison plot
# ============================================================
signals_to_compare = {
    'Ideal LFM Chirp': tx_chirp_f,
    'ISAC Signal':     tx_isac_f,
    'Vouras Signal':   tx_vouras_f
}
plot_correlation_comparison(signals_to_compare, SNR_dB, true_distance, fs, c,
                            title_suffix=" (filtered to [0, B])")

# ============================================================
# 7. RMSE vs SNR sweep
# ============================================================
print("\n" + "="*55)
print("RMSE vs SNR sweep (filtered signals)...")
print("="*55)

SNR_range_dB = np.arange(-30, 5, 2)
N_trials_sweep = 2000

rmse_results = {
    'Ideal LFM Chirp': np.zeros(len(SNR_range_dB)),
    'ISAC Signal':     np.zeros(len(SNR_range_dB)),
    'Vouras Signal':   np.zeros(len(SNR_range_dB)),
}
signals_for_sweep = {
    'Ideal LFM Chirp': tx_chirp_f,
    'ISAC Signal':     tx_isac_f,
    'Vouras Signal':   tx_vouras_f,
}

for snr_idx, snr_val in enumerate(SNR_range_dB):
    print(f"\n--- SNR = {snr_val} dB ({snr_idx+1}/{len(SNR_range_dB)}) ---")
    for name, sig in signals_for_sweep.items():
        _, _, rmse_dist = calculate_roc_for_signal(
            sig, snr_val, N_trials_sweep, true_distance, fs, name
        )
        rmse_results[name][snr_idx] = rmse_dist

plt.figure(figsize=(10, 7))
plt.semilogy(SNR_range_dB, rmse_results['Ideal LFM Chirp'],
             marker='o', linewidth=2.2, color='blue',  label='Ideal LFM Chirp')
plt.semilogy(SNR_range_dB, rmse_results['ISAC Signal'],
             marker='s', linewidth=2.2, color='green', label='ISAC Signal')
plt.semilogy(SNR_range_dB, rmse_results['Vouras Signal'],
             marker='^', linewidth=2.2, color='red', linestyle='-.', label='Vouras Signal')
plt.title('Range Estimation RMSE vs. SNR (filtered to [0, B])', fontsize=14)
plt.xlabel('SNR [dB]', fontsize=12)
plt.ylabel('RMSE [meters] (log scale)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()

plt.show()


# ============================================================
# 8. Plot Ambiguity Functions
# ============================================================
print("\n" + "="*55)
print("Calculating Ambiguity Functions...")
print("="*55)

# Define maximum Doppler shift (e.g., 50 kHz - adjust based on your target's expected velocity)
MAX_DOPPLER = 50e3  

# --- Ideal LFM Chirp ---
delays_c, dopplers_c, af_chirp = calculate_ambiguity_function(tx_chirp_f, fs, max_doppler=MAX_DOPPLER)
plot_ambiguity_function(delays_c, dopplers_c, af_chirp, title="Ambiguity Function: Ideal LFM Chirp")

# --- ISAC Signal ---
delays_i, dopplers_i, af_isac = calculate_ambiguity_function(tx_isac_f, fs, max_doppler=MAX_DOPPLER)
plot_ambiguity_function(delays_i, dopplers_i, af_isac, title="Ambiguity Function: ISAC Signal")

# --- Vouras Signal ---
delays_v, dopplers_v, af_vouras = calculate_ambiguity_function(tx_vouras_f, fs, max_doppler=MAX_DOPPLER)
plot_ambiguity_function(delays_v, dopplers_v, af_vouras, title="Ambiguity Function: Vouras Signal")

plt.show()

# ============================================================
# 8. Plot 3D Ambiguity Functions
# ============================================================


#print("\n" + "="*55)
#print("Calculating and Plotting 3D Ambiguity Functions...")
#print("="*55)
#
#MAX_DOPPLER = 20e3  
#
## --- Ideal LFM Chirp ---
#delays_c, dopplers_c, af_chirp = calculate_ambiguity_function(tx_chirp_f, fs, max_doppler=MAX_DOPPLER)
#plot_ambiguity_function_3d(delays_c, dopplers_c, af_chirp, title="3D Ambiguity Function: Ideal LFM Chirp")
#
## --- ISAC Signal ---
#delays_i, dopplers_i, af_isac = calculate_ambiguity_function(tx_isac_f, fs, max_doppler=MAX_DOPPLER)
#plot_ambiguity_function_3d(delays_i, dopplers_i, af_isac, title="3D Ambiguity Function: ISAC Signal")
#
## --- Vouras Signal ---
#delays_v, dopplers_v, af_vouras = calculate_ambiguity_function(tx_vouras_f, fs, max_doppler=MAX_DOPPLER)
#plot_ambiguity_function_3d(delays_v, dopplers_v, af_vouras, title="3D Ambiguity Function: Vouras Signal")
#
#plt.show()
