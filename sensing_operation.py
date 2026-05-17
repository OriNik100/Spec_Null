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

SNR_dB = -20
N_trials = 10000
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
def calculate_roc_for_signal(tx_signal, SNR_dB, N_trials, true_distance, fs, name):

    estimation_errors = np.zeros(N_trials)
    delay = (2 * true_distance) / c
    Ps = np.mean(np.abs(tx_signal)**2)
    noise_power = Ps * 10**(-SNR_dB / 10)

    total_samples = len(tx_signal) + int(2 * 3000 / c * fs)
    tx_padded = np.pad(tx_signal, (0, total_samples - len(tx_signal)))
    X_f = fft(tx_padded)
    freqs = fftfreq(total_samples, 1/fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    X_f_delayed = X_f * phase_shift
    rx_signal_clean = ifft(X_f_delayed)

    max_vals_noise_only = np.zeros(N_trials)
    max_vals_signal_plus_noise = np.zeros(N_trials)

    for i in range(N_trials):
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        complex_noise = noise_i + 1j * noise_q

        rx_noise_only = complex_noise
        rx_sig_noise  = rx_signal_clean + complex_noise

        corr_noise_only = np.correlate(rx_noise_only, tx_signal, mode='valid')
        corr_sig_noise  = np.correlate(rx_sig_noise,  tx_signal, mode='valid')

        max_vals_noise_only[i] = np.max(np.abs(corr_noise_only))
        max_vals_signal_plus_noise[i] = np.max(np.abs(corr_sig_noise))

        est_delay_idx  = np.argmax(np.abs(corr_sig_noise))
        est_delay_time = est_delay_idx / fs
        estimation_errors[i] = est_delay_time - delay

    min_thresh = min(np.min(max_vals_noise_only), np.min(max_vals_signal_plus_noise))
    max_thresh = max(np.max(max_vals_noise_only), np.max(max_vals_signal_plus_noise))
    thresholds = np.linspace(min_thresh, max_thresh, 500)

    P_fa = np.zeros(len(thresholds))
    P_d  = np.zeros(len(thresholds))
    for idx, thresh in enumerate(thresholds):
        P_fa[idx] = np.sum(max_vals_noise_only > thresh) / N_trials
        P_d[idx]  = np.sum(max_vals_signal_plus_noise > thresh) / N_trials

    rmse_time = np.sqrt(np.mean(estimation_errors**2))
    rmse_distance = rmse_time * c / 2
    print(f"{name}: RMSE = {rmse_distance:.3f} m")
    return P_fa, P_d, rmse_distance


# ============================================================
# 1. Load signals + create chirp
# ============================================================
k = B / T
tx_chirp = np.exp(1j * np.pi * k * t**2)

with np.load(filename) as data:
    tx_isac   = data['isac_siganl']
    tx_vouras = data['vouras_siganl']
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
print(f"\nRunning ROC at SNR = {SNR_dB} dB ({N_trials} trials each)...")
P_fa_chirp,  P_d_chirp,  _ = calculate_roc_for_signal(tx_chirp_f,  SNR_dB, N_trials, true_distance, fs, "Chirp")
P_fa_isac,   P_d_isac,   _ = calculate_roc_for_signal(tx_isac_f,   SNR_dB, N_trials, true_distance, fs, "ISAC")
P_fa_vouras, P_d_vouras, _ = calculate_roc_for_signal(tx_vouras_f, SNR_dB, N_trials, true_distance, fs, "Vouras")

plt.figure(figsize=(10, 8))
plt.plot(P_fa_chirp,  P_d_chirp,  linewidth=2.5, color='blue',  label='Ideal LFM Chirp')
plt.plot(P_fa_isac,   P_d_isac,   linewidth=2.5, color='green', linestyle='-',  label='ISAC Signal')
plt.plot(P_fa_vouras, P_d_vouras, linewidth=2.5, color='red',   linestyle='-.', label='Vouras Signal')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title(f'ROC (filtered to [0, B], SNR = {SNR_dB} dB)', fontsize=14)
plt.xlabel('Probability of False Alarm ($P_{FA}$)', fontsize=12)
plt.ylabel('Probability of Detection ($P_D$)', fontsize=12)
plt.xlim([-0.01, 1.0]); plt.ylim([0.0, 1.05])
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
