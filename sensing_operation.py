import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import helper_functions as hlp

from scipy.signal import firwin, lfilter


c = 3e8      
B = 2e6           
fs = 5 * B          
T = 60e-6       
t = np.arange(0, T, 1/fs)

SNR_dB = -20
N_trials = 1000
true_distance = 1500 
filename = 'simulation_results.npz'


def plot_correlation_comparison(signals_dict, SNR_dB, true_distance, fs, c):
    
    plt.figure(figsize=(12, 6))
    delay = (2 * true_distance) / c
    
    # צבעים וסגנונות שיתאימו לגרף ה-ROC שיצרנו מקודם
    colors = ['blue', 'green', 'red']
    styles = ['-', '-', '-.']
    
    for idx, (name, tx_signal) in enumerate(signals_dict.items()):
        Ps = np.mean(np.abs(tx_signal)**2)
        noise_power = Ps * 10**(-SNR_dB / 10)
        
        # 1. יצירת האות המושהה בסביבה
        total_samples = len(tx_signal) + int(2 * 3000 / c * fs)
        tx_padded = np.pad(tx_signal, (0, total_samples - len(tx_signal)))
        X_f = fft(tx_padded)
        freqs = fftfreq(total_samples, 1/fs)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
        X_f_delayed = X_f * phase_shift
        rx_signal_clean = ifft(X_f_delayed)
        
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        rx_signal_single = rx_signal_clean + 0*(noise_i + 1j * noise_q)
        
        
        correlation_single = np.correlate(rx_signal_single, tx_signal, mode='valid')
        
        
        time_corr = np.arange(len(correlation_single)) / fs
        plt.plot(time_corr * 1000, np.abs(correlation_single), 
                 color=colors[idx % len(colors)], 
                 linestyle=styles[idx % len(styles)], 
                 linewidth=2,
                 label=name)
    
    # סימון המיקום האמיתי של המטרה
    plt.axvline(x=delay * 1000, color='black', linestyle=':', linewidth=2, label=f'True Delay ({delay*1000:.4f} ms)')
    
    plt.title(f'Cross-Correlation Comparison (SNR = {SNR_dB}dB)', fontsize=14)
    plt.xlabel('Delay Time (ms)', fontsize=12)
    plt.ylabel('Correlation Magnitude', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    # -- טיפ למחקר: אם תרצה לראות את ההבדלים בפיקים מקרוב (זום-אין), --
    # -- פשוט תוריד את סולמית ההערה מהשורה הבאה: --
    #plt.xlim((delay * 1000) - 0.005, (delay * 1000) + 0.005)
    
    plt.tight_layout()
def calculate_roc_for_signal(tx_signal, SNR_dB, N_trials, true_distance, fs):

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
    true_delay_sample = int(delay * fs)

    for i in range(N_trials):
        
        noise_i = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        noise_q = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal_clean))
        complex_noise = noise_i + 1j * noise_q
        
        rx_noise_only = complex_noise  #H0
        rx_sig_noise = rx_signal_clean + complex_noise #H1
        
        
        corr_noise_only = np.correlate(rx_noise_only, tx_signal, mode='valid')  
        corr_sig_noise = np.correlate(rx_sig_noise, tx_signal, mode='valid')    
        
        
        max_vals_noise_only[i] = np.max(np.abs(corr_noise_only))
        max_vals_signal_plus_noise[i] = np.max(np.abs(corr_sig_noise))

   
    min_thresh = min(np.min(max_vals_noise_only), np.min(max_vals_signal_plus_noise))
    max_thresh = max(np.max(max_vals_noise_only), np.max(max_vals_signal_plus_noise))
    thresholds = np.linspace(min_thresh, max_thresh, 500)

    P_fa = np.zeros(len(thresholds))
    P_d = np.zeros(len(thresholds))

    for idx, thresh in enumerate(thresholds):
        P_fa[idx] = np.sum(max_vals_noise_only > thresh) / N_trials
        P_d[idx] = np.sum(max_vals_signal_plus_noise > thresh) / N_trials

    return P_fa, P_d




k = B / T 
tx_chirp = np.exp(1j * np.pi * k * t**2)


try:
    with np.load(filename) as data:
        tx_isac = data['isac_siganl']
        #tx_isac = tx_isac *np.hamming(len(tx_isac))
        tx_vouras = data['vouras_siganl'] 
        #tx_vouras = tx_vouras *np.hamming(len(tx_vouras))
        print("Successfully loaded custom signals.")
except FileNotFoundError:
    print(f"Error: Could not find {filename}. Generating dummy signals for testing.")


freqs_nulled, S_nulled = hlp.spectrum(tx_isac, fs, 2**14)
S_nulled_db = 20*np.log10(np.abs(S_nulled) + 1e-80)

freqs, S_lin = hlp.spectrum(tx_vouras, fs, 2**14)
S_linear_db = 20*np.log10(np.abs(S_lin)+1e-80)

freqs_orig, S_orig = hlp.spectrum(tx_chirp, fs, 2**14)
S_orig_db = 20*np.log10(np.abs(S_orig) + 1e-80)

plt.figure()


plt.plot(freqs/1e6, S_nulled_db, label='Optimized Signal', color='green', linewidth=1)
plt.plot(freqs/1e6, S_linear_db, label='Linearized Solution', color='red')
plt.plot(freqs/1e6, S_orig_db, label='Chirp', color='blue')

plt.title("Power Spectrum [dB]")
plt.xlabel("Frequency [MHz]")

#plt.legend(loc='upper right')
plt.grid(True, alpha=0.5)
#plt.xlim(-0.5, 0.5)
#plt.ylim(-78, 5)
plt.tight_layout()

tx_chirp = tx_chirp / np.sqrt(np.mean(np.abs(tx_chirp)**2))
tx_isac = tx_isac / np.sqrt(np.mean(np.abs(tx_isac)**2))
tx_vouras = tx_vouras / np.sqrt(np.mean(np.abs(tx_vouras)**2))

print(f"\nRunning {N_trials} Monte Carlo trials at SNR = {SNR_dB} dB...")
print("1/3: Calculating ROC for Ideal LFM Chirp...")
P_fa_chirp, P_d_chirp = calculate_roc_for_signal(tx_chirp, SNR_dB, N_trials, true_distance, fs)

print("2/3: Calculating ROC for ISAC Signal...")
P_fa_isac, P_d_isac = calculate_roc_for_signal(tx_isac, SNR_dB, N_trials, true_distance, fs)

print("3/3: Calculating ROC for Vouras Signal...")
P_fa_vouras, P_d_vouras = calculate_roc_for_signal(tx_vouras, SNR_dB, N_trials, true_distance, fs)

print("Finished! Plotting results.")


plt.figure(figsize=(10, 8))


plt.plot(P_fa_chirp, P_d_chirp, linewidth=2.5, color='blue', label='Ideal LFM Chirp')
plt.plot(P_fa_isac, P_d_isac, linewidth=2.5, color='green', linestyle='-', label='ISAC Signal')
plt.plot(P_fa_vouras, P_d_vouras, linewidth=2.5, color='red', linestyle='-.', label='Vouras Signal')


plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

plt.title(f'ROC Comparison: Radar vs. ISAC Signals (SNR = {SNR_dB} dB)', fontsize=14)
plt.xlabel('Probability of False Alarm ($P_{FA}$)', fontsize=12)
plt.ylabel('Probability of Detection ($P_D$)', fontsize=12)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()

# === תוספת: הצגת דגימות בודדות לאחר חישוב ה-ROC ===
print("\nGenerating correlation comparison visualization...")

# אורזים את כל הסיגנלים למילון כדי לשלוח לפונקציה
signals_to_compare = {
    'Ideal LFM Chirp': tx_chirp,
    'ISAC Signal': tx_isac,
    'Vouras Signal': tx_vouras
}

# קוראים לפונקציה החדשה פעם אחת
plot_correlation_comparison(signals_to_compare, SNR_dB, true_distance, fs, c)
plt.show()


