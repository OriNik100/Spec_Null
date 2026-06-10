import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_mf_pslr_islr_torch(signal, B, Fs, db=False):
    # Calculate the matched filter (auto-correlation)
    matched_filter = torch.conj(torch.flip(signal, dims=[0]))
    conv_len = signal.shape[0] + matched_filter.shape[0] - 1
    
    # Perform linear convolution via the frequency domain
    mf = torch.fft.ifft(torch.fft.fft(signal, n=conv_len) * torch.fft.fft(matched_filter, n=conv_len))
    
    mf_abs = torch.abs(mf)
    mf_db = 20 * torch.log10(mf_abs + 1e-80)
    
    # The peak location is known a priori for auto-correlation (N-1)
    N = signal.shape[0]
    center_idx = N - 1 
    
    # Calculate the boundaries of the main lobe
    main_lobe_width_seconds = 2 / B
    margin_samples = int((main_lobe_width_seconds / 2) * Fs)
    
    # Ensure no array bounds violation (clipping)
    start_idx = max(0, center_idx - margin_samples)
    end_idx = min(len(mf_abs), center_idx + margin_samples + 1)
    
    # Split into regions (linear values only, DB calculated later if needed)
    main_lobe_linear = mf_abs[start_idx:end_idx]
    left_side_linear = mf_abs[:start_idx]
    right_side_linear = mf_abs[end_idx:]
    sidelobes_linear = torch.cat((left_side_linear, right_side_linear))
    
    # Calculate energies using sum of squares (standard for discrete signals)
    main_lobe_energy = torch.sum(main_lobe_linear ** 2)
    sidelobe_energy = torch.sum(sidelobes_linear ** 2)
    
    # Find maximum values
    max_mainlobe_linear = torch.max(main_lobe_linear)
    max_sidelobe_linear = torch.max(sidelobes_linear)
    
    if db:
        # Work directly with logarithmic values
        max_sidelobe_db = torch.max(mf_db[:start_idx].max() if len(left_side_linear) > 0 else torch.tensor(-float('inf')),
                                    mf_db[end_idx:].max() if len(right_side_linear) > 0 else torch.tensor(-float('inf')))
        max_mainlobe_db = torch.max(mf_db[start_idx:end_idx])
        
        pslr = max_sidelobe_db - max_mainlobe_db
        islr = 10 * torch.log10(sidelobe_energy / (main_lobe_energy + 1e-80))
    else:
        # Linear power ratio calculation
        # Note: For NN optimization, a differentiable L-p norm (e.g., p=10) 
        # can replace torch.max here to push down all high sidelobes evenly.
        pslr = (max_sidelobe_linear / (max_mainlobe_linear + 1e-80)) ** 2
        islr = sidelobe_energy / (main_lobe_energy + 1e-80)
        
    return mf_db, pslr, islr

import numpy as np

def calculate_mf_pslr_islr_numpy(signal, B, Fs, db=False):
    # Calculate the matched filter (auto-correlation)
    matched_filter = np.conj(signal[::-1])
    conv_len = len(signal) + len(matched_filter) - 1
    
    # Perform linear convolution via the frequency domain
    mf = np.fft.ifft(np.fft.fft(signal, n=conv_len) * np.fft.fft(matched_filter, n=conv_len))
    
    mf_abs = np.abs(mf)
    mf_db = 20 * np.log10(mf_abs + 1e-80)
    
    # The peak location is known a priori for auto-correlation (N-1)
    N = len(signal)
    center_idx = N - 1 
    
    # Calculate the boundaries of the main lobe
    main_lobe_width_seconds = 2 / B
    margin_samples = int((main_lobe_width_seconds / 2) * Fs)
    
    # Ensure no array bounds violation (clipping)
    start_idx = max(0, center_idx - margin_samples)
    end_idx = min(len(mf_abs), center_idx + margin_samples + 1)
    
    # Split into regions (linear values only, DB calculated later if needed)
    main_lobe_linear = mf_abs[start_idx:end_idx]
    left_side_linear = mf_abs[:start_idx]
    right_side_linear = mf_abs[end_idx:]
    sidelobes_linear = np.concatenate((left_side_linear, right_side_linear))
    
    # Calculate energies using sum of squares (standard for discrete signals)
    main_lobe_energy = np.sum(main_lobe_linear ** 2)
    sidelobe_energy = np.sum(sidelobes_linear ** 2)
    
    # Find maximum values safely
    max_mainlobe_linear = np.max(main_lobe_linear)
    max_sidelobe_linear = np.max(sidelobes_linear) if len(sidelobes_linear) > 0 else 0.0
    
    if db:
        # Work directly with logarithmic values
        max_left_db = np.max(mf_db[:start_idx]) if len(left_side_linear) > 0 else -np.inf
        max_right_db = np.max(mf_db[end_idx:]) if len(right_side_linear) > 0 else -np.inf
        max_sidelobe_db = max(max_left_db, max_right_db)
        
        max_mainlobe_db = np.max(mf_db[start_idx:end_idx])
        
        pslr = max_sidelobe_db - max_mainlobe_db
        islr = 10 * np.log10(sidelobe_energy / (main_lobe_energy + 1e-80))
    else:
        # Linear power ratio calculation
        pslr = (max_sidelobe_linear / (max_mainlobe_linear + 1e-80)) ** 2
        islr = sidelobe_energy / (main_lobe_energy + 1e-80)
        
    return mf_db, pslr, islr

# After defining the functions, let us use them to examine the metrics for our ISAC signals.
if __name__ == "__main__":
    T = 60e-6           
    B = 2e6             
    Fs = 5 * B          
    N = int(np.round(T * Fs)) 
    
    N_fft = 2**14
    
    
    t_np = np.linspace(0, T, N, endpoint=False)
    t = torch.tensor(t_np, dtype=torch.float64)
    
    t_norm = t / torch.max(t)
    
    b_slope = B/T
    psi = 2 * np.pi * (b_slope/2) * t**2
    a = 1
    s_base = a * torch.complex(torch.cos(psi), torch.sin(psi))
    s_base_np = np.array(s_base)
    
    filename = 'simulation_results.npz'
    
    try:
        with np.load(filename) as data:
            tx_isac = data['isac_signal']
            #tx_isac = tx_isac *np.hamming(len(tx_isac))
            tx_vouras = data['vouras_signal'] 
            #tx_vouras = tx_vouras *np.hamming(len(tx_vouras))
            print("Successfully loaded custom signals.")
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Generating dummy signals for testing.")
    
    chirp_mf, chirp_pslr, chirp_islr = calculate_mf_pslr_islr_numpy(s_base_np, B=B, Fs=Fs, db=True)
    isac_mf, isac_pslr, isac_islr = calculate_mf_pslr_islr_numpy(tx_isac, B=B, Fs=Fs, db=True)
    vouras_mf, vouras_pslr, vouras_islr = calculate_mf_pslr_islr_numpy(tx_vouras, B=B, Fs=Fs, db=True)
    
    plt.figure(figsize=(12, 6))
    #plt.plot(mf_orig_db, label=f'Original LFM (PSLR={pslr_orig:.1f}dB)', color='blue', alpha=0.6)
    plt.plot(isac_mf, label=f'Optimized LFM (PSLR={isac_pslr:.1f}dB)', color='red', linestyle='--')
    plt.plot(vouras_mf, label=f'Linearized LFM (PSLR={vouras_pslr:.1f}dB)', color='green', linestyle='-.')
    #plt.xlim(570, 630)
    plt.title("Matched Filter Comparison")
    plt.xlabel("index")
    plt.ylabel("Amplitude [dB]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("matched_filter_comparison_three_nulls.png", dpi=1000)
    
    print(f"--- PSLR Performence ---")
    print(f"Selected Model based on Minimum Total Loss")
    print(f"Original LFM PSLR:  {chirp_pslr:.2f} dB")
    print(f"Optimized LFM PSLR: {isac_pslr:.2f} dB")
    print(f"Linearized LFM PSLR: {vouras_pslr:.2f} dB")
    print(f"Degradation:        {vouras_pslr - isac_pslr:.2f} dB")
    print("Notice, PSLR should be as low as possible, so negative degradation means worse performance.\n")
    
    
    print(f"--- ISLR Performance ---")
    print(f"Original LFM ISLR:  {chirp_islr:.2f} dB")
    print(f"Optimized LFM ISLR: {isac_islr:.2f} dB")
    print(f"Linearized LFM ISLR: {vouras_islr:.2f} dB")
    print(f"Degradation:        {vouras_islr - isac_islr:.2f} dB")
    print("Notice, ISLR should be as small as possible, so negative degradation means worse performance.\n")
    
    plt.show()
    