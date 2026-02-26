import numpy as np
import os
import helper_functions as hlp
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import null_depth_control as dpth
import null_width_control as wdth

# ---- parameters ----
T = 60e-6           # duration (s)
B = 2e6             # bandwidth (Hz)
fs = 5 * B         # sampling rate- number of samples per second (10 and not two for over sampling)(1/s)
N = int(np.round(T * fs))  # (s/s - number)
t = np.linspace(0, T, N, endpoint=False) #array of time values from 0 to T spaced evenly with N points
N_FFT = 2**14

# LFM chirp phase (baseband) # center time optional, f0 = 0# amplitude (rect), replace with window if desired
b = B/T
psi = np.pi *b* t**2
a = np.ones_like(t)
s1 = a * np.exp(1j*psi)


# compute baseband spectrum
    
freqs, S = hlp.spectrum(s1, fs, N_FFT)

# Nulls real frequencies for N_FFT = 2**14
null_targets = [0.4e6]
nulls = []

for f_val in null_targets:
    df = fs / N_FFT
    bin_idx = int(np.round(f_val / df))
    nulls.append(bin_idx * df)


K=len(nulls)
z = hlp.build_z(a,psi,t,nulls)
c = np.real(z)
s = np.imag(z)
ones = np.ones((N,1))
A = np.hstack([c,s])
A_inner = hlp.inner_product_mat(A , A,t).real
y = hlp.inner_product_mat(np.hstack([-s,c]),ones,t)
gamma = hlp.matrix_inverse(A_inner) @ y
phi_hat = (A @ gamma)
s_adapted = a * np.exp(1j*psi + 1j * phi_hat.flatten())
freqs2, S_adapted = hlp.spectrum(s_adapted, fs, N_FFT)


# Trying to find the TRUE depth of the null.
S_adapted_db = 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted)) + 1e-40)
for k, f_val in enumerate(nulls):
    depth = np.min(S_adapted_db[np.abs(freqs2 - f_val) < 0.05e6])  # search within 100 kHz around the null frequency
    print(f"Null at around {f_val/1e6:.2f} MHz: Depth = {depth:.2f} dB")
# It changes with NFFT.


def getphi(nulls=nulls):
    t = np.linspace(0, T, N, endpoint=False)
    psi = np.pi *b* t**2
    a = np.ones_like(t)
    z = hlp.build_z(a,psi,t,nulls)
    c = np.real(z)
    s = np.imag(z)
    ones = np.ones((N,1))
    A = np.hstack([c,s])
    A_inner = hlp.inner_product_mat(A , A,t).real

    y = hlp.inner_product_mat(np.hstack([-s,c]),ones,t)

    gamma = hlp.matrix_inverse(A_inner) @ y

    phi_hat = (A @ gamma)
    return phi_hat

def calculate_mf_and_pslr(signal):
    
    matched_filter = np.conj(signal[::-1])
    
    mf = np.convolve(signal, matched_filter, mode='full')
    
    mf_abs = np.abs(mf)
    mf_db = 20 * np.log10(mf_abs + 1e-20)
    
    # PSLR Calculation
    center_idx = len(mf_db) // 2
    main_lobe_width_seconds = 2 / B
    margin_samples = int((main_lobe_width_seconds / 2) * fs)
    
    left_side = mf_db[:center_idx - margin_samples]
    right_side = mf_db[center_idx + margin_samples:]
    
    sidelobe_region = np.concatenate((left_side, right_side))
    max_sidelobe = np.max(sidelobe_region)
    max_mainlobe = np.max(mf_db)
    
    pslr = max_mainlobe - max_sidelobe


    # ISLR calculation
    n_main = np.arange(len(mf_db[center_idx - margin_samples : center_idx + margin_samples]))
    n_left = np.arange(len(left_side))
    n_right = np.arange(len(right_side))

    main_lobe_linear = mf_abs[center_idx - margin_samples : center_idx + margin_samples]
    left_side_linear = mf_abs[:center_idx - margin_samples]
    right_side_linear = mf_abs[center_idx + margin_samples:]

    # discrete options:
    # main_lobe_energy = np.sum(mf_db[center_idx - margin_samples : center_idx + margin_samples] ** 2)
    # sidelobe_energy = np.sum(left_side ** 2) + np.sum(right_side ** 2)    

    main_lobe_energy = np.trapezoid(main_lobe_linear ** 2, x=n_main)
    sidelobe_energy = np.trapezoid(left_side_linear ** 2, x=n_left) + np.trapezoid(right_side_linear ** 2, x=n_right)
    islr = 10 * np.log10(sidelobe_energy / main_lobe_energy + 1e-40)


    return mf_db, pslr, islr

if __name__ == "__main__":
    phi_hat = getphi(nulls).flatten()
    plt.figure()
    plt.plot(t,np.real(s1), label ="Real part (I)")
    plt.plot(t,np.imag(s1),color ='red', label ="Imag part (Q)")
    plt.xlim(0, T)
    plt.xlabel('Time')
    plt.ylabel("Amplitude")
    plt.title('LFM')
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted))))
    plt.plot(freqs/1e6, 20*np.log10(np.abs(S)/np.max(np.abs(S))), color ='red')
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Unadapted LFM spectrum')
    plt.grid()


    plt.figure()
    plt.plot(t*1e6, phi_hat*180/np.pi)
    plt.xlabel("Time (µs)")
    plt.ylabel("Phase offset φ(t) (rad)")
    plt.title("Computed φ̂(t) from equation (8)")
    plt.grid()
    plt.show()



    #############################################################
    # ----------------Depth Control ----------------------------#
    #############################################################


    phi__depth_control = dpth.solve_nulling_problem(
        A=A,
        b=y,
        phi_hat = phi_hat,
        beta = 1e4,
        W = 4000*np.eye(2*K) ,
        M=np.eye(N),
        max_iter=20
    )

    s_depth_control = s1 * np.exp(1j * phi__depth_control.flatten())

    freqs3, S_depth_control = hlp.spectrum(s_depth_control, fs, N_FFT)

    beta_arr = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
    phi_arr = [dpth.solve_nulling_problem(A=A,
        b=y,
        phi_hat = phi_hat,
        beta = beta_val,
        W = 4000*np.eye(2*K) ,
        M=np.eye(N),
        max_iter=20) for beta_val in beta_arr]
    s_depth_arr = [s1 * np.exp(1j * phi_val.flatten()) for phi_val in phi_arr]

# Plotting the spectra for different beta values
    plt.figure(figsize=(12, 8))
    for beta_val, s_depth in zip(beta_arr, s_depth_arr):
        freqs_depth, S_depth = hlp.spectrum(s_depth, fs, N_FFT)
        plt.plot(freqs_depth/1e6, 20*np.log10(np.abs(S_depth)/np.max(np.abs(S_depth))), label=f'β={beta_val:.0e}')
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Depth Control LFM spectrum for different β values')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(freqs3/1e6, 20*np.log10(np.abs(S_depth_control)/np.max(np.abs(S_depth_control))))
    plt.xlim(-B/1e6-3, B/1e6 +3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Depth Control LFM spectrum')
    plt.grid()

    plt.figure()
    plt.plot(t*1e6, phi__depth_control*180/np.pi)
    plt.xlabel("Time (µs)")
    plt.ylabel("Phase offset φ(t) (rad)")
    plt.title("Computed φ̂(t) from equation (9) -Depth control")
    plt.grid()
    plt.show()
    
# calculating PSLR and ISLR for depth control

    results = [calculate_mf_and_pslr(s_depth_val.flatten()) for s_depth_val in s_depth_arr]
    mf_depth, pslr_depth, islr_depth = zip(*results)
    for beta_val, pslr_val, islr_val in zip(beta_arr, pslr_depth, islr_depth):
        print(f"Depth Control (β={beta_val:.0e}) - PSLR: {pslr_val:.2f} dB, ISLR: {islr_val:.2f} dB")



    #############################################################
    # ----------------Width Control ----------------------------#
    #############################################################
    phi_width_control = wdth.compute_phi_hat(a, psi, t, nulls)

    s_width_control = s1 * np.exp(1j * phi_width_control.flatten())

    freqs4, S_width_control = hlp.spectrum(s_width_control, fs, N_FFT)

    plt.figure()
    plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted))), '--')
    plt.plot(freqs4/1e6, 20*np.log10(np.abs(S_width_control)/np.max(np.abs(S_width_control))))
    plt.xlim(0.35,0.45)
    plt.ylim(-45,0)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title('Width Control LFM spectrum')
    plt.grid()
    plt.show()

    #############################################################
    # ----------------Matched Filter----------------------------#
    #############################################################

    mf_self = hlp.apply_matched_filter(s1,s1)
    mf_basic = hlp.apply_matched_filter(s_adapted.flatten(), s1)
    #mf_dpth = hlp.apply_matched_filter(s_depth_control.flatten(), s1)

    plt.figure()
    plt.plot(mf_basic, label="Basic Adapted Chirp")
    #plt.plot(mf_dpth, ':', label="Depth Control")
    # plt.plot(mf_width, ':', label="Width Controlled (Deriv. Const.)")
    plt.xlabel('Index (Time Samples)')
    plt.ylabel('Power (dB)')
    plt.title('Nulled-Chirp Matched Filter Output')
    plt.grid(True)
    plt.show()



    ################################################
    # -------Non linear ---------------------------#
    ################################################
'''
     print("\n--- Loading Optimized Phasor from file ---")
     
     filename = 'optimal_phasor.npy'
     if os.path.exists(filename):
         # 1. טעינת הקובץ
         correction_phasor = np.load(filename)
         print(f"Loaded '{filename}' successfully.")
     else:
         correction_phasor = 0
         print(f"Error: File '{filename}' not found.")
     
     
     
     # 2. בניית האות האופטימלי
     s_opt = s1 * correction_phasor
     
     # 3. חישוב ספקטרום
     freqs_opt, S_opt = hlp.spectrum(s_opt, fs,N_FFT)
     
     
     plt.figure(figsize=(12, 8))
     norm_factor = np.max(np.abs(S)) 
     
     
     S_orig_db = 20*np.log10(np.abs(S)/norm_factor + 1e-40)
     S_adapted_db = 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S_adapted)))
     S_opt_db = 20*np.log10(np.abs(S_opt)/np.max(np.abs(S_opt)) + 1e-40)
     
     # גרף 1: מקורי (כחול)
     plt.plot(freqs/1e6, S_orig_db, label='Original LFM', color='blue', alpha=0.3)
     
     # גרף 2: אנליטי (ירוק)
     plt.plot(freqs2/1e6, S_adapted_db, label='Analytical', color='green', linestyle='--', alpha=0.6 )
     
     # גרף 3: אופטימלי (אדום)
     plt.plot(freqs_opt/1e6, S_opt_db, label='Optimized', color='red')
     
     # --- סימון החורים וכתיבת העומק ---
     for f_val in nulls:
         plt.axvline(f_val/1e6, color='k', linestyle=':', alpha=0.3)
     
         search_bw = 0.05e6
     
         # 1. כתיבת עומק עבור האופטימלי (אדום)
         mask = np.abs(freqs_opt - f_val) < search_bw
         idx_opt = np.argmin(S_opt_db[mask])
         depth_opt = S_opt_db[mask][idx_opt]
         plt.text(f_val/1e6, depth_opt + 2, f"{depth_opt:.2f} dB", 
                     color='red', fontweight='bold', rotation=90, 
                     verticalalignment='bottom', horizontalalignment='right')
     
         # 2. כתיבת עומק עבור האנליטי (ירוק)
         mask = np.abs(freqs2 - f_val) < search_bw
         idx_ana = np.argmin(S_adapted_db[mask]) 
         depth_ana = S_adapted_db[mask][idx_ana]
         
         # הזזנו את הטקסט הירוק קצת שמאלה (horizontalalignment='left') כדי שלא יתנגש באדום
         plt.text(f_val/1e6, depth_ana + 2, f"{depth_ana:.2f} dB", 
                     color='green', fontweight='bold', rotation=90, 
                     verticalalignment='bottom', horizontalalignment='left')
     
     plt.title("Spectrum Comparison: Original vs Analytical vs Optimized")
     plt.xlabel("Frequency [MHz]")
     plt.ylabel("Normalized Magnitude [dB]")
     plt.legend()
     plt.grid(True)
     plt.xlim(0, B/1e6 + 0.2)
     plt.ylim(-120, 5)
     plt.tight_layout()
     
     
     
     # ---  השוואת פאזות (אנליטי מול אופטימלי) - במעלות ---
     print("\n--- Phase Correction Comparison ---")
     
     # 1. המרת הפאזה האנליטית למעלות (משתמשים ב-phi_hat הקיים)
     phi_ana_deg = np.degrees(phi_hat)
     
     # 2. חילוץ הפאזה האופטימלית והמרה למעלות
     # correction_phasor הוא exp(j*phi), אז נוציא את הזווית
     phi_opt_rad = np.angle(correction_phasor)
     phi_opt_deg = np.degrees(phi_opt_rad)
     
     # 3. חישוב המרחק האוקלידי (בין וקטורי המעלות)
     diff_norm = np.linalg.norm(phi_opt_deg - phi_ana_deg)
     print(f"Euclidean Distance (in degrees space): {diff_norm:.4f}")
     
     # 4. ציור הגרף
     plt.figure(figsize=(12, 6))
     t_us = t * 1e6  # ציר זמן במיקרו-שניות
     
     # גרף הפאזה האנליטית (ירוק מקווקו)
     plt.plot(t_us, phi_ana_deg, label='Analytical Phase (phi_hat)', 
                 color='green', linestyle='--', linewidth=2)
     
     # גרף הפאזה האופטימלית (אדום רציף)
     plt.plot(t_us, phi_opt_deg, label='Optimized Phase (DL Model)', 
                 color='red', alpha=0.7, linewidth=2)
     
     plt.title(f"Phase Correction Comparison [Degrees]\n(Euclidean Diff: {diff_norm:.2f})")
     plt.xlabel("Time [us]")
     plt.ylabel("Phase [Degrees]") # שינינו למעלות
     plt.legend()
     plt.grid(True, alpha=0.5)
     plt.tight_layout()
     plt.show()
     
    '''

    
