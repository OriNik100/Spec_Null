import numpy as np
import helper_functions as hlp
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import null_depth_control
import null_width_control

# ---- parameters ----
T = 60e-6           # duration (s)
B = 2e6             # bandwidth (Hz)
fs = 10 * B         # sampling rate- number of samples per second (10 and not two for over sampling)(1/s)
N = int(np.round(T * fs))  # (s/s - number)
t = np.linspace(0, T, N, endpoint=False) #array of time values from 0 to T spaced evenly with N points

# LFM chirp phase (baseband) # center time optional, f0 = 0# amplitude (rect), replace with window if desired
b = B/T
psi = 2*np.pi * (b/2) * t**2 
a = np.ones_like(t)
s1 = a * np.exp(1j*psi)


# compute baseband spectrum

freqs, S = hlp.spectrum(s1, fs)

nulls = [0.4e6]
K=len(nulls)
z = hlp.build_z(a,psi,t,nulls)
c = np.real(z)
s = np.imag(z)
ones = np.ones((N,1))
A = np.hstack([c,s])
A_inner = hlp.inner_product_mat(A , A)

y = hlp.inner_product_mat(np.hstack([-s,c]),ones)

gamma = hlp.matrix_inverse(A_inner) @ y

phi_hat = (A @ gamma)

s_adapted = a * np.exp(1j*psi + 1j * phi_hat.flatten())
freqs2, S_adapted = hlp.spectrum(s_adapted, fs)


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
plt.plot(freqs/1e6, 20*np.log10(np.abs(S)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('Unadapted LFM spectrum')
plt.grid()

plt.figure()
plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('adapted LFM spectrum')
plt.grid()

plt.figure()
plt.plot(t*1e6, phi_hat*180/np.pi)
plt.xlabel("Time (µs)")
plt.ylabel("Phase offset φ(t) (rad)")
plt.title("Computed φ̂(t) from equation (8)")
plt.grid()
plt.show()


phi__depth_control = null_depth_control.solve_nulling_problem(
    A=A,
    y=y,
    phi_hat = phi_hat,
    beta = 10000,
    W = 4000*np.eye(2*K) ,
    M=np.eye(N),
    max_iter=20
)

s_depth_control = s1 * np.exp(1j * phi__depth_control.flatten())

freqs3, S_depth_control = hlp.spectrum(s_depth_control, fs)

plt.figure()
plt.plot(freqs3/1e6, 20*np.log10(np.abs(S_depth_control)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('adapted LFM spectrum')
plt.grid()

plt.figure()
plt.plot(t*1e6, phi__depth_control*180/np.pi)
plt.xlabel("Time (µs)")
plt.ylabel("Phase offset φ(t) (rad)")
plt.title("Computed φ̂(t) from equation (9)")
plt.grid()
plt.show()

print(f"shape of z :{np.shape(z)}")
print(f"shape of c :{np.shape(c)}")
print(f"shape of s :{np.shape(s)}")
print(f"shape of A :{np.shape(A)}")
print(f"shape of y :{np.shape(y)}")
print(f"shape of phi_hat :{np.shape(phi_hat)}")
print(f"shape of phi_hat_depth :{np.shape(phi__depth_control)}")

phi_width_control = null_width_control.compute_phi_hat(a, psi, t, nulls)


s_width_control = s1 * np.exp(1j * phi_width_control.flatten())

freqs4, S_width_control = hlp.spectrum(s_width_control, fs)

plt.figure()
plt.plot(freqs2/1e6, 20*np.log10(np.abs(S_adapted)/np.max(np.abs(S))), '--')
plt.plot(freqs4/1e6, 20*np.log10(np.abs(S_width_control)/np.max(np.abs(S))))
plt.xlim(-B/1e6, B/1e6 +1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB)')
plt.title('comon LFM spectrum')
plt.grid()
plt.show()
