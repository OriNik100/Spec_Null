# Integrated Sensing and Communication (ISAC) via Spectral Shaping

**Developers:** Harel Naveh and Ori Nikcha  
**Supervisor:** Dr. Amir Weiss, Bar-Ilan University  

## Overview & Motivation
The transition to 6G and the emergence of the Internet of Everything (IoE) necessitate a paradigm shift from pure data-transfer networks to intelligent infrastructures capable of perceiving their physical surroundings. **Integrated Sensing and Communication (ISAC)** realizes this by utilizing the same hardware, spectrum, and waveforms for both communication and environmental sensing.

At the core of the ISAC architecture lies an inherent conflict: sensing requires deterministic structures with low autocorrelation sidelobes for high resolution, while communication thrives on the randomness of data symbols to maximize spectral efficiency.

This repository presents a novel, metric-agnostic waveform design framework that resolves this conflict. By carving out designated interference-free spectral "holes" within an optimal, constant-envelope radar waveform (e.g., LFM) and actively embedding high-capacity OFDM symbols into these sub-bands, we fundamentally decouple these conflicting physical requirements.

## Our Approach: Numerical Optimization
Historically, spectral nulling methods relied on constrained optimization problems using linear approximations (e.g., Taylor series) of a highly non-linear, non-convex phase problem. This legacy approach treats waveform distortion as an uncontrollable byproduct.

We transcend these limitations by treating the added phase perturbation, $\phi(t)$, as a set of learned parameters. By employing a gradient descent-based algorithm (Adam optimizer), we directly tackle the non-convex problem to find the optimal phase that minimizes a multi-objective Lagrangian loss function:

$$
\mathcal{L} = \sum_{k\in\mathcal{F}}|X_s(f_k)|^2 + \beta_\text{norm}\|\phi\|_2^2 + \beta_\text{smooth}\|\mathbf{D}\phi\|_2^2 + \beta_\text{PSLR}\gamma_\text{PSLR}^2 + \beta_\text{ISLR}\gamma_\text{ISLR}^2
$$

This framework empowers designers to dynamically control the fundamental trade-off between spectral null depth (to accommodate embedded OFDM data) and resulting radar performance metrics like the Peak Sidelobe Level Ratio (PSLR) and Integrated Sidelobe Ratio (ISLR).

## Experimental Setup & System Parameters

The simulation framework operates in a baseband representation configured with the following parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Bandwidth ($B$)** | 2 MHz | Baseline-chirp bandwidth |
| **Sampling Rate ($f_s$)** | 10 MHz | Computed as $5B$ |
| **Symbol Duration ($T_s$)** | $60\mu\text{s}$ | Duration of the OFDM symbol |
| **FFT Size ($N_{FFT}$)** | $2^{14}$ | Spectral analysis size |
| **OFDM Subcarriers** | 64 | 48 data, 4 pilots, 1 DC, 11 guard |
| **Modulation** | QPSK | Implemented with Gray coding |

## System Implementation & Results
We have successfully developed a complete Python-based simulation framework, realizing the full ISAC architecture from waveform optimization to receiver evaluation:

* **Deep Learning Spectral Nulling:** Achieved a significantly deeper spectral null (**-74.91 dB**) by moving away from legacy linear approximations, far outperforming the traditional baseline (which equalizes at approximately -37 dB to -39.80 dB).
* **Null Broadening:** Implemented first and second derivative constraints on the frequency spectrum to generate wider, more robust nulls capable of accommodating finite-duration OFDM $\text{sinc}(\cdot)$ combinations.
* **Pareto-Frontier Analysis:** Utilized the **Optuna** framework to systematically sweep regularization coefficients ($\boldsymbol{\beta}$), mapping the multi-dimensional trade-offs between communication integrity (Null Energy) and sensing shape preservation (PSLR, ISLR), and selected the optimal parameters for our subcarrier configuration.
* **Complete ISAC Transmitter:** Engineered the communication subsystem by successfully mapping QPSK-modulated OFDM data precisely into the optimal spectral nulls of the radar chirp.
* **Receiver Design & AWGN Evaluation:** Architected a custom **Maximum-Likelihood (ML) receiver** and comprehensively evaluated the dual-functionality performance under an AWGN channel across varying Sensing-to-Communication (StC) ratios. Performance was validated using standard metrics:
    * **Sensing:** Probability of Detection (POD) against targets (e.g., $1500\text{m}$ delay simulations).
    * **Communication:** Symbol Error Rate (SER).
