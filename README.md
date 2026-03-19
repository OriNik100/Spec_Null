# Spec_Null: Integrated Sensing and Communication (ISAC) via Spectral Shaping

[cite_start]**Developers:** Harel Naveh and Ori Nikcha [cite: 7]
[cite_start]**Supervisor:** Dr. Amir Weiss, Bar-Ilan University [cite: 7, 8]

## Overview
[cite_start]This repository contains the development of a novel transmitting/receiving method for Integrated Sensing and Communication (ISAC) systems[cite: 2, 13]. [cite_start]Our approach focuses on spectral shaping by creating spectral nulls within a chirp (LFM) signal using a time-dependent phase, and seamlessly embedding a multi-carrier OFDM communication signal into those nulled frequencies[cite: 15, 85]. 

[cite_start]While previous methods relied on linearized Taylor series approximations [cite: 17, 139][cite_start], our project goes beyond the linear approximation by leveraging Machine Learning (ML) tools[cite: 18, 251]. [cite_start]By treating the added phase as learned parameters and using the Adam optimizer, we directly minimize a highly non-linear, multi-objective loss function to find the exact, optimal phase[cite: 96, 252, 254].

## Current Capabilities & Milestones Achieved
[cite_start]To date, we have successfully developed a Python-based simulation framework [cite: 187] and implemented the following:
* [cite_start]**Baseline Replication:** Successfully reproduced the analytical, linear-approximation results from prior literature to serve as a baseline[cite: 234, 235].
* [cite_start]**Deep Learning Optimization:** Transitioned to an ML-based optimization approach, achieving a significantly deeper null (-74.91 dB) compared to the linear baseline (-39.80 dB)[cite: 287, 288].
* [cite_start]**Multi-Objective Loss Function:** Designed a comprehensive loss function that balances several inherently conflicting objectives[cite: 262, 271]:
  * [cite_start]**Null Energy:** Penalizing energy at target frequencies to create deep nulls[cite: 258].
  * [cite_start]**Phase Smoothness:** Punishing rapid changes in the phase to maintain signal stability[cite: 265].
  * [cite_start]**Null Width Control:** Expanding the width of the nulls by enforcing constraints on the first and second-order derivatives of the spectrum[cite: 194, 268].
  * [cite_start]**Radar Performance (PSLR & ISLR):** Penalizing high Integrated Sidelobe Ratio (ISLR) and low Peak Sidelobe Ratio (PSLR) to preserve the matched filter's sensing capabilities[cite: 266, 267].
* [cite_start]**Pareto-Frontier Analysis:** Utilized the Optuna framework to systematically explore the hyperparameter space, generating a Pareto Frontier that illustrates the trade-offs between Null Energy, PSLR, and ISLR[cite: 275, 421].

## Future Work / To-Do
[cite_start]To complete the proof-of-concept and fully realize the ISAC architecture, the following milestones remain[cite: 560, 563]:
* [cite_start]**Finalize Spectral Shaping Parameters:** Select the optimal regularization coefficients from the Pareto-frontier and determine the exact number and frequencies of the nulls required for the OFDM scheme[cite: 561, 562, 564].
* [cite_start]**ISAC Transmitter Development:** Implement the communication subsystem by mapping QPSK-modulated data onto selected OFDM sub-carriers, embedding them precisely within the radar chirp's spectral nulls[cite: 565, 566, 567].
* [cite_start]**Receiver Design:** Architect a custom Maximum-Likelihood receiver with an optimal hypothesis-testing decision rule to accurately demodulate the transmitted QPSK data[cite: 18, 569, 570].
* [cite_start]**AWGN Channel Testing:** Evaluate the dual-functionality performance under an Additive White Gaussian Noise (AWGN) channel[cite: 571]. [cite_start]Sensing efficacy will be measured via Probability of Detection (POD), while communication robustness will be quantified using Bit Error Rate (BER) and Symbol Error Rate (SER)[cite: 572].
