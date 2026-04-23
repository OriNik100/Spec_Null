# Spec_Null: Integrated Sensing and Communication (ISAC) via Spectral Shaping

**Developers:** Harel Naveh and Ori Nikcha  
**Supervisor:** Dr. Amir Weiss, Bar-Ilan University  

## Overview
This repository contains the development of a novel transmitting and receiving method for **Integrated Sensing and Communication (ISAC)** systems. Our approach focuses on spectral shaping by creating spectral nulls within a chirp (LFM) signal using a time-dependent phase, while seamlessly embedding a multi-carrier OFDM communication signal into those nulled frequencies. 

While previous methods relied on linearized Taylor series approximations, our project goes beyond the linear approximation by leveraging **Machine Learning (ML)** tools. By treating the added phase as learned parameters and using the **Adam optimizer**, we directly minimize a highly non-linear, multi-objective loss function to find the exact, optimal phase.

## Current Capabilities & Milestones
To date, we have successfully developed a Python-based simulation framework and implemented the following:

* **Baseline Replication:** Successfully reproduced the analytical, linear-approximation results from prior literature to serve as a reliable baseline.
* **Deep Learning Optimization:** Transitioned to an ML-based optimization approach, achieving a significantly deeper null (**-74.91 dB**) compared to the linear baseline (-39.80 dB).
* **Multi-Objective Loss Function:** Designed a comprehensive loss function that balances several inherently conflicting objectives, including:
    * **Null Energy Minimization:** Penalizing energy at target frequencies.
    * **Phase Smoothness:** Maintaining signal stability.
    * **Null Width Control:** Enforcing constraints on spectral derivatives.
    * **Radar Performance (PSLR & ISLR):** Preserving the matched filter's sensing capabilities.
* **Pareto-Frontier Analysis:** Utilized the **Optuna** framework to systematically explore the hyperparameter space, generating a Pareto Frontier that illustrates the trade-offs between Null Energy, PSLR, and ISLR.

## Future Work
To complete the proof-of-concept and fully realize the ISAC architecture, the following milestones remain:

* **Finalize Spectral Shaping Parameters:** Select the optimal regularization coefficients from the Pareto-frontier and determine the exact number and frequencies of the nulls required for the OFDM scheme.
* **ISAC Transmitter Development:** Implement the communication subsystem by mapping QPSK-modulated data onto selected OFDM sub-carriers, embedding them precisely within the radar chirp's spectral nulls.
* **Receiver Design & Channel Testing:** Architect a custom **Maximum-Likelihood receiver**, and evaluate the system's dual-functionality performance under an **AWGN channel** using standard metrics:
    * **Sensing:** Probability of Detection (POD).
    * **Communication:** Bit Error Rate (BER) and Symbol Error Rate (SER).
