import numpy as np
import pandas as pd

# --- energy grid -----------------------------------------------------------
E = np.linspace(115.0, 135.0, 400)          # GeV

# --- true model parameters -------------------------------------------------
m_H, Gamma, N = 125.0, 0.004, 1.0
alpha, S0, kappa, c = 0.04, 0.10, 0.9, 1.00

bw = N * (Gamma**2) / ((E - m_H)**2 + (Gamma**2)/4)
S_eff = alpha * S0 * (1 + c * np.log(E / m_H))
true_obs = bw + kappa * S_eff

# add 3 % Gaussian noise
rng = np.random.default_rng(42)
noisy_obs = rng.normal(true_obs, 0.01 * true_obs)

pd.DataFrame({"Energy": E, "Obs": noisy_obs}).to_csv("toy.csv", index=False)
print("Wrote toy.csv (400 rows)")
