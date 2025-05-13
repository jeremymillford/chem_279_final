
import pandas as pd
import matplotlib.pyplot as plt

# Relative energy values (all made negative for stability representation)
data = {
    "Molecule": ["CH4", "C2H4", "C2H6", "C4H10", "C6H6"],
    "Relative Energy (eV)_CNDO2": [-0.0, -65.998966, -103.238826, -921.926525, -536.630555],
    "Relative Energy (eV)_CNDOS": [-0.0, -910.636454, -1469.868409, -5917.285987, -8038.621992],
    "Relative Energy (eV)_MINDO": [-0.0, -454.622, -602.426, -2211.76, -3122.85],
    "Relative Energy (eV)_HF": [-0.0, -1003.529869, -1062.158217, -3186.542290, -5183.979308]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(df["Molecule"], df["Relative Energy (eV)_CNDO2"], marker='o', label="CNDO/2")
plt.plot(df["Molecule"], df["Relative Energy (eV)_CNDOS"], marker='s', label="CNDO/S")
plt.plot(df["Molecule"], df["Relative Energy (eV)_MINDO"], marker='^', label="MINDO")
plt.plot(df["Molecule"], df["Relative Energy (eV)_HF"], marker='x', linestyle='--', label="Hartree-Fock")

plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("Relative Energy (Normalized (eV)")
plt.title("Relative Total Energies by Size (CH₄-Referenced, All Negative")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
