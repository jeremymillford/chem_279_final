import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Molecule": ["Methane (CH₄)", "Ethane (C₂H₆)", "Ethylene (C₂H₄)", "Butane (C₄H₁₀)", "Benzene (C₆H₆)"],
    "CNDO2_Energy": [-195.195, -329.004, -260.41, -546.207, -733.116],
    "CNDOS_Energy": [2686.57, 5860.48, 4891.64, 11799.6, 20908.1],
    "Expected_Energy": [1093.767318408271, 2155.925535327978, 2097.297187600075, 4280.309608704617, 6277.746626506433]
}

df = pd.DataFrame(data)
# Convert to positive values
df["CNDO2_Energy_Pos"] = df["CNDO2_Energy"].abs()
df["CNDOS_Energy_Pos"] = df["CNDOS_Energy"].abs()
df["Expected_Energy_Pos"] = df["Expected_Energy"].abs()

# Plot
x = np.arange(len(df))
width = 0.25

plt.figure()
plt.bar(x - width, df["CNDO2_Energy_Pos"], width, label="CNDO/2")
plt.bar(x, df["Expected_Energy_Pos"], width, label="Expected")
plt.bar(x + width, df["CNDOS_Energy_Pos"], width, label="CNDO/S")
plt.xticks(x, df["Molecule"], rotation=45, ha="right")
plt.ylabel("Absolute Energy (eV)")
plt.title("CNDO/2, Expected, and CNDO/S Absolute Energies")
plt.legend()
plt.tight_layout()
plt.show()
