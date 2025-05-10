import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data including MINDO and HF
data = {
    "Molecule": [
        "Methane (CH₄)",
        "Ethane (C₂H₆)",
        "Ethylene (C₂H₄)",
        "Butane (C₄H₁₀)",
        "Benzene (C₆H₆)"
    ],
    "CNDO2_Energy": [-179.87, -300.56, -237.52, 762.19, -672.25],
    "MINDO_Energy": [68.0005, 516.552, 618.529, 2021.28, 3023.2],
    "CNDOS_Energy": [1196.98, 5821.40, 4885.50, 11141.60, 20810.41],
    "HF_Energy": [1093.767318408271, 2155.925535327978, 2097.297187600075, 4280.309608704617, 6277.746626506433]
}

df = pd.DataFrame(data)

# Convert to absolute values
df["CNDO2_Pos"] = df["CNDO2_Energy"].abs()
df["MINDO_Pos"] = df["MINDO_Energy"].abs()
df["CNDOS_Pos"] = df["CNDOS_Energy"].abs()
df["HF_Pos"] = df["HF_Energy"].abs()

# Plot grouped bar chart with order: CNDO/2, MINDO, HF, CNDO/S
x = np.arange(len(df))
width = 0.2

plt.figure()
plt.bar(x - 1.5*width, df["CNDO2_Pos"], width, label="CNDO/2")
plt.bar(x - 0.5*width, df["MINDO_Pos"], width, label="MINDO")
plt.bar(x + 0.5*width, df["HF_Pos"],   width, label="HF")
plt.bar(x + 1.5*width, df["CNDOS_Pos"], width, label="CNDO/S")

plt.xticks(x, df["Molecule"], rotation=45, ha="right")
plt.ylabel("Absolute Energy (eV)")
plt.title("Comparison of CNDO/2, MINDO, HF, and CNDO/S Energies")
plt.legend()
plt.tight_layout()
plt.show()
