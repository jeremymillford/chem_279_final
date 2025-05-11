#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
import pandas as pd

# Define a harmonic function for fitting
def harmonic(x, x0, k, E0):
    """Harmonic potential function: E = E0 + 0.5 * k * (x - x0)^2"""
    return E0 + 0.5 * k * (x - x0)**2

def debug_data(data, method_name):
    """Print diagnostic information about the data"""
    print(f"\n{method_name} Data Diagnostics:")
    print(f"  Shape: {data.shape}")
    print(f"  Energy Range: {np.min(data[:, 1]):.6f} to {np.max(data[:, 1]):.6f} eV")
    print(f"  Energy Difference (max-min): {np.max(data[:, 1]) - np.min(data[:, 1]):.6f} eV")
    
    # Check for suspicious values
    if np.max(np.abs(data[:, 1])) > 1000:
        print(f"  WARNING: Very large energy values detected!")
    
    # Check for uniformity in displacement steps
    disp_steps = np.diff(data[:, 0])
    if not np.allclose(disp_steps, disp_steps[0], rtol=1e-3):
        print(f"  WARNING: Displacement steps are not uniform!")
        print(f"  Step sizes: {disp_steps}")
    
    # Check if minimum is near zero displacement
    min_idx = np.argmin(data[:, 1])
    if abs(data[min_idx, 0]) > 0.1:
        print(f"  WARNING: Energy minimum is not near zero displacement!")
        print(f"  Minimum at: {data[min_idx, 0]:.6f} Å")

def main():
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python improved_plot_displacement.py <cndo2_file> <cndos_file> [output_file]")
        sys.exit(1)
    
    cndo2_file = sys.argv[1]
    cndos_file = sys.argv[2]
    
    output_file = None
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    # Extract molecule name from filename
    molecule_name = os.path.basename(cndo2_file).split('_')[0]
    
    # Load data with more robust error handling
    try:
        # Try pandas first
        try:
            df_cndo2 = pd.read_csv(cndo2_file)
            cndo2_data = df_cndo2.values
            print(f"Successfully loaded {cndo2_file} with pandas")
            print(f"Columns: {df_cndo2.columns.tolist()}")
        except Exception as e:
            print(f"Error loading with pandas: {e}")
            print("Trying numpy loadtxt...")
            cndo2_data = np.loadtxt(cndo2_file, delimiter=',', skiprows=1)
        
        try:
            df_cndos = pd.read_csv(cndos_file)
            cndos_data = df_cndos.values
            print(f"Successfully loaded {cndos_file} with pandas")
            print(f"Columns: {df_cndos.columns.tolist()}")
        except Exception as e:
            print(f"Error loading with pandas: {e}")
            print("Trying numpy loadtxt...")
            cndos_data = np.loadtxt(cndos_file, delimiter=',', skiprows=1)
            
    except Exception as e:
        print(f"Failed to load data files: {e}")
        print("\nFirst few lines of CNDO/2 file:")
        with open(cndo2_file, 'r') as f:
            print(''.join(f.readlines()[:5]))
        print("\nFirst few lines of CNDO/S file:")
        with open(cndos_file, 'r') as f:
            print(''.join(f.readlines()[:5]))
        sys.exit(1)
    
    # Print diagnostics
    debug_data(cndo2_data, "CNDO/2")
    debug_data(cndos_data, "CNDO/S")
    
    # Extract columns - with more careful handling
    cndo2_disp = cndo2_data[:, 0]
    cndo2_energy = cndo2_data[:, 1]
    
    cndos_disp = cndos_data[:, 0]
    cndos_energy = cndos_data[:, 1]
    
    # Check for NaN or inf values
    if np.any(np.isnan(cndo2_energy)) or np.any(np.isinf(cndo2_energy)):
        print("WARNING: CNDO/2 data contains NaN or inf values!")
    if np.any(np.isnan(cndos_energy)) or np.any(np.isinf(cndos_energy)):
        print("WARNING: CNDO/S data contains NaN or inf values!")
    
    # Display absolute energy ranges
    print(f"\nAbsolute Energy Ranges:")
    print(f"CNDO/2: {np.min(cndo2_energy):.6f} to {np.max(cndo2_energy):.6f} eV")
    print(f"CNDO/S: {np.min(cndos_energy):.6f} to {np.max(cndos_energy):.6f} eV")
    
    # Shift energies to have minimum at zero
    cndo2_energy_shifted = cndo2_energy - np.min(cndo2_energy)
    cndos_energy_shifted = cndos_energy - np.min(cndos_energy)
    
    print(f"\nRelative Energy Ranges (after shifting):")
    print(f"CNDO/2: 0.0 to {np.max(cndo2_energy_shifted):.6f} eV")
    print(f"CNDO/S: 0.0 to {np.max(cndos_energy_shifted):.6f} eV")
    
    # Check if the energy range is reasonable for bond displacement
    if np.max(cndo2_energy_shifted) > 10.0:
        print(f"WARNING: CNDO/2 energy range seems unusually large (> 10 eV)!")
    if np.max(cndos_energy_shifted) > 10.0:
        print(f"WARNING: CNDO/S energy range seems unusually large (> 10 eV)!")
    
    # Fit harmonic potentials
    fit_success_cndo2 = False
    fit_success_cndos = False
    
    try:
        # Check if we have enough points for fitting
        if len(cndo2_disp) >= 5:
            # Only fit near the minimum for better harmonic approximation
            min_idx = np.argmin(cndo2_energy)
            fit_range = min(2, len(cndo2_disp) // 3)  # Use about 1/3 of points centered at minimum
            start_idx = max(0, min_idx - fit_range)
            end_idx = min(len(cndo2_disp), min_idx + fit_range + 1)
            
            fit_disp = cndo2_disp[start_idx:end_idx]
            fit_energy = cndo2_energy[start_idx:end_idx]
            
            # Initial guess: x0 at minimum, k roughly estimated, E0 at minimum energy
            x0_guess = cndo2_disp[min_idx]
            # Rough estimate of k using neighboring points
            if min_idx > 0 and min_idx < len(cndo2_disp) - 1:
                dx = cndo2_disp[min_idx+1] - cndo2_disp[min_idx-1]
                dy = cndo2_energy[min_idx+1] - cndo2_energy[min_idx-1]
                k_guess = abs(dy / dx) * 2  # Rough approximation
            else:
                k_guess = 10.0  # Fallback
            
            popt_cndo2, pcov_cndo2 = curve_fit(
                harmonic, fit_disp, fit_energy, 
                p0=[x0_guess, k_guess, np.min(fit_energy)],
                maxfev=10000
            )
            
            # Check if the fit parameters make sense
            if popt_cndo2[1] > 0:  # k should be positive
                fit_success_cndo2 = True
                print(f"\nCNDO/2 Harmonic Fit Results:")
                print(f"  Equilibrium position (x0): {popt_cndo2[0]:.6f} Å")
                print(f"  Force constant (k): {popt_cndo2[1]:.6f} eV/Å²")
                print(f"  Minimum energy (E0): {popt_cndo2[2]:.6f} eV")
                
                # Convert force constant to mdyn/Å
                eV_to_mdyn_A = 160.2177
                cndo2_k_mdyn = popt_cndo2[1] * eV_to_mdyn_A
                print(f"  Force constant: {cndo2_k_mdyn:.4f} mdyn/Å")
            else:
                print("Warning: CNDO/2 fit produced negative force constant, which is physically unrealistic")
    except Exception as e:
        print(f"Could not fit CNDO/2 data: {e}")

    try:
        # Similar approach for CNDO/S
        if len(cndos_disp) >= 5:
            min_idx = np.argmin(cndos_energy)
            fit_range = min(2, len(cndos_disp) // 3)
            start_idx = max(0, min_idx - fit_range)
            end_idx = min(len(cndos_disp), min_idx + fit_range + 1)
            
            fit_disp = cndos_disp[start_idx:end_idx]
            fit_energy = cndos_energy[start_idx:end_idx]
            
            x0_guess = cndos_disp[min_idx]
            if min_idx > 0 and min_idx < len(cndos_disp) - 1:
                dx = cndos_disp[min_idx+1] - cndos_disp[min_idx-1]
                dy = cndos_energy[min_idx+1] - cndos_energy[min_idx-1]
                k_guess = abs(dy / dx) * 2
            else:
                k_guess = 10.0
            
            popt_cndos, pcov_cndos = curve_fit(
                harmonic, fit_disp, fit_energy, 
                p0=[x0_guess, k_guess, np.min(fit_energy)],
                maxfev=10000
            )
            
            if popt_cndos[1] > 0:
                fit_success_cndos = True
                print(f"\nCNDO/S Harmonic Fit Results:")
                print(f"  Equilibrium position (x0): {popt_cndos[0]:.6f} Å")
                print(f"  Force constant (k): {popt_cndos[1]:.6f} eV/Å²")
                print(f"  Minimum energy (E0): {popt_cndos[2]:.6f} eV")
                
                eV_to_mdyn_A = 160.2177
                cndos_k_mdyn = popt_cndos[1] * eV_to_mdyn_A
                print(f"  Force constant: {cndos_k_mdyn:.4f} mdyn/Å")
            else:
                print("Warning: CNDO/S fit produced negative force constant, which is physically unrealistic")
    except Exception as e:
        print(f"Could not fit CNDO/S data: {e}")
    
    # Calculate numerical derivatives for forces
    # Use central difference where possible for better accuracy
    cndo2_force = np.zeros_like(cndo2_disp)
    cndos_force = np.zeros_like(cndos_disp)
    
    for i in range(len(cndo2_disp)):
        if i == 0:
            # Forward difference for first point
            cndo2_force[i] = -(cndo2_energy[i+1] - cndo2_energy[i]) / (cndo2_disp[i+1] - cndo2_disp[i])
        elif i == len(cndo2_disp) - 1:
            # Backward difference for last point
            cndo2_force[i] = -(cndo2_energy[i] - cndo2_energy[i-1]) / (cndo2_disp[i] - cndo2_disp[i-1])
        else:
            # Central difference for interior points
            cndo2_force[i] = -(cndo2_energy[i+1] - cndo2_energy[i-1]) / (cndo2_disp[i+1] - cndo2_disp[i-1])
    
    for i in range(len(cndos_disp)):
        if i == 0:
            cndos_force[i] = -(cndos_energy[i+1] - cndos_energy[i]) / (cndos_disp[i+1] - cndos_disp[i])
        elif i == len(cndos_disp) - 1:
            cndos_force[i] = -(cndos_energy[i] - cndos_energy[i-1]) / (cndos_disp[i] - cndos_disp[i-1])
        else:
            cndos_force[i] = -(cndos_energy[i+1] - cndos_energy[i-1]) / (cndos_disp[i+1] - cndos_disp[i-1])
    
    # Check for unusually large forces
    if np.max(np.abs(cndo2_force)) > 100:
        print(f"WARNING: Very large CNDO/2 forces detected! Max: {np.max(np.abs(cndo2_force)):.2f} eV/Å")
    if np.max(np.abs(cndos_force)) > 100:
        print(f"WARNING: Very large CNDO/S forces detected! Max: {np.max(np.abs(cndos_force)):.2f} eV/Å")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create two subplots
    ax1 = plt.subplot(211)  # Energy plot
    ax2 = plt.subplot(212)  # Force plot
    
    # Plot energy curves with absolute values
    ax1.plot(cndo2_disp, cndo2_energy, 'bo-', label='CNDO/2 (absolute)')
    ax1.plot(cndos_disp, cndos_energy, 'rs-', label='CNDO/S (absolute)')
    
    # Add dotted lines for the shifted curves
    ax1.plot(cndo2_disp, cndo2_energy_shifted, 'b--', alpha=0.5, label='CNDO/2 (shifted)')
    ax1.plot(cndos_disp, cndos_energy_shifted, 'r--', alpha=0.5, label='CNDO/S (shifted)')
    
    if fit_success_cndo2:
        # Generate smooth curve for fitted function
        x_fine = np.linspace(np.min(cndo2_disp), np.max(cndo2_disp), 100)
        y_cndo2_fit = harmonic(x_fine, *popt_cndo2)
        
        # Plot the fitted curve
        ax1.plot(x_fine, y_cndo2_fit, 'b-', linewidth=1, alpha=0.7,
                label=f'CNDO/2 Fit (k = {popt_cndo2[1]:.2f} eV/Å²)')
    
    if fit_success_cndos:
        x_fine = np.linspace(np.min(cndos_disp), np.max(cndos_disp), 100)
        y_cndos_fit = harmonic(x_fine, *popt_cndos)
        
        ax1.plot(x_fine, y_cndos_fit, 'r-', linewidth=1, alpha=0.7,
                label=f'CNDO/S Fit (k = {popt_cndos[1]:.2f} eV/Å²)')
    
    # Plot forces
    ax2.plot(cndo2_disp, cndo2_force, 'bo-', label='CNDO/2')
    ax2.plot(cndos_disp, cndos_force, 'rs-', label='CNDO/S')
    
    # Add linear force lines from harmonic fits if available
    if fit_success_cndo2:
        # For a harmonic potential V = 0.5*k*(x-x0)^2, the force is F = -k*(x-x0)
        force_line = -popt_cndo2[1] * (x_fine - popt_cndo2[0])
        ax2.plot(x_fine, force_line, 'b-', linewidth=1, alpha=0.7,
                label=f'CNDO/2 Harmonic Force')
    
    if fit_success_cndos:
        force_line = -popt_cndos[1] * (x_fine - popt_cndos[0])
        ax2.plot(x_fine, force_line, 'r-', linewidth=1, alpha=0.7,
                label=f'CNDO/S Harmonic Force')
    
    # Set labels and titles
    ax1.set_title(f'Energy vs. Displacement for {molecule_name}')
    ax1.set_ylabel('Energy (eV)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    ax2.set_xlabel('Displacement (Å)')
    ax2.set_ylabel('Force (eV/Å)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add a second figure with the relative (shifted) energies only
    plt.figure(figsize=(12, 8))
    
    plt.plot(cndo2_disp, cndo2_energy_shifted, 'bo-', label='CNDO/2')
    plt.plot(cndos_disp, cndos_energy_shifted, 'rs-', label='CNDO/S')
    
    if fit_success_cndo2:
        y_cndo2_fit_shifted = y_cndo2_fit - np.min(y_cndo2_fit)
        plt.plot(x_fine, y_cndo2_fit_shifted, 'b--', 
                label=f'CNDO/2 Fit (k = {popt_cndo2[1]:.2f} eV/Å²)')
    
    if fit_success_cndos:
        y_cndos_fit_shifted = y_cndos_fit - np.min(y_cndos_fit)
        plt.plot(x_fine, y_cndos_fit_shifted, 'r--', 
                label=f'CNDO/S Fit (k = {popt_cndos[1]:.2f} eV/Å²)')
    
    plt.title(f'Relative Energy vs. Displacement for {molecule_name}')
    plt.xlabel('Displacement (Å)')
    plt.ylabel('Relative Energy (eV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    # Save or show the plots
    if output_file:
        # Save both figures
        plt.figure(1)  # First figure (absolute energies + forces)
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
        # Save the relative energy plot
        plt.figure(2)  # Second figure (relative energies)
        base, ext = os.path.splitext(output_file)
        relative_output = f"{base}_relative{ext}"
        plt.savefig(relative_output, dpi=300)
        print(f"Relative energy plot saved to {relative_output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()