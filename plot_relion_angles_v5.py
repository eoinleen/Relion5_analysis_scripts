#!/usr/bin/env python3
"""
================================================================================
RELION Angular Distribution Analysis and Visualization Tool
================================================================================

Author: Eoin Leen UoL 
Version: 5.0
Date: December 2024

Description:
    Comprehensive analysis tool for assessing angular distribution quality in 
    cryo-EM single particle reconstructions processed with RELION. This script
    analyzes particle orientation distributions from RELION refinement iterations
    and generates diagnostic plots to identify potential preferred orientation
    issues that could limit reconstruction quality.

Key Features:
    - Reads RELION star files containing particle Euler angles
    - Converts Euler angles to viewing directions on the unit sphere
    - Performs antipodal symmetry folding for orientation analysis
    - Calculates multiple anisotropy metrics (entropy, spherical variance, etc.)
    - Generates publication-quality diagnostic visualizations
    - Produces both RELION-style and CryoSPARC-style angular distribution plots
    - Automatically generates methods text for scientific manuscripts

Input:
    RELION star files (run_it###_data.star) containing particle orientations

Output:
    - angular_diagnostics_it###.png: Three-panel diagnostic figure
    - viewing_dirs_cryosparc_style_it###.png: Mollweide projection plot
    - angular_distribution_methods.txt: Methods section for manuscript
    - Console output with quantitative metrics

Usage:
    python plot_relion_angles_v4.py
    
    Modify USER SETTINGS section below to customize analysis parameters:
    - ITERATIONS: List of iteration numbers to analyze
    - JOB_DIR: Directory containing RELION job files
    - COLORMAP: Matplotlib colormap for visualization
    - Angular binning parameters for hemisphere discretization

Requirements:
    - Python 3.6+
    - NumPy
    - Matplotlib

    IF ON 300304 "conda activate plotting" will load these modules

References:
    - Scheres, S.H.W. (2012) RELION: Implementation of a Bayesian approach to
      cryo-EM structure determination. J Struct Biol 180, 519-530.
    - Naydenova, K. & Russo, C.J. (2017) Measuring the effects of particle 
      orientation to improve the efficiency of electron cryomicroscopy. 
      Nat Commun 8, 629.

================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# ============================================================
# USER SETTINGS
# ============================================================

ITERATIONS = [15, 20]
JOB_DIR = "./"
COLORMAP = "viridis"
FIGSIZE = (16, 5)
DPI = 600

# Angular binning for folded hemisphere
N_LON = 36   # 10° bins for longitude
N_LAT = 18   # 10° bins for latitude

# Thresholds for methods text interpretation
ENTROPY_THRESHOLD_GOOD = 0.85  # Above this is considered good coverage
ENTROPY_THRESHOLD_BAD = 0.70   # Below this indicates poor coverage
MAX_BIN_THRESHOLD = 5.0        # Maximum acceptable % in single bin

# ============================================================
# STAR FILE READER (minimal, robust)
# ============================================================

def read_star_angles(starfile):
    """
    Parse RELION star file to extract Euler angles.
    
    Parameters:
        starfile (str): Path to RELION star file
        
    Returns:
        tuple: Arrays of (rot, tilt, psi) angles and particle count
    """
    rot, tilt, psi = [], [], []

    with open(starfile) as f:
        lines = f.readlines()

    cols = {}
    in_loop = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Detect loop start
        if line == "loop_":
            in_loop = True
            continue

        # Read column definitions
        if in_loop and line.startswith("_rln"):
            name, idx = line.split()
            cols[name] = int(idx[1:]) - 1
            continue

        # Data line: must start with a number or minus sign
        if in_loop and (line[0].isdigit() or line[0] == "-"):
            parts = line.split()

            # Make sure required columns exist
            if "_rlnAngleRot" not in cols or "_rlnAngleTilt" not in cols:
                continue
            if max(cols["_rlnAngleRot"], cols["_rlnAngleTilt"]) >= len(parts):
                continue

            rot.append(float(parts[cols["_rlnAngleRot"]]))
            tilt.append(float(parts[cols["_rlnAngleTilt"]]))

            if "_rlnAnglePsi" in cols and cols["_rlnAnglePsi"] < len(parts):
                psi.append(float(parts[cols["_rlnAnglePsi"]]))

    return np.array(rot), np.array(tilt), len(rot)

# ============================================================
# RELION EULER → VIEWING DIRECTION
# ============================================================

def relion_to_viewing(rot, tilt):
    """
    Convert RELION Euler angles to viewing directions.
    
    The viewing direction represents the direction from which the particle
    is viewed in the microscope reference frame.
    
    Parameters:
        rot (array): Rotation angles in degrees
        tilt (array): Tilt angles in degrees
        
    Returns:
        tuple: Cartesian coordinates (vx, vy, vz) of viewing directions
    """
    r = np.deg2rad(rot)
    t = np.deg2rad(tilt)

    vx =  np.sin(r) * np.sin(t)
    vy = -np.cos(r) * np.sin(t)
    vz =  np.cos(t)

    return vx, vy, vz

# ============================================================
# ANISOTROPY METRICS (folded)
# ============================================================

def angular_metrics(vx, vy, vz):
    """
    Calculate quantitative metrics for angular distribution quality.
    
    Parameters:
        vx, vy, vz (arrays): Cartesian coordinates of viewing directions
        
    Returns:
        tuple: (entropy, max_bin_percent, spherical_variance, anisotropy_index)
    """
    # Apply antipodal folding
    mask = vz < 0
    vx, vy, vz = vx.copy(), vy.copy(), vz.copy()
    vx[mask] *= -1
    vy[mask] *= -1
    vz[mask] *= -1

    lon = np.arctan2(vy, vx)
    lat = np.arcsin(vz)

    H, _, _ = np.histogram2d(
        lon, lat,
        bins=[N_LON, N_LAT],
        range=[[-np.pi, np.pi], [0, np.pi/2]]
    )

    counts = H[H > 0]
    p = counts / counts.sum()

    entropy = -np.sum(p * np.log(p)) / np.log(len(p))
    max_bin = counts.max() / counts.sum() * 100
    mean_vec = np.array([vx.mean(), vy.mean(), vz.mean()])
    spherical_variance = 1 - np.linalg.norm(mean_vec)
    anisotropy_index = np.std(counts) / np.mean(counts)

    return entropy, max_bin, spherical_variance, anisotropy_index, len(counts)

# ============================================================
# METHODS TEXT GENERATOR
# ============================================================

def generate_methods_text(iterations_data, output_file="angular_distribution_methods.txt"):
    """
    Generate methods section text for scientific manuscript based on analysis results.
    
    Parameters:
        iterations_data (list): List of dictionaries containing metrics for each iteration
        output_file (str): Path to output methods text file
    """
    
    # Analyze the final iteration (typically the most relevant)
    final_data = iterations_data[-1]
    
    # Determine overall quality assessment
    entropy = final_data['entropy']
    max_bin = final_data['max_bin']
    
    if entropy > ENTROPY_THRESHOLD_GOOD and max_bin < MAX_BIN_THRESHOLD:
        quality = "good"
        quality_desc = "well-distributed"
    elif entropy < ENTROPY_THRESHOLD_BAD or max_bin > MAX_BIN_THRESHOLD * 2:
        quality = "poor"
        quality_desc = "showing significant preferred orientation"
    else:
        quality = "moderate"
        quality_desc = "with moderate angular coverage"
    
    # Generate methods text
    methods_text = f"""
================================================================================
ANGULAR DISTRIBUTION ANALYSIS - METHODS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

Cryo-EM Data Processing - Angular Distribution Analysis

The angular distribution of particle orientations was assessed using custom 
Python scripts to evaluate potential preferred orientation issues that could 
limit reconstruction quality. Particle orientation parameters were extracted 
from RELION refinement star files at iterations {', '.join(map(str, [d['iteration'] for d in iterations_data]))}.

Euler angles (rot, tilt, psi) were converted to unit sphere viewing directions 
using the RELION convention. Antipodal symmetry was applied by folding all 
viewing directions to the upper hemisphere (v ≡ -v for vz < 0), as particles 
viewed from opposite directions provide equivalent information in single 
particle reconstruction.

The folded hemisphere was discretized into {N_LON} × {N_LAT} angular bins 
(10° resolution in both longitude and latitude), resulting in {N_LON * N_LAT} 
total bins covering the unique angular space. The distribution of {final_data['n_particles']:,} 
particles across these bins was analyzed.

Quantitative metrics were calculated to assess angular coverage quality:

1. Distribution entropy: {final_data['entropy']:.3f} (normalized to [0,1], where 1 
   indicates perfectly uniform distribution)
   
2. Maximum bin occupancy: {final_data['max_bin']:.1f}% of particles in the most 
   populated bin
   
3. Spherical variance: {final_data['spherical_variance']:.3f} (measures clustering, 
   0 = perfectly uniform, 1 = all particles in one orientation)
   
4. Anisotropy index: {final_data['anisotropy_index']:.2f} (coefficient of variation 
   of bin counts, lower values indicate more uniform coverage)

5. Angular coverage: {final_data['occupied_bins']} of {N_LON * N_LAT // 2} possible bins 
   contained at least one particle ({final_data['occupied_bins'] / (N_LON * N_LAT // 2) * 100:.1f}% coverage)

Assessment: The angular distribution was {quality_desc}, with an entropy of 
{final_data['entropy']:.3f} (threshold for good coverage: >{ENTROPY_THRESHOLD_GOOD:.2f}) and 
maximum bin occupancy of {final_data['max_bin']:.1f}% (threshold: <{MAX_BIN_THRESHOLD:.1f}%). 
"""

    if quality == "poor":
        methods_text += f"""
The observed preferred orientation may limit the achievable resolution, 
particularly along certain directions in Fourier space. Strategies such as 
tilted data collection or modified grid preparation may be beneficial for 
improving angular coverage in future data collections.
"""
    elif quality == "moderate":
        methods_text += f"""
While the angular distribution shows some degree of preferred orientation, 
the coverage is sufficient for structure determination. Minor anisotropy in 
the reconstruction may be expected.
"""
    else:
        methods_text += f"""
The well-distributed angular coverage indicates that preferred orientation is 
not a significant limiting factor for this reconstruction.
"""

    methods_text += f"""
Visualization: Angular distributions were visualized using Mollweide projections 
of the viewing sphere, with color intensity representing particle density. Both 
folded hemisphere and full sphere (CryoSPARC-style) representations were generated 
for comprehensive assessment.

Software: Analysis performed using RELION {get_relion_version()} with custom 
Python scripts utilizing NumPy {np.__version__} and Matplotlib {plt.matplotlib.__version__} 
for quantitative analysis and visualization.

================================================================================
RAW METRICS SUMMARY
================================================================================

"""
    
    # Add table of metrics for all iterations
    methods_text += f"{'Iteration':<12} {'Particles':<12} {'Entropy':<10} {'Max Bin %':<12} {'Sph. Var.':<12} {'Aniso. Index':<12}\n"
    methods_text += "-" * 80 + "\n"
    
    for data in iterations_data:
        methods_text += f"{data['iteration']:<12} {data['n_particles']:<12,} {data['entropy']:<10.3f} "
        methods_text += f"{data['max_bin']:<12.2f} {data['spherical_variance']:<12.3f} {data['anisotropy_index']:<12.3f}\n"
    
    methods_text += """
================================================================================
INTERPRETATION GUIDE
================================================================================

Entropy (0-1):
  > 0.85  : Excellent coverage
  0.70-0.85: Moderate coverage  
  < 0.70  : Poor coverage, significant preferred orientation

Maximum Bin %:
  < 2%    : Excellent distribution
  2-5%    : Acceptable distribution
  > 5%    : Concerning concentration of particles

Spherical Variance (0-1):
  < 0.3   : Good isotropy
  0.3-0.5 : Moderate anisotropy
  > 0.5   : Strong preferred orientation

Anisotropy Index:
  < 0.5   : Uniform distribution
  0.5-1.0 : Moderate variation
  > 1.0   : High variation, non-uniform coverage

================================================================================
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(methods_text)
    
    print(f"\nMethods text written to: {output_file}")
    
    return methods_text

def get_relion_version():
    """Try to determine RELION version from environment or return generic version."""
    # This is a placeholder - in practice you might read from version file or environment
    return "4.0"

# ============================================================
# MAIN PLOTTING
# ============================================================

def plot_iteration(iteration, rot, tilt, vx, vy, vz, n_particles):
    """
    Generate comprehensive visualization of angular distribution for one iteration.
    
    Parameters:
        iteration (int): Iteration number
        rot, tilt (arrays): Euler angles
        vx, vy, vz (arrays): Viewing direction coordinates
        n_particles (int): Total number of particles
        
    Returns:
        dict: Dictionary containing calculated metrics
    """

    # -------------------------
    # Antipodal folding
    # -------------------------
    mask = vz < 0
    vx_f, vy_f, vz_f = vx.copy(), vy.copy(), vz.copy()
    vx_f[mask] *= -1
    vy_f[mask] *= -1
    vz_f[mask] *= -1

    lon = np.arctan2(vy_f, vx_f)
    lat = np.arcsin(vz_f)

    # Explicit binning
    lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
    lat_edges = np.linspace(0, np.pi/2, N_LAT + 1)
    H, _, _ = np.histogram2d(lon, lat, bins=[lon_edges, lat_edges])

    # =====================================================
    # 3-panel diagnostic figure
    # =====================================================

    fig = plt.figure(figsize=FIGSIZE)
    fig.suptitle(
        f"RELION Angular Distribution Analysis – Iteration {iteration} ({n_particles:,} particles)",
        fontsize=11, fontweight="bold"
    )

    # Panel 1: Euler angles
    ax1 = plt.subplot(131)
    h = ax1.hist2d(rot, tilt, bins=50, cmap=COLORMAP, cmin=1)
    ax1.set_xlabel("Rot (degrees)", fontsize=9)
    ax1.set_ylabel("Tilt (degrees)", fontsize=9)
    ax1.set_title("Euler angle distribution (Rot vs Tilt)", fontsize=10)
    ax1.tick_params(axis='both', labelsize=8)
    cbar1 = plt.colorbar(h[3], ax=ax1)
    cbar1.set_label("Particle count", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: Tilt marginal
    ax2 = plt.subplot(132)
    ax2.hist(tilt, bins=40, edgecolor="black", alpha=0.7)
    ax2.axvline(90, color="green", linestyle=":", label="Top view (90°)")
    ax2.axvline(np.mean(tilt), color="red", linestyle="--",
                label=f"Mean: {np.mean(tilt):.1f}°")
    ax2.set_xlabel("Tilt (degrees)", fontsize=9)
    ax2.set_ylabel("Number of particles", fontsize=9)
    ax2.set_title("Tilt angle marginal distribution", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.grid(alpha=0.3, axis="y")

    # Panel 3: Folded viewing directions (explicit zero bins)
    ax3 = plt.subplot(133, projection="mollweide")
    ax3.set_title("Viewing direction coverage (antipodally folded)", fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.tick_params(axis='both', labelsize=8)

    # Zero bins (light grey)
    for i in range(N_LON):
        for j in range(N_LAT):
            if H[i, j] == 0:
                ax3.scatter(
                    0.5 * (lon_edges[i] + lon_edges[i+1]),
                    0.5 * (lat_edges[j] + lat_edges[j+1]),
                    s=18,
                    color="#e6e6e6",
                    marker="h",
                    zorder=1
                )

    # Non-zero bins
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c, indexing="ij")

    sc = ax3.scatter(
        LON[H > 0],
        LAT[H > 0],
        c=H[H > 0],
        s=30,
        cmap=COLORMAP,
        zorder=2
    )
    cbar3 = plt.colorbar(sc, ax=ax3)
    cbar3.set_label("Particle count", fontsize=9)
    cbar3.ax.tick_params(labelsize=8)

    # Longitude labels below equator (reduced font size)
    lon_degs = np.arange(-150, 180, 30)
    lon_rads = np.deg2rad(lon_degs)
    ax3.set_xticks(lon_rads)
    ax3.set_xticklabels([])

    for lr, d in zip(lon_rads, lon_degs):
        ax3.text(lr, np.deg2rad(-10), f"{d}°",
                 ha="center", va="top", fontsize=7)

    ax3.axhline(0, color="black", lw=0.6, alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_main = f"angular_diagnostics_it{iteration:03d}.png"
    plt.savefig(out_main, dpi=DPI)
    plt.close()

    # =====================================================
    # CryoSPARC-style visualization
    # =====================================================

    vx_m = np.concatenate([vx_f, -vx_f])
    vy_m = np.concatenate([vy_f, -vy_f])
    vz_m = np.concatenate([vz_f, -vz_f])

    lon_m = np.arctan2(vy_m, vx_m)
    lat_m = np.arcsin(vz_m)

    fig2 = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111, projection="mollweide")
    hb = ax.hexbin(lon_m, lat_m, gridsize=35, cmap=COLORMAP, mincnt=1)
    
    # Reduced title font size
    ax.set_title("Viewing direction coverage (CryoSPARC-style visualization)", fontsize=10)
    ax.grid(alpha=0.3)
    
    # Reduced colorbar label font size
    cbar = plt.colorbar(hb, ax=ax, label="Particle count")
    cbar.set_label("Particle count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Reduced tick label sizes
    ax.tick_params(axis='both', labelsize=8)

    annotation = (
        "NOTE: Viewing directions were antipodally folded (v ≡ −v),\n"
        "then mirrored across the equator for visualization only.\n"
        "No additional orientation information is introduced."
    )
    fig2.text(0.5, 0.02, annotation, ha="center", fontsize=7)

    out_cs = f"viewing_dirs_cryosparc_style_it{iteration:03d}.png"
    plt.savefig(out_cs, dpi=DPI)
    plt.close()

    # =====================================================
    # Metrics
    # =====================================================

    ent, maxb, svar, aniso, occupied = angular_metrics(vx, vy, vz)

    print(f"\nIteration {iteration}")
    print(f"  Particles           : {n_particles:,}")
    print(f"  Entropy             : {ent:.3f}")
    print(f"  Max bin %           : {maxb:.3f}")
    print(f"  Spherical variance  : {svar:.3f}")
    print(f"  Anisotropy index    : {aniso:.3f}")
    print(f"  Occupied bins       : {occupied} / {N_LON * N_LAT // 2}")
    print(f"  Saved: {out_main}")
    print(f"  Saved: {out_cs}")
    
    # Return metrics for methods text generation
    return {
        'iteration': iteration,
        'n_particles': n_particles,
        'entropy': ent,
        'max_bin': maxb,
        'spherical_variance': svar,
        'anisotropy_index': aniso,
        'occupied_bins': occupied
    }

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """
    Main execution function - processes all specified iterations and generates reports.
    """
    print("\n" + "="*80)
    print("RELION ANGULAR DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Processing iterations: {ITERATIONS}")
    print(f"Job directory: {JOB_DIR}")
    print(f"Angular binning: {N_LON} × {N_LAT} bins")
    print("="*80)
    
    iterations_data = []
    
    for it in ITERATIONS:
        star = os.path.join(JOB_DIR, f"run_it{it:03d}_data.star")
        if not os.path.exists(star):
            print(f"\nIteration {it}: file not found ({star})")
            continue

        print(f"\nProcessing iteration {it}...")
        rot, tilt, n = read_star_angles(star)
        vx, vy, vz = relion_to_viewing(rot, tilt)
        metrics = plot_iteration(it, rot, tilt, vx, vy, vz, n)
        iterations_data.append(metrics)
    
    # Generate methods text if we processed any iterations
    if iterations_data:
        print("\n" + "="*80)
        print("GENERATING METHODS TEXT")
        print("="*80)
        generate_methods_text(iterations_data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll output files have been generated.")
    print("Check the current directory for:")
    print("  - angular_diagnostics_it###.png (diagnostic plots)")
    print("  - viewing_dirs_cryosparc_style_it###.png (CryoSPARC-style plots)")
    print("  - angular_distribution_methods.txt (methods section text)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
