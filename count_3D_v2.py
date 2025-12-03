#!/usr/bin/env python3
import os
import sys
import re
from datetime import datetime

"""
count3D.py — RELION 3D Classification Analysis & Timing Summary
--------------------------------------------------------------

This script analyzes a RELION 3D classification job directory
(Class3D/jobXXX/) and extracts:

  • Number of classes
  • Per-iteration class statistics from run_itXXX_model.star
  • Actual particle assignments from run_itXXX_optimiser.star
  • Per-iteration particle counts and class distributions
  • Total particle count (from *_data.star or fallback to optimiser)
  • Optics group composition
  • Iteration runtime durations based on file modification timestamps
  • Total job runtime and average iteration time

Outputs:
  • Full analysis printed to the terminal
  • A summary file "analysis.txt" written to Class3D/jobXXX/

Usage:
    python count3D.py <jobnumber>

Example:
    python count3D.py 035

This will analyze:
    Class3D/job035/run_it000_model.star
    Class3D/job035/run_it000_optimiser.star
    … up to final iteration

Notes:
  • Works with long-running classification jobs (hours to days).
  • Supports any number of iterations.
  • Handles missing _rlnTotalParticles by falling back to counting
    particle assignments in run_itXXX_optimiser.star.
  • Compatible with RELION 3.x–5.x STAR file formats.

Author: ChatGPT5 (for Eoin Leen)
Last updated: 2025-02-14
"""

# -----------------------------------------------
# Utility functions
# -----------------------------------------------

def format_number(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"

def read_star_classes(model_star_path):
    """Read class stats from model.star (data_model_classes block)."""
    classes = []
    with open(model_star_path, "r") as f:
        lines = f.readlines()

    in_block = False
    for line in lines:
        if line.strip().startswith("data_model_classes"):
            in_block = True
            continue
        if in_block and line.strip().startswith("loop_"):
            continue
        if in_block and line.strip().startswith("_rln"):
            continue
        if in_block and ".mrc" in line:
            parts = line.split()
            try:
                classes.append({
                    "class": len(classes) + 1,
                    "distribution": float(parts[1]),
                    "angacc": float(parts[2]),
                    "transacc": float(parts[3]),
                    "resolution": float(parts[4])
                })
            except Exception:
                # ignore malformed lines
                pass
        if in_block and line.strip().startswith("data_model_class_"):
            break
    return classes

def read_total_particles(data_star_path):
    """Extract total particles from *_data.star if present."""
    if not os.path.isfile(data_star_path):
        return None
    with open(data_star_path, "r") as f:
        for line in f:
            if line.startswith("_rlnTotalParticles"):
                try:
                    return int(float(line.split()[1]))
                except:
                    return None
    return None

# -------------------------------------------------------------
# Robust STAR parsers
# -------------------------------------------------------------

def parse_table_columns(lines, target_block):
    """
    Generic RELION STAR loop parser.
    Returns (headers_list, rows_list) for the first loop encountered under target_block.
    """
    in_block = False
    in_loop = False
    headers = []
    rows = []
    for line in lines:
        s = line.strip()
        if s.startswith("data_"):
            in_block = (s == target_block)
            in_loop = False
            continue
        if not in_block:
            continue
        if not s or s.startswith("#"):
            continue
        if s == "loop_":
            in_loop = True
            continue
        if in_loop and s.startswith("_rln"):
            headers.append(s.split()[0])
            continue
        if in_loop:
            # treat other lines as data rows while loop is active
            parts = s.split()
            # A simple heuristic: if first token is numeric or looks like a path (.mrc or @) accept it
            if parts:
                rows.append(parts)
                continue
            else:
                # end of loop
                break
    return headers, rows

def read_particle_classes_from_star(star_path, data_block="data_particles"):
    """
    Robust: parse the given star file for class numbers under `data_block`.
    Returns list of ints (class numbers). Works regardless of header ordering.
    """
    if not os.path.isfile(star_path):
        return []

    with open(star_path, "r") as f:
        lines = f.readlines()

    headers, rows = parse_table_columns(lines, data_block)

    if not headers or not rows:
        return []

    # map header name to index
    header_map = {}
    for idx, h in enumerate(headers):
        header_map[h] = idx

    if "_rlnClassNumber" not in header_map:
        return []

    cidx = header_map["_rlnClassNumber"]

    classes = []
    for r in rows:
        if len(r) <= cidx:
            continue
        try:
            classes.append(int(float(r[cidx])))
        except:
            pass
    return classes

def read_particle_classes(optimiser_star_path, data_star_path):
    """
    Try optimizer file first; if that yields no particles, fallback to data.star.
    Returns list of class numbers.
    """
    # 1) try optimiser
    classes = read_particle_classes_from_star(optimiser_star_path, data_block="data_particles")
    if classes:
        return classes

    # 2) fallback to data.star (often contains full particle table)
    classes = read_particle_classes_from_star(data_star_path, data_block="data_particles")
    if classes:
        return classes

    # 3) some RELION versions may store particle table under 'data_images' or other blocks;
    # try a few common alternatives
    for alt in ("data_images", "data_micrographs", "data_optics", "data_particles"):
        if alt == "data_particles":
            continue  # already tried
        classes = read_particle_classes_from_star(optimiser_star_path, data_block=alt)
        if classes:
            return classes
        classes = read_particle_classes_from_star(data_star_path, data_block=alt)
        if classes:
            return classes

    # nothing found
    return []

# -----------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python count3D.py <jobnumber>")
    sys.exit(1)

job = sys.argv[1]
job_dir = f"Class3D/job{job}"

if not os.path.isdir(job_dir):
    print(f"Error: directory {job_dir} not found.")
    sys.exit(1)

output_buffer = []   # capture everything printed

def out(x=""):
    print(x)
    output_buffer.append(x)

out(f"=== Analyzing Class3D Job {job} ===")
out(f"Directory: {job_dir}\n")

# ---------------------------------------------------------
# Detect all iterations
# ---------------------------------------------------------
iters = []
for fname in os.listdir(job_dir):
    m = re.match(r"run_it(\d+)_model\.star", fname)
    if m:
        iters.append(int(m.group(1)))

iters = sorted(iters)

if not iters:
    out("ERROR: No model.star iterations found!")
    sys.exit(1)

# ---------------------------------------------------------
# Detect number of classes
# ---------------------------------------------------------
first_model = f"{job_dir}/run_it{iters[0]:03d}_model.star"
class_stats = read_star_classes(first_model)
num_classes = len(class_stats)

out(f"Detected {num_classes} classes in the classification\n")

# ---------------------------------------------------------
# Determine total particle count
# ---------------------------------------------------------
data_star = f"{job_dir}/run_it{iters[0]:03d}_data.star"
total_particles = read_total_particles(data_star)

fallback_needed = (total_particles is None) or (total_particles == 0)

if fallback_needed:
    out("Warning: _rlnTotalParticles missing or zero — counting particles manually from optimiser/data.star.")
    first_opt = f"{job_dir}/run_it{iters[0]:03d}_optimiser.star"
    assigned_tmp = read_particle_classes(first_opt, data_star)
    total_particles = len(assigned_tmp)

# final safety
if total_particles is None:
    total_particles = 0

out(f"Total particles in dataset: {format_number(total_particles)}\n")

# ---------------------------------------------------------
# ITERATION LOOP
# ---------------------------------------------------------

timestamps = {}

for it in iters:
    model_path = f"{job_dir}/run_it{it:03d}_model.star"
    optimiser_path = f"{job_dir}/run_it{it:03d}_optimiser.star"
    data_path = f"{job_dir}/run_it{it:03d}_data.star"

    # timestamp for timing summary (model file mtime is reliable)
    if os.path.isfile(model_path):
        timestamps[it] = os.path.getmtime(model_path)
    else:
        timestamps[it] = os.path.getmtime(job_dir)  # fallback

    classes = read_star_classes(model_path)
    assigned = read_particle_classes(optimiser_path, data_path)

    out(f"======== Iteration {it} ========")
    out(f"Particles in this iteration: {format_number(len(assigned))}\n")

    # Class stats
    out("Class Statistics:")
    out("Class | Particles |   (%)  | Ang.Acc | Trans.Acc | Resolution")
    out("------|-----------|--------|---------|-----------|------------")

    for c in classes:
        pct = c["distribution"] * 100
        n = int(total_particles * c["distribution"]) if total_particles > 0 else 0
        out(f"{c['class']:>5} | {n:>9} | {pct:6.2f} | "
            f"{c['angacc']:.2f}° | {c['transacc']:.2f} Å | {c['resolution']:.2f} Å")
    out("")

    # Actual distribution
    counts = {i: 0 for i in range(1, num_classes+1)}
    for num in assigned:
        if num in counts:
            counts[num] += 1

    denom = total_particles if total_particles > 0 else sum(counts.values())

    out("Actual particle distribution:")
    for k in sorted(counts):
        if denom > 0:
            pct = 100 * counts[k] / denom
            out(f"  Class {k}: {counts[k]:7} particles ({pct:5.1f}%)")
        else:
            out(f"  Class {k}: {counts[k]:7} particles (n/a)")
    out(f"  Total:   {format_number(sum(counts.values()))} particles\n")

    # Optics group (assume 1)
    out("Optics Group Composition:")
    if denom > 0:
        og_pct = 100.0
        out(f"  Optics Group 1: {format_number(sum(counts.values()))} particles ({og_pct:.1f}%)")
    else:
        out(f"  Optics Group 1: {format_number(sum(counts.values()))} particles (n/a)")
    out(f"  Total:   {format_number(sum(counts.values()))}\n")

    # Matrix
    out("Optics Group vs Class Matrix:")
    header = "     OpticGroup"
    for c in range(1, num_classes + 1):
        header += f"         Class_{c}"
    header += "           Total"
    out(header)

    row = "  OpticsGroup_1"
    og_total = sum(counts.values())
    for c in range(1, num_classes + 1):
        if og_total > 0:
            pct = 100 * (counts[c] / og_total)
            row += f"  {counts[c]} ({pct:5.1f}%)"
        else:
            row += f"  {counts[c]} (n/a)"
    row += f"          {format_number(og_total)}"
    out(row)

    total_row = "          Total "
    for c in range(1, num_classes + 1):
        total_row += f"       {counts[c]}"
    total_row += f"       {format_number(sum(counts.values()))}"
    out(total_row)
    out("")

# ---------------------------------------------------------
# TIMING SUMMARY
# ---------------------------------------------------------

out("\n===== Timing Summary =====")

sorted_iters = sorted(timestamps.keys())
start_time = datetime.fromtimestamp(timestamps[sorted_iters[0]])
end_time = datetime.fromtimestamp(timestamps[sorted_iters[-1]])

for i, it in enumerate(sorted_iters):
    t_end = datetime.fromtimestamp(timestamps[it])
    if i == 0:
        out(f"Iteration {it}: End = {t_end}   (first iteration)")
    else:
        t_start = datetime.fromtimestamp(timestamps[sorted_iters[i-1]])
        delta = t_end - t_start
        out(f"Iteration {it}: {delta}   (from {t_start.time()} to {t_end.time()})")

total_runtime = end_time - start_time
out(f"\nTotal job runtime: {total_runtime}")

if len(sorted_iters) > 1:
    avg_runtime = total_runtime / (len(sorted_iters) - 1)
    out(f"Average per iteration: {avg_runtime}\n")

# ---------------------------------------------------------
# WRITE analysis.txt
# ---------------------------------------------------------

output_path = f"{job_dir}/analysis.txt"
with open(output_path, "w") as f:
    for line in output_buffer:
        f.write(line + "\n")

out(f"Analysis written to: {output_path}")
