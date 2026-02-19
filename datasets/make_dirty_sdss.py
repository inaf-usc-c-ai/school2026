#!/usr/bin/env python3
"""
make_dirty_sdss.py

Utility to create a 'dirty' version of the SDSS17 stellar classification dataset.
All indentation uses TAB characters.

A script that creates a modified (say dirtier) version of the csv dataset in which I artificially corrupt some data in the following way: 
  - add missing data (NaNs/inf) in the flux band variables (u, v, etc) with a configurable missing fraction per column 
  - add duplicated observations with configurable parameters (e.g. fraction of duplicated rows) 
  - add outliers in the flux band variables (e.g. unreasonable/unphysical flux values) with a configurable fraction per each column 
  - add anomalies in the flux band variables with a configurable fraction per each column. This would have physical flux values but for example peculiar combinations of multiple band fluxes. 
  - add mislabeled class types with a configurable fraction, e.g. adding one extra blank space at the beginning/end of the class label string, or inserting the class with capital letters, etc 
  - (optionally) add saturated clipped values for flux variables with a configurable fraction per column I need the updated dataset to make a tutorial on data preparation/cleaning.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def parse_kv_fractions(items: Optional[Sequence[str]]) -> Dict[str, float]:
	if not items:
		return {}
	out: Dict[str, float] = {}
	for it in items:
		if "=" not in it:
			raise ValueError(f"Expected KEY=VALUE, got: {it}")
		k, v = it.split("=", 1)
		v = float(v)
		if not (0.0 <= v <= 1.0):
			raise ValueError(f"Fraction must be in [0,1], got {k}={v}")
		out[k.strip()] = v
	return out


def choose_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
	k = int(round(frac * n))
	if k <= 0:
		return np.array([], dtype=int)
	k = min(k, n)
	return rng.choice(n, size=k, replace=False)


def ensure_cols_exist(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
	missing = [c for c in cols if c not in df.columns]
	if missing:
		raise KeyError(f"Columns not found: {missing}")
	return list(cols)


def infer_label_col(df: pd.DataFrame) -> str:
	for c in ["class", "Class", "label", "target"]:
		if c in df.columns:
			return c
	obj_cols = [c for c in df.columns if df[c].dtype == "object"]
	if not obj_cols:
		raise KeyError("Could not infer label column. Use --label-col.")
	return obj_cols[0]


@dataclass
class CorruptionReport:
	n_rows_in: int
	n_rows_out: int
	flux_cols: List[str]
	label_col: str
	seed: int
	duplicated_rows_added: int
	missing_injected: Dict[str, int]
	inf_injected: Dict[str, int]
	outliers_injected: Dict[str, int]
	saturated_injected: Dict[str, int]
	saturation_mode: str
	saturation_q_low: float
	saturation_q_high: float
	anomalies_rows: int
	mislabeled_rows: int


def inject_missing_and_inf(df, cols, col_fracs, rng, inf_share=0.25):
	""" Add missing and inf values in input data frame """
	df = df.copy()
	n = len(df)
	nan_counts, inf_counts = {}, {}
	for c in cols:
		frac = col_fracs.get(c, 0.0)
		idx = choose_indices(n, frac, rng)
		k_inf = int(round(inf_share * len(idx)))
		inf_idx = idx[:k_inf]
		nan_idx = idx[k_inf:]
		if len(inf_idx):
			signs = rng.choice([-1.0, 1.0], size=len(inf_idx))
			df.loc[df.index[inf_idx], c] = signs * np.inf
		if len(nan_idx):
			df.loc[df.index[nan_idx], c] = np.nan
		nan_counts[c] = len(nan_idx)
		inf_counts[c] = len(inf_idx)
	return df, nan_counts, inf_counts
	
def inject_outliers(df, cols, col_fracs, rng, outlier_mult_min=8.0, outlier_mult_max=15.0):
	df = df.copy()
	n = len(df)
	counts = {}

	# Initialize outlier flag column
	if "is_outlier" not in df.columns:
		df["is_outlier"] = 0

	for c in cols:
		frac = col_fracs.get(c, 0.0)
		idx = choose_indices(n, frac, rng)
		if not len(idx):
			counts[c] = 0
			continue

		x = pd.to_numeric(df[c], errors="coerce").values
		mu = np.nanmean(x)
		sd = np.nanstd(x)
		sd = sd if np.isfinite(sd) and sd > 0 else 1.0

		mult = rng.uniform(outlier_mult_min, outlier_mult_max, size=len(idx))
		signs = rng.choice([-1.0, 1.0], size=len(idx))

		df.loc[df.index[idx], c] = mu + signs * mult * sd
		
		# Mark these rows as outliers
		df.loc[df.index[idx], "is_outlier"] = 1

		counts[c] = len(idx)

	return df, counts	

def inject_saturation(df, cols, col_fracs, rng, mode="replace", q_low=0.001, q_high=0.999):
	""" Add saturated values and create censoring flags """
	df = df.copy()
	n = len(df)
	counts = {}

	for c in cols:
		frac = col_fracs.get(c, 0.0)
		flag_col = f"{c}_is_censored"

		# Initialize flag column to 0
		if flag_col not in df.columns:
			df[flag_col] = 0

		x = pd.to_numeric(df[c], errors="coerce")
		lo = np.nanquantile(x.values, q_low)
		hi = np.nanquantile(x.values, q_high)

		if mode == "clip":
			if frac <= 0.0:
				counts[c] = 0
				continue

			mask_low = x < lo
			mask_high = x > hi
			mask = mask_low | mask_high

			# Count censored values
			counts[c] = int(mask.sum(skipna=True))

			# Apply clipping
			df[c] = x.clip(lower=lo, upper=hi)

			# Set censoring flag
			df.loc[mask, flag_col] = 1

		else:
			# Replacement-only mode
			idx = choose_indices(n, frac, rng)
			if not len(idx):
				counts[c] = 0
				continue

			half = len(idx) // 2
			rows_low = df.index[idx[:half]]
			rows_high = df.index[idx[half:]]

			df.loc[rows_low, c] = lo
			df.loc[rows_high, c] = hi

			# In replace mode, we also flag them as censored
			df.loc[df.index[idx], flag_col] = 1

			counts[c] = len(idx)

	return df, counts

def inject_anomalies_multivariate(df, cols, frac_rows, rng):
	""" Add anomalies"""
	df = df.copy()
	n = len(df)
	
	# Initialize anomaly flag column
	if "is_anomaly" not in df.columns:
		df["is_anomaly"] = 0
	
	idx_rows = choose_indices(n, frac_rows, rng)
	for ridx in idx_rows:
		k = int(rng.integers(2, len(cols) + 1))
		chosen = rng.choice(cols, size=k, replace=False)
		donor = int(rng.integers(0, n))
		while donor == ridx:
			donor = int(rng.integers(0, n))
		df.loc[df.index[ridx], chosen] = df.loc[df.index[donor], chosen].values
		
		# Mark row as anomalous
		df.loc[df.index[ridx], "is_anomaly"] = 1

	return df, len(idx_rows)


def corrupt_labels(df, label_col, frac_rows, rng, mode="all"):
	""" Corrupt class labels """
	df = df.copy()
	n = len(df)
	idx = choose_indices(n, frac_rows, rng)
	s = df[label_col].astype(str).values

	if mode == "space_case":
		# Only: leading/trailing blanks and upper/lower case
		ops = ["lead_space", "trail_space", "upper", "lower"]
	else:
		# Original full set
		ops = ["lead_space", "trail_space", "upper", "lower", "title", "inner_space", "tab", "newline"]

	for i in idx:
		val = s[i]
		op = ops[int(rng.integers(0, len(ops)))]
		if op == "lead_space":
			s[i] = " " + val
		elif op == "trail_space":
			s[i] = val + " "
		elif op == "upper":
			s[i] = val.upper()
		elif op == "lower":
			s[i] = val.lower()
		elif op == "title":
			s[i] = val.title()
		elif op == "inner_space":
			pos = int(rng.integers(1, len(val))) if len(val) > 1 else 1
			s[i] = val[:pos] + " " + val[pos:]
		elif op == "tab":
			s[i] = val + "\t"
		elif op == "newline":
			s[i] = val + "\n"

	df[label_col] = s
	return df, len(idx)


def add_duplicates(df, dup_frac, rng):
	""" Add duplicated rows """
	n = len(df)
	k = int(round(dup_frac * n))
	if k <= 0:
		return df.copy(), 0
	dup_idx = rng.choice(n, size=min(k, n), replace=False)
	dup_rows = df.iloc[dup_idx].copy()
	out = pd.concat([df, dup_rows], ignore_index=True)
	out = out.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
	return out, len(dup_rows)

########################################
####      MAIN
########################################
def main():
	""" Main program """
	
	# - Define and parse arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", required=True)
	ap.add_argument("--output", required=True)
	ap.add_argument("--output-base", default=None, help="Optional output file containing only original columns (no diagnostic flags)")
	ap.add_argument("--report", default=None)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--flux-cols", nargs="+", default=["u","g","r","i","z"])
	ap.add_argument("--label-col", default=None)
	ap.add_argument("--missing", nargs="*", default=[])
	ap.add_argument("--inf-share", type=float, default=0.25)
	ap.add_argument("--outliers", nargs="*", default=[])
	ap.add_argument("--anomalies", type=float, default=0.0)
	ap.add_argument("--dup-frac", type=float, default=0.0)
	ap.add_argument("--mislabeled", type=float, default=0.0)
	ap.add_argument("--label-corrupt-mode", choices=["all","space_case"], default="all", help="Label corruption modes: all=all operations, space_case=only leading/trailing blanks + case changes")
	ap.add_argument("--saturation", nargs="*", default=[])
	ap.add_argument("--saturation-mode", choices=["replace","clip"], default="replace")
	ap.add_argument("--saturation-q-low", type=float, default=0.001)
	ap.add_argument("--saturation-q-high", type=float, default=0.999)

	print("Parsing arguments ...")
	args = ap.parse_args()
	
	if not (0.0 < args.saturation_q_low < args.saturation_q_high < 1.0):
		raise ValueError("--saturation-q-low and --saturation-q-high must satisfy 0 < low < high < 1")
 	
	# - Init random seed
	print(f"Setting random seed to {args.seed} ...")
	rng = np.random.default_rng(args.seed)
	
	# - Read input data
	print(f"Reading input data {args.input} ...")
	df = pd.read_csv(args.input)

	# - Extract flux & label data columns
	print(f"Extract flux data columns {args.flux_cols} ...")
	flux_cols = ensure_cols_exist(df, args.flux_cols)
	
	label_col = args.label_col or infer_label_col(df)
	print(f"Extracted label data column: {label_col} ...")
	
	# - Parse missing data fractions
	print(f"Parsing fractions of missing values: {args.missing}")
	missing_fracs = parse_kv_fractions(args.missing)
	print(missing_fracs)
	
	print(f"Parsing fractions of outliers: {args.outliers}")
	outlier_fracs = parse_kv_fractions(args.outliers)
	print(outlier_fracs)
	
	print(f"Parsing fraction of saturated values: {args.saturation}")
	sat_fracs = parse_kv_fractions(args.saturation)
	print(sat_fracs)

	# - Add missing values
	#   NB: A fraction of these (inf-share) will be set to +-inf, rest to nan
	print("Adding missing values ...")
	df1, nan_counts, inf_counts = inject_missing_and_inf(df, flux_cols, missing_fracs, rng, args.inf_share)
	print("--> nan_counts")
	print(nan_counts)
	print("--> inf_counts")
	print(inf_counts)
	
	# - Add saturated values
	print("Adding saturated values ...")
	df2, sat_counts = inject_saturation(df1, flux_cols, sat_fracs, rng, mode=args.saturation_mode, q_low=args.saturation_q_low, q_high=args.saturation_q_high) if sat_fracs else (df1, {c:0 for c in flux_cols})
	
	# - Add outliers
	#   NB: Adding this after saturation, otherwise the outliers would be clipped (so no outlier would be generated!)
	print("Adding outliers ...")
	df3, out_counts = inject_outliers(df2, flux_cols, outlier_fracs, rng)
	print("--> outlier_counts")
	print(out_counts)
	
	# - Adding anomalies
	print("Adding anomalies ...")
	df4, n_anom = inject_anomalies_multivariate(df3, flux_cols, args.anomalies, rng)
	
	# - Adding corrupted labels
	print("Adding corrupted labels ...")
	df5, n_mis = corrupt_labels(df4, label_col, args.mislabeled, rng, mode=args.label_corrupt_mode)
	
	# - Add duplicated values
	print("Adding duplicated values ...")
	df_out, n_dup = add_duplicates(df5, args.dup_frac, rng)

	# - Save modified full dataset (with flags)
	print(f"Saving full updated data frame (with flags) to file {args.output} ...")
	df_out.to_csv(args.output, index=False)

	# - Save base dataset (only original columns)
	if args.output_base:
		print(f"Saving base data frame (original columns only) to file {args.output_base} ...")
		df_base = df_out[df.columns]  # keep only original input columns
		df_base.to_csv(args.output_base, index=False)

	# - Print corruption report
	print("Producing a corruption report ...")
	report = CorruptionReport(
		n_rows_in=len(df),
		n_rows_out=len(df_out),
		flux_cols=flux_cols,
		label_col=label_col,
		seed=args.seed,
		duplicated_rows_added=n_dup,
		missing_injected=nan_counts,
		inf_injected=inf_counts,
		outliers_injected=out_counts,
		saturated_injected=sat_counts,
		saturation_mode=args.saturation_mode,
		saturation_q_low=args.saturation_q_low,
		saturation_q_high=args.saturation_q_high,
		anomalies_rows=n_anom,
		mislabeled_rows=n_mis,
	)

	if args.report:
		print(f"Saving corruption report to file {args.report} ...")
		with open(args.report, "w") as f:
			json.dump(asdict(report), f, indent=2)

	print("--> Corruption report")
	print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
	main()
