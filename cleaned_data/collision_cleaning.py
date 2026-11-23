"""Shared loader for the SDOT collision export.

Every script that reads collisions should use `load_collisions()` so the same
records are excluded and the same codes are used everywhere.

Cleaning applied:
1. Drop the 30 rows at the top of the raw export that come from a different
   extract. They are recognizable because ADDRTYPE is spelled out ("Block" /
   "Intersection") while every other row uses "B" / "I", and they have no
   ADDBY and no ADDR_SEGINTKEY. In those rows SEVERITYCODE 3 means fatality
   (4 everywhere else) and 0 means unknown, 9 of 30 are fatalities (0.5%
   elsewhere), and 7 duplicate a collision that is also in the main export
   (same LOCATION and date).
2. Standardize ADDRTYPE: "B" -> "Block", "I" -> "Intersection".
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_COLLISIONS = ROOT / "data/Crashes-Analysis-Export-2023_09_20-2026_09_19.csv"


def load_collisions(path=RAW_COLLISIONS, usecols=None, verbose=True):
    needed = {"ADDRTYPE", "ADDBY", "ADDR_SEGINTKEY"}
    read_cols = None if usecols is None else list(dict.fromkeys(list(usecols) + sorted(needed)))
    df = pd.read_csv(path, usecols=read_cols, low_memory=False)

    other_extract = df["ADDRTYPE"].isin(["Block", "Intersection"])
    # Guard: every spelled-out row must carry the other extract's signature,
    # so a future re-download with different formatting fails loudly.
    signature = df["ADDBY"].isna() & df["ADDR_SEGINTKEY"].isna()
    assert signature[other_extract].all(), "Spelled-out ADDRTYPE rows no longer match the known extra extract"

    df = df.loc[~other_extract].reset_index(drop=True)
    df["ADDRTYPE"] = df["ADDRTYPE"].replace({"B": "Block", "I": "Intersection"})
    assert set(df["ADDRTYPE"].dropna()) <= {"Block", "Intersection"}, set(df["ADDRTYPE"])

    if verbose:
        print(f"collisions: dropped {int(other_extract.sum())} rows from the other extract; {len(df):,} remain")
    if usecols is not None:
        df = df[list(usecols)]
    return df
