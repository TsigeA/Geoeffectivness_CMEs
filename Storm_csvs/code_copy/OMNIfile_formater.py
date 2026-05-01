"""
Convert OMNI text files from:
    year sec_of_year data...
to:
    year doy hour minute data...

The script preserves comment/header lines and updates the two plain-text
header rows so the output matches the older OMNI layout expected by the
downstream workflow.
@author: TsigeA
@date: Apr 24, 2026
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


FNAME = Path("omni_5min_20160507-20160509_BxyzPdynSYMH.txt")


def sec_of_year_to_doy_hm(year: int, sec_of_year: float) -> tuple[int, int, int]:
    dt = datetime(year, 1, 1) + timedelta(seconds=sec_of_year)
    return dt.timetuple().tm_yday, dt.hour, dt.minute



lines = FNAME.read_text().splitlines()
output_lines: list[str] = []

for line in lines:
    stripped = line.strip()

    if not stripped:
        output_lines.append(line)
        continue

    if stripped.startswith("EPOCH_TIME"):
        output_lines.append("#YYYY DOY HR MN      1       2       3       4       5       6       7     8     9 ")
        continue

    if stripped.startswith("Year____Secs-of-year"):
        continue

    if stripped.startswith("#"):
        output_lines.append(line)
        continue

    parts = stripped.split()
    if len(parts) < 11:
        output_lines.append(line)
        continue

    year = int(parts[0])
    sec_of_year = float(parts[1])
    doy, hour, minute = sec_of_year_to_doy_hm(year, sec_of_year)
    values = parts[2:]

    output_lines.append(
        f"{year:4d} {doy:3d} {hour:2d} {minute:2d} "
        + " ".join(f"{value:>7}" for value in values)
    )
# write the output to a new file
new_fname = FNAME.with_name(FNAME.stem + "_formatted" + FNAME.suffix)
new_fname.write_text("\n".join(output_lines) + "\n")
# FNAME.write_text("\n".join(output_lines) + "\n")


# if __name__ == "__main__":
#     convert_omni_file(FNAME)
