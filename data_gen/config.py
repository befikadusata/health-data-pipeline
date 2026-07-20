"""Central constants for synthetic data generation. Keep every knob here so the
generator, ground-truth ledger, and downstream docs stay consistent."""

from datetime import date

SEED = 42

N_FACILITIES = 50
N_MONTHS = 24
START_MONTH = date(2024, 1, 1)  # first-of-month convention throughout

REGIONS = ["North", "Central", "South", "East", "West"]

# Facility-months eligible for injection: every (facility, month) pair.
# Rates are independent per issue type and apply to the full facility-month grid.
MISSING_MONTH_RATE = 0.03
DUPLICATE_RATE = 0.02
OUTLIER_RATE = 0.02
SEVERE_DELAY_RATE = 0.05

# "Normal" reporting delay (days after month-end the report lands), vs. a severely
# delayed report used for the injected issue.
NORMAL_DELAY_DAYS = (1, 10)
SEVERE_DELAY_DAYS = (45, 90)

OUTPUT_DIR = "data_gen/output"
