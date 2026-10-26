"""Analyze immutable runs for SHR-003; no model implementation here."""
from ncm_lab.reporting.reports import analyze
import json

if __name__ == "__main__":
    print(json.dumps(analyze("SHR-003"), indent=2))
