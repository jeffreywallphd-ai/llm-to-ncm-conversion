"""Analyze immutable runs for IS-003; no model implementation here."""
from ncm_lab.reporting.reports import analyze
import json

if __name__ == "__main__":
    print(json.dumps(analyze("IS-003"), indent=2))
