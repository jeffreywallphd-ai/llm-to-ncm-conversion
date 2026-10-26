"""Analyze immutable runs for ACL-004; no model implementation here."""
from ncm_lab.reporting.reports import analyze
import json

if __name__ == "__main__":
    print(json.dumps(analyze("ACL-004"), indent=2))
