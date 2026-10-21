# results/ holds records of past runs (reports, logs, the analysis scripts behind a write-up), not tests:
# a script there named like a test would be collected and import numpy in CI's pytest-only job.
collect_ignore = ["results"]
