DATA_CREW_PROMPT = f"""
Given the combined text for multiple versions...
Your tasks:
- For each table, output a JSON object with metric names as keys, each containing a dict mapping version to values for ATLs Fixed and BTLs Fixed.
- For example:
"release_scope": {{
  "Release Epics": {{
    "45.1.15.0": {{"ATLs Fixed": X, "BTLs Fixed": Y}},
    "45.1.16.0": {{"ATLs Fixed": ..., "BTLs Fixed": ...}},
    ...
  }},
  ...
}}
...
Return only JSON, no commentary.
"""
