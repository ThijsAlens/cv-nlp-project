#!/usr/bin/env python3
"""Analyze label distribution in the TACO dataset."""

import json
from collections import Counter
from pathlib import Path

ann_file = Path("data/raw/taco/annotations.json")

data = json.loads(ann_file.read_text())

# Map category id -> name
categories = {c["id"]: c["name"] for c in data["categories"]}

# Count object instances
counts = Counter(ann["category_id"] for ann in data["annotations"])

print("\nTACO label distribution (instances):\n")

for cat_id, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{categories[cat_id]:30} {count}")
