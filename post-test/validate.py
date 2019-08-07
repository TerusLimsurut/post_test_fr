#!/usr/bin/env python

"""
    post-eval/validate.py
"""

import os
import sys
import json

sys.path.append('dist')
from dist.lib import validate

import main

if __name__ == "__main__":
    results = validate(main)
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        print(json.dumps(results))
        print(json.dumps(results), file=f)