#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Force sentence finalization via API.

No parameters. Optionally respects ML_BASE_URL env var (defaults to http://localhost:8000).
"""

import json
import os
import sys
import urllib.request
import urllib.error

BASE_URL = os.environ.get("ML_BASE_URL", "http://localhost:8000")
URL = f"{BASE_URL.rstrip('/')}/api/force_end_sentence"

def main() -> int:
    try:
        req = urllib.request.Request(URL, data=b"{}", method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # nosec B310
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            print(json.dumps(data, indent=2))
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}")
        try:
            detail = e.read().decode("utf-8")
            print(detail)
        except Exception:
            pass
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        print(f"Make sure the API is running at {BASE_URL}")
        return 2
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())

