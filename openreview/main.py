"""Compatibility entrypoint for tests and legacy deployments.

The service implementation lives in `review.py`; expose the same module under
the historical `main` name so globals patched in tests affect runtime handlers.
"""
from __future__ import annotations

import sys

import review as _review

sys.modules[__name__] = _review
