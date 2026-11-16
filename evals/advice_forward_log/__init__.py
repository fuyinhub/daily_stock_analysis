# -*- coding: utf-8 -*-
"""Forward paper log of the system's published per-ticker advice.

``forward_log`` records what each daily run actually advised (append-only
JSONL), ``evaluate`` scores the accumulated log against the prices that
followed, and ``run.py`` is the command-line entry point. Evaluation only:
nothing here calls an LLM, places an order or talks to a broker.
"""
