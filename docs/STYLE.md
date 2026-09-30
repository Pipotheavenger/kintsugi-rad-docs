# Documentation style guide

Goal: someone who knows ML can **locate things fast**. Short, specific, never
over-explained. If a sentence would be true of any PyTorch code, delete it.

## Hard rules

1. **Never change code.** Only add or edit docstrings and `#` comments.
   `python tools/check_only_docs.py` must print `OK`.
2. **English.** Plain words, no marketing, no filler ("This function is used to…").
3. **The first docstring line stands alone.** The code map on the website shows
   only that line next to the function name. ≤ 90 characters, ends with a period.
4. **Keep the authors' existing docstrings.** Do not delete or rewrite them. If
   one exists and is accurate, leave it. If its first line is missing or vague,
   insert a one-line summary above it. You may append `Shapes:` / `Hardcoded:` lines.
5. **Be exact.** Every claim must be true of *this* code. Check shapes, defaults
   and call sites by reading the code, not by assuming.

## Docstring template (functions and methods)

```python
def get_window_starts(method, audio_len, inference_window_samples, max_overlap_frac):
    """Start sample of each 30 s window for one recording.

    "all": evenly spaced with np.linspace, overlap <= max_overlap_frac.
    Shapes: returns int or list[int] of sample offsets.
    Hardcoded: none (window length comes from window_duration).
    """
```

- Line 1: what it returns or does, in the pipeline's terms.
- Then at most ~4 lines, only if useful, from this menu:
  - `Shapes:` input/output tensor shapes (use B = recordings, ΣW = windows, D = dim).
  - `Hardcoded:` fixed values that a replicator may need to change.
  - `Called by:` only when the caller is not obvious.
  - `Note:` one gotcha (e.g. "does not resample; only warns").
- Do not document arguments whose meaning is obvious from name and type hints.
- Private helpers and one-liners: one line is enough.

## Module docstrings

2–4 lines at the top of each file: its role in the pipeline and which stage it
belongs to (launch, data, model, loss, thresholds/metrics, outputs, utils).

## Inline comments

Only for magic numbers and non-obvious steps. Prefix fixed values with
`# HARDCODED:` so they are greppable, e.g. `# HARDCODED: 16 kHz expected, not resampled`.
Do not comment obvious lines.
