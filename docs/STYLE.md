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

## Block comments inside functions

Docstrings say what a function is; block comments say what each **block** of its body
does, so a reader can skim a function without reading every line.

- **One comment per logical block** (a run of lines that does one thing, usually set off
  by a blank line, a loop, a branch or a `with`). Place it on the line above the block.
- **1 line, 2 at most**, ≤ 88 characters per line (the repo's black / flake8 limit).
- Say **what the block achieves**, in the pipeline's terms ("cut audio into 30 s windows"),
  not how Python works ("loop over files").
- Skip blocks that are obvious from one glance (1–3 trivial lines), plain imports,
  getters and one-line returns. A function whose docstring already covers a 3-line body
  needs no block comment.
- Do not repeat the docstring or the code. No "# set x to 1".
- Keep every existing author comment and every `# HARDCODED:` marker; do not duplicate them.
- Never move, reformat or edit code lines to fit a comment. Comments only.
- Comments go above the code, never trailing at the end of a code line, except `# HARDCODED:`.

Example:

```python
def get_window_starts(...):
    overflow_len = audio_len - inference_window_samples

    # Fixed-position methods: nothing to compute.
    if method == "single_start":
        return 0

    # "all": evenly spaced starts, hop sized so overlap <= max_overlap_fraction.
    if method == "all":
        ...
```
