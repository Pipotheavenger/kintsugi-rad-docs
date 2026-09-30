"""MkDocs hook: do not fail --strict on tensor-shape notation inside docstrings.

Docstrings write shapes like `x["audio"] [ΣW, 80, 3000]`. Markdown reads `[a] [b]` as a
reference link, so mkdocs-autorefs reports "Could not find cross-reference target".
These are not real links. For API reference pages only, such warnings are downgraded
to INFO (still printed); every other warning keeps failing the strict build.
"""

import logging
import re

AUTOREFS_LOGGER = "mkdocs.plugins.mkdocs_autorefs._internal.plugin"
REF_PAGE = re.compile(r"^(mkdocs_autorefs: )?reference/")


class _ShapeRefFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Could not find cross-reference target" in msg and REF_PAGE.match(msg):
            record.levelno, record.levelname = logging.INFO, "INFO"
            record.msg, record.args = f"(shape notation, not a link) {msg}", ()
        return True


def on_startup(command, dirty):
    log = logging.getLogger(AUTOREFS_LOGGER)
    if not any(isinstance(f, _ShapeRefFilter) for f in log.filters):
        log.addFilter(_ShapeRefFilter())
