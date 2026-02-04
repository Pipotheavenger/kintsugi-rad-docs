# kintsugi-rad/research/stable

This directory contains experiments that have been shown to be effective and
are probably going to make it into production in some capacity.

# Experiment Index

| Name                                                         | Description                                                                                         |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| [internal_data_whisper_medium](internal_data_whisper_medium) | A clone of the Whisper Large -> Mean Pool -> FC layer model, but using Whisper Medium as a backbone |

# Code Review SOP

Experiments in this directory will go through the same rigorous code review
process as production code. The code is expected to be clean, performant, and
tidy. Each directory should have a README describing the experiment, with
a baseline performance and experiment performance. The README should also have
a link to a W&B run that logs standard metrics and includes model files.
