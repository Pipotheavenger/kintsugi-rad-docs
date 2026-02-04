# kintsugi-rad

|      State | In use                                |
| ---------: | :------------------------------------ |
| Version 💻 | 0.97.0                                |
|   Owner 1️⃣ | [@colinvaz](colin@kintsugihealth.com) |

This repo contains all of the experiment runs for the R&D team at Kintsugi.

# Environment Setup

All of the experiments use the [`kipy-gpu-full`](envs/kipy-gpu-full.yaml)
environment unless otherwise specified.

# Experiment Structure

All of our experiments live in the `research` directory. We have two
sub-directories that house different kinds of experiments and have different
SOPs for code reviews and code quality. The `stable` directory contains
experiments that have been shown to be effective and are probably going to
make it into production in some capacity. Experiments in the `stable` directory
go through the same rigorous code review process as production code. They are
expected to be clean and tidy. All other experiments go into the `experimental`
directory. The experiments do not have to be effective or clean. They exist
just to have a record of what we have tried in the past. The only requirements
for experiments in the `experimental` directory are having a clear README
and a linked W&B report.

For more details, see
[research/stable/README.md](research/stable/README.md)
and
[research/experimental/README.md](research/experimental/README.md)

# kirad

`kirad` contains all the utility functions that we use at Kintsugi.
`kirad` can be installed locally as a Python package by running
`pip install -e .`
