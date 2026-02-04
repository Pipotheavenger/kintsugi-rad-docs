# kipy/envs

This directory contains `yaml` files defining `conda` environments used both by
MLEs and in the production environment. See [the package README](../README.md)
for steps on how to set up and use these environments.

## Maintaining

|               Name | Human editable | Description                                                                                                      |
| -----------------: | -------------- | :--------------------------------------------------------------------------------------------------------------- |
|           cpu.yaml | Yes            | Dependencies needed on CPU and not GPU                                                                           |
|           gpu.yaml | Yes            | Dependencies needed on GPU and not CPU                                                                           |
|        common.yaml | Yes            | Common dependencies between kipy and kirad                                                                       |
|         kirad.yaml | Yes            | Dependencies used by kirad but not kipy (kept in kipy package so kipy will run tests against these dependencies) |
| kipy-cpu-full.yaml | No             | Full cpu + common env. Regenerate with rebuild.sh when above dependencies are edited.                            |
| kipy-gpu-full.yaml | No             | Full gpu + common + kirad env. Regenerate with rebuild.sh when above dependencies are edited.                    |

## Environments

|          Name | Description                                        |
| ------------: | :------------------------------------------------- |
| kipy-cpu-full | An environment that runs `kipy` on CPU             |
| kipy-gpu-full | An environment that runs `kipy` and `kirad` on GPU |
