Advanced Workflows
==================

This section covers specialized workflows for users who already know the basic
GEDAI pipeline.

- ``00_gedai_online.py``: real-time chunk denoising with live W-OLA blending,
  right-edge causal trimming, and decoupled low-latency feature scoring.
- ``10_gedai_forward.py``: non-10-20 subject-specific MRI BEM, 4-shell sphere,
  automatic 3D dipole projection, and SENSAI goodness-of-fit comparison.
- ``20_gedai_opm.py``: denoising Optically Pumped Magnetometer (OPM) MEG
  recordings using a subject-specific forward model.
