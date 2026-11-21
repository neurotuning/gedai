GEDAI Tutorials
===============

This gallery is organized as a learning path so users can start quickly and
then move to advanced workflows.

How to use this gallery
-----------------------

1. Start with the ``basics`` section to understand model behavior and core parameters.
2. Continue with ``practical pipelines`` for recommended and reusable denoising pipelines.
3. Use ``advanced`` if you want to dive into online processing and custom reference covariance workflows.

Which GEDAI model should I use?
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Model
     - How It Works
     - Best For
   * - :class:`~gedai.gedai.Gedai`
     - Single full-spectrum GEVD spatial filter
     - Teaching core concepts and fast broadband cleaning
   * - :class:`~gedai.gedai.MultibandGedai`
     - Fixed-window Haar MODWT subband denoising (+ broadband pass)
     - Fixed-epoch offline pipelines and low-latency online streaming
   * - :class:`~gedai.gedai.AdaptiveMultibandGedai`
     - Cycle-adaptive MODWT epochs (longer windows at lower frequencies)
     - Recommended highest-accuracy offline EEG/MEG denoising
