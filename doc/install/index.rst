.. _install:
.. include:: ../links.inc

Install
=======

``gedai`` requires Python ``3.10`` or higher.

``gedai`` works best with the latest stable release of MNE-Python. To
ensure MNE-Python is up-to-date, see the
`MNE installation instructions <mne install_>`_.

Methods
-------

.. tab-set::

    .. tab-item:: PyPI [Standard]

        Standard lightweight CPU installation using NumPy (~50 MB download):

        .. code-block:: bash

            $ pip install gedai

    .. tab-item:: PyPI [Accelerated with PyTorch]

        Accelerated performance (up to 2.2x faster) using vectorized PyTorch linear algebra:

        .. code-block:: bash

            $ pip install "gedai[torch]"

        To install lightweight CPU-only PyTorch wheels on Windows/Linux without large GPU/CUDA binaries:

        .. code-block:: bash

            $ pip install torch --index-url https://download.pytorch.org/whl/cpu
            $ pip install gedai

        ``gedai`` defaults to ``engine="auto"``, which automatically activates PyTorch acceleration when available and falls back cleanly to NumPy otherwise.

    .. tab-item:: Snapshot of the current version

        ``gedai`` can be installed from `GitHub <project github_>`_:

        .. code-block:: bash

            $ pip install git+https://github.com/neurotuning/gedai
            # or with PyTorch acceleration:
            $ pip install "gedai[torch] @ git+https://github.com/neurotuning/gedai"


    .. tab-item:: Development version

        ``gedai`` can be installed by cloning the repository and installing:

        .. code-block:: bash

            $ git clone https://github.com/neurotuning/gedai.git
            $ cd gedai
            $ pip install -e .[all]