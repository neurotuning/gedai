.. _install:
.. include:: ../links.inc

Install
=======

``gedai`` requires Python ``3.12`` or higher.

``gedai`` works best with the latest stable release of MNE-Python. To
ensure MNE-Python is up-to-date, see the
`MNE installation instructions <mne install_>`_.

Methods
-------

.. tab-set::

    .. tab-item:: Pypi [Recommended]

        ``gedai`` can be installed from `Pypi <project pypi_>`_:

        .. code-block:: bash

            $ pip install gedai


    .. tab-item:: Snapshot of the current version

        ``gedai`` can be installed from `GitHub <project github_>`_:

        .. code-block:: bash

            $ pip install git+https://github.com/neurotuning/gedai


    .. tab-item:: Development version

        ``gedai`` can be installed by cloning the repository and installing:

        .. code-block:: bash

            $ git clone https://github.com/neurotuning/gedai.git
            $ cd gedai
            $ pip install -e .[all]