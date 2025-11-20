.. include:: links.inc

Install
=======

``pyGEDAI`` requires Python ``3.12`` or higher.

``pyGEDAI`` works best with the latest stable release of MNE-Python. To
ensure MNE-Python is up-to-date, see the
`MNE installation instructions <mne install_>`_.

Methods
-------

.. tab-set::

    .. tab-item:: Snapshot of the current version

        ``pyGEDAI`` can be installed from `GitHub <project github_>`_:

        .. code-block:: bash

            $ pip install git+https://github.com/neurotuning/pyGEDAI


    .. tab-item:: Development version

        ``pyGEDAI`` can be installed by cloning the repository and installing:

        .. code-block:: bash

            $ git clone https://github.com/neurotuning/pyGEDAI.git
            $ cd pyGEDAI
            $ pip install -e .[all]