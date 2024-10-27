

Installation
============

SEDR package is mainly based on python.

1. Python
---------------------

.. code-block:: python

    git clone https://github.com/JinmiaoChenLab/SEDR.git

    cd SEDR

    python setup.py build

    python setup.py.install


2. Anaconda
------------
We highly recommend users to create a separate environment for SEDR.

.. code-block:: python

    conda create -n SEDR python=3.11

    conda activate SEDR

    git clone https://github.com/JinmiaoChenLab/SEDR.git

    cd SEDR

    python setup.py build

    python setup.py install


To use SEDR in notebook,

.. code-block:: bash

    pip install ipykernel

    python -m ipykernel install --user --name=SEDR_Env


The mclust R package is needed for running the notebooks. To install it, run the following lines in a python cell:

.. code-block:: bash

    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector

    # set R package names
    packnames = ('mclust',)

    # import R's utility package
    utils = rpackages.importr('utils')

    utils.chooseCRANmirror(ind=1) # select the first mirror in the list

    # list and install missing packages
    packnames_to_install = [x for x in packnames if not rpackages.isinstalled(x)]

    if len(packnames_to_install) > 0:
        utils.install_packages(StrVector(packnames_to_install))
