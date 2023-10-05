

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

    conda create -n SEDR_Env python=3.11

    conda activate SEDR

    git clone https://github.com/JinmiaoChenLab/SEDR.git

    cd SEDR

    python setup.py build

    python setup.py.install


To use SEDR in notebook,

.. code-block:: bash

    pip install ipykernel

    python -m ipykernel install --user --name=SEDR_Env

