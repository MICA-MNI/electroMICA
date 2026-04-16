.. electroMICA documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

electroMICA
===========

**Integrate electrophysiological and brain imaging data.**

`electroMICA` is developed by `MICA Lab <https://mica-mni.github.io>`_ at McGill University for use at 
`the Neuro <https://www.mcgill.ca/neuro/>`_, McConnell Brain Imaging Center (`BIC <https://www.mcgill.ca/bic/>`_).

The main goal of `electroMICA` is to provide a robust framework to integrate electrophysiological data 
(scalp EEG, intracranial EEG) with information derived from multimodal MR images processed 
by `micapipe <https://micapipe.readthedocs.io>`_and `hippunfold <https://hippunfold.readthedocs.io>`. It contains two pipelines:

**electroMICA_iEEG:**
    Projects intracranial EEG features (e.g. event rates) onto cortical and hipocampal surfaces.

.. image:: ../img/workflow-iEEG.png
   :alt: electroMICA iEEG Workflow
   :width: 90%

**electroMICA_scalp**

    Computes an Electric Source Imaging solution for scalp EEG data on the cortical and hippocampal surfaces.

.. image:: ../img/workflow-scalp.png
   :alt: electroMICA Scalp EEG Workflow
   :width: 90%

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   algorithm
   faq
   contributing

Highlights
----------

- **Multimodal integration**: Seamlessly integrates electrophysiology with `micapipe` structural and functional outputs.
- **Detailed individualized surfaces**: Projects features onto cortical and (optionally) hippocampal surfaces.
- **BIDS-compliant**: Follows Brain Imaging Data Structure (BIDS) conventions.

Quick Start
-----------

Install from ``requirements.txt``:

.. code-block:: bash

   pip install -r requirements.txt

Then edit an example script with your BIDS paths and run:

.. code-block:: bash

   python example_electroMICA_iEEG.py
   # or
   python example_electroMICA_scalp.py

See :ref:`installation` and :ref:`usage` for detailed instructions.

Key Features
------------

**intracerebral or stereo EEG**
   Depth electrodes are modeled as line segments in a single-layer homogeneous conductor.
   Unique distributed dipolar sheets generator model with no mathematical singularity.
   Contact sensitivity profiles are computed via BEM with analytic element integration.
   Features are mapped to cortical/hippocampal vertices using contact sensitivity weighting.

**Scalp EEG**
   A three-layer head model (scalp, skull, brain) is constructed from T1w MRI.
   Forward problem solved using BEM to compute leadfields from sources to electrodes.
   Inverse problem solved with modified eLORETA with spatial correlation weighting.
   Multiple SNR-dependent feature maps are generated.

**Hippocampal Surfaces**
   When `hippunfold <https://hippunfold.readthedocs.io>`_ outputs are available,
   hippocampal surfaces are included in the source space.

Reference
---------

If you use `electroMICA` in your research, please cite:

   (Citation details to be added)

Additionally, when using `electroMICA`, you should cite:

- `micapipe` — Cruces, R. R., et al. (2022). Micapipe: a pipeline for multimodal neuroimaging and connectome analysis. NeuroImage, 119612.

If you use HippUnfold, please cite

- `hippunfold` — DeKraker, J., et al. (2022), Automated hippocampal unfolding for morphometry and subfield segmentation with HippUnfold. elife 11: e77945.


See :ref:`algorithm` for detailed algorithmic references.

License
-------

This project is licensed under the GPL v3 license. See ``LICENSE`` for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
