.. _usage:

Usage
=====

Overview
--------

`electroMICA` provides two main functions to project electrophysiological features onto cortical and hippocampal surfaces:

- :func:`electroMICA_iEEG` — Project **intracranial EEG** (iEEG / stereo-EEG) features to surfaces.
- :func:`electroMICA_ScalpEEG` — Project **scalp EEG** features to surfaces.

Both functions accept standardized parameters and work with BIDS-formatted datasets.

Basic Workflow
--------------

1. **Prepare your BIDS datasets** with:
   
   - `BIDS-iEEG` or `BIDS-EEG` data (electrode locations, associated anatomical image)
   - `micapipe` derivatives (T1w, surfaces, transforms)
   
2. **Create feature files** (TSV format) containing the electrophysiological metrics you want to project.

3. **Call the appropriate `electroMICA` function** with paths to your data.

4. **Retrieve projected features** from the ``maps/`` folder in the `electroMICA` derivatives.

Intracranial EEG (iEEG)
-----------------------

The :func:`electroMICA_iEEG` function projects depth electrode features onto cortical surfaces.

Basic usage:

.. code-block:: python

   from electroMICA import electroMICA_iEEG
   
   # Paths
   output_folder = "/path/to/derivatives/electroMICA/sub-PX050/ses-01"
   feature_file = "/path/to/features/sub-PX050_ses-01_Interictal.tsv"
   electrodes_folder = "/path/to/BIDS_iEEG/iEEG/"
   micapipe_derivatives = "/path/to/derivatives/micapipe"
   
   # Project iEEG features to surfaces
   electroMICA_iEEG(
       output_folder=output_folder,
       feature=feature_file,
       electrodes=electrodes_folder,
       micapipe_derivatives=micapipe_derivatives
   )

With optional hippocampal surfaces:

.. code-block:: python

   electroMICA_iEEG(
       output_folder=output_folder,
       feature=feature_file,
       electrodes=electrodes_folder,
       micapipe_derivatives=micapipe_derivatives,
       hippunfold_derivatives="/path/to/hippunfold/derivatives"
   )

for new features once the pipeline has been run previously:

.. code-block:: python

   electroMICA_iEEG(
       output_folder=output_folder,
       feature=feature_file
   )


Parameters
~~~~~~~~~~

**output_folder** (str)
   Path to the `electroMICA` derivatives folder where results are stored. This also identifies the subject and session. Ends in ``sub-XX`` or ``sub-XX/ses-YY``.

**feature** (str)
   Path to a feature TSV file, or feature label (accepts wildcard "*"). If the string includes no path,
   the function searches for matching file in the ``feat/`` folder of the `electroMICA` derivatives.

**electrodes** (str, optional)
   Path to the BIDS iEEG folder, or directly to an ``electrodes.tsv`` file. An assocaited image must also be available.
   This input can be ommited if the pipeline ran previously for the same subjet/session, existing sensitivity profiles are 
   then used, speeding up the computations significantly.

**micapipe_derivatives** (str, optional)
   Path to the `micapipe` derivatives. It can be simply the root ``derivatives/micapipe/`` folder, in which case the first session for the subject is used. If a different sessions is desired, it must point to the specific session folder (``derivatives/micapipe/sub-XX/ses-YY/``). 
   It can be ommited if the pipeline ran previously for the same subjet/session, existing sensitivity profiles are 
   then used, speeding up the computations significantly.

**hippunfold_derivatives** (str, optional)
   Path to `hippunfold` derivatives. If provided, hippocampal surfaces are included in the projection.
   It can be simply the root ``derivatives/micapipe/`` folder, in which case the first session for the subject is used. If a different sessions is desired, it must point to the specific session folder (``derivatives/micapipe/sub-XX/ses-YY/``).

Scalp EEG
---------

The :func:`electroMICA_ScalpEEG` function projects scalp electrode features onto cortical surfaces.

Basic usage:

.. code-block:: python

   from electroMICA import electroMICA_ScalpEEG
   
   # Paths
   output_folder = "/path/to/derivatives/electroMICA/sub-PX050/ses-01"
   feature_file = "/path/to/features/sub-PX050_ses-01_AvgSpike.tsv"
   electrodes_folder = "/path/to/BIDS_EEG/EEG/"
   micapipe_derivatives = "/path/to/derivatives/micapipe"
   
   # Project scalp EEG features to surfaces
   electroMICA_ScalpEEG(
       output_folder=output_folder,
       feature=feature_file,
       electrodes=electrodes_folder,
       micapipe_derivatives=micapipe_derivatives
   )

With optional hippocampal surfaces:

.. code-block:: python

   electroMICA_ScalpEEG(
       output_folder=output_folder,
       feature=feature_file,
       electrodes=electrodes_folder,
       micapipe_derivatives=micapipe_derivatives,
       hippunfold_derivatives="/path/to/hippunfold/derivatives"
   )

for new features once the pipeline has been run previously:

.. code-block:: python

   electroMICA_ScalpEEG(
       output_folder=output_folder,
       feature=feature_file
   )


Parameters
~~~~~~~~~~

**output_folder** (str)
   Path to the `electroMICA` derivatives folder where results are stored. This also identifies the subject and session. Ends in ``sub-XX`` or ``sub-XX/ses-YY``.

**feature** (str)
   Path to a feature TSV file, or feature label (accepts wildcard "*"). If the string includes no path,
   the function searches for matching file in the ``feat/`` folder of the `electroMICA` derivatives.

**electrodes** (str, optional)
   Path to the BIDS EEG folder, or directly to an ``electrodes.tsv`` file. 
   It can be ommited if a subset of the 10-10 or 10-20 standard electrode set is used.

**micapipe_derivatives** (str, optional)
   Path to the `micapipe` derivatives. It can be simply the root ``derivatives/micapipe/`` folder, in which case the first session for the subject is used. If a different sessions is desired, it must point to the specific session folder (``derivatives/micapipe/sub-XX/ses-YY/``). 
   It can be ommited if the pipeline ran previously for the same subjet/session, existing sensitivity profiles are 
   then used, speeding up the computations significantly.

**hippunfold_derivatives** (str, optional)
   Path to `hippunfold` derivatives. If provided, hippocampal surfaces are included in the projection.
   It can be simply the root ``derivatives/micapipe/`` folder, in which case the first session for the subject is used. If a different sessions is desired, it must point to the specific session folder (``derivatives/micapipe/sub-XX/ses-YY/``).


Feature File Format
-------------------

Feature files should be in TSV (tab-separated values) format with the following structure:

For iEEG:

.. code-block::

   ChannelName    feature_name_1   feature_name2   ...
   E1-E2    0.45    10.2    ...
   E2-E3    0.52    6.4     ...
   ...

For scalp EEG:

.. code-block::

   ChannelName   feature_name_1  feature_name_2    ...
   Cz    1.23    -3.1    ...
   Pz    1.15    -2.8    ...
   ...

The first column should match electrode names in your BIDS dataset, the reference is the average of all channels. Do not include bad channels in the feature file.



Detailed Inputs
---------------

iEEG:
- from micapipe/anat/: T1w image, brain mask
- from micapipe/surf/: L and R, midthickness, fsnative, fsLR-32k surfaces
- from hippunfold/surf/ (optional): L and R, midthickness, den-0p5mm, den-2mm, den-8k
- from BIDS-iEEG/: electrodes.tsv, associated image (ideally T1w)
- feature file: channel names and feature values, one column per feature (for bad channels, exclude the channel or assign NaN value to feature)

Scalp EEG:
- from micapipe/anat/: T1w image, brain mask
- from micapipe/surf/: L and R, midthickness, fsLR-32k surfaces
- from micapipe/xfm/: transform files from nativepro to MNI space
- from hippunfold/surf/ (optional): L and R, midthickness, den-2mm or den-8k
- from BIDS-EEG/: electrodes.tsv (optional if standard 10-10/10-20 set is used)
- feature file: electrode names and feature values, one column per feature (for bad channels, exclude the channel or assign NaN value to feature)


Output Structure
----------------

After running `electroMICA`, the ``electroMICA`` derivatives folder is organized as:

.. code-block::

   sub-XX/ses-YY/
   ├── anat/                    # Anatomical images (T1w, brain mask)
   ├── feat/                    # Input feature files
   ├── maps/                    # Projected feature maps (main output)
   ├── model/                   # Leadfield / sensitivity matrices (.mat files)
   ├── surf/                    # Surface GIFTI files (cortical, hippocampal)
   └── xfm/                     # Transform files (electrode to MRI alignment)

For each feature and surface, gifti files are created in the with /maps folder:
iEEG:
FEATURE-NAME_SURFACE-NAME.gii : Piecewise constant projection (based on channel with maximum sesnsitivity)
FEATURE-NAME_SURFACE-NAME_smooth.gii : Weighted average projection (based on relative sensitivities)

Scalp EEG:
FEATURE-NAME_SURFACE-NAME_VeryLowSNR.gii
FEATURE-NAME_SURFACE-NAME_LowSNR.gii
FEATURE-NAME_SURFACE-NAME_MediumSNR.gii
FEATURE-NAME_SURFACE-NAME_HighSNR.gii
FEATURE-NAME_SURFACE-NAME_VeryHighSNR.gii

Results for SNR changes in steps of 5, from very low SNR for noisy single events barely distinguisheable from the background to very high SNR of average of high number of clear events.
