from electroMICA import electroMICA_ScalpEEG

# Input strings for the electroMICA_ScalpEEG pipeline:

# The output folder indicate not only the location, but also the subject (and session if applicable) 
# for which the pipeline will be run. This subject/session should match the BIDS_EEG folder.
output_folder = "your_BIDS_EEG_folder/derivatives/electroMICA/sub-yoursubject"
# or
output_folder = "your_BIDS_EEG_folder/derivatives/electroMICA/sub-yoursubject/ses-yoursession"

# The feature string should be the path to the feature file (or files) that contain the electric potential
# values for which the source imaging will be produced. It could point to a folder, a file, 
# or a set of files using wildcards.
feature = "any_path" # all tsv files in the folder will be used.
# or
feature = "any_path/your_feature.tsv" # only this file will be used.
# or
feature = "any_path/*name*.tsv" # all tsv files in the folder with "name" in their name will be used.

# The electrode information. The electrode file should contain the 3D coordinates of the nasion,
# inion, and pre-auricular points, as well as the 3D coordinates of the electrodes. Or a BIDs complinat
# coordsystem.json file with this information should exist in the same folder.
electrodes = "your_BIDS_EEG_folder/EEG/" # electrode file from the same subject/session will be used.
# or
electrodes = "any_path/any_name_electrode.tsv"                                             
# or
electrodes = None # If the electrode placement folows the 10-10 nameing and placement system, 
                  # no electrode file is needed.

# The micapipe output folder, wherever it is located.
micapipe_derivatives = "your_BIDS_folder/derivatives/micapipe" # Will run for the subject/session
                                                               # indicated in output_folder
# or    
micapipe_derivatives = "your_BIDS_folder/derivatives/micapipe/sub_yoursubject/ses-yourmicapipesession" 
# Will run for the subject/session indicated in the path, this is useful when the EEG session is
# different from the micapipe session.

# Optionally, the hippunfold output folder, wherever it is located.
hippunfold_derivatives = "your_BIDS_folder/derivatives/hippunfold" # Will run for the subject/session
                                                                   # indicated in output_folder
# or    
hippunfold_derivatives = "your_BIDS_folder/derivatives/hippunfold/sub_yoursubject/ses-yourhippunfoldsession" 
# Will run for the subject/session indicated in the path, this is useful when the EEG session is
# different from the hippunfold session.


# Running the pipeline:

# To run the EEG pipeline including hippocampal surfaces, run the following line:
electroMICA_ScalpEEG(output_folder,feature,electrodes,micapipe_derivatives,hippunfold_derivatives)

# If the hippocampal surfaces are not available, or of no interest, run the following line:
electroMICA_ScalpEEG(output_folder,feature,electrodes,micapipe_derivatives)

# If the pipeline has been run already (with or without hippocamapl surfaces), and the results 
# for a different feature are needed, it is faster to just run the following line:
electroMICA_ScalpEEG(output_folder,feature)
