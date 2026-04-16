from electroMICA import electroMICA_iEEG

# Input strings for the electroMICA_iEEG pipeline:

# The output folder indicate not only the location, but also the subject (and session if applicable) 
# for which the pipeline will be run. This subject/session should match the BIDS_iEEG folder.
output_folder = "your_BIDS_iEEG_folder/derivatives/electroMICA/sub-yoursubject"
# or
output_folder = "your_BIDS_iEEG_folder/derivatives/electroMICA/sub-yoursubject/ses-yoursession"

# The feature string should be the path to the feature file (or files) that will be projected onto the
# cortical surfaces. It could point to a folder, a file, or a set of files using wildcards.
feature = "any_path" # all tsv files in the folder will be used.
# or
feature = "any_path/your_feature.tsv" # only this file will be used.
# or
feature = "any_path/*name*.tsv" # all tsv files in the folder with "name" in their name will be used.

# The electrode information could come directly from the BIDS_iEEG folder, or from a file..
electrodes = "your_BIDS_iEEG_folder/iEEG/" # electrode file from the same subject/session will be used.
# or
electrodes = "any_path/any_name_electrode.tsv" # A BIDs compliant any_name_coordsystem.json file is
                                               # expected in the same folder, and an image file.

# The micapipe output folder, wherever it is located.
micapipe_derivatives = "your_BIDS_folder/derivatives/micapipe" # Will run for the subject/session
                                                               # indicated in output_folder
# or    
micapipe_derivatives = "your_BIDS_folder/derivatives/micapipe/sub_yoursubject/ses-yourmicapipesession" 
# Will run for the subject/session indicated in the path, this is useful when the iEEG session is
# different from the micapipe session.

# Optionally, the hippunfold output folder, wherever it is located.
hippunfold_derivatives = "your_BIDS_folder/derivatives/hippunfold" # Will run for the subject/session
                                                                   # indicated in output_folder
# or    
hippunfold_derivatives = "your_BIDS_folder/derivatives/hippunfold/sub_yoursubject/ses-yourhippunfoldsession" 
# Will run for the subject/session indicated in the path, this is useful when the iEEG session is
# different from the hippunfold session.


# Running the pipeline:

# To run the iEEG pipeline including hippocampal surfaces, run the following line:
electroMICA_iEEG(output_folder,feature,electrodes,micapipe_derivatives,hippunfold_derivatives)

# If the hippocampal surfaces are not available, or of no interest, run the following line:
electroMICA_iEEG(output_folder,feature,electrodes,micapipe_derivatives)

# If the pipeline has been run already (with or without hippocamapl surfaces), and the results 
# for a different feature are needed, it is faster to just run the following line:
electroMICA_iEEG(output_folder,feature)
