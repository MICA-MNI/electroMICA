from electroMICA import electroMICA_iEEG
from electroMICA import electroMICA_ScalpEEG

# Only line to modify: point it to the example folder
root_folder="D:/test/"

# iEEG pipeline example
output_folder = root_folder + "BIDS_iEEG/derivatives/electroMICA/sub-HC010/ses-01"
feature_info = root_folder + "BIDS_iEEG/derivatives/some_software/sub-HC010/ses-01/feat/*SpikeRate*.tsv"
electrode_info = root_folder + "BIDS_iEEG/ieeg"
micapipe_folder = root_folder + "BIDS/derivatives/micapipe" 
hippunfold_folder = root_folder + "BIDS/derivatives/hippunfold"
electroMICA_iEEG(output_folder,feature_info,electrode_info,micapipe_folder,hippunfold_folder)

# ScalpEEG pipeline example
output_folder = root_folder + "BIDS_EEG/derivatives/electroMICA/sub-HC010/ses-01"
feature_info = root_folder + "BIDS_EEG/derivatives/some_software/sub-HC010/ses-01/feat/*AvgSpike*.tsv"
electrode_info = None
micapipe_folder = root_folder + "BIDS/derivatives/micapipe" 
hippunfold_folder = root_folder + "BIDS/derivatives/hippunfold"
electroMICA_ScalpEEG(output_folder,feature_info,electrode_info,micapipe_folder,hippunfold_folder)
