import os
import re 
from typing import List, Tuple, Dict
from mne.io import read_raw, Raw, read_raw_fif 

import os
import re
from typing import List, Dict, Union
from mne.io import Raw, read_raw_fif

def extract_run_number(filename: str) -> int:
    """
    Extracts the run number from a filename.
    
    Args:
        filename (str): The filename to extract the run number from.
    
    Returns:
        int: The run number, or a large number if the run number cannot be determined.
    """
    match = re.search(r'Run (\d+)', filename)
    return int(match.group(1)) if match else float('inf')
  

def load_eeg_data(baseOutputPath: str, 
                  participants: Union[str, List[str]] = 'all') -> Dict[str, List[Raw]]:
    """
    Loads all processed EEG runs from the base output path into a dictionary.
    
    Args:
        baseOutputPath (str): The base path where the processed EEG data is stored.
        participants (Union[str, List[str]]): 'all' to load data for all participants, 
                                              a list of participant IDs to load specific participants, 
                                              or a single participant ID to load one participant.
    
    Returns:
        Dict[str, List[Raw]]: A dictionary where keys are participant IDs and values are lists of MNE Raw objects.
    """
    eeg_data = {}

    if participants == 'all':
        participant_dirs = sorted(os.listdir(baseOutputPath))
    elif isinstance(participants, str):
        participant_dirs = [participants]
    else:
        participant_dirs = participants

    for participant_dir in participant_dirs:
        participant_path = os.path.join(baseOutputPath, participant_dir)
        if not os.path.isdir(participant_path):
            continue

        eeg_data[participant_dir] = []

        # Sort EEG files by run number
        eeg_files = sorted([f for f in os.listdir(participant_path) if f.endswith('.fif')], key=extract_run_number)
        for eeg_file in eeg_files:
            eeg_file_path = os.path.join(participant_path, eeg_file)
            try:
                raw = read_raw_fif(eeg_file_path, preload=True, verbose=False)
                eeg_data[participant_dir].append(raw)
            except Exception as e:
                print(f"Error loading file {eeg_file_path}: {e}")

    return eeg_data
  
  
  
### =================================================================

def save_processed_files(baseOutputPath: str, sub_fol_name: str, name_of_file: str, eegData: Dict[str, List[Raw]]) -> None:
    os.makedirs(baseOutputPath, exist_ok=True)
    
    for subject_id, raws in eegData.items():
        subject_output_path = os.path.join(baseOutputPath, f'{sub_fol_name}/{subject_id}')
        os.makedirs(subject_output_path, exist_ok=True)
        
        for run_index, raw in enumerate(raws):
            output_file_path = os.path.join(subject_output_path, f'{name_of_file} {run_index + 1} raw.fif')
            raw.save(output_file_path, overwrite=True, verbose=False)
        
        print(f'Saved processed runs for {subject_id} to {subject_output_path}')
        
        
# ============================================================================= #
