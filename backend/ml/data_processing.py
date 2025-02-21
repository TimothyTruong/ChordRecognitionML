import numpy as np
import csv
import pandas as pd
import torch 
from torch.utils.data import Dataset
import os

class ChordDataset(Dataset):
    def __init__(self, chroma_data, chord_labels, window_size=5, sequential=True):
        self.window_size = window_size
        self.inputs = []
        self.labels = []

        for chroma, chord in zip(chroma_data, chord_labels):
            if(window_size > 0):
                for i in range(len(chroma) - window_size):
                    self.inputs.append(chroma[i:i+window_size])
                    self.labels.append(chord[i+window_size  - 1])
            else:
                self.inputs.append(chroma)
                self.labels.append(chord)
        
        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.inputs)
    
    def __str__(self):
        return f"ChordDataset: {len(self.inputs)} samples"

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class ChordMapper:
    _chord_map = {
        "X": 0, "N" : 0, "nan" : 0,  # Unknown or no chord
        "C:maj": 1, "C:min": 2,  
        "C#:maj": 3, "C#:min": 4, "Db:maj": 3, "Db:min": 4,  
        "D:maj": 5, "D:min": 6,  
        "D#:maj": 7, "D#:min": 8, "Eb:maj": 7, "Eb:min": 8,  
        "E:maj": 9, "E:min": 10, "Fb:maj": 9, "Fb:min": 10,
        "F:maj": 11, "F:min": 12,  
        "F#:maj": 13, "F#:min": 14, "Gb:maj": 13, "Gb:min": 14,  
        "G:maj": 15, "G:min": 16,  
        "G#:maj": 17, "G#:min": 18, "Ab:maj": 17, "Ab:min": 18,  
        "A:maj": 19, "A:min": 20,  
        "A#:maj": 21, "A#:min": 22, "Bb:maj": 21, "Bb:min": 22,  
        "B:maj": 23, "B:min": 24, "Cb:maj": 23, "Cb:min": 24
    }

    @classmethod
    def get_index(cls, chord):
        if(pd.isna(chord)):
            print(chord)
            return cls._chord_map["X"]
        return cls._chord_map[chord]
    
    def get_chord(cls, index):
        for chord, idx in cls._chord_map.items():
            if idx == index:
                return chord
        return "X"

def process_lab_file(file_path): 

    file = open(file_path, "r")

    start_times = []
    end_times = []
    chords = []

    df = pd.DataFrame(columns=["Start Time", "End Time", "Chord"])

    reader = csv.reader(file, delimiter="\t")
    for l in reader:
        if len(l) > 0:
            start_times.append(float(l[0]))
            end_times.append(float(l[1]))
            chords.append(l[2])

    df["Start Time"] = start_times
    df["End Time"] = end_times
    df["Chord"] = chords

    return df

def process_chroma_features(file_path):
    chroma_features = pd.read_csv(file_path, header=None)
    chroma_features = chroma_features[chroma_features.columns[1:14]]
    chroma_features.columns = ["Timestamp", "Chroma_1", "Chroma_2", "Chroma_3", "Chroma_4", "Chroma_5", "Chroma_6", "Chroma_7", "Chroma_8", "Chroma_9", "Chroma_10", "Chroma_11", "Chroma_12"]

    return chroma_features

def filter_label(chord_df):
    rows_to_drop = []

    for idx, row in chord_df.iterrows():
        if(row['Start Time'] == row['End Time']):
            rows_to_drop.append(idx)
        elif(row['Start Time'] > row['End Time']):
            rows_to_drop.append(idx)
        elif(row['Start Time'] == 0 and row['End Time'] == 0):
            rows_to_drop.append(idx)
    
    chord_df.drop(rows_to_drop, inplace=True)

def align_chroma_to_chords(chroma_df, chord_df):
    filter_label(chord_df)

    chord_labels = chord_df["Chord"]
    bin_edges = np.concatenate(([0], chord_df['End Time'].values)) #Creates a list of the end times of the chords, that will act as the intervals
  
    chroma_df['Chord'] = pd.cut(chroma_df['Timestamp'], bins=bin_edges, labels=chord_labels, right=False, ordered=False)

    return chroma_df

def format_tensor(aligned_df):
    chord_map = ChordMapper()
    chroma_tensors = []
    label_tensors = []
 
    for _, row in aligned_df.iterrows():
        chroma = row.iloc[1:13].values.astype(np.float32)  # Ensure numeric dtype
        label = chord_map.get_index(row.iloc[13])  # Use iloc for safe indexing
        
        chroma_tensors.append(chroma)
        label_tensors.append(label)
    
    chroma_tensors = np.array(chroma_tensors, dtype=np.float32)
    label_tensors = np.array(label_tensors, dtype=np.int64)

    return torch.tensor(chroma_tensors), torch.tensor(label_tensors)

def generate_dataset(chroma_path, lab_path, window_size=5):
    chroma_data_per_song = []
    chord_labels_per_song = []
    
    chroma_dir = sorted(os.listdir(chroma_path))
    lab_dir = sorted(os.listdir(lab_path))

    if(".DS_Store" in chroma_dir):
        chroma_dir.remove(".DS_Store")
    if(".DS_Store" in lab_dir):
        lab_dir.remove(".DS_Store")

    if(len(chroma_dir)) != len(lab_dir):
       raise ValueError("The number of chroma features and lab files do not match")

    for i in range(len(chroma_dir)):
        if chroma_dir[i] != lab_dir[i]:
            raise ValueError("The files do not match")
        
        chroma_file = os.path.join(chroma_path, chroma_dir[i], "bothchroma.csv")
        lab_file = os.path.join(lab_path, lab_dir[i], "majmin.lab")
        
        chroma_features = process_chroma_features(chroma_file)
        chord_data = process_lab_file(lab_file)

        aligned_df = align_chroma_to_chords(chroma_features, chord_data)
        print(chroma_dir[i])
        print(aligned_df.sample(5))
        chroma_tensors, label_tensors= format_tensor(aligned_df)
   
        chroma_data_per_song.append(chroma_tensors)
        chord_labels_per_song.append(label_tensors)
    
    print(chord_labels_per_song)

    return ChordDataset(chroma_data_per_song, chord_labels_per_song, window_size)

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
dataset = generate_dataset(data_path + "/chroma_features", data_path + "/lab_files", window_size=5)
torch.save(dataset, 'chord_dataset.pt')

