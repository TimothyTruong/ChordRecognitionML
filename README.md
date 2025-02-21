# Chordentify (New name later)

Based off of Chordify:
- Machine learning based chord recogniition software to identify chords within pop songs
- What it will do: search a song (maybe connect to spotify), and break the song into timestamps and recognize all of the chords throughout a song
- Challenges:
    - Realtime chord recognition (songs are long, how will I break it up to identify all of the chords)
    - Lack of data for a full-fledged neural network (maybe use transfer learning?)
    - Overfitting - I don't know what kinds of songs are in this dataset... May have to somehow create my own
    - Basically the model has to ALSO recognize the timestamp that this chord appears...
- Backbone: CNN or RNN trained using audio info + chord annotations

Steps:
- Find the dataset, and format the dataset properly into python (McGill Dashboard Dataset)
- Create the script to produce the dataset - attach .lab file chord info with the chroma feature audio data
- Ensure that the shape is suitable for a neural network
- Create the model architecture for the neural network (research what the structure should be like, and using tensorflow or pytorch)
- Train and test the model on McGill dataset
- Create the pipeline for formatting audio files into the correct input for the model
- Test if it works on new songs... 
- Create a script where it can divide up the song, identify the chords, and display it to the user
- Create the UI that moves in realtime

