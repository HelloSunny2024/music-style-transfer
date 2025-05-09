
# 🎵 Music Style Transfer with CycleGAN

This project leverages deep learning techniques to perform music style transfer using a CycleGAN model. The goal is to transform music from one genre to another, such as converting Pop music to Rock music, while preserving the original content and structure. The workflow involves preprocessing raw audio into Mel spectrograms, applying CycleGAN for genre transformation, and then reconstructing the audio waveform using DiffWave for high-quality sound generation. Additionally, a Genre Classifier is trained to evaluate the generated music and ensure it belongs to the correct genre.

---

## 📁 Project Structure

```
music-style-transfer/

│
├── cyclegan/                    # CycleGAN related files
│   ├── cyclegan.py              
│   ├── module.py                
│   ├── preprocess_data.py       
│   ├── test.py                 
│   └── train.py                
│
├── dataset/                     # Dataset directory
│   └── wav_fma_split/           
│       ├── Pop/                
│       └── Rock/                
│
├── diffwave/                    # DiffWave-related files for waveform generation
│
├── classifier/                  # Genre classifier related files
│   ├── evaluate_npy_mels.ipynb    
│   └── genre_classifier_train.ipynb
│               
└── README.md                    # Project documentation
```

---

## 🔧 Features

- 🎶 **Mel Spectrogram Extraction**: Preprocess raw audio files into Mel spectrograms for CycleGAN input.
- 🔁 **CycleGAN Model**: Performs music style transfer between genres, such as Pop and Rock.
- 🎧 **DiffWave Reconstruction**: Utilizes DiffWave for reconstructing high-quality audio waveforms from the transformed spectrograms. https://github.com/lmnt-com/diffwave
- 📊 **Genre Classifier**: Classifies music into genres, verifying the accuracy of the style transfer.


---

## 🚀 Steps

#### 1. Train the CycleGAN
Train the CycleGAN model on your dataset to perform music style transfer between genres.

#### 2. Test the CycleGAN
After training, test the CycleGAN model by generating style-transferred music, for example, converting Pop to Rock.

#### 3. Reconstruct the waveform
Use DiffWave to reconstruct the waveform of the generated audio from the Mel spectrograms.

#### 4. Train the Genre Classifier
Train a genre classifier to classify the music into genres, such as Rock, Pop, or others.

#### 5. Evaluate the Converted spectrograms with the Classifier
Use the trained genre classifier to evaluate the generated music and validate the effectiveness of the style transfer. The classifier will check if the generated spectrograms correspond to the target genre (e.g., Rock from Pop).


---



