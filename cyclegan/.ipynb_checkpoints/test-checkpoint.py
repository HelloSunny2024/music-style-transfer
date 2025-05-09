import torch
import numpy as np
from cyclegan import Generator
import torchaudio as T
from torch.utils.data import DataLoader
import os

from preprocess_data import AudioDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# load test dataloader
pop_dir = "/home/quincy/DATA/music/dataset/wav_fma_split/Pop/pop_val"
pop_dataset = AudioDataset(pop_dir, target_time_steps=2580, save_features=True)
pop_dataloader = DataLoader(pop_dataset, batch_size=1, shuffle=False)

# initial the generator
model = torch.load('/home/quincy/DATA/music/cyclegan_path/cyclegan_checkpoint.pth', map_location=device)
print(type(model))
print(model.keys())
generatorG = Generator(ngf=64, input_channels=80, output_channels=80)
generatorG.load_state_dict(model['model_generatorG'])
total_params = sum(p.numel() for p in generatorG.parameters())
print(f"Total number of generatorG parameters: {total_params}")
generatorG.to(device)
generatorG.eval()

def test_model(generator, test_loader, output_dir):
    with torch.no_grad():
        for idx, (test_data, file_name) in enumerate(test_loader):  
            input_features = test_data.to(device)
            generated_features = generator(input_features)
            generated_features = generated_features.cpu().numpy()
            output_filename = os.path.join(output_dir, f"{file_name[0]}_converted.npy")
            np.save(output_filename, generated_features)

            print(f"Generated features saved for {file_name[0]}.")

output_dir = "/home/quincy/DATA/music/converted_features"
os.makedirs(output_dir, exist_ok=True)

# Test pop -> rock
test_model(generatorG, pop_dataloader, output_dir)
