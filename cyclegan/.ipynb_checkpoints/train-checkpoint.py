from torch.utils.data import DataLoader
import os
from cyclegan import *
from module import *
from preprocess_data import AudioDataset
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt


# load pre trained weights

def load_checkpoint(model, optimizer_generator, optimizer_discriminatorA, optimizer_discriminatorB, checkpoint_path):
	if os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		print("Checkpoint keys:", checkpoint.keys())

		# load generator and discriminator
		model.generatorG.load_state_dict(checkpoint['model_generatorG'], strict=False)
		model.generatorF.load_state_dict(checkpoint['model_generatorF'], strict=False)
		model.discriminatorA.load_state_dict(checkpoint['model_discriminatorA'], strict=False)
		model.discriminatorB.load_state_dict(checkpoint['model_discriminatorB'], strict=False)

		# load optimizer
		optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
		optimizer_discriminatorA.load_state_dict(checkpoint['optimizer_discriminatorA'])
		optimizer_discriminatorB.load_state_dict(checkpoint['optimizer_discriminatorB'])

		start_epoch = checkpoint['epoch'] + 1
		best_loss = checkpoint['best_loss']
		print(f"Checkpoint loaded: Starting from epoch {start_epoch}, Best Loss: {best_loss:.4f}")
		return start_epoch, best_loss
	else:
		print("No checkpoint found. Starting fresh training.")
		return 0, float('inf')


def save_checkpoint(model, optimizer_generator, optimizer_discriminatorA, optimizer_discriminatorB, epoch, best_loss,
					checkpoint_path):
	torch.save({
		'epoch': epoch,
		'best_loss': best_loss,
		'model_generatorG': model.generatorG.state_dict(),
		'model_generatorF': model.generatorF.state_dict(),
		'model_discriminatorA': model.discriminatorA.state_dict(),
		'model_discriminatorB': model.discriminatorB.state_dict(),
		'optimizer_generator': optimizer_generator.state_dict(),
		'optimizer_discriminatorA': optimizer_discriminatorA.state_dict(),
		'optimizer_discriminatorB': optimizer_discriminatorB.state_dict(),
	}, checkpoint_path)
	print(f"Checkpoint saved at epoch {epoch} with best loss: {best_loss:.4f}")


def save_losses_to_csv(filepath, epoch, generator_loss, discriminatorA_loss, discriminatorB_loss):
	write_header = not os.path.exists(filepath)

	with open(filepath, mode='a', newline='') as file:
		writer = csv.writer(file)
		if write_header:
			writer.writerow(['Epoch', 'Generator Loss', 'DiscriminatorA Loss', 'DiscriminatorB Loss'])  # Header
		writer.writerow([epoch, generator_loss, discriminatorA_loss, discriminatorB_loss])


# load train dataloader
pop_dir = "/home/quincy/DATA/music/dataset/wav_fma_split/Pop/pop_train"
pop_dataset = AudioDataset(pop_dir, target_time_steps=2580, save_features=True)
pop_train_dataloader = DataLoader(pop_dataset, batch_size=32, shuffle=True)
# load train dataloader
rock_dir = "/home/quincy/DATA/music/dataset/wav_fma_split/Rock/rock_train"
rock_dataset = AudioDataset(rock_dir, target_time_steps=2580, save_features=True)
rock_train_dataloader = DataLoader(rock_dataset, batch_size=32, shuffle=True)


device = (
	"cuda" if torch.cuda.is_available()
	else "cpu"
)


# train
import matplotlib.pyplot as plt

def train(model, dataloaderA, dataloaderB, optimizer_generator, optimizer_discriminatorA, optimizer_discriminatorB,
          epochs, checkpoint_path, loss_csv_path):
    # load pretrained
    start_epoch, best_loss = load_checkpoint(
        model, optimizer_generator, optimizer_discriminatorA, optimizer_discriminatorB, checkpoint_path
    )

    # Lists to store loss values for plotting
    generator_losses = []
    discriminatorA_losses = []
    discriminatorB_losses = []

    for i in range(start_epoch, epochs):
        print("----------Training Epoch:{} start----------".format(i + 1))
        total_step = 0
        total_generator_loss = 0.0
        total_discriminatorA_loss = 0.0
        total_discriminatorB_loss = 0.0
        
        for (realA, realB) in tqdm(zip(dataloaderA, dataloaderB), desc=f"Epoch {i+1}", total=len(dataloaderA)):
            realA, _ = realA
            realB, _ = realB
            realA = realA.to(device)
            realB = realB.to(device)
            
            # train discriminator
            fakeB = model.generatorG(realA)
            fakeA = model.generatorF(realB)
            optimizer_discriminatorA.zero_grad()
            optimizer_discriminatorB.zero_grad()
            
            # calculate the loss
            discriminatorA_loss, discriminatorB_loss = model.DiscriminatorLoss(realA, realB, fakeA, fakeB)
            discriminatorA_loss.backward()
            discriminatorB_loss.backward()
            optimizer_discriminatorA.step()
            optimizer_discriminatorB.step()
            
            # train generator
            optimizer_generator.zero_grad()
            
            # calculate the loss
            generator_loss = model.GeneratorLoss(realA, realB)
            generator_loss.backward()
            optimizer_generator.step()
            
            total_generator_loss += generator_loss.item()
            total_discriminatorA_loss += discriminatorA_loss.item()
            total_discriminatorB_loss += discriminatorB_loss.item()
            total_step = total_step + 1

        # scheduler
        scheduler_generator.step()
        scheduler_discriminatorA.step()
        scheduler_discriminatorB.step()
        
        avg_generator_loss = total_generator_loss / total_step
        avg_discriminatorA_loss = total_discriminatorA_loss / total_step
        avg_discriminatorB_loss = total_discriminatorB_loss / total_step

        print(f"Avg Generator Loss: {avg_generator_loss:.4f}")
        print(f"Avg DiscriminatorA Loss: {avg_discriminatorA_loss:.4f}")
        print(f"Avg DiscriminatorB Loss: {avg_discriminatorB_loss:.4f}")
        save_losses_to_csv(loss_csv_path, i + 1, avg_generator_loss, avg_discriminatorA_loss, avg_discriminatorB_loss)

        # Append losses to lists for plotting
        generator_losses.append(avg_generator_loss)
        discriminatorA_losses.append(avg_discriminatorA_loss)
        discriminatorB_losses.append(avg_discriminatorB_loss)

        # save model
        if avg_generator_loss < best_loss:
            best_loss = avg_generator_loss
            save_checkpoint(model, optimizer_generator, optimizer_discriminatorA, optimizer_discriminatorB, i+1,
                                best_loss, checkpoint_path)
        # Plotting the loss curves
        plt.figure(figsize=(10, 5))
        epochs_so_far = len(generator_losses)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs_so_far + 1), generator_losses, label='Generator Loss')
        plt.plot(range(1, epochs_so_far + 1), discriminatorA_losses, label='Discriminator A Loss')
        plt.plot(range(1, epochs_so_far + 1), discriminatorB_losses, label='Discriminator B Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig('/home/quincy/DATA/music/cyclegan_path/loss_curve.png')  # Save the plot as a .png file
        plt.show()
                

    print(f"Training completed. Best Loss achieved at epoch {i + 1}: {best_loss:.4f}")


# initial model
generatorG = Generator(ngf=64, input_channels=80, output_channels=80)
generatorF = Generator(ngf=64, input_channels=80, output_channels=80)
discriminatorA = Discriminator(input_channels=80, ndf=64)
discriminatorB = Discriminator(input_channels=80, ndf=64)

# optimizer
optimizer_generator = torch.optim.Adam(
    list(generatorG.parameters()) + list(generatorF.parameters()),
    lr=0.0002, betas=(0.5, 0.999)
)
optimizer_discriminatorA = torch.optim.Adam(
    discriminatorA.parameters(),
    lr=0.0002, betas=(0.5, 0.999)
)
optimizer_discriminatorB = torch.optim.Adam(
    discriminatorB.parameters(),
    lr=0.0002, betas=(0.5, 0.999)
)


# learning rate scheduler
# scheduler_generator = torch.optim.lr_scheduler.StepLR(optimizer_generator, step_size=20, gamma=0.5)
# scheduler_discriminatorA = torch.optim.lr_scheduler.StepLR(optimizer_discriminatorA, step_size=20, gamma=0.5)
# scheduler_discriminatorB = torch.optim.lr_scheduler.StepLR(optimizer_discriminatorB, step_size=20, gamma=0.5)

def lambda_rule(epoch):
    start_decay = 100
    total_epochs = 1000
    if epoch < start_decay:
        return 1.0
    else:
        return 1.0 - (epoch - start_decay) / (total_epochs - start_decay)

scheduler_generator = torch.optim.lr_scheduler.LambdaLR(optimizer_generator, lr_lambda=lambda_rule)
scheduler_discriminatorA = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminatorA, lr_lambda=lambda_rule)
scheduler_discriminatorB = torch.optim.lr_scheduler.LambdaLR(optimizer_discriminatorB, lr_lambda=lambda_rule)

# initial model
model = CycleGAN(generatorG, generatorF, discriminatorA, discriminatorB)
model.to(device)

epochs = 1000
checkpoint_path = '/home/quincy/DATA/music/cyclegan_path/cyclegan_checkpoint.pth'
# train(model, folk_new_1_train_dataloader, pop_new_1_train_dataloader, optimizer_generator, optimizer_discriminatorA,
# 	optimizer_discriminatorB, epochs, checkpoint_path)
loss_csv_path = '/home/quincy/DATA/music/cyclegan_path/loss.csv'
train(model, pop_train_dataloader, rock_train_dataloader, optimizer_generator, optimizer_discriminatorA,
	  optimizer_discriminatorB, epochs, checkpoint_path, loss_csv_path=loss_csv_path)
