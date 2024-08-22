import torch
from modules.loss import *
from modules.architecture import *
import matplotlib.pyplot as plt
import os

# trainer multiscale loss ------------------------------------------------------------------
def train_multiscale_loss(model, optimizer, num_epochs, device, dataloader, save_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        overall_loss = 0
        
        for batch in dataloader:
            x = batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            x_hat, mean, var = model(x)
            
            # Ensure everything is on the same device
            x_hat, mean, var = x_hat.to(device), mean.to(device), var.to(device)

            # Compute main loss
            loss_1 = multiscale_spectrogram_loss(x, x_hat).to(device)
            loss_2 = -0.5 * torch.sum(1 + safe_log(var) - mean.pow(2) - var).to(device)  # KLD
            loss = (loss_1 + loss_2).to(device)
            overall_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate the average loss for this epoch
        avg_loss = overall_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\tEpoch {epoch + 1} \tAverage Loss: {avg_loss}")

        # Save the best model if this epoch's loss is the lowest so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, f"{save_dir}/best_model.pth")
        
        # Save the latest model
        save_model(model, f"{save_dir}/last_model.pth")

    # After all epochs, plot the loss history
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_plot.png")
    plt.show()  # Show the plot only once at the end

    print(f"Training complete. Best model saved to {save_dir}/best_model.pth and the latest model saved to {save_dir}/last_model.pth")

# Trainer statistics loss ------------------------------------------------------------------
def train_statistics_loss(model, optimizer, num_epochs, device, dataloader, save_dir, N_filter_bank, frame_size, sample_rate):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    erb_bank = fb.EqualRectangularBandwidth(frame_size, sample_rate, N_filter_bank, 20, sample_rate // 2)
    new_size = frame_size // 4
    new_sample_rate = sample_rate // 4
    log_bank = fb.Logarithmic(new_size, new_sample_rate, 6, 20, new_sample_rate // 2)
    
    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        overall_loss = 0
        
        for batch in dataloader:
            x = batch.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            x_hat, mean, var = model(x)

            # Ensure everything is on the same device
            x_hat, mean, var = x_hat.to(device), mean.to(device), var.to(device)

            # Compute main loss
            loss_1 = batch_statistics_loss(x, x_hat, N_filter_bank, sample_rate, erb_bank, log_bank).to(device)
            loss_2 = -0.5 * torch.sum(1 + safe_log(var) - mean.pow(2) - var).to(device)  # KLD
            loss = (loss_1 + loss_2).to(device)
            overall_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Calculate the average loss for this epoch
        avg_loss = overall_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\tEpoch {epoch + 1} \tAverage Loss: {avg_loss}")

        # Save the best model if this epoch's loss is the lowest so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, f"{save_dir}/best_model.pth")
        
        # Save the latest model
        save_model(model, f"{save_dir}/last_model.pth")

    # After all epochs, plot the loss history
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_plot.png")
    plt.show()  # Show the plot only once at the end

    print(f"Training complete. Best model saved to {save_dir}/best_model.pth and the latest model saved to {save_dir}/last_model.pth")
