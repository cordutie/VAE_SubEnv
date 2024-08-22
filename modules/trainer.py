import torch
from modules.loss import *
from modules.architecture import *
import matplotlib.pyplot as plt

# Updated Trainer
def train(model, optimizer, num_epochs, device, dataloader, save_dir):
    best_loss = float('inf')
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        overall_loss = 0
        
        for batch in dataloader:
            x = batch
            x = x.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            x_hat, mean, var = model(x)
            
            # Compute main loss
            loss = loss_function(x, x_hat, mean, var)
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

        # Plot the loss history
        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/loss_plot.png")
        plt.close()  # Close the figure to free up memory

    print(f"Training complete. Best model saved to {save_dir}/best_model.pth and the latest model saved to {save_dir}/last_model.pth")
