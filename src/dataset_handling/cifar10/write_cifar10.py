import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

CIFAR_N = 60_000
CIFAR_DIM = 3072
CIFAR_DIR = r"/home/hphi344/Documents/GS-DBSCAN-Analysis/data/CIFAR10"

def write_cifar10(dtype="f16"):

    # Define the transformation to flatten and normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to mean=0 and std=1 for each channel
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a 1D vector
    ])

    # Download the CIFAR-10 train and test datasets with the specified transformations
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Combine the datasets into a single list
    full_dataset = train_dataset + test_dataset

    # Extract data and labels separately
    data = np.array([np.array(sample[0].numpy()) for sample in full_dataset])  # Data as a 2D NumPy array
    labels = np.array([sample[1] for sample in full_dataset])  # Labels as a 1D NumPy array

    print("Data shape:", data.shape)  # Should be (60000, 3072) for CIFAR-10
    print("Labels shape:", labels.shape)  # Should be (60000,)

    np_dtype = np.float32 if dtype == "f32" else np.float16

    data.astype(np_dtype).tofile(os.path.join(CIFAR_DIR, f"cifar10_data_{dtype}.bin"))
    labels.astype(np.uint8).tofile(os.path.join(CIFAR_DIR, f"cifar10_labels_{dtype}.bin"))


if __name__ == "__main__":
    # write_cifar10("f32")
    write_cifar10("f16")