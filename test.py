import torch
import numpy as np

from model import CNN_RegDrop
def inv_sigmoid(x):
    if 0.00000001 < x < 0.999999:
        return np.log(x / (1 - x))
    elif x <= 0.00000001:
        return np.log(0.00000001 / (1 - 0.00000001))
    else:
        return np.log(0.999999 / (1 - 0.999999))


# Function to preprocess a single file and run inference
def predict_single_file(model, file_path, device='cpu'):
    # Load the .npy file and preprocess
    data = np.load(file_path).reshape(1, 1, 128, 431).astype(
        np.float32)  # Shape: (batch_size=1, channels=1, height=128, width=431)
    data_tensor = torch.tensor(data).to(device)  # Convert to PyTorch tensor

    # Set model to evaluation mode
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(data_tensor)  # Raw logits
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities

    # Interpret results
    pred_class = np.argmax(probabilities)  # Predicted class index
    pred_probability = probabilities[0][pred_class]  # Probability of predicted class
    anomaly_score = inv_sigmoid(1 - probabilities[0][pred_class])

    return {
        "predicted_class": int(pred_class),
        "predicted_probability": float(pred_probability),
        "anomaly_score": float(anomaly_score)
    }


# Example usage
if __name__ == "__main__":
    # Set the device to 'cpu'
    device = 'cpu'

    # Load the trained model
    model_path = "CNN_RegDrop.pt"
    model = CNN_RegDrop()  # Replace CNN_RegDrop with your model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to CPU

    # Path to the single file you want to predict
    single_file_path = "data/328000__hali-pinson__scream-5.npy"

    # Run prediction
    result = predict_single_file(model, single_file_path, device=device)
    print("Prediction Results:")
    print(result)
