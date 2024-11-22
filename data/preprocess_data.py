import numpy as np
import json

# Path to the .npy file
file_path = "Lz0Oh2cxoTU_28133-28296.npy"

# Load the .npy file
data = np.load(file_path).reshape(1, 1, 128, 431).astype(np.float32)

# Convert the numpy array to a list for JSON serialization
data_list = data.tolist()

# Create the JSON payload
payload = {
    "data": data_list
}

# Save the payload to a file
with open("payload.json", "w") as f:
    json.dump(payload, f)

print("Payload saved to data/payload.json")
