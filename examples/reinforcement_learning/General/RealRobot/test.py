import pickle
import torch

# Function to recursively move tensors to CPU
def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    else:
        return data

# Step 1: Load the original pickle file
with open('best_reward.pkl', 'rb') as f:
    data = pickle.load(f)

# Step 2: Move all tensors in the data to CPU
data_cpu = move_to_cpu(data)

# Step 3: Save the updated data back to a new pickle file
with open('final_no_noise_cpu.pkl', 'wb') as f:
    pickle.dump(data_cpu, f)

print("Tensors have been successfully moved to CPU and saved.")

# import pickle
#
# # Specify the path to the .pkl file
# file_path = 'final_no_noise.pkl'
#
# # Open the .pkl file and load the dictionary
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#
# # Check if 'replay_buffers' exists and delete it
# if 'replay_buffers' in data:
#     del data['replay_buffers']
#
# # Save the modified dictionary back to the .pkl file
# with open(file_path, 'wb') as file:
#     pickle.dump(data, file)
#
# print("Entry 'replay_buffers' has been deleted and the file has been saved.")
