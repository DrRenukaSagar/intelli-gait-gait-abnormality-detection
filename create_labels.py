import pickle
import os

# The gait types from the GAVD dataset, sorted alphabetically.
# This is the standard order the training script would have used.
class_names = [
    'antalgic_gait', 
    'ataxic_gait', 
    'bilateral_spastic_gait', 
    'choreiform_gait',
    'diplegic_gait', 
    'hemiplegic_gait', 
    'myopathic_gait', 
    'neuropathic_gait',
    'normal', 
    'parkinsonian_gait', 
    'unilateral_spastic_gait'
]

# Define the path to your model folder
MODEL_FOLDER = 'model'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Define the path for the new labels file
LABEL_MAP_SAVE_PATH = os.path.join(MODEL_FOLDER, "gait_labels_nn.pkl")

# Save the list to the .pkl file
with open(LABEL_MAP_SAVE_PATH, 'wb') as f:
    pickle.dump(class_names, f)

print(f"✅ Successfully created the labels file at: {LABEL_MAP_SAVE_PATH}")
print("You can now run your main application.")
