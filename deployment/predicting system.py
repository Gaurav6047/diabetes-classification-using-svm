import pickle
import numpy as np

# Load the saved model/pipeline
with open(r"C:\Users\lenovo\Desktop\github\diabetes classification using svm\deployment\diabetes_pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Input sample (8 features)
sample = (5, 166, 72, 19, 175, 25.8, 0.587,51)

# Convert to NumPy array
sample_data_as_numpy_array = np.asarray(sample)

# Reshape (1 row, 8 columns)
sample_data_reshaped = sample_data_as_numpy_array.reshape(1, -1)

# Predict using loaded pipeline
prediction = loaded_pipeline.predict(sample_data_reshaped)

print("Raw Prediction:", prediction)
print("Prediction:", "Diabetes" if prediction[0] == 1 else "No Diabetes")
