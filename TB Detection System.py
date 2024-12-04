#!/usr/bin/env python
# coding: utf-8

# # TB Detection System

# In[40]:


import os
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[41]:


# Set paths for your images
data_path = r'C:\Users\Vivek\OneDrive\Desktop\DS_DA\AI\TB_Chest_Radiography_Database'  

categories = ['Tuberculosis', 'Normal']
image_paths = []
labels = []

# Collect all image paths and labels
for category in categories:
    category_path = os.path.join(data_path, category)
    for img_name in os.listdir(category_path):
        image_paths.append(os.path.join(category_path, img_name))
        labels.append(category)

# Convert labels to binary (0 for Normal, 1 for Tuberculosis)
labels = [1 if label == 'Tuberculosis' 
          else 0 
          for label in labels]

# Split the data into train and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.3, random_state=42)


# # Image data preprocessing

# In[42]:


# Function to display sample images from the dataset

def display_samples(image_paths, labels, num_samples=5):
    """Displays a few images from the dataset."""
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        img_path = image_paths[i]
        label = labels[i]
        img = image.load_img(img_path, target_size=(150, 150))  # Resize image
        img_array = image.img_to_array(img) / 255.0  # Normalize image
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_array)
        plt.title('Tuberculosis' if label == 1 else 'Normal')
        plt.axis('off')
    plt.show()
    
display_samples(train_paths, train_labels, num_samples=5)


# In[43]:


# Custom data generator to load images dynamically
def custom_data_generator(image_paths, labels, batch_size=32, target_size=(150, 150)):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            images = []
            for img_path in batch_paths:
                img = image.load_img(img_path, target_size=target_size)
                img_array = image.img_to_array(img) / 255.0  # Normalize image
                images.append(img_array)
            yield np.array(images), np.array(batch_labels)
        


# # Model architecture

# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN model architecture
model = Sequential()

# Add convolutional layers with MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout to avoid overfitting
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[45]:


# Create data generators for training and testing
train_generator = custom_data_generator(train_paths, train_labels)
test_generator = custom_data_generator(test_paths, test_labels)


history = model.fit(
    train_generator,
    epochs=5,  
    steps_per_epoch=len(train_paths) // 32,
    validation_data=test_generator,
    validation_steps=len(test_paths) // 32)


# In[46]:


# Evaluate the model on test data

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_paths) // 32)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# In[54]:


def predict_tb(image_path, model):
    # Preprocess the input image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return "Tuberculosis positive"
    else:
        return "Tuberculosis Negative"
    
# Example usage:
new_image_path = r"C:\Users\Vivek\OneDrive\Desktop\DS_DA\AI\TB_Chest_Radiography_Database\Test_Img\Tuberculosis-2.png"
result = predict_tb(new_image_path, model)
print(result)


# In[ ]:





# In[ ]:




