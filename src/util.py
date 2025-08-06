import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def get_data_list(file_path: str):
    with open(file_path, 'r') as file:
        return file.read().splitlines()
    
def get_df():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    image_dir = os.path.join(project_root, 'data', 'images')
    extenstion = '.jpg'
    
    train_list = get_data_list(os.path.join(project_root, 'data', 'meta', 'train.txt'))
    test_list = get_data_list(os.path.join(project_root, 'data', 'meta', 'test.txt'))
    data = []

    for file in train_list:
        class_name, file_id = file.split('/')
        # Add .jpg extension if not present
        path = os.path.join(image_dir, class_name, file_id) + extenstion
        data.append({'class_name': class_name, 'file_id': file_id, 'path': path, 'split': 'train'})
    
    for file in test_list:
        class_name, file_id = file.split('/')
        # Add .jpg extension if not present
        path = os.path.join(image_dir, class_name, file_id) + extenstion
        data.append({'class_name': class_name, 'file_id': file_id, 'path': path, 'split': 'test'})
    
    return pd.DataFrame(data)

def get_df_stats(df: pd.DataFrame):
    stats = {
        'num_samples': len(df),
        'num_classes': df['class_name'].nunique(),
        'samples_per_class': df['class_name'].value_counts().to_dict(),
        'splits': df['split'].value_counts().to_dict()
    }
    return pd.DataFrame([stats])

def get_data_generators(image_df, target_size=(224, 224), batch_size=32):
    """
    Split Pandas dataframe and return Keras ImageDataGenerators
    """

    # Split DataFrames
    train_df = image_df[image_df['split'] == 'train']
    test_df   = image_df[image_df['split'] == 'test']

    # Train: with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation: no augmentation, just rescale
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='class_name',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_generator = val_datagen.flow_from_dataframe(
        test_df,
        x_col='path',
        y_col='class_name',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, test_generator

def get_random_image_from_class(df, class_name):
    return df[df['class_name'] == class_name].sample(1).iloc[0]['path']

def model_predict(model, image_path):
    """ Predict the class of an image using the trained model.
    Args:
        model: Trained Keras model.
        image_path: Path to the image file.
    Returns:
        Predicted class name.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

def display_image(image_path):
    """ Display an image from a given path.
    Args:
        image_path: Path to the image file.
    """
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0
    plt.imshow(img_array[0])
    plt.axis('off')
    plt.show()

