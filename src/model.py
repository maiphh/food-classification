import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_food101_model(num_classes, input_shape=(224, 224, 3), dropout_rate=0.2):
    """
    Builds an EfficientNetB0-based classifier for `num_classes` outputs.
    Returns the full model and the base MobileNetV2 model.
    """
    # Load EfficientNetB0 without the top layer, using ImageNet weights
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model initially

    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create new model
    model = models.Model(inputs=base_model.input, outputs=outputs, name="Food101_MobileNetV2")
    return model, base_model