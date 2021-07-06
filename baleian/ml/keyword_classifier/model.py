import tensorflow as tf
import tensorflow.keras.layers as layers

from .transformer import VALID_CHARS


class InputLayer(layers.Layer):
    
    def __init__(self, num_class, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.num_class = num_class
        self.reshape_layer = layers.Reshape((-1, 3, num_class))
        
    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=tf.int32)
        x = tf.one_hot(x, self.num_class)
        x = self.reshape_layer(x)
        return x
    
    def get_config(self):
        config = super(InputLayer, self).get_config()
        config.update({'num_class': self.num_class})
        return config
    

class ClassifierLayer(layers.Layer):

    def __init__(self, num_class, **kwargs):
        super(ClassifierLayer, self).__init__(**kwargs)
        self.num_class = num_class
        self.conv1_layers = [
            layers.Conv2D(kernel_size=2, strides=1, filters=32, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=2, strides=1, filters=64, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=2, strides=1, filters=128, padding='same', activation='relu'),
            layers.GlobalMaxPool2D()
        ]
        self.conv2_layers = [
            layers.Conv2D(kernel_size=2, strides=2, filters=32, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=3, strides=2, filters=64, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=4, strides=2, filters=128, padding='same', activation='relu'),
            layers.GlobalMaxPool2D()
        ]
        self.conv3_layers = [
            layers.Conv2D(kernel_size=4, strides=1, filters=32, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=5, strides=1, filters=64, padding='same', activation='relu'),
            layers.Conv2D(kernel_size=6, strides=1, filters=128, padding='same', activation='relu'),
            layers.GlobalMaxPool2D()
        ]
        self.concat_layer = layers.Concatenate(axis=-1)
        self.dense_layer = layers.Dense(256, activation='relu')
        self.output_layer = layers.Dense(num_class, activation='softmax')

    def call(self, x, **kwargs):
        h1 = self._call_sequential(x, self.conv1_layers)
        h2 = self._call_sequential(x, self.conv2_layers)
        h3 = self._call_sequential(x, self.conv3_layers)
        x = self.concat_layer([h1, h2, h3])
        x = self.dense_layer(x)
        x = self.output_layer(x)
        return x
    
    def _call_sequential(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x
    
    def get_config(self):
        config = super(ClassifierLayer, self).get_config()
        config.update({'num_class': self.num_class})
        return config
    

def create_training_model(input_shape, num_class, **kwargs):
    x = inputs = layers.Input(shape=input_shape, dtype=tf.int32)
    x = InputLayer(len(VALID_CHARS))(x)
    x = outputs = ClassifierLayer(num_class)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_ensemble_model(input_shape, num_class, folds, **kwargs):
    x = inputs = layers.Input(shape=input_shape, dtype=tf.int32, name='feature')
    x = InputLayer(len(VALID_CHARS))(x)
    classifier_layers = [ClassifierLayer(num_class, name=fold.name) for fold in folds]
    x = [layer(x) for layer in classifier_layers]
    x = outputs = layers.Average(name='predicted')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    for layer, fold in zip(classifier_layers, folds):
        layer.set_weights(fold.get_weights())
        
    return model
