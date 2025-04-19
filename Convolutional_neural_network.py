from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, InputLayer
from tensorflow.keras.models import Model
from visualization import PCA, TSNE

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  

batch_size = 32

inp = Input(shape=(28, 28, 1))
print("Input shape:", inp.shape)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
x = MaxPooling2D((2, 2), padding='same')(x)  # (28,28) -> (14,14)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # (14,14) -> (7,7)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # (7,7) -> (4,4)

x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # (4,4) -> (2,2)

print("Encoded shape:", x.shape)  # (2,2,8)

feature_vector = Flatten()(x)  # (2,2,8) → (32,)
print("Flattened shape:", feature_vector.shape)

x = Reshape((2, 2, 8))(feature_vector)  # Restores back to (2,2,8)
print("Reshaped shape:", x.shape)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)  # (2x2) → (4x4)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)  # (4x4) → (8x8)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)  # (8x8) → (16x16)

x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)  # (16x16) → (32x32)


x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

out = x
print("Final output shape:", out.shape)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(x_train, x_train,epochs=100,batch_size=128,shuffle=True,validation_data=(x_test, x_test))
scores = autoencoder.evaluate(x_test, x_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

orig_input = Input(shape=(28, 28, 1), dtype='float32')
x = orig_input
layers = [layer for layer in autoencoder.layers if not isinstance(layer, InputLayer)]
n_half = len(layers) // 2  # Use first half of layers as encoder
batch_size = 64  
numbers_input = input("Enter labels (space-separated): ")
selected_labels = [int(n) for n in numbers_input.strip().split() if n.isdigit()]
print("Selected labels:", selected_labels)
train_mask = np.isin(y_train, selected_labels)
x_train = x_train[train_mask]
y_train = y_train[train_mask]

test_mask = np.isin(y_test, selected_labels)
x_test = x_test[test_mask]
y_test = y_test[test_mask]
for i, layer in enumerate(layers[:n_half]):
    x = layer(x)
    encoder = Model(orig_input, x)
    encoder.summary()

    output_shape = encoder.output_shape[1:]
    feature_dim = np.prod(output_shape)
    print(f"Feature dimension: {feature_dim}")
    type_of_network = "Convolutional"
    PCA(feature_dim, batch_size, x_train, encoder, y_train, output_shape, selected_labels, type_of_network)
    TSNE(feature_dim, batch_size, x_train, encoder, y_train, output_shape, selected_labels, type_of_network)
