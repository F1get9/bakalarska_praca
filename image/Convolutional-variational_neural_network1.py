import tensorflow as tf
import numpy as np
import time
from IPython import display
import cv2
from visualizationOpenCV import display, PCA, TSNE, MEAN
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os

(train_images, train_categories), (test_images, test_categories) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

# Use *tf.data* to batch and shuffle the data
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=16, kernel_size=3, strides=(1, 1), activation='relu',padding='same'),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=16, kernel_size=3, strides=(2, 2), activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=32, kernel_size=3, strides=(1, 1), activation='relu',padding='same'),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=32, kernel_size=3, strides=(2, 2), activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=64, kernel_size=3, strides=(1, 1), activation='relu',padding='same'),
            tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
                filters=64, kernel_size=3, strides=(1, 1), activation='relu',padding='same'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim,kernel_initializer=self.initializer,),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu, kernel_initializer=self.initializer,),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=64, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=32, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=1, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
      
  def call(self, x):
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    x_logit = self.decode(z)
    return x_logit, mean, logvar

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 230
latent_dim = 2
num_examples_to_generate = 16
model = CVAE(latent_dim)

CHECKPOINT_ROOT = 'checkpoints-var'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
    loss = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
        x_logit, _, _ = model(test_x)
        x_hat = tf.sigmoid(x_logit)
        accuracy_metric.update_state(
            tf.reshape(test_x, [test_x.shape[0], -1]),
            tf.reshape(x_hat,   [test_x.shape[0], -1])
        )
    elbo = -loss.result()
    acc  = accuracy_metric.result()
    print(f'Epoch: {epoch}, Test set ELBO: {elbo:.4f}, '
          f'Time: {(end_time - start_time):.2f}s, Accuracy: {acc:.4f}')
    if epoch % 10 == 0:
        ckpt_dir = os.path.join(CHECKPOINT_ROOT, f'epoch_{epoch}')
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = checkpoint.save(file_prefix=os.path.join(ckpt_dir, 'ckpt'))
        print(f'→ Saved checkpoint for epoch {epoch}: {save_path}')
batch_size = 512
y_codes = np.zeros((len(train_images),2),np.float32)    
for i in range(0,len(train_images),batch_size):
    a, b = i, min(len(train_images),i+batch_size)
    input_images = train_images[a:b] 
    output_codes = model.encoder.predict(input_images)
    y_codes[a:b] = output_codes[:,:2]

points = y_codes
graph = display(points,train_categories,range(10), True)
cv2.imwrite('latent-space-conv_var_ae.png',graph)

encoder_input = Input(shape=(28, 28, 1), dtype=tf.float32)
x = encoder_input
batch_size = 64
numbers_input = input("Enter labels (space-separated): ")
selected_labels = [int(n) for n in numbers_input.strip().split() if n.isdigit()]
print("Selected labels:", selected_labels)
train_mask = np.isin(train_categories, selected_labels)
train_images = train_images[train_mask]
train_categories = train_categories[train_mask]
flatten_layer_dim = 4
for idx, layer in enumerate(model.encoder.layers):
    x = layer(x)
    layer_model = Model(encoder_input, x)
    print(f"\nEncoder Layer {idx}:")
    layer_model.summary()
    output_shape = layer_model.output_shape[1:]
    feature_dim = np.prod(layer_model.output_shape[1:])
    print(f"Feature Dimension: {feature_dim}")
    type_of_network = "Convolutional-variational"
    if feature_dim == flatten_layer_dim:
        MEAN(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels, type_of_network)
    PCA(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels, type_of_network)
    TSNE(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels, type_of_network)
