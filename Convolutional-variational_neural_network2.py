import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from visualization import PCA, MEAN, TSNE

(train_images, train_categories), (test_images, test_categories) = tf.keras.datasets.mnist.load_data()
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = len(train_images)
test_size = len(test_images)
batch_size = 32

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder with separate branches for mean and log variance."""
  
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.latent_dim = latent_dim

    self.encoder_shared = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=16, kernel_size=3, strides=2, activation='relu', padding='same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=32, kernel_size=3, strides=1, activation='relu', padding='same'),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=32, kernel_size=3, strides=2, activation='relu', padding='same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
          tf.keras.layers.Conv2D(kernel_initializer=self.initializer,
              filters=64, kernel_size=3, strides=1, activation='relu', padding='same'),
          tf.keras.layers.Flatten(),
      ]
    )

    self.encoder_mean = tf.keras.Sequential([
        tf.keras.layers.Dense(latent_dim, kernel_initializer=self.initializer)
    ])
    self.encoder_logvar = tf.keras.Sequential([
        tf.keras.layers.Dense(latent_dim, kernel_initializer=self.initializer)
    ])

    self.decoder = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu, kernel_initializer=self.initializer),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=1, kernel_size=3, strides=1, padding='same', activation='relu'),
          tf.keras.layers.Conv2DTranspose(kernel_initializer=self.initializer,
              filters=1, kernel_size=3, strides=1, padding='same'),
      ]
    )

  def encode(self, x):
    shared_features = self.encoder_shared(x)
    mean = self.encoder_mean(shared_features)
    logvar = self.encoder_logvar(shared_features)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      return tf.sigmoid(logits)
    return logits

  def call(self, x, training=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=False)
        return x_logit, mean, logvar

  optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
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
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 230
latent_dim = 2
num_examples_to_generate = 16
model = CVAE(latent_dim)

CHECKPOINT_ROOT = 'checkpoints-mod-var'
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

encoder_input = Input(shape=(28, 28, 1), dtype=tf.float32)
x = encoder_input
batch_size = 64
numbers_input = input("Enter labels (space-separated): ")
selected_labels = [int(n) for n in numbers_input.strip().split() if n.isdigit()]
print("Selected labels:", selected_labels)
train_mask = np.isin(train_categories, selected_labels)
train_images = train_images[train_mask]
train_categories = train_categories[train_mask]
for idx, layer in enumerate(model.encoder_shared.layers):
    x = layer(x)
    layer_model = Model(encoder_input, x)
    print(f"\nShared Layer {idx}:")
    layer_model.summary()
    output_shape = layer_model.output_shape[1:]
    feature_dim = np.prod(layer_model.output_shape[1:])
    print(f"Feature Dimension: {feature_dim}")
    if feature_dim < 500:
        PCA(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels)
        TSNE(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels)

print("\n--- Encoder Mean Branch Layers ---")
for idx, layer in enumerate(model.encoder_mean.layers):
    x = layer(x)
    layer_model = Model(encoder_input, x)
    print(f"\nShared Layer {idx}:")
    layer_model.summary()
    output_shape = layer_model.output_shape[1:]
    feature_dim = np.prod(layer_model.output_shape[1:])
    print(f"Feature Dimension: {feature_dim}")
    if feature_dim == 2:
        MEAN(feature_dim, batch_size, train_images, layer_model, train_categories, output_shape, selected_labels)
