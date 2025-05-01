import matplotlib.pyplot as plt
from openTSNE import TSNE as OpenTSNE
import numpy as np
import os
import tensorflow as tf

def TSNE(feature_dim, batch_size, x_train, encoder, y_train, shape, selected_labels):
    y_codes = []
    for i in range(0, len(x_train), batch_size):
        a, b = i, min(len(x_train), i + batch_size)
        input_images = x_train[a:b]
        batch = x_train[a:b]
        output_codes = encoder.predict(input_images)
        codes = encoder.predict(batch)
        y_codes.append(output_codes.reshape(len(input_images), -1))
    y_codes = np.concatenate(y_codes, axis=0)
    tsne = OpenTSNE(perplexity=30, random_state=42)
    points_np = tsne.fit(y_codes)
    display_and_save(points_np, y_train, selected_labels, "TSNE", feature_dim, shape)

def PCA(feature_dim, batch_size, x_train, encoder, y_train, shape, selected_labels):
    y_codes = []
    for i in range(0, len(x_train), batch_size):
        a, b = i, min(len(x_train), i + batch_size)
        input_images = x_train[a:b]
        output_codes = encoder.predict(input_images)
        y_codes.append(output_codes.reshape(len(input_images), -1))
    y_codes = tf.convert_to_tensor(np.concatenate(y_codes, axis=0), dtype=tf.float32)
    mean = tf.reduce_mean(y_codes, axis=0)
    y_centered = y_codes - mean
    cov_matrix = tf.matmul(tf.transpose(y_centered), y_centered) / tf.cast(tf.shape(y_codes)[0] - 1, tf.float32)
    eigvals, eigvecs = tf.linalg.eigh(cov_matrix)
    top_components = eigvecs[:, -2:]
    points = tf.matmul(y_centered, top_components)
    points_np = points.numpy()
    display_and_save(points_np, y_train, selected_labels, "PCA", feature_dim, shape)


def MEAN_PCA(feature_dim, batch_size, x_train, encoder, y_train, shape, selected_labels):
    y_codes = []
    for i in range(0, len(x_train), batch_size):
        a, b = i, min(len(x_train), i + batch_size)
        input_images = x_train[a:b]
        output_codes = encoder.predict(input_images)
        if isinstance(output_codes, (list, tuple)):
            mean_code = output_codes[0]
        else:
            mean_code = output_codes
        y_codes.append(mean_code[:, :2].reshape(len(input_images), -1))
    points_np = np.concatenate(y_codes, axis=0).astype(np.float32)
    display_and_save(points_np, y_train, selected_labels, "MEAN_PCA", feature_dim, shape)

def display_and_save(points_np, y_train, selected_labels, name_of_reduction_algorithm, feature_dim, shape):
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.tab10
    labels = np.unique(y_train)
    for lbl in labels:
        mask = (y_train == lbl)
        plt.scatter(
            points_np[mask, 0], points_np[mask, 1],
            label=str(lbl),
            s=5, alpha=0.8,
            color=cmap(int(lbl) % 10)
        )
    plt.legend(title='Digit Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    label_str = ",".join(str(lbl) for lbl in selected_labels)
    plt.title(f'{name_of_reduction_algorithm} visualization for labels [{label_str}] (feat_dim={feature_dim})')
    plt.tight_layout()
    save_dir = f"{name_of_reduction_algorithm}_plots"
    os.makedirs(save_dir, exist_ok=True)
    label_str = ",".join(str(lbl) for lbl in selected_labels)
    filename = os.path.join(save_dir, f'MCVAE_latent-space_{name_of_reduction_algorithm}_{shape[0]}_{label_str}.png')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"{name_of_reduction_algorithm} visualization saved to {filename}")
