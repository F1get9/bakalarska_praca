import tensorflow as tf
import cv2
import numpy as np
from openTSNE import TSNE as OpenTSNE

colors = { 0 : (0,0,255), 1 : (0,255,255), 2 : (0,255,0), 3 : (255,255,0), 4 : (255,255,255), 5: (160,160,160), 6: (255,0,0), 7: (255,0,255), 8: (80,80,0), 9 : (80,0,0) }

def display(points,types, numbers, normal_distribution):
    if normal_distribution == True:
        ext = 1.6448536
        points = (points+ext)/(2*ext)
    else:
        m = np.max(np.abs(points))
        points = (points+m)/(2*m)
    v = 800
    graph = np.zeros((v,v,3),np.uint8)
    for i in range(len(points)):
        cv2.circle(graph,(int(v*points[i,0]),int(v*points[i,1])),2,colors[types[i]],cv2.FILLED)
    for j in numbers:
        cv2.rectangle(graph,(j*32,0),((j+1)*32,32),colors[j],cv2.FILLED)
        cv2.putText(graph,str(j),(j*32+8,32-8),0,0.9,(0,0,0))
    return graph

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

    graph = display(points_np, y_train, selected_labels, False)
    label_str = "_".join(str(lbl) for lbl in selected_labels)
    filename = f"latent-space_{label_str}_{feature_dim}.png"
    cv2.imwrite(filename, graph)
    print(f"Saved PCA visualization to {filename}")

def TSNE(feature_dim, batch_size, x_train, encoder, y_train, shape, selected_labels):
    y_codes = []

    for i in range(0, len(x_train), batch_size):
        a, b = i, min(len(x_train), i + batch_size)
        input_images = x_train[a:b]
        output_codes = encoder.predict(input_images)
        y_codes.append(output_codes.reshape(len(input_images), -1))

    y_codes = np.concatenate(y_codes, axis=0)

    tsne = OpenTSNE(perplexity=30, random_state=42) 
    points_np = tsne.fit(y_codes)  

    graph = display(points_np, y_train, selected_labels, False)
    label_str = ",".join(str(lbl) for lbl in selected_labels)
    filename = f"latent-space_tsne_{label_str}_{feature_dim}.png"
    cv2.imwrite(filename, graph)

    print(f"t-SNE visualization saved to {filename}")

def MEAN(feature_dim, batch_size, x_train, encoder_model, y_train, shape, selected_labels):
    y_codes = []
    for i in range(0, len(x_train), batch_size):
        a, b = i, min(len(x_train), i + batch_size)
        input_images = x_train[a:b]
        encoded_output = encoder_model.predict(input_images)  
        mean = encoded_output  
        mean = mean[:, :2]  
        y_codes.append(mean)  
    y_codes = np.concatenate(y_codes, axis=0) 
    shape = y_codes.shape[1]
    if y_codes.shape[1] != 2:
        raise ValueError(f"Expected (n_samples, 2), but got {y_codes.shape}")
    graph = display(y_codes, y_train, selected_labels, True)
    label_str = ",".join(str(lbl) for lbl in selected_labels)
    filename = f"latent-space_tsne_{label_str}_{feature_dim}.png"
    cv2.imwrite(f'latent-space_MEAN_{label_str}_{shape}.png', graph)
    print(f"Mean visualization saved to {filename}")
