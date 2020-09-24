import pickle
import numpy as np
import os
import scipy.stats as stats
import cv2


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data


def classification_accuracy(prediction, ground_truth):
    ground_truth = ground_truth[:prediction.shape[0]]
    n_images = prediction.shape[0]
    correct_predictions = np.sum(prediction == ground_truth)
    accuracy = correct_predictions / n_images

    return accuracy*100


def load_training_data(dataset_path):
    train_images = np.zeros([50000, 3072])
    train_labels = np.zeros([50000])

    start = 0
    n_images_in_a_file = 10000
    for i in range(1, 6):
        path = os.path.join(dataset_path, "data_batch_{}".format(i))
        data_dict = unpickle(path)
        train_images[start: start + n_images_in_a_file, :] = data_dict["data"]
        train_labels[start: start + n_images_in_a_file] = data_dict["labels"]
        start += n_images_in_a_file

    return np.asarray(train_images, dtype=np.int), np.asarray(train_labels, dtype=np.int)


def load_test_data(dataset_path):

    path = os.path.join(dataset_path, "test_batch")
    datadict = unpickle(path)
    test_images = datadict["data"]
    test_labels = datadict["labels"]
    return np.asarray(test_images, dtype=np.int), np.asarray(test_labels, dtype=np.int)


def cifar_10_color(images):
    reshaped_image = np.reshape(images, [images.shape[0], 3, -1])
    mean_images = np.mean(reshaped_image, axis=2)
    return mean_images


def naive_bayes_learn(mean_channel_images, train_labels, num_classes=10):
    train_images_sort = mean_channel_images[np.argsort(train_labels)]
    train_images_sort = np.reshape(train_images_sort, [num_classes, -1, 3])
    mu = np.mean(train_images_sort, axis=1)
    sigma = np.std(train_images_sort, axis=1)
    prior = 0.1 + np.zeros([num_classes, 1])
    return mu, sigma, prior


def cifar10_classifier_naivebayes(mean_channel_test_images, mu, sigma, p):
    p_c_x = np.zeros([10000, 10])

    for j in range(10):
        term1 = stats.norm.pdf(mean_channel_test_images[:, 0], mu[j, 0], sigma[j, 0])
        term2 = stats.norm.pdf(mean_channel_test_images[:, 1], mu[j, 1], sigma[j, 1])
        term3 = stats.norm.pdf(mean_channel_test_images[:, 2], mu[j, 2], sigma[j, 2])
        p_c_x[:, j] = term1 * term2 * term3 * p[j]

    pred = np.argmax(p_c_x, axis=1)
    return pred


def bayes_learn(mean_channel_images, train_labels, num_classes=10):
    train_images_sort = mean_channel_images[np.argsort(train_labels)]
    train_images_sort = np.reshape(train_images_sort, [num_classes, 5000, -1])
    mu = np.mean(train_images_sort, axis=1)
    covariance = np.zeros([10, mu.shape[1], mu.shape[1]])
    for i in range(10):
        train_class = train_images_sort[i, :, :].T
        cov = np.cov(train_class)
        covariance[i, :, :] = cov
    prior = 0.1 + np.zeros([10, 1])
    return mu, covariance, prior


def cifar10_classifier_bayes(mean_channel_test_images, mu, covariance, p):
    p_c_x = np.zeros([10000, 10])
    for j in range(10):
        term1 = stats.multivariate_normal.logpdf(mean_channel_test_images, mu[j, :], covariance[j, :, :])
        p_c_x[:, j] = term1 * p[j]

    pred = np.argmax(p_c_x, axis=1)
    return pred


def cifar10_color_resize(images, size=(2, 2)):
    reshaped_image = np.reshape(images, [images.shape[0], 3, 32, 32]).transpose([0, 2, 3, 1])
    reshaped_image = np.array(reshaped_image, dtype='uint8')
    out = np.zeros([images.shape[0], size[0], size[1], 3])
    for i in range(images.shape[0]):
        out[i, :, :, :] = cv2.resize(reshaped_image[i, :, :, :], size)
    out = np.transpose(out, [0, 3, 1, 2])
    out = np.reshape(out, [out.shape[0], -1])
    return np.array(out, dtype=np.int)


