import numpy as np
import utils
import matplotlib.pyplot as plt
import time

train_images, train_labels = utils.load_training_data("cifar-10-batches-py")  # 50000,3072
test_images, test_labels = utils.load_test_data("cifar-10-batches-py")  # 10000,3072


mean_channel_train_images = utils.cifar_10_color(train_images)  # 50000,3
mean_channel_test_images = utils.cifar_10_color(test_images)   # 10000,3


# task 1
tic = time.time()
mu, sigma, prior = utils.naive_bayes_learn(mean_channel_train_images, train_labels)  # 10x3 10x3  10x1
prediction = utils.cifar10_classifier_naivebayes(mean_channel_test_images, mu, sigma, prior)
accuracy = utils.classification_accuracy(prediction, test_labels)
print("naive bayes accuracy = {}".format(accuracy))
toc = time.time()
print("Time taken for Naive Bayes = ", toc - tic)
print("----------------------------------")

# # task 2
tic = time.time()
mu, covariance, prior = utils.bayes_learn(mean_channel_train_images, train_labels)
prediction = utils.cifar10_classifier_bayes(mean_channel_test_images, mu, covariance, prior)
accuracy = utils.classification_accuracy(prediction, test_labels)
print("Bayes accuracy = {}".format(accuracy))
toc = time.time()
print("Time taken for  Bayes = ", toc - tic)
print("----------------------------------")

# task 3
tic = time.time()
acc = np.zeros(6)
for i in range(0, 6):
    mean_train_images = utils.cifar10_color_resize(train_images, size=(2**i, 2**i))
    mean_test_images = utils.cifar10_color_resize(test_images, size=(2**i, 2**i))
    mu, covariance, prior = utils.bayes_learn(mean_train_images, train_labels)
    prediction = utils.cifar10_classifier_bayes(mean_test_images, mu, covariance, prior)

    accuracy = utils.classification_accuracy(prediction, test_labels)
    acc[i] = accuracy
    print("Bayes accuracy for size {} x {} = {}".format(2**i, 2**i, accuracy))
    np.save("accuracy.npy", acc)
toc = time.time()
print("Total Time taken for all sizes = ", toc - tic)

plt.plot(['1x1', '2x2', '4x4', '8x8', '16x16', '32x32'], acc, 'b-', label="img size vs acc")
plt.xlabel("size")
plt.ylabel("accuracy")
plt.legend(loc='best')
plt.savefig("accuracy.jpeg", bbox_inches="tight")
plt.clf()
