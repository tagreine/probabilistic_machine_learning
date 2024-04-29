import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import matplotlib.pyplot as plt
tfd = tfp.distributions


def em(dataset, n_classes, n_iterations, random_seed):
    n_samples = dataset.shape[0]
    np.random.seed(random_seed)

    # initial guesses for the parameters
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    class_probs = np.random.dirichlet(np.ones(n_classes))

    for _ in tqdm(range(n_iterations)):
        "E-step"
        # for looping for intuition
        #responsibilities = np.zeros((n_samples, n_classes))
        #for i in range(n_samples):
        #    for c in range(n_classes):
        #        responsibilities[i, c] = class_probs[c] * tfd.Normal(loc=mus[c], scale=sigmas[c]).prob(dataset[i])

        # broadcasting for speed
        responsibilities = tfd.Normal(loc=mus, scale=sigmas).prob(dataset.reshape(-1, 1)).numpy() * class_probs

        # normalize over the class axis
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)
        # sum over the samples axis
        class_responsibilities = np.sum(responsibilities, axis=0)

        for c in range(n_classes):
            "M-steps"
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(np.sum(responsibilities[:, c] * (dataset - mus[c]) ** 2) / class_responsibilities[c])

    return class_probs, mus, sigmas


def main():
    class_probs_true = [0.6, 0.4]
    mus_true = [2.5, 4.8]
    sigmas_true = [0.6, 0.3]
    random_seed = 42
    n_samples = 1000
    n_iterations = 50
    n_classes = 3

    # generate the data
    univariate_gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=class_probs_true),
                                           components_distribution=tfd.Normal(loc=mus_true, scale=sigmas_true)
                                           )
    dataset = univariate_gmm.sample(n_samples).numpy()
    class_probs, mus, sigmas = em(dataset, n_classes, n_iterations, random_seed)

    x = np.linspace(mus_true[0] - 3 * sigmas_true[0], mus_true[1] + 3 * sigmas_true[1], 100)

    plt.figure(figsize=(5, 5))
    plt.plot(x, stats.norm.pdf(x, mus_true[0], sigmas_true[0]))
    plt.plot(x, stats.norm.pdf(x, mus_true[1], sigmas_true[1]))
    for i in range(n_classes): 
        plt.plot(x, stats.norm.pdf(x, mus[i], sigmas[i]))

    plt.hist(dataset, bins=20, density=True)
    legends = ["True class 1", "True class 2"]
    for i in range(n_classes):
        legends.append(["Estimated {}".format(i)]) 
    plt.legend(["True class 1", "True class 2", "Estimated 1", "Estimated 2"])
    plt.show()


if __name__ == "__main__":
    main()
