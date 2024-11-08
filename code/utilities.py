import matplotlib.pyplot as plt

def plot_house_sizes_and_prices(sizes, prices, title):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.scatter(sizes, prices)
    ax.set_title(title)
    ax.set_xlabel("Size [m²]")
    ax.set_ylabel("Price [€]")
    figure.tight_layout()
    plt.show()

def plot_house_results(sizes, prices, estimations, linear_sizes, linear_prices, linear_estimations, title):
    figure = plt.figure()
    ax = figure.add_subplot()
    # plot real and estimated data points for dataset
    ax.scatter(sizes, prices, c="#377eb8")
    ax.scatter(sizes, estimations, c="#ff7f00")
    # plot real and estimated data points for a line
    ax.plot(linear_sizes, linear_prices, "#377eb8")
    ax.plot(linear_sizes, linear_estimations, "#ff7f00")
    ax.set_title(title)
    ax.set_xlabel("Size [m²]")
    ax.set_ylabel("Price [€]")
    ax.legend([
        "Actual house prices",
        "Estimated house prices",
        "Unknown function f(x)",
        "Learned function g(x)",
    ])
    plt.show()

def plot_images_and_labels(images, labels, predictions = None):
    images = images[:20]
    labels = labels[:20]
    predictions = None if predictions is None else predictions[:20]
    figure = plt.figure()
    for i in range(len(images)):
        ax = figure.add_subplot(4, 5, i + 1)
        ax.imshow(images[i][0], cmap="gray", interpolation="none")
        ax.set_title(f"Label: {labels[i]}" if predictions is None else f"Label: {labels[i]}\nPrediction: {predictions[i]}")
        ax.set_xticks([])
        ax.set_yticks([])
    figure.tight_layout()
    plt.show()

def plot_training_process(train_losses, valid_losses):
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.plot(train_losses, label="Training loss")
    ax.plot(valid_losses, label="Validation loss")
    ax.set_title("Training process")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    figure.tight_layout()
    plt.show()