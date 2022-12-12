import torch

from datasets.mnist_dataset import MNISTDataset
from models.models import Classifier, ClassifierVariableLayers
from options.classification_options import ClassificationOptions
from utilities import utils
from utilities.utils import init_pytorch, test_classification_model, train_classification_model, classify_images

if __name__ == "__main__":
    options = ClassificationOptions()
    init_pytorch(options)

    # create and visualize the MNIST dataset
    dataset = MNISTDataset(options)
    dataset.show_examples()

    """START TODO: fill in the missing parts"""
    print(
        'options.batch_size_test = ', options.batch_size_test, '\n',
        'options.batch_size_train = ', options.batch_size_train, '\n',
        'options.lr = ', options.lr, '\n',
        'options.num_epochs = ', options.num_epochs, '\n',
        'options.hidden_sizes = ', options.hidden_sizes, '\n',
        sep=''
    )
    # create a Classifier instance named model
    model = Classifier(options)
    model.to(options.device)
    # define the opimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=options.lr)
    # train the model
    utils.train_classification_model(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        options=options
    )
    """END TODO"""

    # Test the model
    print("The Accuracy of the model is: ")
    test_classification_model(model, dataset, options)
    classify_images(model, dataset, options)

    # save the model
    utils.save(model, options)
