import torch
from torch.utils.data import DataLoader
from torch import optim

from utilities import utils
from datasets.houses_dataset import HousesDataset
from models.models import LinearRegression
from options.linear_regression_options import LinearRegressionOptions

if __name__ == "__main__":
    options = LinearRegressionOptions()
    utils.init_pytorch(options)

    # create and visualize datasets
    train_dataset = HousesDataset(options, train=True)
    test_dataset = HousesDataset(options, train=False)
    train_dataset.plot_data()
    test_dataset.plot_data()

    # create dataloaders for easy access
    train_dataloader = DataLoader(train_dataset, options.batch_size_train)
    test_dataloader = DataLoader(test_dataset, options.batch_size_test)

    # create a LinearRegression instance named model
    model = LinearRegression()
    model.to(options.device)

    # define the opimizer
    optimizer = optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)

    # train the model
    utils.train_lin_model(model=model, optimizer=optimizer, train_data=train_dataloader, options=options)

    # test the model
    print("Testing the model...\n")

    print("On the train set:")
    utils.test_lin_reg_model(model, train_dataloader)
    utils.test_lin_reg_plot(model, train_dataloader, options)

    print("On the test set:")
    utils.test_lin_reg_model(model, test_dataloader)
    utils.test_lin_reg_plot(model, test_dataloader, options)
    utils.print_lin_reg(model, options)

    # save the model
    utils.save(model, options)
