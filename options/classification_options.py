from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 500
        self.batch_size_train = 50

        # hyperparameters
        self.lr = 0.001
        self.num_epochs = 12
        self.hidden_sizes = [784, 196, 98, 10]
