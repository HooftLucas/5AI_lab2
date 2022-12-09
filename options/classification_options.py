from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 256
        self.batch_size_train = 64

        # hyperparameters
        self.lr = 0.00001
        self.num_epochs = 20
        self.hidden_sizes = [784, 196, 98, 10]
