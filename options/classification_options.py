from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 512
        self.batch_size_train = 256

        # hyperparameters
        self.lr = 0.001
        self.num_epochs = 20
        self.hidden_sizes = [784, 198, 98, 10]
