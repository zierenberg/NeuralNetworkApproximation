
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class DenseNN(nn.Module):
    """
    This class implements a fully connected neural network with a variable number of hidden layers.
    Example
    -------
    >> model = DenseNN(input_size=3, hidden_sizes=[42,42,42], output_size=2)
    >> model(torch.tensor([1,2,3], dtype=torch.float32))
    Parameters
    ----------
    input_size: int
        Number of input neurons.
    hidden_sizes: list
        List of hidden layer sizes.
    output_size: int
        Number of output neurons.
    act: torch.nn.Module
        Activation function (default nn.Tanh()).
    """

    def __init__(self, input_size, hidden_sizes, output_size, act=nn.Tanh()):
        super().__init__()
        # create activation function
        self.act = act
        self.input_size = input_size
        self.output_size = output_size
        # create layers
        self.hidden = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))

        return self.output(x)


class nnapprox:
    """
    This class implements a pipeline to preprocess input, to feed it to a neural network (general function approximator), and to postprocess the output.
    Example
    -------
    initialize function approximation
    >> func = nnfunction(input=['lambda', 'window', 'h'], output=['a', 'b'])
    specify a map from input to NN (inverse not needed): since parameters are logarithmically distributed we map them to log scale (for lambda we use 1-lambda because we are interested in getting close to 1)
    >> func.set_map('lambda', lambda x: np.log10(1-x))
    >> func.set_map('window', lambda x: np.log10(x))
    >> func.set_map('h',      lambda x: np.log10(x))
    specify a map from NN to output and its inverse (for training!)
    >> func.set_map_output('a', lambda Y: 10**Y, lambda y: np.log10(y))
    >> func.set_map_output('b', lambda Y: 10**Y, lambda y: np.log10(y))
    training (takes care of scaling and shuffling)
    >> loss = func.train(dataframe, epochs=10000, batch_size=300)
    usage
    >> func([0.9,1,1e-3]) # returns a,b

    Parameters
    ----------
    input: list
        List of input variables.
    output: list
        List of output variables.
    """

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        if "params" in kwargs:
            self.params = kwargs["params"]
        else:
            self.params = None
        if "filename" in kwargs:
            filename = kwargs["filename"]
            if self.verbose:
                print(
                    f"Loading function approximation from {filename} and ignore additional arguments!"
                )
            self.load(filename)
        else:
            self.init(**kwargs)

    def init(self, **kwargs):
        self.input_names = kwargs["input"]
        self.output_names = kwargs["output"]
        assert len(set(self.input_names).intersection(set(self.output_names))) == 0
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        self.map_input = [lambda x: x for i in range(self.input_dim)]
        self.map_output = [lambda x: x for i in range(self.output_dim)]
        self.map_output_inv = [lambda x: x for i in range(self.output_dim)]
        # ranges for input that the model is trained on (initialize with None)
        self.input_range = [(None, None) for i in range(self.input_dim)]

        if "model" in kwargs:
            self.model = kwargs["model"]
            assert self.model.input_size == self.input_dim
            assert self.model.output_size == self.output_dim
        else:
            if self.verbose:
                print(
                    "Initializing default model (choose own model with kwargs `model`)."
                )
            self.model = DenseNN(
                input_size=self.input_dim,
                hidden_sizes=[42, 42, 42],
                output_size=self.output_dim,
            )
        if self.verbose:
            print(self.model)

        # Attention: scaler has to match the activation function:
        # tanh: [-1,1]
        # sigmoid: [0,1]
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.Y_scaler = MinMaxScaler(feature_range=(-1, 1))

        if self.verbose:
            print(
                "Next steps:\n>> func.set_map() # optimize parameter distribution over activation function domain (input) and range (output) \n>> func.train() # train neural network model\n>> func.save(filename) # save function approximation to file"
            )

    def set_map_to_NN(self, label, func):
        """
        Specifies a mapping for parameter `name` from either input or output to the NN.
        The pipeline is thus as follows: input -> func -> NN -> func_inv -> output
        """
        if label in self.input_names:
            i = self.input_names.index(label)
            self.map_input[i] = func
        elif label in self.output_names:
            i = self.output_names.index(label)
            self.map_output[i] = func

    def set_map_from_NN(self, label, func):
        """
        Specifies a mapping for parameter `name` from either input or output to the NN.
        The pipeline is thus as follows: input -> func -> NN -> func_inv -> output
        """
        if label in self.output_names:
            i = self.output_names.index(label)
            self.map_output_inv[i] = func
        elif label in self.input_names:
            raise ValueError("Inverse mapping only required for output variables!")
        
    def prepare_data(self, dataframe):
        assert self.model.training
        
        # extract training data from dataframe into numpy arrays
        Xs = dataframe[self.input_names].values
        Ys = dataframe[self.output_names].values

        # specify input ranges used for training
        for i in range(self.input_dim):
            self.input_range[i] = (np.min(Xs[:, i]), np.max(Xs[:, i]))

        # map inputs and outputs so that data is equidistant
        # (e.g. into logspace if data comes from a log-normal distribution)
        for i in range(self.input_dim):
            Xs[:, i] = self.map_input[i](Xs[:, i])
        for i in range(self.output_dim):
            Ys[:, i] = self.map_output[i](Ys[:, i])

        # rescale data into [-1,1] range (this restricts the fit to the range of the training data!!)
        Xs = self.X_scaler.fit_transform(Xs)
        Ys = self.Y_scaler.fit_transform(Ys)

        return Xs, Ys

    def train(self, Xs, Ys, custom_loss=None, epochs=1000, lr=0.005, device=None):
        """
        Trains the neural network.
        """
        # select suitable device for training
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if self.verbose:
            print("Device for training: {}".format(device))

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float().to(device)
        Y_tensor = torch.from_numpy(Ys).float().to(device)
        model_ = self.model.to(device)
        if self.verbose:
            print(
                f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}."
            )

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction="mean")
        if custom_loss is None:
            custom_loss = lambda Y_pred, Y, X: loss_fn(Y_pred, Y)

        history_loss = []
        for epoch in tqdm(range(epochs), desc="Training"):
            # forward pass
            Y_pred = model_(X_tensor)
            loss = custom_loss(Y_pred, Y_tensor, X_tensor)
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            # for logging
            history_loss.append(loss.item())
        # fetch model from device
        self.model = model_.to("cpu")
        # get data from device (does this free memory on the device?)
        Xs = X_tensor.to("cpu").detach().numpy()
        Ys = Y_tensor.to("cpu").detach().numpy()
        # set training to false to enter evaluation mode
        self.model.train(False)
        if self.verbose:
            print(
                "Training complete; left training mode for faster evaluation! To re-enter training mode call `func.model.train(True)`."
            )

        return history_loss

    def save(self, filename):
        """
        Saves the function approximation to a pickle file.
        """

        # step-by-step save all the relevant objects for running the function approximation (not training!)
        def to_string(lambda_function):
            # getsource returns "func.set_map_to_NN('lambda', lambda x: f(x) )\n" or "func.set_map_from_NN('lambda', lambda x: f_inv(x) )"
            # we want to return "lambda x: f(x)"
            return inspect.getsource(lambda_function).split(",")[1][:-2]

        # create datastructure that can be iteratively passed to pickle.dump
        data = {}
        data["params"] = self.params
        data["input_names"] = self.input_names
        data["output_names"] = self.output_names
        # convert lambda functions to strings
        data["map_input"] = [to_string(m) for m in self.map_input]
        data["map_output"] = [to_string(m) for m in self.map_output]
        data["map_output_inv"] = [to_string(m) for m in self.map_output_inv]
        data["X_scaler"] = self.X_scaler
        data["Y_scaler"] = self.Y_scaler
        data["model"] = self.model
        data["input_range"] = self.input_range

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Loads class objects from a pickle file.
        """
        data = pickle.load(open(filename, "rb"))
        self.params = data["params"]
        self.input_names = data["input_names"]
        self.output_names = data["output_names"]
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        # convert strings back to lambda functions
        self.map_input = [eval(m) for m in data["map_input"]]
        self.map_output = [eval(m) for m in data["map_output"]]
        self.map_output_inv = [eval(m) for m in data["map_output_inv"]]
        self.X_scaler = data["X_scaler"]
        self.Y_scaler = data["Y_scaler"]
        self.model = data["model"]
        self.input_range = data["input_range"]
        self.model.train(False)

    def __call__(self, *x):
        """
        Returns the approximation of y1,y2, ... = func(x1,x2,...) where x1,x2 are the arguments in the same order as specified in `func.input_names` and y1, y2, ... are the outputs in the same order as specified in `func.output_names`.

        xi currently have to be lists or numpy arrays of the same length.
        """
        if self.model.training:
            raise ValueError("FunctionApproximation has not finished training yet!")

        x = list(x)

        if len(x) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input values, but got {len(x)}!"
            )

        # iterate over inputs to transform scalars into arrays and apply mapping
        for i in range(self.input_dim):
            if np.isscalar(x[i]):
                x[i] = np.array([x[i]])
            else:
                x[i] = np.array(x[i])

            # apply mapping
            x[i] = self.map_input[i](x[i])

        # stack input list into one array
        X = np.stack(x, axis=1)

        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.model(torch.from_numpy(X_scaled).float()).detach().numpy()
        Y = self.Y_scaler.inverse_transform(Y_scaled)

        # unstack output array into separate arrays
        y = [Y[:, i] for i in range(self.output_dim)]
        # apply inverse mapping
        for i in range(self.output_dim):
            y[i] = self.map_output_inv[i](y[i])
            # if output is a scalar, return scalar
            if len(y[i]) == 1:
                y[i] = y[i][0]

        return tuple(y)