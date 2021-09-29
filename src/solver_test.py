from helpers import rel_error
from solver import Solver
from cnn import ThreeLayerConvNet
from keras.datasets import mnist

def test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000,1,28,28).astype("float64")
    x_test = x_test.reshape(10000,1,28,28).astype("float64")

    num_train = 500

    data = {
        "X_train": x_train[:num_train],
        "y_train": y_train[:num_train],
        "X_val": x_test,
        "y_val": y_test
    }

    

    model = ThreeLayerConvNet(weight_scale=1e-2)

    solver = Solver(
        model,
        data,
        num_epochs=15,
        batch_size=50,
        update_rule='adam',
        optim_config={'learning_rate': 1e-3,},
        verbose=True,
        print_every=1
    )
    solver.train()
    ## should be able to overfit
    train_acc = solver.check_accuracy(data['X_train'], data['y_train'])
    assert rel_error(train_acc, 1.0) < 10**-10
    ## gets suprisingly good validation accuracy
    validation_acc = solver.check_accuracy(data["X_val"], data["y_val"])
    assert validation_acc > 0.8