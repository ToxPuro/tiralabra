import numpy as np
from cnn import ThreeLayerConvNet
from solver import Solver
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sys
import time





def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000,1,28,28).astype("float64")
    x_test = x_test.reshape(10000,1,28,28).astype("float64")
    num_train = int(sys.argv[1])

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
        num_epochs=int(sys.argv[2]),
        batch_size=int(sys.argv[3]),
        update_rule=sys.argv[4],
        optim_config={'learning_rate': float(sys.argv[5]),},
        verbose=True,
        print_every=1
    )
    start_time = time.time()
    solver.train()
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    print(
    "training accuracy:",
    solver.check_accuracy(data['X_train'], data['y_train'])
    )


if __name__ == "__main__":
    main()
