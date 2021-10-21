# tiralabra

Run the program with python src/main.py
Running the program will train a CNN-neural net with training size of 500 and show training and validation error

Test the program with: pytest

Check code style: pylint

Required libraries: pytest, numpy, matplotlib, pylint

The program takes 5 command line arguments:

first is the training size, maximum is 50000

Second is the number of epochs, meaning iterations, to train

Third is the batch size ie. how many data points to train in batches

Fourth is the optimization algorithm, which can be: adam, sgd, sgd_momentum, rmsprop

And the final is the learning rate meaning how fast the gradient descent is done.

For example run use: python main.py 50 15 50 adam 1e-3. If you want the whole data set increase train size to 50000
