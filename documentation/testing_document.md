You can find find htmlcov inside this folder that includes detailed testing report.
Index.html includes the full report

Testing is mainly focused on testing that the layers of the neural net compute what they are supposed to compute.
This is achieved in two main ways. First so called forward pass is tested against simulated values which were either easily simulated, max_pooling, ReLU, or against other public implementations.

Secondly so called backward pass is tested with gradient checking. Since they should calculate gradients it is possible to simulate the derivatives with small differences in the inputs and see how they change.

Neural net is tested with gradient checking and also checking that it can overfit a small amount of data.

Testing is done with numpy array inputs that they should get in runtime.

All tests can be run by entering to the root folder on typing "pytest".

500_train.png includes training plot with training size of 500 and validation of 10000. Number of epochs was 15
It took 103 seconds to train, training accuracy of 100% and validation accuracy of 89.8%

50000_train.png includes training plot with training size of 50000 and validation of 10000. Number of epochs was 15
It took 50 minutes to train, training accuracy 99.4% and validation accuracy of 98.4%



Batch size was 50 for both of them