You can find find htmlcov inside this folder that includes detailed testing report.
Index.html includes the full report

Testing is mainly focused on testing that the layers of the neural net compute what they are supposed to compute.
This is achieved in two main ways. First so called forward pass is tested against simulated values which were either easily simulated, max_pooling, ReLU, or against other public implementations.

Secondly so called backward pass is tested with gradient checking. Since they should calculate gradients it is possible to simulate the derivatives with small differences in the inputs and see how they change.

Testing is done with numpy array inputs that they should get in runtime-

All tests can be run by entering to the root folder on typing "pytest".

After I have done the neural net will put some slides here.