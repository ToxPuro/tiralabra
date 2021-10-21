The program 

The program is compromised of four main elements. Layers of the neural net, the neural net that uses layers, a solver object that gets a model that it trains according to
training rules in optim-module.



Time complexity is rather surprisingly either O(n*t(ij + jk)) O(n*t), depending on the viewpoint we view the runtime.
If we keep the 3-layered architecture fixed time complexity is O(n*t) where n is the size of training set and t epoch size, since all the same matrix multiplications are done for all data points
If we also thing of the layers as variables we get O(n*t(ij+jk))) where i,j and k are node sizes of the layers.
(read more here if interested https://ai.stackexchange.com/questions/5728/what-is-the-time-complexity-for-training-a-neural-network-using-back-propagation/20281#20281)

The program could get some kind of input from the user that it tries to classify, but now it just trains itself and plots accuracy.


Sources:

https://arxiv.org/abs/1502.03167
https://arxiv.org/abs/1502.03167
https://towardsdatascience.com/lets-code-convolutional-neural-network-in-plain-numpy-ce48e732f5d5
