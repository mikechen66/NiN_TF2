# keras_nin_tf2
Build the script based on Keras 2.4.3 and TensorFlow 2.2

NIN(Network in Network) is the creative DNN model in which GAN(Global Averag Pooling) is adpted to elimitate the 
large quantity of parameters. It deals with 10 label classes. In contrast, AlexNet is a combination of CNN+FC 
that incurs a huge RAM occupation. 

CNN is responsible for extracting features, and FC is responsible for feature classification. NIN uses mlpconv 
and GAP to organicall combine the two parts of CNN and streamline FC with making it more interpretable.

Make the necessary changes to adapt to the new environment of TensorFlow 2.2, Keras 2.4.3, CUDA Toolkit 11.0, 
cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated code. I would 
like to thank all of the creators and interptretors for their contributions.
