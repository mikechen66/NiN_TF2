# keras_nin_tf2
Build the script based on Keras 2.4.3 and TensorFlow 2.2

NIN(Network in Network) is the creative DNN model in which GAN(Global Averag Pooling) is adpted to elimitate the 
large quantity of parameters. It deals with 10 label classes.  In contrast, AlexNet deep learning structure is a 
combination of CNN+FC that incurs a huge RAM occupation. CNN is responsible for extracting features, and FC is 
responsible for feature classification. NIN uses mlpconv and GAP to organicall combine the two parts of CNN and 
streamline FC with making it more interpretable.
