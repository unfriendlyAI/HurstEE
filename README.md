# HurstEE
Differentiable Neural Network Layer for Estimating Hurst and Anomalous Diffusion Exponents, TensorFlow and PyTorch

Roman Lavrynenko, Lyudmyla Kirichenko, Nataliya Ryabova and Sophia Lavrynenko

Neural networks have shown excellent performance in the task of estimating the Hurst exponent, but their primary drawback is a lack of explainability. We introduce a specialized neural network layer to address this. This layer is an implementation of the second-order Generalized Hurst Exponent method, designed as a non-trainable, differentiable layer compatible with any deep learning framework. We have implemented it for both TensorFlow and PyTorch. The layer also seamlessly handles missing values, which facilitates its integration into complex neural network architectures designed for analyzing heterogeneous trajectories. Our differentiable Hurst exponent estimation layer offers simplicity in deployment, as it eliminates the need for training and is ready to process time series of any length. While the convenience of not requiring training and its flexibility with series length are clear advantages, the key novelty of our work is the ability to provide interpretability by successfully translating a classical statistical method into a core deep learning component.

http://dx.doi.org/10.1088/1751-8121/ae17f9

Kaggle code:

https://www.kaggle.com/code/unfriendlyai/hurstee-layer-for-estimating-hurst

https://www.kaggle.com/code/unfriendlyai/hurstautograd
