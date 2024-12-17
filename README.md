## **MTFNet**

We perform deep feature extraction by integrating three types of features, enabling the identification of gas types and concentration prediction for 10 gases within the dynamic concentration range of 10ppm to 50ppm. Detailed information about this method can be found in our paper. The Python_Code folder contains data and related the py files to simulate the training and validation process of the proposed method. The Data folder includes three Excel files, each corresponding to the electronic nose response data under three temperature modulation methods used in the experiments. The MTFNet.py file implements the MTFNet method described in the paper.

## **Installation**

This is to clarify that the newly added code must be executed in a Python environment. If running the code in a different environment, it is necessary to install the required packages, including tensorflow, numpy, scikit-learn, and pandas.

## **Usage**

Download the code and run Run_Network.py directly. You can freely adjust parameters such as the sliding window size, number of iterations, batch size, kernel size, filter size, and learning rate, and even define the MTFNet configuration you desire.