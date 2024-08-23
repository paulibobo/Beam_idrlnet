## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  The available code was developed using Python 3.8.10.
>📋  To run the code the libraries available in the "requirements.txt" file were used, with the referred version.
>📋  It is recommended to install them with the given command and with the same version of each library, although other versions might still work.

## Training

To train the model(s) in the paper, run this command:

```train
python beam_idrlnet.py -max_iter 5000 -plots True/False -savepath <path to save files to/>
```
>📋 The input parameters for the training script are:
    	* -max_iter: Max number if iterations for the model
   	* -plots: Whether to plot the inputs and outputs from the dataset or not
	* -savepath: The path where to save the weights, model and training history

    


