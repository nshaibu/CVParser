# CVParser
Python library for parsing resumes using natural language processing and machine learning.

### Setup
## Installation on Linux and Mac OS

* [Follow the guide here](https://help.github.com/articles/fork-a-repo) on how to clone or fork a repo
* [Follow the guide here](http://simononsoftware.com/virtualenv-tutorial/) on how to create virtualenv

* To create a normal virtualenv (example _myvenv_) and activate it (see Code below).

  ```
  $ virtualenv --python=python3 myvenv
  
  $ source myvenv/bin/activate

  (myvenv) $ pip install -r requirements.txt

### Usage

```
from cvparser.parser import CVParser

CVParser.download_nlk_data()


parser = CVParser(file_path="path/to/file.[pdf|doc|docx|png|jpeg]")
parser.parse()
print(parser.json())
```

### Re-training the Model
* `cd` into the `train` folder.
* Delete the folder `model` and the file `train.json`.
* Copy your new training data into the `train` folder. The train data must be in `json`. This can be generated using the data annotation tool called `Dataturk`. The file containing the training data must be named `train.json`.
* Then, start re-training the model by execute the python script in the `train` folder named `manual_training.py`.
* Then test your new model by #usage .
