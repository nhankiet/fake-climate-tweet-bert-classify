# fake-climate-tweet-bert-classify
Using BERT language model to classify fake climate change tweets. Natural Language Processing subject from UniMelb

<br>
<br>

For implementation
1. Extract the zip files

2. Go into the folder and open a command line

3. Make sure anaconda is installed, type in the command line

	conda create -n transformers python pandas tqdm

	conda activate transformers

if with GPU

    conda install pytorch cudatoolkit=10.1 -c pytorch

if no GPU

    conda install pytorch cpuonly -c pytorch

Lastly

	pip install simpletransformers

4. Run the file "not_misinformation_preprocessing.py" to (re)create the file "train_not_misinfo"

5. Run the file "training.py" with input parameter 0 for begin training anew, another number other than 0 to continue training
   <br>
   e.g.: "python training.py 0" -> training anew
   <br>
   e.g.: "python training.py 1" -> continue training
   <br>
   The output model or weight will be putting in /outputs/ folder in the same directory, the output models will consume a huge storage capacity

6. Run the file "predicting.py" to get the predictions from "test-unlabelled.json -> test-output.json" and "dev.json -> dev-output.json "

7. Type in command line "python scoring.py --groundtruth dev.json --predictions dev-output.json" to see the performance of the model
   against the dev dataset.



Specs was use for this implementation:
<br>
CPU: Intel Core I7-9750H - GPU: NVIDIA RTX-2070
<br>
OS: Windows 10 Home
<br>
Anaconda 4.8.3, Python 3.7.3
<br>
CUDA cuda_10.1.105_418.96_win10
<br>
cuDNN cudnn-10.1-windows10-x64-v7.6.5.32

END




