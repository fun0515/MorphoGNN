# MorphEmbedding
Morphological embedding for single neuron with graph neuron networks.
## Data preprocessing
Make sure that the morphology data is composed of ''.swc'' files, each of which contains the morphological structure of a single neuron. Activate your python environment and using:
```python
python SWC2H5PY.py --swc_dir=./neuron7
```
to generate trainable corresponding point cloud datasets. ''--swc_dir'' is the path where you store the morphological data. Then you will see two ''.h5'' files in your current directory, the training set and the test set.
## Train MorphoGNN
When the trainable data is generated, run:
```python
python MorphoGNN.py
```
to train the MorphoGNN model. After running 50 epoches, the model file named ''MorphoGNN.t7'' appears in the current directory.
## Retrieval
''retrieval.py'' helps you retrieve nerve fibers based on the MorphoGNN model you trained. Firstly, you should run:
```python
python retrieval.py --task=ExtractFeature --model_path=./MorphoGNN.t7 --swc_dir=./neuron7
```
to build a feature library for each neuron, which is saved as ''.npy'' file. Then 
```python
python retrieval.py --task=QueryTest --query_times=100
```
to retrieve the most similar neurons in this library ''100'' times. Or you can using this command:
```python
python retrieval.py --task=Visualize
```
to visulize features distribution.
## Morphometrics
We also provide an example of classifying neurons using sixteen traditional morphometrics in ''morphometrics.py''. Morphometrics are captured through [NeuroM](https://github.com/BlueBrain/NeuroM). Run:
```python
python morphometrics.py --swc_dir=./neuron7
```
to genrate datasets with traditional morphometrics and train a simple multilayer perceptron to classify.
