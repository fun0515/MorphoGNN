# MorphEmbedding
Morphological embedding for single neuron with graph neuron networks.
## Data preprocessing
Make sure that the morphology data consists of `.swc` files, each of which contains the morphological structure of a single neuron.  
```python
run SWC2H5PY.py
```
to generate trainable corresponding point cloud datasets. This dataset is stored in `.h5py` files, with the points of neurons filled up to a same limit. Don't forget to modify ```neuron_list``` for your own `.swc` directory.
## Train MorphoGNN
Replace the file path in `dataSet.py` with the dataset you generated in the previous step. Then
```python
run MorphoGNN.py
```
to train the MorphoGNN model. `num_classes` needs to be set according to your own morphology category.
## Retrieval
`retrieval.py` helps you retrieve nerve fibers based on the MorphoGNN you trained. Firstly, you should 
```python
run ExtractFeature()
```
to build a feature library for each neuron, which is saved as `.npy` file. Then 
```python
run QueryTest(feature_library, rounds)
```
to retrieve the most similar neurons in this library by `rounds` times. Or you can
```python
run Tsne(feature_library)
```
to visulize features distribution.
## Morphometrics
We also provide an example of classifying neurons using several traditional morphometrics in `morphometrics.py`. Morphometrics are captured through [NeuroM](https://github.com/BlueBrain/NeuroM).
```python
run morphometrics.py
```
to genrate datasets with traditional morphometrics and train a simple multilayer perceptron to classify.
