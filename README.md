# TensorFlow
Repo con material para las sesiones de estudio de TensorFlow.

<div style="margin: 0 auto">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" />
</div>

## Prerrequisitos
- [Colab: Python and Colab Primer](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb)
- [Python-StudyClub Data Engineering Latam 🐍](https://github.com/DataEngineering-LATAM/Python-StudyClub)
## Contenido

### Beginners

#### Quickstarts

- [TensorFlow 2 quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [TensorFlow 2 quickstart for experts](https://www.tensorflow.org/tutorials/quickstart/advanced)

#### ML basics with Keras

- [Basic classification: Classify images of clothing](https://www.tensorflow.org/tutorials/keras/classification)
- [Basic text classification](https://www.tensorflow.org/tutorials/keras/text_classification)
- [Text classification with TensorFlow Hub: Movie reviews](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
- [Basic regression: Predict fuel efficiency](https://www.tensorflow.org/tutorials/keras/regression)
- [Overfit and underfit](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
- [Save and load models](https://www.tensorflow.org/tutorials/keras/save_and_load)
- [Introduction to the Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)

#### Loading Data

- [Load CSV data](https://www.tensorflow.org/tutorials/load_data/csv)
- [Load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images)
- [Load video data](https://www.tensorflow.org/tutorials/load_data/video)
- [Load NumPy data](https://www.tensorflow.org/tutorials/load_data/numpy)
- [Load a pandas DataFrame](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe)
- [TFRecord and tf.train.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [Load text](https://www.tensorflow.org/tutorials/load_data/text)

### Deep Dive

- [Install TensorFlow with pip](https://www.tensorflow.org/install/pip)
- [Install TensorFlow with Docker](https://www.tensorflow.org/install/docker)

#### [TensorFlow basics](https://www.tensorflow.org/guide/basics)

- [Introduction to Tensors](https://www.tensorflow.org/guide/tensor)
- [Introduction to Variables](https://www.tensorflow.org/guide/variable)
- [Introduction to gradients and automatic differentiation](https://www.tensorflow.org/guide/autodiff)
- [Introduction to graphs and tf.function](https://www.tensorflow.org/guide/intro_to_graphs)
- [Introduction to modules, layers, and models](https://www.tensorflow.org/guide/intro_to_modules)
- [Basic training loops](https://www.tensorflow.org/guide/basic_training_loops)

#### Keras: The high-level API for TensorFlow

- [Keras Overview](https://www.tensorflow.org/guide/keras)
- [The Sequential model](https://www.tensorflow.org/guide/keras/sequential_model)
- [The Functional API](https://www.tensorflow.org/guide/keras/functional_api)
- [Training & evaluation with the built-in methods](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [Making new layers and models via subclassing](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing)
- [Save, serialize, and export models](https://www.tensorflow.org/guide/keras/serialization_and_saving)
- [Customizing Saving and Serialization](https://www.tensorflow.org/guide/keras/customizing_saving_and_serialization)
- [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers)
- [Customizing what happens in fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
- [Writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [Working with RNNs](https://www.tensorflow.org/guide/keras/working_with_rnns)
- [Understanding masking & padding](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding)
- [Writing your own callbacks](https://www.tensorflow.org/guide/keras/writing_your_own_callbacks)
- [Transfer learning & fine-tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Multi-GPU and distributed training](https://www.tensorflow.org/guide/keras/distributed_training)

## Glosario de Términos

- Feature: El(los) input(s) para nuestro modelo
- Examples: Un par de entrada/salida usados para el entrenamiento
- Labels: La salida del modelo
- Layer: Una colección de nodos conectados dentro de una red neuronal
- Model: La representación de nuestra red neuronal
- Dense and Fully Connected (FC): Cada nodo en una capa está conectada con cada nodo de la capa anterior.
- Weights and biases: Son variables internas del modelo
- Loss: La discrepancia entre la salida deseada y la real
- MSE: Error cuadrado de la media (Mean squared error), es un tipo de función de pérdida que cuenta un número pequeño de grandes discrepancias como algo peor que un gran número de pequeñas discrepancias.
- Gradient Descent: Un algoritmo que cambia las variables internas un poco cada vez para reducir la función de pérdida.
- Optimizer: Una implementación específica del algoritmo de gradiente descendiente. (Hay muchos algoritmos para esto. Un tipo de implementación considerada como "best practice" es “Adam” Optimizer, que significa ADAptive con Momentum.)
- Learning rate:  El "step size" para mejorar la pérdida durante el descenso del gradiente.
- Batch: El conjunto de ejemplos utilizados durante el entrenamiento de la red neuronal.
- Epoch: Un recorrido completo por todo el conjunto de datos de entrenamiento
- Forward propagation (forward pass): El cálculo de los valores de salida a partir de la entrada.
- Backpropagation (backward pass): El cálculo de los ajustes de las variables internas de acuerdo con el algoritmo optimizador, comenzando desde la capa de salida y retrocediendo a través de cada capa hasta la entrada.

## Otros Notebooks

- [The Basics: Training Your First Model](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb)
- [Simple Linear Regression with Synthetic Data](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb)
- [Linear Regression with a Real Dataset](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb)
- [Code examples](https://keras.io/examples/)

## Libros Recomendados

- [AI and Machine Learning for Coders](https://www.oreilly.com/library/view/ai-and-machine/9781492078180)
- [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632)
- [Deep Learning](https://www.deeplearningbook.org)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
- [Learning TensorFlow.js](https://www.oreilly.com/library/view/learning-tensorflowjs/9781492090786)
- [Deep Learning with JavaScript](https://www.manning.com/books/deep-learning-with-javascript)

## Playlists Recomendadas

- [TensorFlow in Google Colaboratory](https://www.youtube.com/playlist?list=PLQY2H8rRoyvyK5aEDAI3wUUqC_F0oEroL)
- [ML Zero to Hero](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwWuPiWnuTDBHe7I0fMSsfO)
- [Natural Language Processing (NLP) Zero to Hero](https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S)
- [Building recommendation systems with TensorFlow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz)
- [TensorFlow and Google Cloud](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwN2KcgCiApoDsVaxW64tNh)
- [Responsible AI](https://www.youtube.com/playlist?list=PLQY2H8rRoyvw40o-nd2CSrk-3JNMxW6er)

## Certificación

- [TensorFlow Developer Certificate](https://www.tensorflow.org/certificate)

---

## Sobre la comunidad Data Engineering Latam

Data Engineering Latam es la comunidad de datos más grande de América Latina cuya misión es promover el talento de la región a través de la difusión de charlas, talleres, grupos de estudio, ayuda colaborativa y la creación de contenido relevante.

<div style="margin: 0 auto">
  <img src="https://pbs.twimg.com/profile_images/1462605042444341249/xjZALInT_400x400.jpg" />
</div>

## Síguenos en nuestras redes oficiales

- [YouTube](https://youtube.com/c/dataengineeringlatam?sub_confirmation=1)
- [Medium](https://medium.com/@dataengineeringlatam)
- [Twitter](https://twitter.com/DataEngiLatam)
- [Instagram](https://instagram.com/dataengineeringlatam)
- [Facebook](https://facebook.com/dataengineeringlatam)
- [TikTok](https://www.tiktok.com/@dataengineeringlatam)
- [Slack](https://bit.ly/dataengineeringlatam_slack)
- [Telegram](https://t.me/dataengineeringlatam)
- [Linkedin](https://linkedin.com/company/data-engineering-latam)

## ¿Quieres dar charla en la comunidad? 

:microphone: Cuéntanos [aquí](https://docs.google.com/forms/d/e/1FAIpQLSd7CZgRxGHx-rRA7CyAeB0MxNPgVj5rCqQsrjrFiNYhoZxS1w/viewform)
