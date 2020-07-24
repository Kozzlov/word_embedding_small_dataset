import io
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras as k
from tensorflow.keras import layers 
import tensorflow_datasets as tfds

#data shuffling 
def get_batch_data():
  #creating and splitting datasets
  (training_set, test_set), info = tfds.load('imdb_reviews/subwords8k', 
                                            split = (tfds.Split.TRAIN, tfds.Split.TEST), 
                                            with_info = True,
                                            as_supervised = True)
  encoder = info.features['text'].encoder
  #print(encoder.subwords[:20])
  padded_shapes = ([None], ())
  training_betches = training_set.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
  test_betches = test_set.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
  return training_betches, test_betches, encoder

#creating model itself
def get_model(encoder, embedding_dim=16):
  model = k.Sequential([
            layers.Embedding(encoder.vocab_size, embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
  return model

#
def plot_data(history):
  history_dict = history.history
  accur = history_dict['accuracy']
  val_accur = history_dict['val_accuracy']
  epochs = range(1, len(accur) + 1)

  plt.figure(figsize = (12, 9))
  plt.plot(epochs, accur, 'bo', label = 'Training accur')
  plt.plot(epochs, val_accur, 'b', label = 'Validation accur')
  plt.title(' Training & validation accuracy ')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc = 'lower right')
  plt.ylim((0.5, 1))
  plt.show()

#visualize the embeddings
def retrieve_embeddings(model, encoder):
  out_vectors = io.open('vecs.tsv', 'w', encoding = 'utf-8')
  out_metadata = io.open('meta.tsv', 'w', encoding = 'utf-8')
  weights = model.layers[0].get_weights()[0]

  for number, word in enumerate(encoder.subwords):
    vector = weights[number+1]
    out_metadata.write(word + '\n')
    out_vectors.write('\t'.join([str(x) for x in vector]) + '\n')
  out_vectors.close()
  out_metadata.close()


training_betches, test_betches, encoder=get_batch_data()
model=get_model(encoder)
history=model.fit(training_betches, 
                  epochs = 5, #10 provokes overfitting
                  validation_data = test_betches,
                  validation_steps = 20)

#plot_data(history) model performance

retrieve_embeddings(model, encoder)
