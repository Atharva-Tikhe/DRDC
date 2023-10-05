
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow_datasets as tfds

tf.config.run_functions_eagerly(True)

# Load the AG News dataset from TensorFlow Datasets
data, info = tfds.load("ag_news_subset", as_supervised=True, with_info=True)

# Split the dataset into train and test sets
train_data, test_data = data['train'], data['test']

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and preprocess the data
def tokenize_reviews(text, label):
    # Tokenize the text using the tokenizer
    inputs = tokenizer(text.numpy(), padding='max_length', truncation=True, max_length=128, return_tensors='tf', return_token_type_ids=False, return_attention_mask=False, verbose=False)

    return inputs, label

# Apply tokenization and preprocessing to train_data
train_data = train_data.map(tokenize_reviews, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and shuffle the data
batch_size = 32
train_data = train_data.shuffle(10000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Tokenize and preprocess the test data similarly
test_data = test_data.map(tokenize_reviews, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size)

# Load the pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)  # AG News has 4 classes

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=3, validation_data=test_data)

# Evaluate the model
results = model.evaluate(test_data)
print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")