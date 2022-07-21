import tensorflow as tf
print("TensorFlor version:", tf.__version__)

mnist = tf.keras.datasets.mnist

#Loading datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0


#Build a machine learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)])

predictions = model(x_train[:1]).numpy()
print(predictions)

print(tf.nn.softmax(predictions).numpy())

#Defining loss function

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1],predictions).numpy())

#Defining the ML model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#Train and evaluate the model
model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))
