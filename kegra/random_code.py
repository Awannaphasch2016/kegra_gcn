import matplotlib.pyplot as plt
from keras.datasets import mnist

from keras.layers import Input, Dropout, Dense, Activation
from keras.models import Model, Sequential

from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import sys
import tensorflow as tf

# tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)
# y_train = x_train.reshape(60000,784)
# print(y_train.shape) # 60000,28,28
x_train = x_train /255

from sklearn.preprocessing import OneHotEncoder
# enc.transform([['Female', 1], ['Male', 4]]).toarray()
y_train = [[x] for x in y_train]
# print(y_train)
# exit()

y_train = OneHotEncoder().fit_transform(y_train).toarray()



##################
# below model works
# sequential
##################
# model = Sequential()
# model.add(Dense(32, input_dim=784, activation = 'relu', name ="Dense_layer1"))
# model.add(Dense(16, activation = 'relu', name ="Dense_layer2"))
# model.add(Dense(10, activation = 'softmax', name ="Dense_layer3"))
#
# print(type(model))
# # intermediate_layer_model =  model(inputs  = model.input)
# intermediate_layer_model = model.get_layer("Dense_layer2").output
# identity = K.identity(intermediate_layer_model)

##########
# functional api
##########
inputs = Input(shape= (784, ), name="input")
x = Dense(32, activation = 'relu', name ="Dense_layer1")(inputs)
x = Dense(16, activation = 'relu', name="Dense_layer2")(x)
output = Dense(10, activation = 'softmax', name="Dense_layer3")(x)

model = Model(inputs=[inputs],outputs=[output])
model.compile(optimizer='rmsprop',
             loss ='categorical_crossentropy',
             metrics=['accuracy'])

# print(model.summary())

# intermediate_layer_model = model.get_layer("Dense_layer3").output
# identity = K.identity(intermediate_layer_model)

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
model.fit(x_train,y_train, batch_size=32, epoch = 3)

# x_placeholder = K.variable(value = x_train)
# y_placeholder = K.variable(value = y_train)
# #model.fit(x_placeholder,y_placeholder, batch_size=32, epochs = 3)
# model.fit(x_placeholder,y_placeholder, steps_per_epoch= 32, epochs = 3)

#######3
# Convert Tensor Object to array using K.function
########3
# f= K.function([intermediate_layer_model], [identity])
# print(f([x_train])[0].shape) # I don't know how to pass input variable to functional api.
# exit()

############33
# Convert Tensor Object to array by RUNING A SESSION
###########3#
# # t = tf.constant(42.0)
# sess = tf.Session()
# with sess.as_default():
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(intermediate_layer_model.eval(session=sess))
#     # print(intermediate_layer_model.eval(session= sess,
#     #                                     feed_dict={x_placeholder: x_train,
#     #                                                y_placeholder: y_train}))
#     # print_op = tf.print(intermediate_layer_model, output_stream=sys.stderr)
#     # sess.run(print_op)
# exit()

############
#### Plot t-sne of the last layer output. ( if not last layer's output, which one is it?)
#> plot animation of tsne of the last layer output for the end of batch process.
############



############
# # Plot training & validation accuracy values
############
# plt.plot(history.history['categorical_accuracy'])
# plt.plot(history.history['val_categorical_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
