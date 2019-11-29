# 1) Final Validation accuracy for Base Network: 83.29

----

# 2) My Model definition with output channel size and receptive field


weight_decay = 1e-4

### Define the model 

model = Sequential()

### Layer 1 

#### Output size - (32,32,48) 

#### Receptive Field size - 3 x 3 

model.add(SeparableConv2D(48, 3, use_bias=False, padding='same', input_shape=(32,32,3), activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



### Layer 2 

#### Output size - (32,32,48)

#### Receptive Field size - 5 x 5

model.add(SeparableConv2D(48, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay))) 

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 3 

#### Output size - (32,32,96)

#### Receptive Field size - 7 x 7

model.add(SeparableConv2D(96, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 4 

#### Output size - (32,32,96) 

#### Receptive Field size - 9 x 9 

model.add(SeparableConv2D(96, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 5 

#### Output size - (32,32,192) 

#### Receptive Field size - 11 x 11 

model.add(SeparableConv2D(192, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 6 

#### Output size - (32,32,156)

#### Receptive Field size - 13 x 13 

model.add(SeparableConv2D(156, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 7 

#### Output size - (32,32,96) 

#### Receptive Field size - 15 x 15 

model.add(SeparableConv2D(96, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(BatchNormalization())

model.add(Dropout(0.2))


### Layer 8 

#### Output size - (32,32,48) 

#### Receptive Field size - 17 x 17 

model.add(SeparableConv2D(48, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))


### Layer 9 

#### Output size - (32,32,10) 

#### Receptive Field size - 19 x 19 

model.add(SeparableConv2D(num_classes, 3, use_bias=False, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))


model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))

----

# 3) 50 epoch logs

##### The best validation accuracy is 85.08 at 48 epoch.

##### The model's test accuracy is 84.10 .

Epoch 1/50

390/390 [==============================] - 46s 117ms/step - loss: 1.6997 - acc: 0.3726 - val_loss: 1.4334 - val_acc: 0.5010

Epoch 2/50

390/390 [==============================] - 40s 103ms/step - loss: 1.2838 - acc: 0.5394 - val_loss: 1.2276 - val_acc: 0.5670

Epoch 3/50

390/390 [==============================] - 40s 103ms/step - loss: 1.1409 - acc: 0.5915 - val_loss: 1.1687 - val_acc: 0.6125

Epoch 4/50

390/390 [==============================] - 40s 102ms/step - loss: 1.0475 - acc: 0.6265 - val_loss: 1.0061 - val_acc: 0.6539

Epoch 5/50

390/390 [==============================] - 40s 103ms/step - loss: 0.9846 - acc: 0.6516 - val_loss: 0.9781 - val_acc: 0.6673

Epoch 6/50

390/390 [==============================] - 40s 102ms/step - loss: 0.9274 - acc: 0.6730 - val_loss: 0.9190 - val_acc: 0.6920

Epoch 7/50

390/390 [==============================] - 40s 103ms/step - loss: 0.8790 - acc: 0.6941 - val_loss: 0.9153 - val_acc: 0.7050

Epoch 8/50

390/390 [==============================] - 40s 102ms/step - loss: 0.8426 - acc: 0.7056 - val_loss: 0.8835 - val_acc: 0.7153

Epoch 9/50

390/390 [==============================] - 40s 103ms/step - loss: 0.8055 - acc: 0.7185 - val_loss: 0.9261 - val_acc: 0.7083

Epoch 10/50

390/390 [==============================] - 40s 102ms/step - loss: 0.7751 - acc: 0.7297 - val_loss: 0.8265 - val_acc: 0.7327

Epoch 11/50

390/390 [==============================] - 40s 102ms/step - loss: 0.7571 - acc: 0.7353 - val_loss: 0.7385 - val_acc: 0.7585

Epoch 12/50

390/390 [==============================] - 40s 102ms/step - loss: 0.7275 - acc: 0.7466 - val_loss: 0.7498 - val_acc: 0.7554

Epoch 13/50

390/390 [==============================] - 40s 101ms/step - loss: 0.7069 - acc: 0.7527 - val_loss: 0.8119 - val_acc: 0.7445

Epoch 14/50

390/390 [==============================] - 40s 102ms/step - loss: 0.6953 - acc: 0.7591 - val_loss: 0.6734 - val_acc: 0.7765

Epoch 15/50

390/390 [==============================] - 40s 102ms/step - loss: 0.6789 - acc: 0.7644 - val_loss: 0.6297 - val_acc: 0.7896

Epoch 16/50

390/390 [==============================] - 40s 102ms/step - loss: 0.6641 - acc: 0.7694 - val_loss: 0.6731 - val_acc: 0.7853

Epoch 17/50

390/390 [==============================] - 39s 101ms/step - loss: 0.6519 - acc: 0.7728 - val_loss: 0.5972 - val_acc: 0.8058

Epoch 18/50

390/390 [==============================] - 40s 102ms/step - loss: 0.6333 - acc: 0.7803 - val_loss: 0.6134 - val_acc: 0.8053

Epoch 19/50

390/390 [==============================] - 39s 101ms/step - loss: 0.6237 - acc: 0.7835 - val_loss: 0.6000 - val_acc: 0.8040

Epoch 20/50

390/390 [==============================] - 40s 102ms/step - loss: 0.6072 - acc: 0.7888 - val_loss: 0.6136 - val_acc: 0.8030

Epoch 21/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5979 - acc: 0.7924 - val_loss: 0.6485 - val_acc: 0.7927

Epoch 22/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5891 - acc: 0.7951 - val_loss: 0.6322 - val_acc: 0.7959

Epoch 23/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5887 - acc: 0.7966 - val_loss: 0.6501 - val_acc: 0.7915

Epoch 24/50

390/390 [==============================] - 40s 102ms/step - loss: 0.5682 - acc: 0.8007 - val_loss: 0.5828 - val_acc: 0.8115

Epoch 25/50

390/390 [==============================] - 40s 102ms/step - loss: 0.5659 - acc: 0.8032 - val_loss: 0.5730 - val_acc: 0.8161

Epoch 26/50

390/390 [==============================] - 40s 102ms/step - loss: 0.5595 - acc: 0.8044 - val_loss: 0.5414 - val_acc: 0.8239

Epoch 27/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5537 - acc: 0.8055 - val_loss: 0.5562 - val_acc: 0.8218

Epoch 28/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5440 - acc: 0.8105 - val_loss: 0.5820 - val_acc: 0.8161

Epoch 29/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5366 - acc: 0.8135 - val_loss: 0.6569 - val_acc: 0.8013

Epoch 30/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5342 - acc: 0.8138 - val_loss: 0.6388 - val_acc: 0.8038

Epoch 31/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5268 - acc: 0.8168 - val_loss: 0.5519 - val_acc: 0.8191

Epoch 32/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5177 - acc: 0.8211 - val_loss: 0.5534 - val_acc: 0.8200

Epoch 33/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5152 - acc: 0.8215 - val_loss: 0.5645 - val_acc: 0.8178

Epoch 34/50

390/390 [==============================] - 40s 101ms/step - loss: 0.5104 - acc: 0.8233 - val_loss: 0.5586 - val_acc: 0.8235

Epoch 35/50

390/390 [==============================] - 39s 101ms/step - loss: 0.5057 - acc: 0.8238 - val_loss: 0.5499 - val_acc: 0.8248

Epoch 36/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4936 - acc: 0.8283 - val_loss: 0.6019 - val_acc: 0.8169

Epoch 37/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4930 - acc: 0.8276 - val_loss: 0.5436 - val_acc: 0.8276

Epoch 38/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4894 - acc: 0.8311 - val_loss: 0.5334 - val_acc: 0.8334

Epoch 39/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4869 - acc: 0.8300 - val_loss: 0.6030 - val_acc: 0.8126

Epoch 40/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4827 - acc: 0.8321 - val_loss: 0.5709 - val_acc: 0.8236

Epoch 41/50

390/390 [==============================] - 40s 101ms/step - loss: 0.4758 - acc: 0.8347 - val_loss: 0.5677 - val_acc: 0.8236

Epoch 42/50

390/390 [==============================] - 40s 101ms/step - loss: 0.4722 - acc: 0.8350 - val_loss: 0.5498 - val_acc: 0.8305

Epoch 43/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4724 - acc: 0.8347 - val_loss: 0.5284 - val_acc: 0.8356

Epoch 44/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4677 - acc: 0.8365 - val_loss: 0.5525 - val_acc: 0.8301

Epoch 45/50

390/390 [==============================] - 40s 101ms/step - loss: 0.4608 - acc: 0.8389 - val_loss: 0.5322 - val_acc: 0.8377

Epoch 46/50

390/390 [==============================] - 40s 102ms/step - loss: 0.4569 - acc: 0.8428 - val_loss: 0.6066 - val_acc: 0.8194

Epoch 47/50

390/390 [==============================] - 40s 101ms/step - loss: 0.4524 - acc: 0.8429 - val_loss: 0.5493 - val_acc: 0.8352

Epoch 48/50

390/390 [==============================] - 39s 101ms/step - loss: 0.4537 - acc: 0.8432 - val_loss: 0.4765 - val_acc: 0.8508

Epoch 49/50

390/390 [==============================] - 40s 102ms/step - loss: 0.4460 - acc: 0.8438 - val_loss: 0.5268 - val_acc: 0.8312

Epoch 50/50

390/390 [==============================] - 40s 102ms/step - loss: 0.4507 - acc: 0.8433 - val_loss: 0.5065 - val_acc: 0.8410
