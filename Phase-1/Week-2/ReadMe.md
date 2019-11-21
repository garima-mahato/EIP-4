### 1) Logs for 20 epochs

Train on 60000 samples, validate on 10000 samples

Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.

60000/60000 [==============================] - 11s 178us/step - loss: 0.2443 - acc: 0.9218 - val_loss: 0.0592 - val_acc: 0.9808


Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0571 - acc: 0.9821 - val_loss: 0.0359 - val_acc: 0.9890

Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0430 - acc: 0.9865 - val_loss: 0.0276 - val_acc: 0.9902

Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0368 - acc: 0.9884 - val_loss: 0.0233 - val_acc: 0.9930

Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.

60000/60000 [==============================] - 5s 87us/step - loss: 0.0336 - acc: 0.9893 - val_loss: 0.0319 - val_acc: 0.9900

Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.

60000/60000 [==============================] - 5s 87us/step - loss: 0.0288 - acc: 0.9911 - val_loss: 0.0248 - val_acc: 0.9921

Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0256 - acc: 0.9921 - val_loss: 0.0211 - val_acc: 0.9934

Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0246 - acc: 0.9925 - val_loss: 0.0199 - val_acc: 0.9942

Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.

60000/60000 [==============================] - 5s 87us/step - loss: 0.0234 - acc: 0.9930 - val_loss: 0.0202 - val_acc: 0.9945

Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.

60000/60000 [==============================] - 5s 90us/step - loss: 0.0225 - acc: 0.9933 - val_loss: 0.0218 - val_acc: 0.9934

Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.

60000/60000 [==============================] - 5s 90us/step - loss: 0.0200 - acc: 0.9940 - val_loss: 0.0244 - val_acc: 0.9922

Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.

60000/60000 [==============================] - 5s 91us/step - loss: 0.0194 - acc: 0.9940 - val_loss: 0.0180 - val_acc: 0.9942

Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0191 - acc: 0.9938 - val_loss: 0.0184 - val_acc: 0.9944

Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0179 - acc: 0.9943 - val_loss: 0.0215 - val_acc: 0.9942

Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0173 - acc: 0.9944 - val_loss: 0.0197 - val_acc: 0.9943

Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0179 - acc: 0.9943 - val_loss: 0.0176 - val_acc: 0.9949

Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.

60000/60000 [==============================] - 5s 87us/step - loss: 0.0147 - acc: 0.9951 - val_loss: 0.0208 - val_acc: 0.9939

Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0154 - acc: 0.9951 - val_loss: 0.0192 - val_acc: 0.9941

Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.

60000/60000 [==============================] - 5s 89us/step - loss: 0.0154 - acc: 0.9950 - val_loss: 0.0208 - val_acc: 0.9938

Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.

60000/60000 [==============================] - 5s 88us/step - loss: 0.0144 - acc: 0.9954 - val_loss: 0.0201 - val_acc: 0.9951
<keras.callbacks.History at 0x7f7d81ddc4a8>


---------

### 2) Result of model.evaluate(on test data)

[0.020135334616147155, 0.9951]


----------

### 3) Strategy taken

1) Since MNIST is image dataset, I used a CNN model having 3 convolutional layers of 16, 32, 64 kernels each involving 3 x 3 convolutions. This gave an accuracy of 96.6%.

2) **Increasing number of kernels** : On further improving the model by increasing number of kernels in each of the previous layers to 32-64-128, the accuracy improved to  97.14% but the number of parameters increased.

3) **Increasing number of layers** : On further improving the model by increasing number of layers from 3 to 4 each with 32-64-128-256, the accuracy improved to 98.82% but the number of parameters increased.

4) Adding a convolution block of 16-32 contributes to approx 5k parameters. Thus, adding 3 such blocks with 1x1 convolutions in between contributed to 19k paraemters with an accuracy of 96.5%.

5) Adding maxpooling layer would improve the receptive field and hence improves the accuracy. For adding maxpooling, number of convolution blocks will have to be reduced from 3 to 2.

6) Maxpooling should not be added just before the last layer. The resulting CNN model has an accuracy score of [0.04112879056819947, 0.986]

7) To improve the accuracy, we can add more layers. Since more blocks cannot be added, I added 1 3x3 convolution layer in the last block with 32 kernels. This crossed the 15k mark of number of parameters. So I changed both the 2nd last convolution layers' number of filters from 32 to 16. This still exceeded. So, I changed the last layers' number of kernels to 16. This had 12,336 parameters with validation score of [0.03090761237440747, 0.9909].

8) Going through the logs shows decrease in validation accuracy as compared to training accuracy. This seems to be a little overfit. So, added Batch Normalization and dropout after each 3x3 convolution. This gave an accuracy score of [0.021835740570277266, 0.9927].

9) Removed dropout from last layer. This improved accuracy to 99.36%.

10) Added 1 more convolution layer with 16 kernels before the last layer since the number of parameters was around 12k. This gave a score of [0.01818742549381859, 0.9947]. But this also increased the total number of parameters to 15,088.

11) So, I removed the last batch normalisation which reduced the total number of parameters to 14,960. After training, the validation score turned out to be [0.020135334616147155, 0.9951]. Thus, achieving **99.51% Vacc**.

