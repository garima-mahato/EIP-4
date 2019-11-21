### 1) Logs for 20 epochs

Train on 60000 samples, validate on 10000 samples

Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.

60000/60000 [==============================] - 15s 253us/step - loss: 0.3612 - acc: 0.9446 - val_loss: 0.1144 - val_acc: 0.9795


Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.

60000/60000 [==============================] - 11s 183us/step - loss: 0.1013 - acc: 0.9820 - val_loss: 0.0730 - val_acc: 0.9838


Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.

60000/60000 [==============================] - 12s 198us/step - loss: 0.0718 - acc: 0.9853 - val_loss: 0.0472 - val_acc: 0.9890


Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.

60000/60000 [==============================] - 12s 197us/step - loss: 0.0601 - acc: 0.9861 - val_loss: 0.0381 - val_acc: 0.9901


Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.

60000/60000 [==============================] - 12s 202us/step - loss: 0.0515 - acc: 0.9879 - val_loss: 0.0342 - val_acc: 0.9909


Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.

60000/60000 [==============================] - 12s 200us/step - loss: 0.0447 - acc: 0.9891 - val_loss: 0.0340 - val_acc: 0.9908


Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.

60000/60000 [==============================] - 11s 181us/step - loss: 0.0412 - acc: 0.9898 - val_loss: 0.0260 - val_acc: 0.9929


Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.

60000/60000 [==============================] - 11s 182us/step - loss: 0.0379 - acc: 0.9903 - val_loss: 0.0276 - val_acc: 0.9922


Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.

60000/60000 [==============================] - 11s 178us/step - loss: 0.0361 - acc: 0.9906 - val_loss: 0.0289 - val_acc: 0.9925


Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.

60000/60000 [==============================] - 11s 176us/step - loss: 0.0331 - acc: 0.9916 - val_loss: 0.0246 - val_acc: 0.9932


Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.

60000/60000 [==============================] - 11s 180us/step - loss: 0.0322 - acc: 0.9913 - val_loss: 0.0256 - val_acc: 0.9927


Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.

60000/60000 [==============================] - 11s 180us/step - loss: 0.0307 - acc: 0.9916 - val_loss: 0.0230 - val_acc: 0.9928


Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.

60000/60000 [==============================] - 11s 177us/step - loss: 0.0294 - acc: 0.9922 - val_loss: 0.0214 - val_acc: 0.9934


Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.

60000/60000 [==============================] - 11s 187us/step - loss: 0.0282 - acc: 0.9927 - val_loss: 0.0225 - val_acc: 0.9933


Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.

60000/60000 [==============================] - 11s 184us/step - loss: 0.0257 - acc: 0.9929 - val_loss: 0.0225 - val_acc: 0.9934


Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.

60000/60000 [==============================] - 11s 186us/step - loss: 0.0269 - acc: 0.9926 - val_loss: 0.0212 - val_acc: 0.9940


Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.

60000/60000 [==============================] - 12s 200us/step - loss: 0.0255 - acc: 0.9930 - val_loss: 0.0228 - val_acc: 0.9936


Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.

60000/60000 [==============================] - 12s 193us/step - loss: 0.0250 - acc: 0.9931 - val_loss: 0.0208 - val_acc: 0.9936


Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.

60000/60000 [==============================] - 11s 188us/step - loss: 0.0250 - acc: 0.9929 - val_loss: 0.0210 - val_acc: 0.9934


Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.

60000/60000 [==============================] - 11s 190us/step - loss: 0.0236 - acc: 0.9935 - val_loss: 0.0201 - val_acc: 0.9943


---------

### 2) Result of model.evaluate(on test data)

[0.02007201824320946, 0.9943]


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
