# handwrite digit recognition

## TODO

1. TODO 1, 2: finish model
2. TODO 10:
    * change the params below
      * LEARNING_RATE = 0.01
      * BATCH_SIZE = 50
      * EPOCHS = 10
    * set train data, valid data to `TRAIN_DATA_PATH`, `VALID_DATA_PATH`
3. TODO 11: transforms
    * finish: may be adding some data augmentations?
    * comfirm download param? true or false?
4. TODO 15
    * `torch.save(model, MODEL_PATH)` or `torch.save(model.state_dict(), MODEL_PATH)` or ???

## Other

https://colab.research.google.com/

`!nvidia-smi` 顯示GPU狀況

要切 validation set

colab 檔案要存到google drive，不然會不見

batch size vs 平行運算
太小會沒效率，若等於1會造成每增加一張照片，就會更新素有的參數

p28 Conv2d convorlution function

n_in = input channel 大小
n_out = ouput channel 大小
p = 外圈要補多少的0
k = kernel size(3)
s = filter 移動的距離 （一般為1格）（pooling 時為2格）

one hot xxx

loss function
