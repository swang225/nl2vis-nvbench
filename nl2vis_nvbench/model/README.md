The baseline model ncnet is directly ported from the orignal ncnet repo:
https://github.com/Thanksyy/ncNet
We have made slight modifications to the structure of the code to suit our research needs.

The nv_bert, nv_bert_cnn, nv_ncbert3 all have new encoders written. The decoder and multihead attention layers uses the same code from the original ncnet model with slight modifications.
The training and testing script for nv_bert, nv_bert_cnn, nv_ncbert3 uses the original ncnet train/test scripts with modifications due to model structure changes.
The evaluation scripts of the models are based on the test script of the models with additional information saved.