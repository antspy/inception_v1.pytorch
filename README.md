# inception_v1.pytorch
An implementation of inception_v1 on pytorch with pretrained weights. 

This code is a pytorch translation of the soumith torch repo: https://github.com/soumith/inception.torch
It implements the original version of the inception architechture; what is known has GoogLeNet.

Pretrained weights can be found at https://mega.nz/#!4RJE1SSY!kcCDyhkum6EQqVtqTc-deHnQuckM3zYSYq16bADbfww

# Disclaimer 
Test accuracy of the pretrained model on imagenet is only 26.38%. If I am not mistaken, this is an issue of the original torch repo - the data loading is done correctly. If you train this model to better accuracy, I would love to get the new set of weights! 

# License
The code is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for detail.
