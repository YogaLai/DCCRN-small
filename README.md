# DCCRN-small
Speech Enhancement(SE) is the task of taking a noisy speech input and producing an enhanced
speech output.
DCCRN [1] has outperformed previous models on SE with complex-valued operation and submitted to the Interspeech 2020 Deep Noise Suppression (DNS) challenge ranked first for the real-time-track. 
Inspire by it and its previous work CRN [2], we found out that though calculating the complex part can help increasing the accuracy,which come behind is the increase of calculation and parameters. Thus,we **aim to design a model which is more lightweight and remains the complex calculation**

- Reference

[1] Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang,
and Lei Xie, “Dccrn: Deep complex convolution recurrent network for phase-aware speech
enhancement,” arXiv preprint arXiv:2008.00264, 2020

[2] Ke Tan and DeLiang Wang, “A convolutional recurrent neural network for real-time speech
enhancement,” Interspeech, 2018

## Experiments
We train and test the model on Interspeech2020 DNS challenge dataset. The inference time runs on a PC with one NVIDIA 2080Ti and
the length of input wav is 3.75s. 
The comparison result. Althogh DCCRN[1] has the best PESQ, **our model is more lightweight and  more faster. The inference time is close to non-complexed calculation. Also, our model still has competitive PESQ result**

![](https://i.imgur.com/kIfIj93.png)
![](https://i.imgur.com/anQhx9V.png)

![](https://i.imgur.com/uU57xpD.png)

## Note
The more details you can see the document.pdf in the repo
