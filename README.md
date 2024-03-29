# mma_sr
This is a crude and non-official implemention of baseline model-mma sr in the paper :_Multimodal Attention with Image Text Spatial Relationship for OCR-Based Image Captioning_ (ACM MM2020)https://dl.acm.org/doi/10.1145/3394171.3413753

![image](https://user-images.githubusercontent.com/49356039/147088105-209facee-7c1e-40f3-a6a1-bc78ab8482f6.png)


The code is based on the mmf-framework, which is a popular framework from Facebook AI Research. The other interfaces and packages are here: https://github.com/facebookresearch/mmf

In this paper, it introduces a model based on the LSTM and attention module, and improves its performance with the spatial relationship. However, I did not implement this component for spatial relationship beacuse of the task that finding the _next_ neighbours ocr' = N(ocr) for the each OCR token has many conditions to discuss.

All the research work belongs to those researchers, I just write it with pytorch by my own for the follow-up study. I find the author of the paper has published a new paper in https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_OCR-Based_Image_Captioning_by_Incorporating_Geometrical_Relationship_CVPR_2021_paper.pdf (CVPR 2021). The structure of the model is similar with the mma_sr and has more powerful performance in OCR-based image captioning task.However, its code is still not open souce. 


If you want to use this program, you just need to install the previous mmf and then put this mma_sr.py in mmf folder location:  mmf/mmf/models. The code of this model is modified by the m4c.py in mmf. The reason why I use this kind of framework is that the dataset processsing in the OCR-based image captioning task is a bit of exhausting. So I directly write the programme in this mmf-framework. 




My programming ability is not professional so I just write the code as this paper says. However, there are some gap on metrics result in the paper.


| Model             | BLEU-4        |Metor          | Rouge-L       |Spice          |CIDEr          |
|----------         |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| baseline in paper |  24.0         |22.6           |47.0           |15.6           |93.7           |
| baseline(my code)          |    22.3       |  21.5         |45.0           |14.5           |86.9           |
| baseline(my code,beam=5)  |    24.0       |  21.6         |46.4           |14.6           |90.9           |



Perhaps there is something goes wrong with my code. I could not find out which part in my code is false. If you could find out the problem in the code, I would appreciate it very much. 

For other componet, I would also modify the code in the later.
