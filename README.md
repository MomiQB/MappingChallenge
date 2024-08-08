# MappingChallenge
Binary segmentation on [AIcrowd Mapping Challenge]([https://pages.github.com/](https://www.aicrowd.com/challenges/mapping-challenge) dataset to detect buildings in satellite images.

> Many parts of the world have not been mapped or in many cases maps are outdated by disaster or conflict. \
> Having detailed maps of areas with a high risk of natural disasters is fundamental to facilitate the prompt response of emergency responders. \
> Nowadays when new maps are needed they are drawn by hand, often by volunteers who participate in the so-called “Mapathons”. \
> Machine Learning and AI can help drawing accurate maps.


The **Mapping Challenge Dataset** (available at https://www.aicrowd.com/challenges/mapping-challenge/dataset_files) consists of 280,741 training images, 60,317 validation images, and 60,697 testing images, all of size 300x300. In addition, for the training and validation set we also have the corresponding annotations in MS COCO format. These provide the information to construct the binary masks to use for training. \

In this project, only the validation part of the dataset will be used, with the corresponding annotations. The **goal** is to design and train a robust model to perform binary segmentation and detect the buildings. Image processing techniques will be applied to further improve the quality of the predicted masks.

***Content:***
- 





