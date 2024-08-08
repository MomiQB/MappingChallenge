# MappingChallenge
Binary segmentation on [AIcrowd Mapping Challenge](https://www.aicrowd.com/challenges/mapping-challenge) dataset to detect buildings in satellite images.

> Many parts of the world have not been mapped or in many cases maps are outdated by disaster or conflict. \
> Having detailed maps of areas with a high risk of natural disasters is fundamental to facilitate the prompt response of emergency responders. \
> Nowadays when new maps are needed they are drawn by hand, often by volunteers who participate in the so-called “Mapathons”. \
> Machine Learning and AI can help drawing accurate maps.


The **Mapping Challenge Dataset** (available at https://www.aicrowd.com/challenges/mapping-challenge/dataset_files) consists of 280,741 training images, 60,317 validation images, and 60,697 testing images, all of size 300x300. In addition, for the training and validation set we also have the corresponding annotations in MS COCO format. These provide the information to construct the binary masks to use for training. \

In this project, only the validation part of the dataset will be used, with the corresponding annotations. The **goal** is to design and train a robust model to perform binary segmentation and detect the buildings. Image processing techniques will be applied to further improve the quality of the predicted masks.

The code is developed using ***Pytorch***.

### _Setting Up_:
- Go to https://www.aicrowd.com/challenges/mapping-challenge/dataset_files and download the file val.tar.gz (830 MB)
-	Rename the zip file ‘mappingDataset’
-	Upload on Kaggle or on Colab

  When running the notebook, be careful to write the correct path to the folder with the mappingDataset.tar file in the cell with the file extraction.

### _Content:_
- [MappingChallenge_VaccariSimone.ipynb](https://github.com/MomiQB/MappingChallenge/blob/main/MappingChallenge_VaccariSimone.ipynb): main notebook
- [final_model.pth](https://github.com/MomiQB/MappingChallenge/blob/main/final_model.pth): weights of trained model (U-Net with ResNet101 backbone)
- [dashboard.py](https://github.com/MomiQB/MappingChallenge/blob/main/dashboard.py): interactive dashboard source code designed with Streamlit
- [requirements.txt](https://github.com/MomiQB/MappingChallenge/blob/main/requirements.txt): packages requirments to run the dashboard
- [000000060315.jpg](https://github.com/MomiQB/MappingChallenge/blob/main/000000060315.jpg), [000000060313.jpg](https://github.com/MomiQB/MappingChallenge/blob/main/000000060313.jpg): examples of images to use in the dashboard





