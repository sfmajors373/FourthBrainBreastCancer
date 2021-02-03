# FourthBrainBreastCancer
This is our final project for Fourth Brain

# Cancer Map
#### Libraries
| Stage       | Libraries |
|--           |--         |
| Prototyping | Pandas, Numpy, Tensorboard    |
| WSI Tools   | OpenSlide |
| DL          | Tensorflow? |
| API         | ?         |
| Front End   | Dash?     |


### Three Phases/Failsafes
| Phase | Goal | Architecture |  Outcome |
|--         |--            |--                    |--|
| 1 | First Thing | Unknown | Unknown |
| 2 | Second Thing | Unknown | Unknown |
| 3 | Third Thing  | Unknown | Unknown |

### Sources
#### White Papers
1. [A Comprehensive Review for Breast Histopathology Image Analysis Using Classical and Deep Neural Networks](https://arxiv.org/pdf/2003.12255v2.pdf)
2. [A Fast and Refined Cancer Regions Segmentation Framework in Whole-slide Breast Pathological Images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7239841/pdf/41598_2020_Article_65026.pdf)
3. [Assessment of Breast Cancer Histology using Densely Connected Convolutional Networks](https://arxiv.org/pdf/1804.04595.pdf)
4. [A Unified Framework for Tumor Proliferation Score Prediction in Breast Histopathology](https://arxiv.org/pdf/1612.07180.pdf)
5. [Deep Learning for Identifying Metastatic Breast Cancer](https://arxiv.org/pdf/1606.05718.pdf) 
6. [Detecting Cancer Metastases on Gigapixel Pathology Images - Google 2017t](https://arxiv.org/pdf/1703.02442)
7. [Multi-Stage Pathological Image Classification using Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Takahama_Multi-Stage_Pathological_Image_Classification_Using_Semantic_Segmentation_ICCV_2019_paper.pdf)
#### Other Works



### Current development / How to use :

base_directory/dataset_folder
```bash
base_directory
├── dataset_folder
    ├── training
    │   ├── lesion_annotations
    │   │   └── tumor_001.xml
    │   ├── normal
    │   │   └── normal_001.tif
    │   └── tumor
    │       └── tumor_001.tif
    │
    └── testing
        ├── lesion_annotations
        │   └── test_001.xml
        └── images
            └── test_001.tif
```

Implemented so far:

- Generate_tiles.py script :

Takes the normal (negative), tumoral (positive) WSIs and corresponding lesion annotations (xml).
And stores the tiles into hdfs files stored in a destination folder.

Note: During the next stage, we will generate augmented tiles that will be used in our training model.
In order to add randomness, the tiles generated with generate_tiles.py should be larger than the ones used in read_tiles.py
In our case, we generate tiles of 312 x 312. Later that tile will be randomly cropped into a 256 x 256 tile.

- Read_tiles.py prepares data for the training and validation process. The generator can be directly plugged into a model.fit() call
The data augmentation and color normalization is made at that level

Next steps:
- Upstream WSI cleaning in generate_tiles.py to improve the quality of the training set generated
- Test dataset generation

To run backend:
- Navigate to /app/backend
- Run `uvicorn app:app --reload`
