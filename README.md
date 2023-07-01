# FAITH
Thank you for your interest in our work! </br>

This is the code for the paper FAITH: Few-Shot Graph Classification with Hierarchical Task Graphs.


### Requirement:
```
torch==1.11.0+cu113
torchvision==0.12.0+cu113  
```


### Code Running:

Replace the 'dataset_name' with a specific dataset name.  
To run the command for image datasets, i.e., 'miniImageNet' and 'FC100':  
```
python main_image.py --dataset dataset_name
```

To run the command for text datasets, i.e., '20newsgroup' and 'huffpost':  
```
python main_text.py --dataset dataset_name
```

### Citation
Welcome to cite our work! </br>

> @inproceedings{wang2022faith,  
  title={Federated Few-shot Learning},  
  author={Wang, Song and Fu, Xingbo and Ding, Kaize and Chen, Chen and Chen, Huiyuan and Li, Jundong},  
  booktitle={SIGKDD},  
  year={2023}  
}

