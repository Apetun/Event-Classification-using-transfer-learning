# Event-Classification-using-transfer-learning
In this task, we have to classify war event images into different classes   
There are total 5 image classes as shown in the table below:
| Class | Image |
| --- | --- |
| Fire |  <img src="https://github.com/Apetun/Event-Classification-using-transfer-learning/assets/114131523/b5e787be-0496-41c5-9bb7-5873fc63f1e0" width="300" /> |
| Destroyed buildings | <img src="https://github.com/Apetun/Event-Classification-using-transfer-learning/assets/114131523/7be5b651-449d-4fef-be0c-65aeb322b598" width="300" /> |
| Humanitarian Aid and rehabilitation | <img src="https://github.com/Apetun/Event-Classification-using-transfer-learning/assets/114131523/b7e55d45-c0a6-4483-b7c5-8573242ae126" width="300" /> |
| Military Vehicles | <img src="https://github.com/Apetun/Event-Classification-using-transfer-learning/assets/114131523/8636a0b1-0e64-43a1-a2dd-c2c9a296648c" width="300" /> |
| Combat | <img src="https://github.com/Apetun/Event-Classification-using-transfer-learning/assets/114131523/e171cb5e-641f-4ae8-ae12-6212786f06a6" width="300" /> |

# Transfer Learning
Transfer learning, used in machine learning, reuses a pre-trained model on a new problem. In transfer learning, a machine exploits the knowledge gained from a previous task to improve generalization about another.
We will be using TensorFlow and Google Mobile-Net V2 model as a starting point changing only the final layer of the model and training it with our dataset to achieve the task at hand.   
These are the labels that will be used:
```
Combat                              = "combat"
Humanitarian Aid and rehabilitation = "humanitarianaid"
Military Vehicles                   = "militaryvehicles"
Fire                                = "fire"
Destroyed buildings                 = "destroyedbuilding"
```
## Steps Taken 
- Data Augmentation using [Data_augmentation.py](Data_augmentation.py) on the original dataset to create a larger augmented dataset provided in the `training` folder
- 
