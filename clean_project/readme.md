# Smart Waste Sorter - Project for CV and NLP
The goal of this project is to create an easy installable system that tells the user where to put the garbage. The system will assist users in correctly sorting their waste. 
The flow is the following:
1. The user shows the waste to the camera and presses a button to start the system.
2. The camera detects waste with YOLO.
3. A descriptive text is made by the LLM which indicates the bins the wast should go into.
4. (Optional) The user asks some questions to the system to answer.

## Structure of the repo
### /demo
This folder contains all the nessecairy files for the POC shown during the presentation. It has all the finished code for the different parts including ``README.md`` where all info about running it can be found.

### /model_training
This folder contains everything to do with the training and experementing done on the YOLO models. This also includes a ``README.md`` which holds all the necessary info.

### /nlp_testing
This folder holds the NLP-part of the project and everything done for it. It also has a ``README.md`` with further instructions.

## Requirements
