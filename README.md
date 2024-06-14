# Anti Spoofing System
This is project for exploration about Machine Learning Operation in Classification Image Task

## Folder Structure
```
│
├── Figures                   <- Example Image for input and output
│
│
├── src                       <- Source code
│   ├── data                      <- Data scripts
│   ├── models                    <- Model scripts
│   ├── metrics                   <- calculation metrics scripts
│   ├── pipelines                 <- machine learning pipeline for training and evaluation scripts
│   ├── utils.py                  <- Utility scripts
│  
│
├── run_training.py           <- Run Training Pipeline
├── run_evaluation.py         <- Run Evaluation Pipeline
├── predict_sample.py         <- Example code how to predict image
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- List python dependencies
├── setup.sh                  <- File for set up python dependencies
└── README.md
```


## References
- [Mobilenet v2 Architecture](https://github.com/tonylins/pytorch-mobilenet-v2)
- [Senet Architecture](https://github.com/moskomule/senet.pytorch)
- [Framework PyTorch](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

### Todo:
- API Development
- Containerization
- CI/CD using Github Action
