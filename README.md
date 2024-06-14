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
│   ├── metrics                   <- Calculation metrics scripts
│   ├── pipelines                 <- Machine learning pipeline for training and evaluation scripts
│   ├── api                       <- Endpoint script
│   ├── service                   <- Service script
│   ├── utils.py                  <- Utility scripts
|
|
│
├── app.py                    <- Run API Endpoint
├── run_training.py           <- Run Training Pipeline
├── run_evaluation.py         <- Run Evaluation Pipeline
├── predict_sample.py         <- Example code how to predict image
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- List python dependencies
├── setup.sh                  <- File for set up python dependencies
└── README.md
```
## Getting Started

### Prerequisites
- Docker
- Docker Compose

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/anti-spoofing-system.git
   cd anti-spoofing-system
2. <b> Build and run the Docker container</b>:
    ```bash
    docker-compose up --build

### Usage

The API can be accessed at `http://localhost:5000`. You can use tools like Postman to interact with the API.

Example Request
To make a prediction, you can send a `POST` request to the `/predict` endpoint with an image file.

## References
- [Mobilenet v2 Architecture](https://github.com/tonylins/pytorch-mobilenet-v2)
- [Senet Architecture](https://github.com/moskomule/senet.pytorch)
- [Framework PyTorch](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

### Todo:
- Containerization
- CI/CD using Github Action
