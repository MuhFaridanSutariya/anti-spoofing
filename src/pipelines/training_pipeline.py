import argparse
import os
import sys
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
import wandb

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from models.ln_model import load_model, ModelInterface
from data.dataset import load_data, load_dataloader
from utils import load_backbone
from metrics.apcer import APCER
from metrics.npcer import NPCER
from metrics.acer import ACER

def training_pipeline(args: argparse.Namespace):
    # Load dataset
    data = load_data(args)

    # Load dataloader
    train_loader, val_loader, test_loader = load_dataloader(data)

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" > Device map:", device)

    # Load backbone
    backbone = load_backbone(args)

    # Load model
    model = load_model(backbone=backbone,
                       input_shape=args.input_shape, 
                       num_classes=args.num_classes)
    model.to(device)
    
    # Load logger
    wandb.login(key=args.wandb_token)
    logger = WandbLogger(name=args.wandb_runname, project="cv-project")

    # Load callbacks
    es_callback = EarlyStopping(monitor="val/apcer", min_delta=0.00, patience=4, verbose=True, mode="min")

    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoint',
        filename=args.modelname,
        save_top_k=3,
        verbose=True,
        mode='min',
        monitor="val/npcer"
    )

    # Load trainer
    trainer = Trainer(max_epochs=args.max_epochs,
                      callbacks=[ckpt_callback],
                      logger=logger)
    
    trainer.fit(model, train_loader, val_loader)

    # Save best model checkpoint
    best_model_path = ckpt_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Load and save model to Hugging Face
    model = ModelInterface.load_from_checkpoint(checkpoint_path=best_model_path, 
                                                model=backbone,
                                                input_shape=args.input_shape, 
                                                num_classes=args.num_classes)

    model.to(device)

    # Hugging Face Hub credentials and repository
    # hf_token = args.hf_token
    # repo_name = args.hf_repo_name
    # HfFolder.save_token(hf_token)
    # model.half()

    # model.push_to_hub(repo_name, use_auth_token=hf_token)

    # model.pretrained_model.config.push_to_hub("faridans27/anti-spoofing")

    if test_loader is not None:
        model.eval()
        apcer_metric = APCER().to(device)
        npcer_metric = NPCER().to(device)
        acer_metric = ACER().to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                all_preds.append(outputs)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        apcer = apcer_metric(all_preds, all_labels)
        npcer = npcer_metric(all_labels, all_labels)

        acer = acer_metric(all_preds, all_labels)

        print("......................")
        print(f"Test APCER: {apcer}")
        print(f"Test NPCER: {npcer}")
        print(f"Test ACER: {acer}")

    
