from bg_remover.src.model.deeplab import deeplabv3_plus
from bg_remover.src.model.loss_fucn import dice_loss,dice_coef,iou
from pathlib import Path
import yaml
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from bg_remover.src.data.data_for_train import  train_data
ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DICT=ROOT_DIR/ "configs" / "train.yaml"
MODEL_DICT=ROOT_DIR/ "configs" / "model.yaml"

with open(TRAIN_DICT) as f:
    train_cfg= yaml.safe_load(f)
with open(MODEL_DICT) as f:
    model_cfg= yaml.safe_load(f)   
model = deeplabv3_plus((model_cfg["H"], model_cfg["W"], 3))
model.compile(loss=dice_loss, optimizer=Adam(train_cfg["lr"]), metrics=[dice_coef, iou, Recall(), Precision()])
dataset=train_data()
train_dataset=dataset[0]
valid_dataset=dataset[1]
model.fit(
        train_dataset,
        epochs=1,
        validation_data=valid_dataset,
        
    )
model.save(f"{model_cfg["model_save_path"]}/my_model")