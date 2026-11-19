# Named intentions on DeMo

This branch adds a named-intention prediction to DeMo. With `num_intentions: 0` (the default) the model and training are exactly the original DeMo.

## 1. Make labels (CPU is enough)

From the raw AV2 Motion Forecasting folders (the ones holding `scenario_*.parquet` and `log_map_archive_*.json`):

```
python intention_labels.py --data_root /path/to/av2 --split train --out data/intention_labels/train.csv
python intention_labels.py --data_root /path/to/av2 --split val   --out data/intention_labels/val.csv
```

Classes: 0 STRAIGHT, 1 LEFT_TURN, 2 RIGHT_TURN, 3 LANE_CHANGE_L, 4 LANE_CHANGE_R, 5 U_TURN, 6 STOP, 7 OTHER.
Turns and stops come from the heading change and distance over the 6 s future. Lane changes come from the AV2 lane graph (left and right neighbours, successors).

## 2. Train or fine-tune

Start from the released DeMo checkpoint and switch the heads on:

```
python train.py datamodule=av2 model=model_forecast \
    pretrained_weights=ckpts/DeMo.ckpt num_intentions=8 \
    'intention_class_weights=[1,3,4,5,6,8,3,1]'
```

`intention_class_weights` is a starting guess against class imbalance; set it from the label counts of your training split.

## 3. What is added

- `intention_labels.py`: automatic labels.
- `Av2Dataset(intention_dir=...)`: attaches the label to the focal agent (`-100` = no label, ignored).
- `ModelForecast(num_intentions=C)`: `intent_logits` [B, C] from the agent's scene token, and `mode_intent_logits` [B, K, C] from the K mode queries, so each mode gets a name.
- `Trainer`: cross-entropy on both (the mode head is trained on the mode closest to the ground truth), weighted by `intention_weight`; validation logs `val_intent_acc` and `val_intent_f1` (macro).
