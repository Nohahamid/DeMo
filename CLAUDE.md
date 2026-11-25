# Named intentions on DeMo (master's thesis, Noha Hamid)

## Goal
Make DeMo (NeurIPS 2024 motion forecaster, Argoverse 2) predict **named maneuvers**
(straight, left/right turn, lane change left/right, U-turn, stop, other) and, above all,
**generate lane-change paths it currently misses**, without hurting minFDE6.

## How we work
- Code is edited here (laptop / VS Code) and pushed to branch `named-intentions`
  of github.com/Nohahamid/DeMo. Heavy work runs on **Google Colab (L4 GPU)**:
  notebook cells `git pull`, then preprocess / train / evaluate.
- Claude cannot run the Colab cells; the user runs them and pastes the output.
- Test changes locally on CPU first (a few hundred AV2 val scenarios are enough).
  Mamba's Triton kernels need a GPU: on CPU, monkeypatch
  `mamba_ssm.ops.triton.layernorm` (layer_norm_fn / rms_norm_fn / RMSNorm) with the
  `*_ref` versions.
- Colab paths: repo at `/content/drive/MyDrive/thesis_demo/DeMo` (`{WORK}/DeMo`);
  big data on local disk, lost when the runtime is deleted:
  `/content/av2_raw` (50k train raw), `/content/data/DeMo_processed` (processed),
  `/content/av2_val5k` + `/content/val5k/data/DeMo_processed` (5k val).
  `preprocess.py` writes to `<cwd>/data/DeMo_processed`, so run it from the folder you want.
- The user prefers short bullet-point answers in simple words.

## Files we added
- `intention_labels.py`: rule labels from the true future + lane graph (`label_states`).
- `analyze_intentions.py`: per-class / per-type analysis, rule-on-DeMo baseline.
- `fusion_intent.py`: `extract` (run DeMo once, label all 6 paths, save npz with
  pi, rule, fde, ade, slot_sem), `fit` (tree model on top), `diagnose`
  (per maneuver: Cov@6, Top1, condFDE, A generation / B ranking / C geometry,
  MR6, diversity@6, what each mode query stands for).
- `semantic_slots.py`: from the lane graph at t=0, which maneuvers are feasible
  + a map anchor for each -> `data/semantic_slots/<split>.csv`.
- Semantic slots in the model: `src/datamodule/av2_dataset.py` (`add_slots`,
  PRIORITY, FREE), `src/model/layers/time_decoder.py` (`enable_semantic_slots`,
  zero-init so step 0 = released DeMo), `src/model/trainer_forecast.py`
  (`train_scope` all|heads|decoder, `semantic_weight`, slot loss),
  `src/datamodule/av2_datamodule.py` (`oversample` {label: factor}).
  Config keys in `conf/config.yaml`.

## Results so far (5,000 val scenarios)
- Plain DeMo: minFDE6 1.258 m (released checkpoint `ckpts/DeMo.ckpt`).
- Full fine-tune made paths worse (1.414 m) -> keep encoders frozen.
- Intention heads (DeMo frozen, 50k train): acc 85.4%, macro F1 0.63.
- Rule on DeMo's top path (no training): acc 90.7%, macro F1 0.66, beats the heads.
- Diagnosis: lane changes are a **mode generation** problem: no lane-change path
  among the 6 in 47% (left) / 57% (right) of real lane changes; ranking only 7-10%.
  Cov@6: straight 99.6%, turns ~93%, LC 53% / 43%. DeMo's mode queries carry no
  maneuver meaning (each gives the data mix).

## Current step
Train semantic slots (Colab cells H0-H6): `train_scope=decoder`, `semantic_slots=true`,
`oversample={1:2,2:2,3:5,4:5}`, from `ckpts/DeMo.ckpt`, 50k train, 5k val.
Targets: LC Cov@6 80-90%, minFDE6 not worse than 1.258 m, macro F1 above 0.66.
