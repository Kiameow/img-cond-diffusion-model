export PYTHONPATH=/home/kia/Codes/img-cond-diffusion-model/
python UPD_study/models/ours/ours_trainer.py --fold=0 -ev=t --modality=OPMED -dw=True --image_size=256 --normalize=True --h_config=UPD_study/models/ours/omega_config_big_eval_split.yaml
