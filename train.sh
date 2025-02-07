export PYTHONPATH=/home/kia/Codes/img-cond-diffusion-model/
accelerate launch \
            --num_processes=4 --mixed_precision=fp16 \
            UPD_study/models/ours/ours_trainer.py --fold=0 --modality=OPMED -dw=True --image_size=256 --normalize=True
