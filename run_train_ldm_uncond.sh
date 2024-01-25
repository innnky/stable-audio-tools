export PYTHONPATH=.
python ./train.py\
    --dataset-config ./stable_audio_tools/configs/dataset_configs/audio_dir_dataset.json\
    --model-config ./stable_audio_tools/configs/model_configs/dance_diffusion/latent_diffusion_1d_uncond.json\
    --name latent_diffusion_1d_uncond\
    --strategy ddp\
    --num-gpus 1\
    --num-nodes 1\
    --batch-size 2\
    --checkpoint-every 5000\
    # --ckpt-path
    # --pretrained-ckpt-path

