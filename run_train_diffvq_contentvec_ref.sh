export PYTHONPATH=.
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3
export WANDB_API_KEY=********
python ./train.py\
    --dataset-config ./stable_audio_tools/configs/dataset_configs/audio_dir_dataset.json\
    --model-config ./stable_audio_tools/configs/model_configs/dance_diffusion/latent_diffvq_contentvec.json\
    --name latent_diffvq_cvec_ref\
    --strategy ddp\
    --num-gpus 1\
    --num-nodes 1\
    --batch-size 8\
    --checkpoint-every 5000\
    # --ckpt-path
    # --pretrained-ckpt-path

