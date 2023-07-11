# iadb-fun

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pharmapsychotic/iadb-fun/blob/main/notebooks/sampling.ipynb)

<video width="256" height="256" controls>
  <source src="media/cat-seed-slerp.mp4" type="video/mp4">
</video>

## What's IADB?

IADB (Iterative Alpha (De)Blending) replaces the rather complex and math-y bits of denoising
diffusion probabilistic models with the very simple alpha-blending that all graphics programmers 
are already familiar with. I love it! Give their paper a read and check out how simple and elegant the derivation is.

[IADB paper](https://arxiv.org/abs/2305.03486) / [IADB github](https://github.com/tchambon/IADB)


## What's this repo all about?

Having fun with the iterative alpha (de)blending technique!

- [x] Created **pytorch-lightning** based trainer to watch pretty graphs in wandb, use exponential moving average to smooth the weights, pre-process training images into LMDB for faster training, and other goodness
- [x] Trained some 64x64 RGB unconditional models
- [x] Config files so can create models with different architectures
- [x] Implemented simple euler and runge-kutta samplers
- [x] Create interpolation animations slerping through noise seeds to see how smooth the learned mappings are
- [x] Created **latent-IADB** by combining Stable Diffusion VAE with 4-channel IADB to denoise latents
- [x] Trained 512px latent-IADB cat model 

This is for fun and learning so don't kick me in the nuts if you try it and something doesn't work.

## Models

You can find some models I've trained at https://huggingface.co/pharma/iadb/tree/main and they're available in the droplist inside the notebook. You can't really make anything useful with them (unless you want to fill your drive up with cat pictures).

## Trainer

See [trainer.py](trainer.py) for full set of parameters. Here's an example:

```bash
python trainer.py \
    --batch_size 64 \
    --image_folder "train/cats" \
    --learning_rate 1e-4 \
    --lmdb_path "train/cats_lmdb" \
    --max_epochs -1 \
    --model_config "train/cats512_vae.yaml" \
    --out_path "cats/cats512_vae" \
    --preview_interval 250 \
    --save_ckpt_mins 30 \
    --wandb_project iadb_cats_vae
```

You need to create a yaml file specifying the number of UNet blocks, channel counts, whether to use VAE etc. Here's example for the 512 cats [cats512_vae.yaml](https://huggingface.co/pharma/iadb/blob/main/cats512_vae.yaml)