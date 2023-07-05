import fire
import lmdb
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torchvision
import wandb

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from diffusers import AutoencoderKL
from ema_pytorch import EMA
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from iadb import config_load, model_from_config, sample_euler, tensor_to_image


#=============================================================================
# VAE
#=============================================================================

device, generator, vae = None, None, None

def vae_encode(image_t: torch.Tensor) -> torch.Tensor:
    image_t = image_t.to(device=device, dtype=torch.float16).mul(2).sub(1)
    with torch.no_grad():
        latent_dist = vae.encode(image_t).latent_dist
    latents = latent_dist.sample(generator=generator)
    latents = 0.18215 * latents
    return latents

def vae_decode(latents: torch.Tensor) -> Image.Image:
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents.half()).sample
    image = (image*0.5 + 0.5).clamp(0, 1)
    return torchvision.transforms.functional.to_pil_image(image[0])


#=============================================================================
# Datasets
#=============================================================================

def images_to_latents(lmdb_path: str, folder: str, resolution: int=512):
    image_paths = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution)])

    # 2x for hflip, 2 bytes per float16
    max_size = int(1.1 * len(image_paths) * 2 * (4*64*64) * 2)

    env = lmdb.open(lmdb_path, map_size=max_size)
    with env.begin(write=True) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            image = Image.open(image_path)
            image = transform(image)
            for f in range(2):
                latent = vae_encode(torchvision.transforms.functional.to_tensor(image).unsqueeze(0))
                txn.put(str(i*2+f).encode('utf-8'), latent.cpu().numpy().tobytes())
                image = transforms.functional.hflip(image)
    env.close()    

class ImagesDataset(Dataset):
    def __init__(self, directory, pre_transform=None, transform=None, num_threads=8, chunk_size=64):
        self.transform = transform
        self.images = []

        # List all the image file paths
        image_paths = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files
                       if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        # Split the file paths into chunks
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

        def load_chunk(image_paths):
            return [pre_transform(Image.open(path).convert('RGB')) for path in image_paths]

        # Use a thread pool to process the chunks in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for loaded_chunk in tqdm(executor.map(lambda chunk: load_chunk(chunk), chunks),
                                     total=len(chunks), desc="Loading images"):
                self.images.extend(loaded_chunk)

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
class LatentsDataset(Dataset):
    def __init__(self, lmdb_path: str, resolution: int=512):
        self.latents = []
        env = lmdb.open(lmdb_path, readonly=True)
        stats = env.stat()
        num_entries = stats['entries']
        with env.begin() as txn:
            for index in tqdm(range(num_entries), desc="Loading latents"):
                buffer = txn.get(str(index).encode('utf-8'))
                tensor = torch.from_numpy(np.frombuffer(buffer, dtype=np.float16))
                latents = tensor.view(4, resolution//8, resolution//8)
                self.latents.append(latents)
        env.close()
        print(f"Loaded {len(self.latents)} latents")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]


#=============================================================================
# Training
#=============================================================================

class IADBLightningModule(pl.LightningModule):
    def __init__(self, config, batch_size: int, learning_rate: float, use_vae: bool):
        super().__init__()
        self.batch_size = batch_size
        self.model = model_from_config(config)
        self.learning_rate = learning_rate
        self.use_vae = use_vae

        self.ema = EMA(
            self.model,
            beta = 0.9999,              # exponential moving average factor
            update_after_step = 100,    # only after this number of .update() calls will it start updating
            update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def forward(self, x_alpha, alpha):
        return self.model(x_alpha, alpha)
    
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: 'Optimizer') -> None:
        self.ema.update()
        optimizer.zero_grad()

    def training_step(self, train_batch, batch_idx):
        if self.use_vae:
            if train_batch.shape[1] == 3:
                with torch.no_grad():
                    x1 = torch.stack([vae_encode(x.unsqueeze(0)).squeeze() for x in train_batch])
            else:
                x1 = train_batch
        else:
            x1 = train_batch.mul(2).sub(1)
        x0 = torch.randn_like(x1)

        alpha = torch.rand(x0.shape[0], device=x1.device)
        x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0

        d = self(x_alpha, alpha)['sample']
        loss = torch.sum((d - (x1-x0))**2)

        self.log('train_loss', loss)
        return loss


class PreviewSamplesCallback(pl.Callback):
    def __init__(self, out_path: str, preview_interval: int, resolution: int, use_vae: bool, wandb_project: str):
        super().__init__()
        self.out_path = out_path
        self.preview_interval = preview_interval
        self.resolution = resolution
        self.use_vae = use_vae
        self.wandb_project = wandb_project
    
    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.preview_interval != 0:
            return
        if self.use_vae:
            x0 = torch.randn(1, 4, self.resolution//8, self.resolution//8, device=pl_module.device)
        else:
            x0 = torch.randn(1, 3, self.resolution, self.resolution, device=pl_module.device)
        sample = sample_euler(pl_module.ema.model, x0, steps=64)
        if self.use_vae:
            image = vae_decode(sample)
        else:
            image = tensor_to_image(sample)
        preview_filename = f'{self.out_path}/preview_{trainer.global_step:08d}.png'
        os.makedirs(self.out_path, exist_ok=True)
        image.save(preview_filename)
        if self.wandb_project:
            trainer.logger.experiment.log({
                'preview': [wandb.Image(preview_filename, caption='Preview')]
            }, step=trainer.global_step)


def main(
    batch_size: int = 64,
    image_folder: str = 'images',
    learning_rate: float = 1e-4,
    lmdb_path: str = 'latents.lmdb',
    max_epochs: int = -1,
    model_config: str = 'model.yaml',
    out_path: str = 'out',
    preview_interval: int = 250,
    resume_ckpt: str = None,
    save_ckpt_mins: float = 30,
    wandb_project: str = None,
):
    config = config_load(model_config)
    resolution = config.get('resolution', 64)
    use_vae = config.get('vae_latents', False)

    if use_vae:
        global device, generator, vae
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device="cuda").manual_seed(0)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae = vae.eval().to(device).half()

    if use_vae and image_folder and lmdb_path and not os.path.exists(lmdb_path):
        images_to_latents(lmdb_path, image_folder)

    iadb_module = IADBLightningModule(
        config, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        use_vae=use_vae
    )

    if use_vae:
        in_memory_dataset = LatentsDataset(lmdb_path, resolution)
    else:
        pre_transform = transforms.Compose([transforms.Resize(resolution), transforms.CenterCrop(resolution)])
        transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])
        in_memory_dataset = ImagesDataset(directory=image_folder, pre_transform=pre_transform, transform=transform)
    dataloader = DataLoader(in_memory_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if wandb_project:
        wandb_logger = pl.loggers.WandbLogger(project=wandb_project)
        wandb_logger.watch(iadb_module.model)
    else:
        wandb_logger = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=out_path,
        train_time_interval=timedelta(minutes=save_ckpt_mins),
        save_top_k=-1,
    )
    preview_callback = PreviewSamplesCallback(
        out_path, 
        preview_interval, 
        resolution, 
        use_vae, 
        wandb_project
    )

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        precision="16-mixed", 
        callbacks=[ckpt_callback, preview_callback], 
        logger=wandb_logger, 
        max_epochs=max_epochs,
    )
    trainer.fit(iadb_module, dataloader, ckpt_path=resume_ckpt)

if __name__ == "__main__":
    fire.Fire(main)
