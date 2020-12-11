import copy
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch.optim import Adam
from torchvision import transforms, utils
from pathlib import Path
from functools import partial
from PIL import Image
from .helper_functions import cycle, num_to_groups


UPDATE_EMA_EVERY = 10
EXTS = ['jpg', 'jpeg', 'png']

RESULTS_FOLDER = Path('./results/simple_ddpm')
RESULTS_FOLDER.mkdir(exist_ok = True)

# backwards loss function used to construct partial function
def loss_backwards(fp16, loss, optimizer, **kwargs):
    # if fp16:
    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward(**kwargs)
    # else:
    loss.backward(**kwargs)


# Exponential moving average decay - in improved DDPMs model, possibly not wholly necessary
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Dataset(data.Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995, # exponential moving average decay
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        save_and_sample_every = 1000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every

        self.ds = Dataset(folder, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.fp16 = fp16
        # if fp16:
        #     (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.reset_parameters()


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(RESULTS_FOLDER / f'model-{milestone}.pt'))


    def load(self, milestone):
        data = torch.load(str(RESULTS_FOLDER / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def train(self):
        # partial is the creation of a partial function, explained here:
        # https://www.geeksforgeeks.org/partial-functions-python/
        # prefills the loos backwards function fp16 variable
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            # loop lets gradient accumulate before optimisation
            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % UPDATE_EMA_EVERY == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(self.image_size, batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                utils.save_image(all_images, str(RESULTS_FOLDER / f'sample-{milestone}.png'), nrow=6)
                self.save(milestone)

            self.step += 1

        print('training completed')


    def sample_images(self, n_row=7):

        batches = num_to_groups(n_row**2, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(self.image_size, batch_size=n), batches))
        all_images = iter(torch.cat(all_images_list, dim=0).cpu())
        fig = plt.figure(figsize=(12,12))
        for i in range(1, n_row**2 + 1):
            img = next(all_images)
            fig.add_subplot(n_row, n_row, i)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img.permute(1,2,0))

        plt.show()