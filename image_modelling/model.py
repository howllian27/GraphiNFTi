import torch
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device)

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

seed = 777

from PIL import Image
init_img = Image.open('/content/monkey.jpg')
init_img.thumbnail((512, 512))
init_img

prompt = ''
generator = torch.Generator(device = device).manual_seed(seed)
img = pipe(prompt = prompt, image = init_img, generator = generator).images[0]
img

import matplotlib.pyplot as plt

prompt = "photograph of an apple on a grass field, mountains in the background"

plt.figure(figsize=(18,8))
for i in range(1, 6):

  strength_val = (i + 4) / 10
  generator = torch.Generator("cuda").manual_seed(seed)
  img = pipe(prompt, image=init_img, strength=strength_val, generator=generator).images[0]

  plt.subplot(1,5,i)
  plt.title('strength: {}'.format(strength_val))
  plt.imshow(img)
  plt.axis('off')
  
plt.show()

modi = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/mo-di-diffusion", torch_dtype=torch.float16)
modi = modi.to("cuda")
prompt = "modern disney, an apple falling from a tree, mountains in the background"
generator = torch.Generator(device=device).manual_seed(seed)
image = modi(prompt=prompt, image=init_img, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
image