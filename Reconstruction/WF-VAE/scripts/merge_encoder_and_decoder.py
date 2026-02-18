import torch

encoder = ""
decoder = ""
output = ""

encoder_ckpt = torch.load(encoder, map_location="cpu")
decoder_ckpt = torch.load(decoder, map_location="cpu")

new_ckpt = encoder_ckpt
# print(encoder_ckpt.keys())


for key, param in decoder_ckpt['ema_state_dict'].items():
    new_ckpt['ema_state_dict'][key] = param

torch.save(new_ckpt, output)