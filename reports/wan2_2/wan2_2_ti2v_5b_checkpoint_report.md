# Wan2.2-TI2V-5B Checkpoint Report

- status: `PASS`
- checkpoint_path: `/mnt/sfs_turbo/fyy/checkpoints/Wan2.2-TI2V-5B`
- shards: `3`
- checkpoint tensors: `825`
- loadable tensors: `820`
- model tensors: `858`
- config matches known TI2V-5B: `True`
- stripped prefix: `None`
- dropped counts: `{'pixel_head': 3, 'vae_patch_embed': 2}`
- missing count: `38`
- backbone missing count: `0`
- unexpected count: `0`
- shape mismatch count: `0`
- companion files: `{'config_json': True, 'configuration_json': True, 'safetensors_index': True, 'vae': True, 'umt5': True}`

## Examples

### Missing

- `action_encoder.W1.W`
- `action_encoder.W1.b`
- `action_encoder.W2.W`
- `action_encoder.W2.b`
- `action_encoder.W3.W`
- `action_encoder.W3.b`
- `action_head.decoder.layer1.W`
- `action_head.decoder.layer1.b`
- `action_head.decoder.layer2.W`
- `action_head.decoder.layer2.b`
- `action_to_latent.0.bias`
- `action_to_latent.0.weight`
- `action_to_latent.1.bias`
- `action_to_latent.1.weight`
- `latent_adapter.proj.0.bias`
- `latent_adapter.proj.0.weight`
- `latent_adapter.proj.2.bias`
- `latent_adapter.proj.2.weight`
- `latent_head.proj.bias`
- `latent_head.proj.weight`

### Unexpected


### Shape Mismatches

- none
