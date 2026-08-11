import os
import json
import requests


def save_checkpoint(model, optimizer, cfg, path):
    state = {
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'cfg': cfg
    }
    torch = __import__('torch')
    torch.save(state, path)

def maybe_upload(checkpoint_path, cfg):
    # Pluggable uploader for moithub.com or other hubs.
    api_url = os.environ.get('MOITHUB_API_URL') or cfg.get('moithub_api_url')
    token = os.environ.get('MOITHUB_TOKEN') or cfg.get('moithub_token')
    if api_url and token:
        print(f"Uploading {checkpoint_path} to {api_url}... (moithub)")
        files = {'file': open(checkpoint_path, 'rb')}
        headers = {'Authorization': f'Bearer {token}'}
        try:
            r = requests.post(api_url.rstrip('/') + '/api/v1/upload', files=files, headers=headers, timeout=60)
            r.raise_for_status()
            print('Upload successful:', r.text)
        except Exception as e:
            print('Upload failed:', e)
    else:
        print('No moithub credentials provided; skipping upload.')