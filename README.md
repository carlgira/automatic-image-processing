# automatic-image-processing
Several tools to format images to 512 x 512 for dreambooth training


# Install
```
python3 -m venv venv
source venv/bin/activate
pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip
pip install -r requirements.txt
```

# Test
```
export HF_AUTH_TOKEN='<hf_token>'
python apps2.py
```
