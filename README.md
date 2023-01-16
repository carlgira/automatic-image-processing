# automatic-image-processing
Several tools to format images to 512 x 512 for dreambooth training

## Preparation  
```
git clone https://github.com/carlgira/automatic-image-processing.git
```

## Install
```
cd automatic-image-processing
python3 -m venv venv
source venv/bin/activate
pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip
pip install -r requirements.txt
```

## Test
```
export HF_AUTH_TOKEN='<hf_token>'
jupyterlab
```

- Open `notebook.ipynb` http://localhost:8888 (make sure to open the ssh tunnel)
- Upload images
- Change the `notebook.ipynb` code with the name of the image
- Run the code
