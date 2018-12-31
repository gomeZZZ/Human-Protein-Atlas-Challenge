from tqdm import tqdm
from PIL import Image
import requests
import os

def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    """Taken from https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984 to download external data
    
    Args:
        pid ([type]): [description]
        image_list ([type]): [description]
        base_url ([type]): [description]
        save_dir ([type]): [description]
        image_size (tuple, optional): Defaults to (512, 512). [description]
    """

    colors = ['red', 'green', 'blue', 'yellow']
    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for color in colors:
            img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
            img_name = i + '_' + color + '.png'
            img_url = base_url + img_path

            # Get the raw response from the url
            r = requests.get(img_url, allow_redirects=True, stream=True)
            r.raw.decode_content = True

            # Use PIL to resize the image and to convert it to L
            # (8-bit pixels, black and white)
            im = Image.open(r.raw)
            im = im.resize(image_size, Image.LANCZOS).convert('L')
            im.save(os.path.join(save_dir, img_name), 'PNG')