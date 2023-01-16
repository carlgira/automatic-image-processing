from transformers import DetrFeatureExtractor, DetrForSegmentation
import numpy as np
import torch
import cv2
import itertools
import seaborn as sns
from copy import deepcopy
from clustering import filter_clusters
from motionnet import get_keypoints, loop_through_people
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
import traceback
import random


prompts = ['in a empty street', 'alone in a office', 'alone in a garden', 'alone in a house']

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-101-panoptic')


auth_token = os.environ['HF_AUTH_TOKEN']
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=auth_token
).to(device)


# point in rectangle
def point_in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

def get_cluster(x, y, l):
    if x < 0 or y < 0 or x >= l.shape[0] or y >= l.shape[1]:
        return []

    r = []
    if l[x,y] == 1:
        r = [(x, y)]
        l[x, y] = 0
        r.extend(get_cluster(x-1, y, l))
        r.extend(get_cluster(x+1, y, l))
        r.extend(get_cluster(x, y-1, l))
        r.extend(get_cluster(x, y+1, l))

    return r


def get_clusters(l):
    clusters = []
    while True:
        x, y = np.where(l == 1)
        if len(x) == 0:
            break
        clusters.append(get_cluster(x[0], y[0], l))
    return clusters


def predict_animal_mask(im,
                        gr_slider_confidence):
    image = Image.fromarray(im) # im: numpy array 3d: 480, 640, 3: to PIL Image
    image = image.resize((200,200))
    #result_image = np.array(image.convert('RGB'))
    result_image = im

    # encoding is a dict with pixel_values and pixel_mask
    encoding = feature_extractor(images=image, return_tensors="pt") #pt=Pytorch, tf=TensorFlow
    outputs = model(**encoding) # odict with keys: ['logits', 'pred_boxes', 'pred_masks', 'last_hidden_state', 'encoder_last_hidden_state']
    logits = outputs.logits # torch.Size([1, 100, 251]); class logits? but  why 251?
    bboxes = outputs.pred_boxes
    masks = outputs.pred_masks # torch.Size([1, 100, 200, 200]); mask logits? for every pixel, score in each of the 100 classes? there is a mask per class

    # keep only the masks with high confidence?--------------------------------
    # compute the prob per mask (i.e., class), excluding the "no-object" class (the last one)
    prob_per_query = outputs.logits.softmax(-1)[..., :-1].max(-1)[0] # why logits last dim 251?
    # threshold the confidence
    keep = prob_per_query > gr_slider_confidence/100.0

    # postprocess the mask (numpy arrays)
    label_per_pixel = torch.argmax(masks[keep].squeeze(),dim=0).detach().numpy() # from the masks per class, select the highest per pixel

    processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])

    result = []
    category = 1 # 1 is the category id for the "person" class
    #color_mask = np.zeros(image.size+(3,))
    color_mask = np.zeros(result_image.shape)
    palette = itertools.cycle(sns.color_palette())
    keypoints = get_keypoints(result_image)


    w, h, c = result_image.shape
    noses = np.array([(k[0][1]*w, k[0][0]*h, k[0][2], i) for i, k in enumerate(keypoints)])
    noses = noses[noses[:, 2] > 0.1]

    for lbl, cat in zip(np.unique(label_per_pixel), segments_info): #enumerate(palette()):
        if cat['category_id'] == category:
            mask = filter_clusters(label_per_pixel == lbl)
            mask = cv2.resize(mask, dsize=(result_image.shape[1], result_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            y, x = np.where(mask != 0)
            nose = None
            rect = (x.min(), y.min(), x.max(), y.max())
            for key in keypoints:
                if point_in_rect(key[0][1] * h, key[0][0] * w, rect):
                    nose = key
                    break

            if nose is not None:
                cv2.rectangle(color_mask, (int(np.min(x)), int(np.min(y))), (int(np.max(x)), int(np.max(y))), (255,0,0), 1)
                color_mask[mask != 0, :] = np.asarray(next(palette))*255 #color
                result.append((mask, nose, (x.min(), y.min(), x.max(), y.max())))


    # Show image + mask
    pred_img = np.array(result_image)*0.25 + color_mask*0.75


    loop_through_people(pred_img, keypoints)

    pred_img = pred_img.astype(np.uint8)

    return pred_img, result


def default_image_response(gr_image_input, output_path):
    real_image_input = cv2.cvtColor(gr_image_input, cv2.COLOR_BGR2RGB)
    h, w, _ = real_image_input.shape
    scale_percent = 512 / max([w, h])
    width = int(w * scale_percent)
    height = int(h * scale_percent)
    dim = (width, height)
    resized = cv2.resize(real_image_input, dim, interpolation=cv2.INTER_AREA)
    sub = resized[:512, :512, :]
    r_result = Image.fromarray(sub)
    r_result.save(output_path)
    return sub, r_result


def test_image(image_path, output_path, gr_slider_confidence=85):
    gr_image_input = cv2.imread(image_path)
    print(image_path)
    
    pred_img, result = None, None
    try:
        pred_img, result = predict_animal_mask(gr_image_input, gr_slider_confidence)
    except Exception as e:
        print('Error in detecting', e)
        traceback.print_exc()
        return default_image_response(gr_image_input, output_path)

    if len(result) == 0:
        print('No person detected')
        return default_image_response(gr_image_input, output_path)

    area = 0
    r_max = None
    for r in result:
        x1, y1, x2, y2 = r[2]
        a = (x2-x1)*(y2-y1)
        if a > area:
            area = a
            r_max = r

    real_image_input = cv2.cvtColor(gr_image_input, cv2.COLOR_BGR2RGB)

    mask_image = np.zeros(real_image_input.shape, dtype=np.uint8)
    mask_image.fill(255)
    masked_apply = cv2.bitwise_and(mask_image, mask_image, mask=(r_max[0] == 0).astype(np.uint8))

    masked = cv2.bitwise_and(real_image_input, real_image_input, mask=(r_max[0] != 0).astype(np.uint8))

    x1, y1, x2, y2 = r_max[2]
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    sub = masked[y1:y2, x1:x2]
    sub_mask = masked_apply[y1:y2, x1:x2]
    scale_percent = 512 / max([w, h])

    width = int(sub.shape[1] * scale_percent)
    height = int(sub.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(sub, dim, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(sub_mask, dim, interpolation=cv2.INTER_AREA)

    yoff = round((512-height)/2)
    xoff = round((512-width)/2)

    final_image = np.zeros((512, 512, 3), dtype=np.uint8)
    final_image[yoff:yoff+height, xoff:xoff+width, :] = resized
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image.fill(255)
    mask_image[yoff:yoff+height, xoff:xoff+width, :] = resized_mask

    output = pipe(prompt=random.choice(prompts), image=Image.fromarray(final_image), mask_image=Image.fromarray(mask_image)).images[0]
    result = np.array(output)

    result_temp = cv2.bitwise_and(result, mask_image)
    final_result = cv2.bitwise_or(result_temp, final_image)

    r_result = Image.fromarray(final_result)
    r_result.save(output_path)

    return final_result, r_result
