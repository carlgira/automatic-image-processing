from transformers import DetrFeatureExtractor, DetrForSegmentation
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
from copy import deepcopy
from clustering import filter_clusters
import motionnet
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
#from clip_interrogator import Config, Interrogator


parts = ['body', 'half', 'face']

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')


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
    image = image.resize((200,200)) #  PIL image # could I upsample output instead? better?
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
    print('colormask', color_mask.shape) # colormask (200, 200, 3)
    palette = itertools.cycle(sns.color_palette())
    keypoints = motionnet.get_keypoints(result_image)


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
                color_mask[mask == 1, :] = np.asarray(next(palette))*255 #color
                result.append((mask, nose, (x.min(), y.min(), x.max(), y.max())))


    # Show image + mask
    pred_img = np.array(result_image)*0.25 + color_mask*0.75


    motionnet.loop_through_people(pred_img, keypoints)

    pred_img = pred_img.astype(np.uint8)

    return pred_img, result



def possible_parts(image, detection):
    result = []
    mask, keypoints, rect = detection
    img_w, img_h, _ = image.shape
    x_min, y_min, x_max, y_max = rect

    if keypoints[motionnet.LABELS['leftShoulder']][2] > motionnet.THRESHOLD and keypoints[motionnet.LABELS['rightShoulder']][2]:
        leftShoulder_y, leftShoulder_x, _ = keypoints[motionnet.LABELS['leftShoulder']]
        rightShoulder_y, rightShoulder_x, _ = keypoints[motionnet.LABELS['rightShoulder']]

        base_line_shoulder = keypoints[motionnet.LABELS['leftShoulder']][0]
        result.append(['face', (rightShoulder_x, base_line_shoulder, leftShoulder_x, y_min)])
    else:
        nose = keypoints[motionnet.LABELS['nose']]
        leftEar = keypoints[motionnet.LABELS['leftEar']]
        rightEar = keypoints[motionnet.LABELS['rightEar']]
        leftEye = keypoints[motionnet.LABELS['leftEye']]
        rightEye = keypoints[motionnet.LABELS['rightEye']]

        width = abs(leftEye[1] - rightEye[1])

        result.append(['face', (rightEar[1] - width, nose[0] + 3 * width, leftEar[1] + width, rightEye[0] - width)])

    if keypoints[motionnet.LABELS['leftHip']][2] > motionnet.THRESHOLD and keypoints[motionnet.LABELS['rightHip']][2]:
        leftHip_y, leftHip_x, _ = keypoints[motionnet.LABELS['leftHip']]
        rightHip_y, rightHip_x, _ = keypoints[motionnet.LABELS['rightHip']]
        base_line_hip = keypoints[motionnet.LABELS['leftHip']][1]

        hip_size = leftHip_x - rightHip_x

        result.append(['half', (rightHip_x - hip_size, base_line_hip, leftHip_x + hip_size, y_min)])

    if keypoints[motionnet.LABELS['leftAnkle']][2] > motionnet.THRESHOLD and keypoints[motionnet.LABELS['rightAnkle']][2]:
        leftAnkle_y, leftAnkle_x, _ = keypoints[motionnet.LABELS['leftAnkle']]
        rightAnkle_y, rightAnkle_x, _ = keypoints[motionnet.LABELS['rightAnkle']]

        leftHip_y, leftHip_x, _ = keypoints[motionnet.LABELS['leftHip']]
        rightHip_y, rightHip_x, _ = keypoints[motionnet.LABELS['rightHip']]
        hip_size = leftHip_x - rightHip_x

        leftEye = keypoints[motionnet.LABELS['leftEye']]
        rightEye = keypoints[motionnet.LABELS['rightEye']]

        width = abs(leftEye[1] - rightEye[1])

        result.append(['body', (rightAnkle_x - hip_size, leftAnkle_y + width, leftAnkle_x + hip_size, y_min)])

    result_filter = []
    for part in result:
        rect = part[1]
        x_min, y_min, x_max, y_max = rect
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > image.shape[1]:
            x_max = image.shape[1]
        if y_max > image.shape[0]:
            y_max = image.shape[0]

        result_filter.append([part[0], (int(x_min * img_w), int(y_min * img_h), int(x_max * img_w), int(y_max))])

    return result_filter


def process_image(filename, output_filename=None):
    image = cv2.imread(filename)[:,:,::-1]
    org_image = image.copy()

    _, detection = predict_animal_mask(image, 85)

    detection_max = None
    area_max = 0
    for part in detection:
        mask, keypoints, rect = part
        x_min, y_min, x_max, y_max = rect
        area = (x_max - x_min) * (y_max - y_min)
        if area > area_max:
            area_max = area
            detection_max = part

    mask, nose, rect = detection_max
    plt.imshow(mask)
    plt.show()

    x1, y1, x2, y2 = rect
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    sub = org_image[y1:y2, x1:x2]
    scale_percent = 512 / max([w, h])

    width = int(sub.shape[1] * scale_percent)
    height = int(sub.shape[0] * scale_percent)
    dim = (width, height)
    print(dim)
    resized = cv2.resize(sub, dim, interpolation=cv2.INTER_AREA)
    print('resized.shape', resized.shape)
    # put resized image in center of final_image
    yoff = round((512-height)/2)
    xoff = round((512-width)/2)

    final_image = np.zeros((512, 512, 3), dtype=np.uint8)
    final_image[yoff:yoff+height, xoff:xoff+width, :] = resized
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image.fill(255)
    mask_image[yoff:yoff+height, xoff:xoff+width, :] = 0

    if output_filename is not None:
        rr = Image.fromarray(final_image)
        rr.save(output_filename)

    '''
    pil_image = Image.fromarray(np.uint8(org_image)).convert('RGB')
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    ci.config.blip_num_beams = 64
    ci.config.chunk_size = 2048
    ci.config.flavor_intermediate_count = 2048

    prompt = ci.interrogate(pil_image)

    # python load auth_token from environment variable
    # os.environ
    
    auth_token = os.environ['HF_AUTH_TOKEN']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token
    ).to(device)

    output = pipe(prompt=prompt, image=Image.fromarray(final_image), mask_image=Image.fromarray(mask_image)).images[0]
    result = np.array(output)
    result[yoff:yoff+height, xoff:xoff+width, :] = resized

    if output_filename is not None:
        result = Image.fromarray(result)
        result.save(output_filename)
    '''

    return rr


def test_image(image_path, gr_slider_confidence):
    gr_image_input = cv2.imread(image_path)
    print(gr_image_input.shape)

    pred_img, result = predict_animal_mask(gr_image_input, gr_slider_confidence)


    detection_max = None
    area_max = 0
    for part in result:
        mask, keypoints, rect = part
        x_min, y_min, x_max, y_max = rect
        area = (x_max - x_min) * (y_max - y_min)
        if area > area_max:
            area_max = area
            detection_max = part

    mask, nose, rect = detection_max
    plt.imshow(mask)
    plt.show()




test_image('example_image_3.jpg', 40)
#test_image('cuerpo1.jpg', 85)

#image_format.process_image('example_image_1.jpg', 'o_2.png')
#image_format.process_image('example_image_2.jpg', 'o_3.png')
#image_format.process_image('example_image_3.jpg', 'o_4.png')
#image_format.process_image('example_image_2.jpeg', 'o_5.png')
#image_format.process_image('face1.jpg', 'o_6.png')
#image_format.process_image('face2.jpg', 'o_7.png')
#image_format.process_image('example_image_3.jpg', 'o_4.png')