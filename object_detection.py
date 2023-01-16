import yolov7
import numpy as np
import torch
import cv2
#from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
#from clip_interrogator import Config, Interrogator


def yolov8_inference(
        image,
        model_path,
        image_size=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
):
    model = YOLO(model_path)
    model.conf = conf_threshold
    model.iou = iou_threshold
    model.classes = [0]
    results = model.predict(image, imgsz=image_size, return_outputs=True)
    object_prediction_list = []
    predictions = []
    for _, image_results in enumerate(results):
        image_predictions_in_xyxy_format = image_results['det']
        for pred in image_predictions_in_xyxy_format:
            x1, y1, x2, y2 = (
                int(pred[0]),
                int(pred[1]),
                int(pred[2]),
                int(pred[3]),
            )
            predictions.append((int(pred[0]),int(pred[1]),int(pred[2]),int(pred[3]), pred[4], pred[5]))
            bbox = [x1, y1, x2, y2]
            score = pred[4]
            category_name = model.model.names[int(pred[5])]
            category_id = pred[5]
            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=int(category_id),
                score=score,
                category_name=category_name,
            )
            object_prediction_list.append(object_prediction)

    #image = read_image(image)
    #output_image = visualize_object_predictions(image=image, object_prediction_list=object_prediction_list)

    return predictions


def process_image(file_path, output_path):
    r = yolov8_inference(file_path, 'kadirnar/yolov8m-v8.0')

    area = 0
    pred_big = None
    for i, p in enumerate(r.pred[0]):
        x, y, w, h, conf, cls = p
        if (w-x)*(h-y) > area:
            area = (w-x)*(h-y)
            pred_big = p

    org_image = Image.open(file_path)
'''

    pil_image = Image.fromarray(np.uint8(org_image)).convert('RGB')
    

    prompt = ci.interrogate(pil_image)

    auth_token = os.environ['HF_AUTH_TOKEN']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token
    ).to(device)

    x1, y1, x2, y2, _, _ = pred_big
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    sub = org_image[y1:y2, x1:x2]
    scale_percent = 512 / max([w, h])

    width = int(sub.shape[1] * scale_percent)
    height = int(sub.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(sub, dim, interpolation=cv2.INTER_AREA)
    yoff = round((512-height)/2)
    xoff = round((512-width)/2)

    final_image = np.zeros((512, 512, 3), dtype=np.uint8)
    final_image[yoff:yoff+height, xoff:xoff+width, :] = resized
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image.fill(255)
    mask_image[yoff:yoff+height, xoff:xoff+width, :] = 0

    output = pipe(prompt=prompt, image=Image.fromarray(final_image), mask_image=Image.fromarray(mask_image)).images[0]
    result = np.array(output)
    result[yoff:yoff+height, xoff:xoff+width, :] = resized

    result = Image.fromarray(result)
    result.save(output_path)
    '''