import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

LABELS = {'nose': 0, 'leftEye': 1, 'rightEye': 2, 'leftEar': 3, 'rightEar': 4, 'leftShoulder': 5, 'rightShoulder': 6, 'leftElbow': 7, 'rightElbow':8, 'leftWrist': 9, 'rightWrist': 10,
          'leftHip': 11, 'rightHip': 12, 'leftKnee': 13, 'rightKnee': 14, 'leftAnkle': 15, 'rightAnkle': 16}

THRESHOLD = 0.1

def overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 > w2 or w1 < x2 or y1 > h2 or h1 < y2)


def loop_through_people(frame, keypoints_with_scores, confidence_threshold=0.1):

    keypoints_with_scores_filter = remove_duplicates(keypoints_with_scores, confidence_threshold)

    for person in keypoints_with_scores_filter:
        draw_connections(frame, person, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def remove_duplicates(keypoints_with_scores, confidence_threshold):
    keypoints_with_scores_filter = [keypoints_with_scores[0]]
    for i in range(1, len(keypoints_with_scores)):
        value = keypoints_with_scores[i]

        calc_one = value.copy()
        calc_one[calc_one[:,2] < confidence_threshold] = 0
        calc_one = np.around(calc_one, decimals=1)

        flag = True
        for e in range(len(keypoints_with_scores_filter)):
            value_filter = keypoints_with_scores_filter[e]
            calc_two = value_filter.copy()
            calc_two[calc_two[:,2] < 0.1] = 0
            calc_two = np.around(calc_two, decimals=1)
            if np.sum(calc_one == calc_two) > 17*3*0.6:
                flag = False
                if np.sum(calc_two == 0) > np.sum(calc_one == 0):
                    keypoints_with_scores_filter[e] = value
                    break

        if flag:
            keypoints_with_scores_filter.append(value)

    return keypoints_with_scores_filter


def draw_keypoints(frame, shaped, confidence_threshold=0.1):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(shaped, [y, x, 1]))
    shaped = shaped[shaped[:, 2] > confidence_threshold]

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 2, (255,255,255), -1)


def draw_connections(frame, shaped, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(shaped, [y, x, 1]))
    shaped = shaped[shaped[:, 2] > confidence_threshold]

    for edge, color in EDGES.items():
        p1, p2 = edge
        if len(shaped) <= p1 or len(shaped) <= p2:
            continue
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)


def get_keypoints(frame):
    h, w, _ = frame.shape
    h, w = (int(256 * w / h), 256) if h > w else (256, int(256 * w / h))
    h, w = h - h % 32, w - w % 32
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), h, w)
    input_img = tf.cast(img, dtype=tf.int32)

    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((-1, 17, 3))

    keypoints_with_scores_filter = remove_duplicates(keypoints_with_scores, 0.1)

    return keypoints_with_scores_filter
