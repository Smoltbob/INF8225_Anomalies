"""
This file contains all the utilities for the second method of abnormal event detection in videos.
"""
import os
import numpy as np
import cv2
import pandas as pd
import time
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from collections import deque
from natsort import natsorted
import scipy.io as scio
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

# Root directory:
root_dir_name = os.getcwd()
using_my_laptop = False
data_name = 'ped1'

training_size_ratio = 0.8

sequence_stride = 3
white_color_value = 255
roi_size = 64
detection_score_threshold = 0.5
max_queue_size = 3

if using_my_laptop:
    path_dir = 'E:'
else:
    path_dir = '/run/media/paroyc/My Passport'
data_dir = os.path.join(path_dir, 'Dataset/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSD{}'.format(data_name))
train_dir_name = 'Train'
train_dir = os.path.join(data_dir, train_dir_name)
test_dir_name = 'Test'
test_dir = os.path.join(data_dir, test_dir_name)
gt_dir = 'groundtruths'
gt_filename = '{}.mat'.format(data_name)

motion_data_name = 'motion_data_v1'
summary_results_filename = 'summary_results.csv'
detect_results_filename = 'detect_results.csv'
global_summary_filename = 'global_summary.log'
global_detect_summary_filename = 'global_detect_summary.log'
test_score_filename = 'test_scores.csv'
result_dir_name = 'results'
model_dir_name = 'models'

accepted_image_extensions = ['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif',
                             '.ppm', '.xbm', '.tiff', '.rgb', '.pgm', '.png', '.pnm']


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
pre_trained_model_path = 'pretrained_models'
retina_net_model_path = os.path.join(path_dir, pre_trained_model_path, 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
retina_net_model = models.load_model(retina_net_model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}


class ExtractedData:
    """
    This class is the one used in this method.
    """
    def __init__(self):
        self.training_set = []
        self.test_set = []
        self.data_name = 'motion_data'

    def add_data(self, video_name, open_if_exist=True, save_data=True):
        """
        Extracts the motion data  which will be used for training or testing of the model.
        The data file will be saved in the provided video folder.
        Each row in the csv file follows the following format: [frame_num, bbox, HOG_feature, Motion_Entropy_feature].
        The first 100 frames are used to stabilize SuBSENSE and thus they are not taken into account for motion data.
        :param video_name:
        :param open_if_exist:
        :param save_data:
        :return:
        """
        print('=====================================================================')
        motion_data = []
        is_train_data = train_dir_name.lower() in video_name.lower()
        video_path = train_dir if is_train_data else test_dir
        video_dir = os.path.join(video_path, video_name)

        data_filename = '{}_{}.h5'.format(video_name, self.data_name)
        data_file_complete_path = os.path.join(train_dir, data_filename)

        if open_if_exist and os.path.isfile(data_file_complete_path):
            print('Video {} ---> importing data from file:'.format(video_name))
            motion_data = pd.DataFrame(pd.read_hdf(data_file_complete_path, 'data')).to_dict('records')
            print('                                                                done!')
        else:
            print('Video {} ---> constructing data from frames:'.format(video_name))
            video_frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and
                            any(f[f.rfind('.'):] == ext for ext in accepted_image_extensions)]
            video_frames = natsorted(video_frames)

            frames_queue = deque([], max_queue_size)
            prev_index = 0
            curr_index = 1
            next_index = 2

            for i, frame_filename in enumerate(video_frames):
                curr_frame = cv2.imread(os.path.join(video_dir, frame_filename))

                if i % sequence_stride == 0:
                    frames_queue.append(curr_frame)
                    if len(frames_queue) == max_queue_size:
                        print('Video {} ---> frame {}:'.format(video_name, i))
                        # Run the detection using RetinaNet
                        # preprocess image for network
                        image = preprocess_image(frames_queue[curr_index])
                        image, scale = resize_image(image)
                        boxes, scores, labels = retina_net_model.predict_on_batch(np.expand_dims(image, axis=0))

                        # correct for image scale
                        boxes /= scale

                        for box, score, label in zip(boxes[0], scores[0], labels[0]):
                            # scores are sorted so we can break
                            if score < detection_score_threshold:
                                break
                            (x1, y1, x2, y2) = box.astype('int')

                            # Get ROI image
                            curr_roi_image = cv2.resize(frames_queue[curr_index][y1:y2, x1:x2], (roi_size, roi_size))
                            prev_roi_image = cv2.resize(frames_queue[prev_index][y1:y2, x1:x2], (roi_size, roi_size))
                            next_roi_image = cv2.resize(frames_queue[next_index][y1:y2, x1:x2], (roi_size, roi_size))

                            # Gray-scaled
                            curr_gray_roi_image = cv2.cvtColor(curr_roi_image, cv2.COLOR_BGR2GRAY)
                            prev_gray_roi_image = cv2.cvtColor(prev_roi_image, cv2.COLOR_BGR2GRAY)
                            next_gray_roi_image = cv2.cvtColor(next_roi_image, cv2.COLOR_BGR2GRAY)

                            # Previous frames difference
                            prev_roi_diff = np.abs(curr_gray_roi_image - prev_gray_roi_image)

                            # Future frames difference
                            next_roi_diff = np.abs(curr_gray_roi_image - next_gray_roi_image)

                            # Normalize images
                            norm_curr_roi = cv2.normalize(curr_gray_roi_image, None, alpha=0, beta=1,
                                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            norm_prev_dif = cv2.normalize(prev_roi_diff, None, alpha=0, beta=1,
                                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            norm_next_dif = cv2.normalize(next_roi_diff, None, alpha=0, beta=1,
                                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                            # Append the results
                            dict_motion_data = dict()
                            dict_motion_data['video_name'] = video_name
                            dict_motion_data['frame_number'] = i
                            dict_motion_data['label'] = label
                            dict_motion_data['bounding_box'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                            dict_motion_data['roi_image'] = norm_curr_roi
                            dict_motion_data['prev_diff'] = norm_prev_dif
                            dict_motion_data['next_diff'] = norm_next_dif
                            motion_data.append(dict_motion_data)
            print('                                                                done!')
            if save_data:
                print('Video {} ---> saving data:'.format(video_name))
                pd.DataFrame(motion_data).to_hdf(data_file_complete_path, 'data')
                print('                                                                done!')

        if is_train_data:
            print('Video {} ---> adding training data:'.format(video_name))
            self.training_set += motion_data
            print('                                                                done!')
        else:
            print('Video {} ---> adding testing data:'.format(video_name))
            self.test_set += motion_data
            print('                                                                done!')

    def get_training_validation_data(self, train_split_ratio=training_size_ratio):
        # Shuffle the training set
        random_state_value = int((time.time() - int(time.time())) * 1e6)
        np.random.seed(random_state_value)
        train_permutation = np.random.permutation(len(self.training_set))
        roi_image_data = np.array([d['roi_image'] for d in self.training_set])
        prev_diff_data = np.array([d['prev_diff'] for d in self.training_set])
        next_diff_data = np.array([d['next_diff'] for d in self.training_set])
        roi_image_data = roi_image_data[train_permutation]
        prev_diff_data = prev_diff_data[train_permutation]
        next_diff_data = next_diff_data[train_permutation]
        # Get the index that separates from training data to validation data
        train_sep_index = int(train_split_ratio * len(self.training_set))
        roi_image_train_data = roi_image_data[:train_sep_index, :]
        prev_diff_train_data = prev_diff_data[:train_sep_index, :]
        next_diff_train_data = next_diff_data[:train_sep_index, :]
        roi_image_validation_data = roi_image_data[train_sep_index:, :]
        prev_diff_validation_data = prev_diff_data[train_sep_index:, :]
        next_diff_validation_data = next_diff_data[train_sep_index:, :]
        return roi_image_train_data, prev_diff_train_data, next_diff_train_data, roi_image_validation_data, \
            prev_diff_validation_data, next_diff_validation_data, random_state_value

    def get_testing_data(self, video_name):
        roi_image_data = np.array([d['roi_image'] for d in self.test_set if d.get('video_name') == video_name])
        prev_diff_data = np.array([d['prev_diff'] for d in self.test_set if d.get('video_name') == video_name])
        next_diff_data = np.array([d['next_diff'] for d in self.test_set if d.get('video_name') == video_name])
        return roi_image_data, prev_diff_data, next_diff_data

    def get_results(self, video_name, scores):
        """
        Gets the frame-level reconstruction error scores for all test sequences.
        :return:
        """
        test_data = [d for d in self.test_set if d.get('video_name') == video_name]
        assert len(test_data) == len(scores), 'Something is wrong with scores.'

        # Associate scores in test data
        result_data = []
        f_test = []
        for data, score in zip(test_data, scores):
            data['score'] = score
            result_data.append(data)
            f_test.append(data['frame_number'])

        # get the total frames of sub video
        num_frames = len(os.listdir(os.path.join(test_dir, video_name)))
        full_frames = list(range(1, num_frames + 1))

        # get all fame number used in the test data
        test_frames = list(set(d['frame_number'] for d in test_data))
        test_frames.sort()

        test_scores = []
        for f in test_frames:
            frame_scores = [d['score'] for d in result_data if d.get('frame_number') == f]
            test_scores.append(np.max(frame_scores))

        # Interpolate in order to fill missing frames
        test_interp = interpolate.interp1d(test_frames, test_scores, fill_value='extrapolate')
        full_scores = test_interp(full_frames)

        # Smooth scores using Gaussian 1d filter
        final_scores = gaussian_filter1d(full_scores, 50)

        # Normalize
        norm_scores = np.interp(final_scores, (final_scores.min(), final_scores.max()), (0, 1))

        return full_frames, norm_scores


def get_gt_results():
    """
    Get the frame-level ground-truth results.
    Inspired from:
    https://github.com/StevenLiuWen/ano_pred_cvpr2018/blob/master/Codes/evaluate.py
    :return:
    """
    abnormal_events = scio.loadmat(os.path.join(gt_dir, gt_filename), squeeze_me=True)['gt']

    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

    num_video = abnormal_events.shape[0]
    video_list = [v for v in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, v))
                  and test_dir_name.lower() in v.lower()]
    video_list.sort()

    assert num_video == len(video_list), 'ground truth does not match the number of testing videos. {} != {}' \
        .format(num_video, len(video_list))

    # get the total frames of sub video
    def get_video_length(sub_video_number):
        # video_name = video_name_template.format(sub_video_number)
        video_name = os.path.join(test_dir, video_list[sub_video_number])
        assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)
        return len(os.listdir(video_name))

    gt = []
    for i in range(num_video):
        length = get_video_length(i)

        sub_video_gt = np.zeros((length,), dtype=np.int8)
        sub_abnormal_events = abnormal_events[i]
        if sub_abnormal_events.ndim == 1:
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

        _, num_abnormal = sub_abnormal_events.shape

        for j in range(num_abnormal):
            # (start - 1, end - 1)
            start = sub_abnormal_events[0, j] - 1
            end = sub_abnormal_events[1, j]

            sub_video_gt[start: end] = 1

        gt.append(sub_video_gt.tolist())

    return gt
