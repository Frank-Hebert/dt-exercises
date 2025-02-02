#!/usr/bin/env python3

import numpy as np
import cv2
from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs

DATASET_DIR = "../../dataset/sim"


npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1
        print(npz_index)


def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    image = seg_img

    mask_duckie = (image == [100, 117, 226]).all(-1) * 1
    mask_cone = (image == [226, 111, 101]).all(-1) * 2
    mask_truck = (image == [116, 114, 117]).all(-1) * 3
    mask_bus = (image == [216, 171, 15]).all(-1) * 4

    # Resize Mask 224x224 !!!!!!!!!!

    mask_duckie = cv2.resize(mask_duckie.copy(), (224, 224), interpolation=cv2.INTER_NEAREST)
    mask_cone = cv2.resize(mask_cone.copy(), (224, 224), interpolation=cv2.INTER_NEAREST)
    mask_truck = cv2.resize(mask_truck.copy(), (224, 224), interpolation=cv2.INTER_NEAREST)
    mask_bus = cv2.resize(mask_bus.copy(), (224, 224), interpolation=cv2.INTER_NEAREST)




    duckie_uint8 = mask_duckie.astype('uint8')
    cone_uint8 = mask_cone.astype('uint8')
    truck_uint8 = mask_truck.astype('uint8')
    bus_uint8 = mask_bus.astype('uint8')


    contours_duck, _ = cv2.findContours(duckie_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_cone, _ = cv2.findContours(cone_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_truck, _ = cv2.findContours(truck_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_bus, _ = cv2.findContours(bus_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    classes = []

    for i in range(len(contours_duck)):
        xmin, ymin, w, h = cv2.boundingRect(contours_duck[i])
        xmax = xmin + w - 1
        ymax = ymin + h - 1
        if (w > 3) and (h > 3):
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(1)

    for i in range(len(contours_cone)):
        xmin, ymin, w, h = cv2.boundingRect(contours_cone[i])
        xmax = xmin + w - 1
        ymax = ymin + h - 1
        if (w > 3) and (h > 3):
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(2)

    for i in range(len(contours_truck)):
        xmin, ymin, w, h = cv2.boundingRect(contours_truck[i])
        xmax = xmin + w - 1
        ymax = ymin + h - 1
        if (w > 3) and (h > 3):
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(3)

    for i in range(len(contours_bus)):
        xmin, ymin, w, h = cv2.boundingRect(contours_bus[i])
        xmax = xmin + w - 1
        ymax = ymin + h - 1
        if (w > 3) and (h > 3):
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(4)

    return np.array(boxes), np.array(classes)

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500
MAX_IMAGES = 1000 #size of dataset
while True:
    if npz_index > MAX_IMAGES:
        break
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)


        if nb_of_steps % 3 == 0:
            # TODO boxes, classes = clean_segmented_image(segmented_obs)
            boxes, classes = clean_segmented_image(segmented_obs)
            if len(boxes) == 0:
                continue
            save_npz(cv2.resize(obs, (224, 224)), boxes, classes)
        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break