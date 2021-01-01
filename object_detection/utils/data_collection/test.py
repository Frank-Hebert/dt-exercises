import numpy as np
import cv2
from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask



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

image = cv2.imread("/home/francoishebert/Documents/ift6757/exercise 3 - 31 decembre/dt-exercises/object_detection/utils/data_collection/image_test.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes, classe = clean_segmented_image(image)
print(classe)



# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
