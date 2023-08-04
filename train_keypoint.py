import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from time import time
import kornia as K
import cv2
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

train = False
lr = 0.0001
epochs = 1000
grid_points = 16
epochs = 100
grid_points = 24
contrast_threshold = 10
label_widht = 180 - 11.5
label_heigth = 60
mm_px = 0.1

# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load image
input_image = cv2.imread(r'G:\Otros\DeepLearning\image_stitching\dataset\kefir_new\0\uw1.png')


# resize the image to the user requirements
input_image = cv2.resize(input_image, (int(label_widht / mm_px), int(label_heigth / mm_px)))

input_image_tensor = (K.color.bgr_to_rgb(K.image_to_tensor(input_image, False)).float() / 255.).to(device)

# plot rectangles in image
output_image = input_image.copy()

# load model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

if train:
    # create grid of rectangles
    grid = []
    for i in range(grid_points):
        for j in range(grid_points):
            grid.append([i*input_image.shape[1]//grid_points, j*input_image.shape[0]//grid_points, (i+1)*input_image.shape[1]//grid_points, (j+1)*input_image.shape[0]//grid_points])

    boxes = []
    keypoints = []
    labels = []

    # iterate
    for i in range(len(grid)):

        # get contrast of rectangle
        contrast = cv2.Laplacian(input_image[grid[i][1]:grid[i][3], grid[i][0]:grid[i][2]], cv2.CV_64F).var()

        if contrast > contrast_threshold:

            # add rectangle to list
            boxes.append(grid[i])

            # add keypoints to list x, y, visuability
            keypoints.append([[(grid[i][0]+grid[i][2])/2, (grid[i][1]+grid[i][3])/2, 1]])

            # create label
            labels.append(i)

            # draw rectangle
            cv2.rectangle(output_image, (grid[i][0], grid[i][1]), (grid[i][2], grid[i][3]), (0, 255, 0), 2)
            
            # draw center
            cv2.circle(output_image, (int((grid[i][0]+grid[i][2])/2), int((grid[i][1]+grid[i][3])/2)), 2, (0, 0, 255), 2)

    boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
    keypoints = torch.tensor(keypoints, dtype=torch.float32).to(device)
    labels = torch.tensor([1]*len(boxes), dtype=torch.int64).to(device)
    # labels = torch.tensor(labels, dtype=torch.int64).to(device)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["keypoints"] = keypoints

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


    model.to(device).train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # start progress bar
    pbar = tqdm(total=epochs)

    for i in range(epochs):

        loss_dict = model(input_image_tensor, [target])

        losses = sum(loss for loss in loss_dict.values())

        # zero gradients
        optimizer.zero_grad()

        # compute gradients
        losses.backward()

        # update weights
        optimizer.step()

        # update progress bar and close it
        pbar.set_postfix_str("loss %s" % round(round(float(losses), 3), 3))
        pbar.update(1)

    # save model
    torch.save(model.state_dict(), os.path.join('models', app_config['APP']['Product'], 'keypoint_detector.pth'))
else:
    model.load_state_dict(torch.load(os.path.join('models', app_config['APP']['Product'], 'keypoint_detector.pth')))
    model.to(device).eval()

    # inference
    start = time()
    predictions = model(input_image_tensor)  
    keypoints = predictions[0]['keypoints'].detach().cpu().numpy()
    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    print(str((time() - start)*1000))

    for i in range(len(predictions[0]['scores'])):
        if predictions[0]['scores'][i] > 0.5:
            cv2.rectangle(output_image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
            for j in range(len(keypoints[i])):
                cv2.circle(output_image, (int(keypoints[i][j][0]), int(keypoints[i][j][1])), 2, (0, 0, 255), 2)
    # imshow
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()