import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

from anchor import generate_anchors, anchors_of_feature_map
from config import Config
from model import Net
from utils import change_coordinate, change_coordinate_inv, seek_model, save_bounding_boxes_image, nms
from evaluation_metrics import softmax
from advertorch.attacks import LinfPGDAttack, L2MomentumIterativeAttack, LinfMomentumIterativeAttack, LBFGSAttack, SinglePixelAttack, LocalSearchAttack

import pdb

device = torch.device(Config.DEVICE)


class Detector(object):

    def __init__(self, model, image_size=Config.IMAGE_SIZE, threshold=Config.PREDICTION_THRESHOLD):
        if type(model) == str:
            checkpoint = torch.load(seek_model(model))
            self.model = Net().to(device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model = model
        self.model.eval()
        self.threshold = threshold
        self.image_size = image_size

        anchor_configs = (
            Config.ANCHOR_STRIDE,
            Config.ANCHOR_SIZE,
        )

    def convert_predictions(self, predictions, path, anchors):
        # get sorted indices by score

        scores, klass = torch.max(softmax(predictions[:, 4:]), dim=1)
        inds = klass != 0
        print(scores.size())

        scores, klass, predictions, anchors = \
            scores[inds], klass[inds], predictions[inds], anchors[inds]

        if len(scores) == 0:
            return None

        scores, inds = torch.sort(scores, descending=True)
        print(scores.size())
        klass, predictions, anchors = klass[inds], predictions[inds], anchors[inds]
        
        
        #with torch.no_grad():
        #    np.save("./scores.npy", scores)
        #    np.save("./klass.npy", klass)
        #    np.save("./predictions.npy", predictions)
        #    np.save("./anchors.npy", anchors)

        # inds = scores > self.threshold
        # scores, klass, predictions, anchors = \
        #     scores[inds], klass[inds], predictions[inds], anchors[inds]

        scores, klass, predictions, anchors = \
            scores[:200], klass[:200], predictions[:200], anchors[:200]

        if len(predictions) == 0:
            return None
        anchors = anchors.to(device).float()

        x = (predictions[:, 0] * anchors[:, 2] + anchors[:, 0])
        y = (predictions[:, 1] * anchors[:, 3] + anchors[:, 1])
        w = (torch.exp(predictions[:, 2]) * anchors[:, 2])
        h = (torch.exp(predictions[:, 3]) * anchors[:, 3])

        bounding_boxes_data = torch.stack((x, y, w, h), dim=1).cpu().data.numpy()
        bounding_boxes_data = change_coordinate_inv(bounding_boxes_data)

        scores_data = scores.cpu().data.numpy()
        klass_data = klass.cpu().data.numpy()
        bboxes_scores = np.hstack(
            (bounding_boxes_data, np.array(list(zip(*(scores_data, klass_data)))))
        )
        #print("bounding_boxes:", bounding_boxes.shape(), bounding_boxes)
        #print("bboxes_scores:", bboxes_scores(), bboxes_scores)

        # nms
        keep = nms(bboxes_scores)
        
        #np.save("./bounding_boxes.npy", bounding_boxes)
        #np.save("./bboxes_scores.npy", bboxes_scores)
        #np.save("./keep.npy", keep)
        #np.save("./bboxes_scores_keep.npy", bboxes_scores[keep])
        bounding_boxes = torch.stack((x, y, w, h), dim=1)
        return bounding_boxes[keep], bounding_boxes, scores

    def forward(self, batched_data):
        """predict with pytorch dataset output

        Args:
            batched_data (tensor): yield by the dataset
        Returns: predicted coordinate and score
        """
        images = batched_data[0].permute(0, 3, 1, 2).to(device).float()
        predictions = list(zip(*list(self.model(images))))
        result = []

        for i, prediction in enumerate(predictions):
            prediction = list(prediction)
            anchors = []
            for k, feature_map_prediction in enumerate(prediction):
                # create anchors of this feature_map_prediction layer

                if (k % 2) == 0:
                    anchors.append( np.array( anchors_of_feature_map(
                        Config.ANCHOR_STRIDE[k//2],
                        Config.ANCHOR_SIZE[k//2],
                        feature_map_prediction.size()[1:])))

                prediction[k] = feature_map_prediction \
                    .view(feature_map_prediction.size()[0], -1) \
                    .permute(1, 0).contiguous()

            reg_preds = torch.cat(prediction[::2])
            cls_preds = torch.cat(prediction[1::2])

            anchors = torch.tensor(np.vstack(anchors))

            result.append(self.convert_predictions(
                torch.cat((reg_preds, cls_preds), dim=1),
                batched_data[2][i], anchors))

        return result

    def infer(self, image):
        image = cv2.imread(image)
        #print(image)
        #print(np.array([104, 117, 123], dtype=np.uint8))
        image = image - np.array([104, 117, 123], dtype=np.uint8)

        _input = torch.tensor(image).permute(2, 0, 1).float() \
            .to(device).unsqueeze(0)

        predictions = self.model(_input)
        #print(predictions.size())
        # flatten predictions
        reg_preds = []
        cls_preds = []
        anchors = []
        for index, prediction in enumerate(predictions):
            if (index % 2) == 0:
                anchors.append( np.array( anchors_of_feature_map (
                    Config.ANCHOR_STRIDE[index//2],
                    Config.ANCHOR_SIZE[index//2],
                    prediction.size()[2:]
                )))

            predictions[index] = prediction.squeeze().view(prediction.size()[1], -1).permute(1, 0)
        
        anchors = torch.tensor(np.vstack(anchors))
        reg_preds = torch.cat(predictions[::2])
        cls_preds = torch.cat(predictions[1::2])
        #print(reg_preds[0:10])
        #print(cls_preds[0:10])
        #print(torch.max(softmax(cls_preds[0:10, :]), dim=1))

        return self.convert_predictions(torch.cat((reg_preds, cls_preds), dim=1), None, anchors)


class loss_attack(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, predictions_1, y):
        
        """ Calculate the identified loss
        Args:
            outputs: include all the proposed bounding boxes whose scores are higher than threshold. It is not the original output 
                    of model, but need to pick up some of them.
            y: the ground truth, could generate from the original result in the main function.
        Returns:
            Calculate the IoU of all the boxes in outputs, and find out Td in which the IoU is higher than threshold and the Fd.
            Then calculate the final loss with Td and Fd.
        """
        
        reg_preds = []
        cls_preds = []
        anchors = []
        for index, prediction in enumerate(predictions_1):
            if (index % 2) == 0:
                anchors.append( np.array( anchors_of_feature_map (
                    Config.ANCHOR_STRIDE[index//2],
                    Config.ANCHOR_SIZE[index//2],
                    prediction.size()[2:]
                )))
            predictions_1[index] = prediction.squeeze().view(prediction.size()[1], -1).permute(1, 0)
        anchors = torch.tensor(np.vstack(anchors))
        reg_preds = torch.cat(predictions_1[::2])
        cls_preds = torch.cat(predictions_1[1::2])
        
        predictions = torch.cat((reg_preds, cls_preds), dim=1)
        scores, klass = torch.max(softmax(predictions[:, 4:]), dim=1)
        inds = klass != 0
        #print(scores.size())

        scores, klass, predictions, anchors = \
            scores[inds], klass[inds], predictions[inds], anchors[inds]

        if len(scores) == 0:
            print("scores=0")
            #rval = x + delta.data
            #return rval

        scores, inds = torch.sort(scores, descending=True)
        #print(scores.size())
        klass, predictions, anchors = klass[inds], predictions[inds], anchors[inds]
        
        
        #with torch.no_grad():
        #    np.save("./scores.npy", scores)
        #    np.save("./klass.npy", klass)
        #    np.save("./predictions.npy", predictions)
        #    np.save("./anchors.npy", anchors)

        # inds = scores > self.threshold
        # scores, klass, predictions, anchors = \
        #     scores[inds], klass[inds], predictions[inds], anchors[inds]

        scores, klass, predictions, anchors = \
            scores[:200], klass[:200], predictions[:200], anchors[:200]

        if len(predictions) == 0:
            print("predictions=0")
            #rval = x + delta.data
            #return rval
        anchors = anchors.to(device).float()

        x_coordinate = (predictions[:, 0] * anchors[:, 2] + anchors[:, 0])
        y_coordinate = (predictions[:, 1] * anchors[:, 3] + anchors[:, 1])
        w = (torch.exp(predictions[:, 2]) * anchors[:, 2])
        h = (torch.exp(predictions[:, 3]) * anchors[:, 3])

        bounding_boxes_data = torch.stack((x_coordinate, y_coordinate, w, h), dim=1).cpu().data.numpy()
        bounding_boxes_data = change_coordinate_inv(bounding_boxes_data)

        scores_data = scores.cpu().data.numpy()
        klass_data = klass.cpu().data.numpy()
        bboxes_scores = np.hstack(
            (bounding_boxes_data, np.array(list(zip(*(scores_data, klass_data)))))
        )
        #print("bounding_boxes:", bounding_boxes.shape(), bounding_boxes)
        #print("bboxes_scores:", bboxes_scores(), bboxes_scores)

        # nms
        keep = nms(bboxes_scores)
        
        #np.save("./bounding_boxes.npy", bounding_boxes)
        #np.save("./bboxes_scores.npy", bboxes_scores)
        #np.save("./keep.npy", keep)
        #np.save("./bboxes_scores_keep.npy", bboxes_scores[keep])
        bounding_boxes = torch.stack((x_coordinate, y_coordinate, w, h), dim=1)
        
        outputs = bounding_boxes
        
        threshold_p = 0.3
    
        Td_index = []
        #Td_scores = []
        Td_log = []
        all_index = list(range(len(outputs)))
        all_scores = scores
        
        for index_outputs, box_outputs in enumerate(outputs):
            x_outputs, y_outputs, w_outputs, h_outputs = box_outputs
            for index_y, box_y in enumerate(y):
                x_y, y_y, w_y, h_y = box_y
                delta_w = min(x_outputs + w_outputs, x_y + w_y) - max(x_outputs, x_y)
                delta_h = min(y_outputs + h_outputs, y_y + h_y) - max(y_outputs, y_y)
                IoU = delta_w * delta_h / (w_outputs * h_outputs + w_y * h_y - delta_w * delta_h)
                if IoU > threshold_p:
                    #Td_scores.append(score_outputs)
                    Td_log.append(torch.log(1 - all_scores[index_outputs]))
                    Td_index.append(index_outputs)
                    break
                else:
                    continue
        Fd =  [x for x in all_index if x not in Td_index]
        Fd_scores = [all_scores[i] for i in Fd]
        Fd_log = [torch.log(x) for x in Fd_scores]
        Td_sum = sum(Td_log)
        Fd_sum = sum(Fd_log)
        
        return Td_sum + Fd_sum

def main(args):
    print('predicted bounding boxes of faces:')
    print(args.image)
    #pdb.set_trace()
    bboxes, bboxes_many, bboxes_many_scores = Detector(args.model).infer(args.image)
    #print(bboxes)
    #if args.save_to:
    #    save_bounding_boxes_image(args.image, bboxes, args.save_to)
    ground_truth = bboxes
    attacks_try = bboxes_many
    #print("ground_truth:",ground_truth)
    #print("attacks_try:", attacks_try)
    
    #loss_try = loss_attack(attacks_try, bboxes_many_scores, ground_truth)
    
    #print(loss_try)
    
    
    adversary = L2MomentumIterativeAttack(
        Detector(args.model).model, loss_fn=loss_attack(), eps=3,
        nb_iter=40, eps_iter=0.1, decay_factor=1., clip_min=0., clip_max=255.,
        targeted=False)
    image = cv2.imread(args.image)
    image = image - np.array([104, 117, 123], dtype=np.uint8)
    image = torch.tensor(image).permute(2, 0, 1).float() \
        .to(device).unsqueeze(0)
    adv_img = adversary.perturb(image, ground_truth)
    print(adv_img)
    #cv2.imshow('adv', adv_img)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--image', type=str,
                        help='image to be predicted')
    parser.add_argument('--model', type=str,
                        help='model to use, could be epoch number, model file '
                             'name or model file absolute path')
    parser.add_argument('--keep', type=int, default=150,
                        help='how many predictions to keep, default: 150')
    parser.add_argument('--save_to', type=str,
                        help='save the image with bboxes to a file')

    args = parser.parse_args()
    main(args)
