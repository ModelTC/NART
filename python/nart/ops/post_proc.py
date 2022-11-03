# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Post-processing operations.
"""
from .op import Op, GroupType
import numpy as np
from ..core import Graph
from ..core.art import Dtype

import logging

logger = logging.getLogger("nart.op.post_proc")


class RetinanetDet(Op, domain="UP", version=1):
    """The retinanet detection operation which is consistent with UP.
    Inputs:
        scores_i: (N, NUM_CLASS * NUM_ANCHOR, H, W), the scores of each class. *NOTE*: we are using **class_is_before** type, which may be false in deploying models.
        bbox_delta_i: (N, NUM_ANCHOR * 4, H, W), the predict bbox delta.
        anchors [optional]: (NUM_LEVEL, NUM_ANCHOR, 4) the anchors, if not given, the anchors are generated from scale & ratio.

    Outputs:
        bbox_pred: (N, aft_top_k, 6), the predict bboxes, each box is [x1, y1, x2, y2, confidence, class_id].
    """

    attr_dict = {
        "num_levels": (Op.INT,),  # indicates the number of level in FPN.
        "num_class": (Op.INT,),  # the number of classes
        "rpn_stride": (Op.LIST_OF_INTS,),
        "rpn_top_n": (Op.LIST_OF_INTS,),
        "aft_top_k": (Op.INT,),  # the maximum number of output after detection.
        "nms_iou_thresh": (Op.FLOAT,),  # the threshold used in nms.
        "confidence_thresh": (
            Op.LIST_OF_FLOATS,
        ),  # the threshold to filter out bbox before nms.
        # attributes related to anchors.
        "anchor_ratios": (Op.LIST_OF_FLOATS, []),
        "anchor_scales": (Op.LIST_OF_FLOATS, []),
        "im_info": (Op.LIST_OF_FLOATS,),  # the width and height of image.
    }

    def infer_shape(self):
        # TODO: add some check one attributes and shapes
        graph = self.owning_graph
        assert isinstance(graph, Graph)
        aft_top_k = self.get_attribute_value("aft_top_k")
        N, C, H, W = graph.get_tensor_shape(self.input[0])
        graph.set_tensor_shape(self.output[0], [N, aft_top_k, 6])

    def get_anchors(self):
        """Generate/Get the anchors in the form of numpy array.
        return: numpy.ndarray with shape [NUM_LEVEL, NUM_RATIOS * NUM_SCALES, 4]
        """
        num_levels = self.get_attribute_value("num_levels")
        anchor_idx = 2 * num_levels
        if self.has_input(anchor_idx):
            # have anchor input
            return self.owning_graph.get_const_tensor_as_array(
                self.input[anchor_idx], False
            )

        # generate anchor from scale & ratio
        scales = self.get_attribute_value("anchor_scales")
        ratios = self.get_attribute_value("anchor_ratios")
        ratios = np.array(ratios, np.float32)
        strides = self.get_attribute_value("rpn_stride")
        mlvl_anchors = []
        for stride in strides:
            w, h = stride, stride
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
            # enumerate on ratios
            size = w * h
            size_ratios = size / ratios
            ws = np.round(np.sqrt(size_ratios))
            hs = np.round(ws * ratios)
            # enumerate on scales
            ws = [scale * ws for scale in scales]
            hs = [scale * hs for scale in scales]
            ws = np.stack(ws, axis=-1)
            hs = np.stack(hs, axis=-1)
            # anchors' shape will be [NUM_RATIOS, NUM_SCALES, 4]
            anchors = np.stack(
                [
                    x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1),
                ],
                axis=-1,
            )
            # reshape it to [NUM_RATIOS * NUM_SCALES, 4]
            anchors = np.reshape(anchors, [anchors.shape[0] * anchors.shape[1], 4])
            mlvl_anchors.append(anchors)
        mlvl_anchors = np.stack(mlvl_anchors)
        return mlvl_anchors

    def numpy_reference_impl(self, bbox_deltas, scores, im_info):
        """A reference implementation with numpy."""
        num_class = self.get_attribute_value("num_class")

        def xyxy2xywh(boxes, stacked=False):
            """(x1, y1, x2, y2) -> (x, y, w, h)"""
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + 0.5 * (w - 1)
            cy = y1 + 0.5 * (h - 1)
            if stacked:
                return np.stack([cx, cy, w, h], dim=1)
            else:
                return cx, cy, w, h

        def offset2bbox(boxes, offset):
            """
            Forward transform that maps proposal boxes to predicted ground-truth
            boxes using bounding-box regression deltas(offset). See bbox_transform_inv
            for a description of the weights argument.

            boxes: the prior boxes, [K, 4]
            offset: the predict bbox delta, [K, 4]
            """
            ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

            dx = offset[:, 0::4]
            dy = offset[:, 1::4]
            dw = offset[:, 2::4]
            dh = offset[:, 3::4]

            # Prevent sending too large values into np.exp()
            # dw = torch.clamp(dw, max=np.log(1000. / 16.))
            # dh = torch.clamp(dh, max=np.log(1000. / 16.))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = np.exp(dw) * widths[:, None]
            pred_h = np.exp(dh) * heights[:, None]

            pred_boxes = np.zeros_like(offset)
            # x1, y1, x2, y2
            im_w, im_h = im_info
            pred_boxes[:, 0::4] = np.clip(pred_ctr_x - 0.5 * (pred_w - 1), 1.0, im_w)
            pred_boxes[:, 1::4] = np.clip(pred_ctr_y - 0.5 * (pred_h - 1), 1.0, im_h)
            pred_boxes[:, 2::4] = np.clip(pred_ctr_x + 0.5 * (pred_w - 1), 1.0, im_w)
            pred_boxes[:, 3::4] = np.clip(pred_ctr_y + 0.5 * (pred_h - 1), 1.0, im_h)

            return pred_boxes

        def cpu_decode_bbox(bbox_deltas, scores):
            mlvl_anchors = self.get_anchors()
            strides = self.get_attribute_value("rpn_stride")

            num_class = self.get_attribute_value("num_class")
            bbox_for_class = list()
            for i in range(num_class):
                bbox_for_class.append(list())

            for level in range(5):
                H, W = bbox_deltas[level].shape[2:4]
                stride = strides[level]
                anchors = mlvl_anchors[level]
                num_anchors = anchors.shape[0]
                prior_boxes = np.broadcast_to(
                    anchors.reshape([num_anchors, 1, 1, 4]), [num_anchors, H, W, 4]
                )
                prior_boxes = np.ascontiguousarray(prior_boxes)

                for i in range(W):
                    for j in range(H):
                        prior_boxes[:, j, i, 0] += stride * i
                        prior_boxes[:, j, i, 2] += stride * i
                        prior_boxes[:, j, i, 1] += stride * j
                        prior_boxes[:, j, i, 3] += stride * j
                prior_boxes = prior_boxes.reshape([num_anchors * H * W, 4])

                loc_delta = bbox_deltas[level]
                loc_delta = np.reshape(loc_delta, [4, num_anchors * H * W]).transpose(
                    [1, 0]
                )
                bbox = offset2bbox(prior_boxes, loc_delta)

                for cls_id in range(num_class):
                    score = scores[level]
                    score = np.reshape(score, [num_class, score.size // num_class])[
                        cls_id
                    ]
                    score = np.reshape(score, [score.size, 1])
                    bbox_for_class[cls_id].append(np.concatenate([score, bbox], axis=1))

            bbox_for_class = [np.concatenate(item, axis=0) for item in bbox_for_class]
            return bbox_for_class

        def iou(box, boxes):
            x1 = np.maximum(box[0], boxes[:, 0])
            y1 = np.maximum(box[1], boxes[:, 1])
            x2 = np.minimum(box[2], boxes[:, 2])
            y2 = np.minimum(box[3], boxes[:, 3])
            area_inter = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
            area0 = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            area1 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            return area_inter / (area0 + area1 - area_inter)

        def nms(boxes, iou_thresh, top_k, conf_thresh):
            indices = np.argsort(
                boxes[:, 0],
            )
            indices = np.flip(indices)
            boxes = boxes[indices]
            num_box = len(boxes)
            suppressed = np.zeros(shape=[num_box], dtype=np.bool8)
            result = list()
            for i in range(num_box):
                if suppressed[i]:
                    continue
                box = boxes[i]
                if box[0] < conf_thresh:
                    break
                result.append(box)
                ious = iou(box[1:5], boxes[:, 1:5])
                # print(ious)
                suppressed[i + 1 :] = np.logical_or(suppressed, ious > iou_thresh)[
                    i + 1 :
                ]
                if len(result) >= top_k:
                    break
            if len(result) == 0:
                return np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], np.float32)
            result = np.stack(result, axis=0)
            return result

        def shuffle_channel(array, group):
            N, C = array.shape[:2]
            SPATIAL = array.shape[2:]
            array = np.reshape(array, [N, group, C // group, *SPATIAL])
            array = np.transpose(array, [0, 2, 1, *range(3, len(array.shape))])
            array = np.reshape(array, [N, C, *SPATIAL])
            return array

        bboxes_for_class = list()

        # transpose bbox_deltas, making it class_before.
        num_anchors = self.get_attribute_value(
            "anchor_ratios"
        ) * self.get_attribute_value("anchor_scales")
        bbox_deltas = [
            shuffle_channel(bbox_delta, num_anchors) for bbox_delta in bbox_deltas
        ]

        decoded_bbox_for_class = cpu_decode_bbox(bbox_deltas, scores)

        nms_iou_thresh = self.get_attribute_value("nms_iou_thresh")
        aft_top_k = self.get_attribute_value("aft_top_k")
        confidence_thresh = self.get_attribute_value("confidence_thresh")

        for cls_id in range(num_class):
            bboxes = decoded_bbox_for_class[cls_id]
            # print(bboxes)
            box_aft_nms = nms(
                bboxes, nms_iou_thresh, aft_top_k, confidence_thresh[cls_id]
            )
            bboxes_for_class.append(box_aft_nms)
        # finally concat bboxes from all class and keep the top 300
        bbox_pred = np.concatenate(bboxes_for_class, axis=0)

        return bbox_pred


class SiamRPN(Op, domain="UP", version=1):
    """The SiamRPN post-processing op for hermes model.

    Inputs:
        score: A tensor with shape [N, 2*num_anchor, edge_size, edge_size],
            represents the score of each anchor, the sencond axis is
            negative score of each anchor followed by positive score.
        bbox_delta: A tensor with shape [N, 4*num_anchor, edge_size, edge_size],
            represent the bbox delta of each anchor, the layout of second axis is
            "delta x, y, width and height" for each anchor.
        last_target_info: [N, 2], the last targets' scale & ratio.
    """

    GROUP = GroupType.TOOLS

    attr_dict = {
        "anchor_scales": (Op.LIST_OF_FLOATS,),
        "anchor_ratios": (Op.LIST_OF_FLOATS,),
        "round_dight": (Op.INT, 2),
        "anchor_stride": (Op.INT, 4),
        "stride": (Op.INT, 4),
        "penality_k": (Op.FLOAT, 0.04),
        "window_influence": (Op.FLOAT, 0.44),
        "lr_attr": (Op.FLOAT, 0.4),
    }

    def infer_shape(self):
        graph = self.owning_graph
        score_shape = graph.get_tensor_shape(self.input[0])
        N, K, H, W = score_shape
        K = K // 2
        graph.set_tensor_shape(self.output[0], [N, 6])

        def check_shape(name, got, expect):
            if expect != got:
                logger.warning(
                    f"the shape of {name} is not as expected, expect {expect}, got {got}"
                )

        check_shape(
            self.input[0], graph.get_tensor_shape(self.input[0]), [N, 2 * K, H, W]
        )
        check_shape(
            self.input[1], graph.get_tensor_shape(self.input[1]), [N, 4 * K, H, W]
        )
        check_shape(self.input[2], graph.get_tensor_shape(self.input[2]), [N, 2])

    def get_anchors(self):
        """Generate/Get the anchors in the form of numpy array.
        return: numpy.ndarray with shape [NUM_ANCHORS, 4]
        """
        import math

        scales = self.get_attribute_value("anchor_scales")
        ratios = self.get_attribute_value("anchor_ratios")
        stride = self.get_attribute_value("stride")
        round_dight = self.get_attribute_value("round_dight")

        def Round2(number, dight):
            return round(number * pow(10, dight)) * pow(0.1, dight)

        num = len(scales) * len(ratios)
        anchors = np.zeros([num, 4], dtype=np.float32)
        size = stride * stride
        for r_id in range(len(ratios)):
            ratio = ratios[r_id]
            ws = Round2(math.sqrt(size / ratio), round_dight)
            hs = (
                int(ws * ratio)
                if round_dight == 0
                else Round2(math.sqrt(size * ratio), round_dight)
            )
            for s_id in range(len(scales)):
                anchor_id = r_id * len(scales) + s_id
                scale = scales[s_id]
                anchors[anchor_id, 0] = 0
                anchors[anchor_id, 1] = 0
                anchors[anchor_id, 2] = ws * scale
                anchors[anchor_id, 3] = hs * scale
        return anchors

    def get_cosine_window(self):
        """Get the cosine coefficient window for this operator.
        return: numpy.ndarray with shape [H, W]
        """
        cos_window = np.zeros([25, 25], dtype=np.float32)
        height, width = self.owning_graph.get_tensor_shape(self.input[0])[2:]

        def hann(i, size):
            import math

            PI2 = math.pi * 2.0
            return 0.5 * (1.0 - math.cos(PI2 * i / (size - 1)))

        for i in range(0, height):
            for j in range(0, width):
                cos_window[i][j] = hann(i, height) * hann(j, width)
        return cos_window

    def numpy_reference_impl(self, scores, bbox_delta, last_target_info):
        # attributes
        stride = self.get_attribute_value("stride")
        penality_k = self.get_attribute_value("penality_k")
        window_influence = self.get_attribute_value("window_influence")
        lr_attr = self.get_attribute_value("lr_attr")

        ## some helper functions ##
        def sigmoid(x):
            return 1.0 / (np.exp(-x) + 1)

        def size(w, h):
            pad = (w + h) * 0.5
            size2 = (w + pad) * (h + pad)
            return np.sqrt(size2)

        def change(x):
            return np.maximum(x, 1 / x)

        N, C, height, width = scores.shape
        anchor = self.get_anchors()
        anchor = np.reshape(anchor, [*anchor.shape, 1, 1])
        # H&W constant, both [h, w]
        h_const = np.arange(height, dtype=np.float32)
        h_const = np.broadcast_to(h_const.reshape([height, 1]), [height, width])
        w_const = np.arange(width, dtype=np.float32)
        w_const = np.broadcast_to(w_const.reshape([1, width]), [height, width])
        # cosine window
        import math

        def hann(i, size):
            PI2 = math.pi * 2
            return 0.5 * (1.0 - math.cos(PI2 * i / (size - 1)))

        cos_window = np.zeros([25, 25], dtype=np.float32)
        for i in range(0, height):
            for j in range(0, width):
                cos_window[i][j] = hann(i, height) * hann(j, width)

        num_anchor = scores.shape[1] // 2
        score_neg = scores[:, 0:num_anchor]
        score_pos = scores[:, num_anchor:]
        score = sigmoid(score_pos - score_neg)

        ori_h = -(height // 2) * stride
        ori_w = -(width // 2) * stride

        tmp_x = (
            ori_w
            + w_const * stride
            + bbox_delta[:, 0:num_anchor, :, :] * anchor[:, 2, :, :]
        )
        tmp_y = (
            ori_h
            + h_const * stride
            + bbox_delta[:, num_anchor : num_anchor * 2, :, :] * anchor[:, 3, :, :]
        )
        tmp_w = (
            np.exp(bbox_delta[:, num_anchor * 2 : num_anchor * 3, :, :])
            * anchor[:, 2, :, :]
        )
        tmp_h = (
            np.exp(bbox_delta[:, num_anchor * 3 : num_anchor * 4, :, :])
            * anchor[:, 3, :, :]
        )

        scales = last_target_info[:, 0]
        ratios = last_target_info[:, 1]

        r_c = change((tmp_w / tmp_h) / ratios.reshape([N, 1, 1, 1]))
        s_c = change(size(tmp_w, tmp_h) / scales.reshape([N, 1, 1, 1]))
        penality = np.exp((1 - r_c * s_c) * penality_k)

        tmp_score = score
        tmp_score1 = tmp_score * penality
        tmp_score2 = (
            tmp_score1 * (1.0 - window_influence) + cos_window * window_influence
        )

        max_indices = np.zeros(shape=[N, 1], dtype=np.int32)
        for tidx in range(N):
            max_indices[tidx, 0] = np.argmax(tmp_score2[tidx])
        # print(max_indices)
        def take_in_spatial(data, indices):
            N = data.shape[0]
            # flatten spatial axes
            data = data.reshape([N, data.size // N])
            return np.take_along_axis(data, indices=indices, axis=1)

        # target position and score [N, 6]: the layout is (x, y, w, h, score, lr)
        targets = np.concatenate(
            [
                take_in_spatial(tmp_x, max_indices),
                take_in_spatial(tmp_y, max_indices),
                take_in_spatial(tmp_w, max_indices),
                take_in_spatial(tmp_h, max_indices),
                take_in_spatial(tmp_score, max_indices),
                take_in_spatial(tmp_score1, max_indices) * lr_attr,
            ],
            axis=1,
        )

        return targets
