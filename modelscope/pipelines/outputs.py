# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.constant import Tasks

TASK_OUTPUTS = {

    # ============ vision tasks ===================

    # image classification result for single sample
    #   {
    #       "labels": ["dog", "horse", "cow", "cat"],
    #       "scores": [0.9, 0.1, 0.05, 0.05]
    #   }
    Tasks.image_classification: ['scores', 'labels'],
    Tasks.image_tagging: ['scores', 'labels'],

    # object detection result for single sample
    #   {
    #       "boxes": [
    #           [x1, y1, x2, y2],
    #           [x1, y1, x2, y2],
    #           [x1, y1, x2, y2],
    #       ],
    #       "labels": ["dog", "horse", "cow", "cat"],
    #       "scores": [0.9, 0.1, 0.05, 0.05]
    #   }
    Tasks.object_detection: ['scores', 'labels', 'boxes'],

    # instance segmentation result for single sample
    #   {
    #       "masks": [
    #           np.array in bgr channel order
    #       ],
    #       "labels": ["dog", "horse", "cow", "cat"],
    #       "scores": [0.9, 0.1, 0.05, 0.05]
    #   }
    Tasks.image_segmentation: ['scores', 'labels', 'boxes'],

    # image generation/editing/matting result for single sample
    # {
    #   "output_png": np.array with shape(h, w, 4)
    #                 for matting or (h, w, 3) for general purpose
    # }
    Tasks.image_editing: ['output_png'],
    Tasks.image_matting: ['output_png'],
    Tasks.image_generation: ['output_png'],

    # pose estimation result for single sample
    # {
    #   "poses": np.array with shape [num_pose, num_keypoint, 3],
    #       each keypoint is a array [x, y, score]
    #   "boxes": np.array with shape [num_pose, 4], each box is
    #       [x1, y1, x2, y2]
    # }
    Tasks.pose_estimation: ['poses', 'boxes'],

    # ocr detection result for single sample
    # {
    #   "det_polygons": np.array with shape [num_text, 8], each box is
    #       [x1, y1, x2, y2, x3, y3, x4, y4]
    # }
    Tasks.ocr_detection: ['det_polygons'],

    # ============ nlp tasks ===================

    # text classification result for single sample
    #   {
    #       "labels": ["happy", "sad", "calm", "angry"],
    #       "scores": [0.9, 0.1, 0.05, 0.05]
    #   }
    Tasks.text_classification: ['scores', 'labels'],

    # text generation result for single sample
    # {
    #   "text": "this is text generated by a model."
    # }
    Tasks.text_generation: ['text'],

    # word segmentation result for single sample
    # {
    #   "output": "今天 天气 不错 ， 适合 出去 游玩"
    # }
    Tasks.word_segmentation: ['output'],

    # sentence similarity result for single sample
    #   {
    #       "labels": "1",
    #       "scores": 0.9
    #   }
    Tasks.sentence_similarity: ['scores', 'labels'],

    # ============ audio tasks ===================

    # ============ multi-modal tasks ===================

    # image caption result for single sample
    # {
    #   "caption": "this is an image caption text."
    # }
    Tasks.image_captioning: ['caption'],

    # visual grounding result for single sample
    # {
    #       "boxes": [
    #           [x1, y1, x2, y2],
    #           [x1, y1, x2, y2],
    #           [x1, y1, x2, y2],
    #       ],
    #       "scores": [0.9, 0.1, 0.05, 0.05]
    # }
    Tasks.visual_grounding: ['boxes', 'scores'],

    # text_to_image result for a single sample
    # {
    #    "image": np.ndarray with shape [height, width, 3]
    # }
    Tasks.text_to_image_synthesis: ['image']
}
