from styx_msgs.msg import TrafficLight

import numpy as np
import os
import tensorflow as tf

from utils import label_map_util

import rospy

SCORE_THRESHOLD = 0.5
LABEL2STATE_MAP = {1: TrafficLight.RED,
                   2: TrafficLight.YELLOW,
                   3: TrafficLight.GREEN}

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Variables
        current_path = os.getcwd()
        PATH_TO_CKPT = os.path.join(current_path, 'light_classification', 'light_graph', 'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(current_path, 'light_classification', 'data', 'light_label_map.pbtxt')
        NUM_CLASSES = 3
        
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        t0 = rospy.get_time()
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:    
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})
  
        state = TrafficLight.UNKNOWN
        # Assuming classes is always a sorted list, then the first element is always the biggest one
        # if we found that this does not stand, copy again peter's logic from his branch.
        detected_class = classes[0][0]
        detected_score = scores[0][0]
        if detected_score > SCORE_THRESHOLD:
            state = LABEL2STATE_MAP[detected_class]

        dt = rospy.get_time() - t0
        rospy.loginfo("TLClassifier::get_classification() - Detection elapsed time: %f", dt)
        rospy.loginfo("TLClassifier::get_classification() - state: %d", state)
        return state
