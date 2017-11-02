from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
class TLClassifier(object):
    # TODO: maybe we don't really need detection
    def __init__(self):
        # stinky graph loading...
        # 1. init a graph
        # 2. "inside" the graph, load a graph def from file
        #    (bytes from file -> graph def from bytes) 
        self.detection_graph = tf.Graph()
        self.PATH_TO_PB = os.path.abspath('light_classification/light_graph/frozen_inference_graph.pb')
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_PB, 'rb') as fid:
                serialized_graph = fid.read() 
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')  

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
               image_expanded = np.expand_dims(image, axis=0)

               (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                 feed_dict={image_tensor: image_expanded})
        
        valid_idc = np.where((scores > 0.5).flatten())[0]
        valid_detections = classes.flatten()[valid_idc]
        # valid_boxes = boxes.reshape((-1, 4))[valid_idc]
        return self.select_state_from_detection(valid_detections)  

    def select_state_from_detection(self, valid_detections, boxes=None):
        # TODO improve logic to identify the art of determine state from detections
        return TrafficLight.UNKNOWN if valid_detections.shape[0] == 0 else int(valid_detections[0]) - 1
         
         

