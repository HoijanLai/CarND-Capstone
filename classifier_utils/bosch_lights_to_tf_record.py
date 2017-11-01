import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import yaml
import os
import cv2
from tqdm import tqdm
from random import random

TRESHOLD_WIDTH = 10 # we ignore traffic lights that are too small

flags = tf.app.flags
flags.DEFINE_string('input_yaml', '', 'Path to input yaml')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '', 'Path to label map')
FLAGS = flags.FLAGS


def get_all(input_yaml):
    """ Gets all labels within label file

    Note that RGB images are 1280x720
    :param input_yaml: Path to yaml file
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())
    for i in range(len(images)):
        if 'test' in input_yaml:
            images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                'rgb', 'test', os.path.basename(images[i]['path'])))
        else:
            images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['path']))
    return images





def create_tf_example(example, label_map):
    height = 720 # Image height
    width = 1280 # Image width
    filename = example['path']
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'png' 
    
    boxes = example['boxes']
    
    xmins = [] 
    xmaxs = [] 
             
    ymins = [] 
    ymaxs = [] 
             
    classes_text = [] 
    classes = [] 

    has_yellow = False # yellow samples are rare this flag will prevent random dropping 

    for box in boxes:
        bw = box['x_max'] - box['x_min']
        bh = box['y_max'] - box['y_min']
        
        # consider small boxes as invalid
        if bw < TRESHOLD_WIDTH or box['label'] == 'off':
            continue
        
        xmins.append(box['x_min']/width)
        xmaxs.append(box['x_max']/width)
        ymins.append(box['y_min']/height)
        ymaxs.append(box['y_max']/height)
        text = box['label'] 
        if 'Green' in text:
            classes_text.append(b'Green')
            classes.append(label_map['Green'])
        elif 'Red' in text:
            classes_text.append(b'Red')
            classes.append(label_map['Red'])
        elif 'Yellow' in text:
            has_yellow = True
            classes_text.append(b'Yellow')
            classes.append(label_map['Yellow'])

    # the not_to_keep condition 
    # 1. randomly drop no-detection samples by a chance 
    # 2. radnomly drop samples with no yellow  
    not_to_keep = (not has_yellow  and random() > 0.7) or \
                  (len(xmins) == 0 and random() > 0.2)
    if not_to_keep:
        return None  
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return (tf_example, has_yellow)




def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    examples = get_all(FLAGS.input_yaml)
     
    for example in tqdm(examples):
        example_tup = create_tf_example(example, label_map)
        if example_tup:
            tf_example, has_yellow = example_tup
            Duplicate_times = 10 if has_yellow else 1
            for _ in range(Duplicate_times):
                writer.write(tf_example.SerializeToString())
                

    writer.close()



if __name__ == '__main__':
    tf.app.run()
