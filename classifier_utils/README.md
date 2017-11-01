## Make TFRecord 
1. Download the [bosch small traffic light dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
2. Install tensorflow and the [object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
3. In the script we have:
   ```python
   from object_detection.utils import dataset_util
   ```
   which requires the script to run in `<installation_dir>/models/research`
   
### Run
example:
```bash
python bosch_lights_to_tf_record.py \
--input_yaml=${your_path_to_this_file}/dataset_train_rgb/train.yaml \
--output_path=${your_path_to_this_file}/light_out.record
--label_map_path=${your_path_to_this_file}/light_label_map.pbtxt
```

## Use the `light_graph` to test
In the object detection API -> tutorial notebook, replace the checkpoint with this graph

experiment.ipynb shows some experiments of mine

## Train
`light.config`, just modify the path to your data
