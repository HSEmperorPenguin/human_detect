import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import rospy
from std_msgs.msg  import Int32MultiArray, String

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

ack_publisher = None
arr = Int32MultiArray()
arr.data = [0,0]

##### Tensorflow #####
tf.debugging.set_log_device_placement(True)

# 텐서를 CPU에 할당
with tf.device('/GPU:0'):
	a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap("/home/hansung/sumi/models/research/object_detection/data/mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect_objects(image_np, sess, detection_graph):
	global boxes
	global scores
	global classes
	global num_detections
	global category_index
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Actual detection.
	(boxes, scores, classes, num_detections) = sess.run(
	[boxes, scores, classes, num_detections],
	feed_dict={image_tensor: image_np_expanded})

	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
	image_np,
	np.squeeze(boxes),
	np.squeeze(classes).astype(np.int32),
	np.squeeze(scores),
	category_index,
	use_normalized_coordinates=True,
	line_thickness=8)
	return image_np




# First test on images
PATH_TO_TEST_IMAGES_DIR = '/home/hansung/sumi/compressed_image'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
	(im_height, im_width, 3)).astype(np.uint8)


for image_path in TEST_IMAGE_PATHS:
	image = Image.open(image_path)
	image_np = load_image_into_numpy_array(image)
	plt.imshow(image_np)
	print(image.size, image_np.shape)

#Load a frozen TF model 
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.io.gfile.GFile("/home/hansung/sumi/models/research/object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb", 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
def callback_learning(msg_android_learning):
	with detection_graph.as_default():
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			for image_path in TEST_IMAGE_PATHS:
				image = Image.open(image_path)
				image_np = load_image_into_numpy_array(image)
				image_process = detect_objects(image_np, sess, detection_graph)
				print(image_process.shape)
				plt.figure(figsize=IMAGE_SIZE)
				plt.imshow(image_process)
				plt.savefig('savefig_default.png')

				classes_list=np.squeeze(classes[0]).astype(np.int32)
				human_index=-1
				for i in range(len(classes_list)):
					if(classes_list[i]==1): 
						human_index=i
						break
				if(human_index!=-1 and scores[0][human_index]>=0.60):
					print("person: "+str(scores[0][human_index]))
				else: print("non-person")
						 
			
				im_width, im_height = image.size
				red_color = (0,0,255)
				draw = ImageDraw.Draw(image)
				box = boxes * np.array([im_height, im_width, im_height, im_width])
				(start_y, start_x, end_y, end_x) = box[0][human_index].astype("int")
				crop_area =  (start_x, start_y, end_x, end_y)
				cropped_img=image.crop(crop_area)
				cropped_img.save('/home/hansung/sumi/somaset/target/0001.jpg')
				draw.rectangle([start_x, start_y, end_x, end_y], outline="red")
				arr.data = [start_x,start_y,end_x,end_y]
				rospy.loginfo(arr)
				ack_publisher.publish(arr)




print("ready")
while not rospy.is_shutdown():
##### ROS #####
	rospy.init_node("detect")
	rospy.Subscriber("car_learning",String, callback_learning)
	ack_publisher = rospy.Publisher('human_xy', Int32MultiArray,queue_size=10)


