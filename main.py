import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))

	resultsForJSON = []
	# add list for json write-in function

	# tl_coord,br_coord, dims, area = [],[],[],[]
	people_json =[]

	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		

		#################################################
        ##added by me
		# if max_indx != 14:
		# 	continue
		if mess != "person":
			continue
        #################################################
		
		# labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
		# "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    	# "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    	# "train", "tvmonitor"]
		

		## CODE TO EXTEND THE CROP AREA TO SQUARE
		# new_h = int(abs(top-bot))
		# new_w = int(abs(left-right))
		# if new_h>new_w:
		# 	ext = int(new_h-new_w)/2
		# 	left = int(left - ext)
		# 	right = int(right + ext)
		# else:
		# 	ext = int(new_w-new_h)/2
		# 	top = int(top - ext)
		# 	bot = int(bot + ext)
		thick = 2 #int((h + w) // 300)
		# if self.FLAGS.json:
		# 	# resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
		# 	resultsForJSON.append({"file_name": os.path.basename(im), "file_ext": 'png', "file_path": outfolder, \
		# 	'count'
		# 	"confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			
		# 	continue
		# cropImg = imgcv[top:bot,left:right] # CROP IMAGE
		# cv2.imwrite("C:\\Users\\ddand\\Documents\\GitHub\\darkflow\\results\\["+mess+"]"+str(left)+"x"+str(right)+".jpg",cropImg) #SAVE TO FOLDER
		
		pw,ph = abs(bot-top),abs(right-left)

		people_json.append({
			"tl_coord": [top,left],
			"br_coord": [bot,right],
			"dims": [ph,pw],
			"area": ph*pw
		})
		
		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

	# json write in function added for living line team

	if self.FLAGS.json:
			resultsForJSON={"file_name": os.path.basename(im), "file_ext": os.path.basename(im).split(".")[-1], "file_path": outfolder, "file_dim": [h,w], "count": len(people_json), "people":people_json}
			
	if not save: return imgcv

	# original outpath and img_name are here

	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON,indent=4)
		# textFile = os.path.splitext(img_name)[0] + ".json"
		textFile = img_name + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)

# python flow --model cfg/yolo.cfg --load bin/yolo.weights --imgdir sample_img_0926/  --gpu 1.0 --labels labels-new.txt --threshold 0.35 --queue 15 --json