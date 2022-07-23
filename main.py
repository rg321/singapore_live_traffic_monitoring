import requests
from bs4 import BeautifulSoup
from download import download_img
from tqdm import tqdm
from yolo_fastest.darknet_images import batch_detection, image_detection

import random
from yolo_fastest import darknet
from os.path import join
import cv2

# main_url = 'https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras.html'
# res = requests.get(main_url)
# bs = BeautifulSoup(res.text, features="html.parser")

# all_area_buttons = bs.find('div', class_="image-showcase map").find_all('button')

# areas = [area_button['id'] for area_button in all_area_buttons]

# print(areas)
areas = [
		 'stg',
		 'mce',
		 'ecp',
		 'pie',
		 'kpe',
		 'cte',
		 'tpe',
		 'aye',
		 'bke',
		 'sle',
		 'kje',
		 'woodlands']
# exit()

for area in areas:

	print(f'======================== {area} ===========================================')

	url = f'https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/{area}.html'

	res = requests.get(url)

	bs = BeautifulSoup(res.text, features="html.parser")

	# images' url lies in class 'road-snapshots'
	class_div = bs.find('div', class_="road-snapshots")
	
	all_img_divs = class_div.find_all('img') if class_div else []
 
	imgs_url=[t['src'] for t in all_img_divs]

	final_img_urls = ['https://'+img_url[2:] for img_url in imgs_url] if imgs_url else []

	# area = 'kje'

	# url='https://datamall.lta.gov.sg/trafficsmart/images/8701_1744_20220722174533_b36a80.jpg'
	filenames = []
	for i, img_url in enumerate(tqdm(final_img_urls)):
		img_name = img_url.split('/'	)[-1]
		print('img_name ', img_name)
		# img_name = '8701_1744_20220722174533_b36a80.jpg'
		try:
			filename = area + '_' + img_name.split('_')[2] + '_' + str(i) + '.jpg'
		except:
			# filename = img_name
			pass
		else:
			filenames.append(filename)
			download_img(img_url, filename)

	print('filenames')
	print(filenames)

	base='/media/raghav/e34065bb-bd49-4111-ba3a-96160e27ffd0/raghu/cctv/Yolo-Fastest/ModelZoo/yolo-fastest-1.1_coco'
	cfg_file = join(base, 'yolo-fastest-1.1_coco.cfg')
	data_file = join(base, 'coco.data')
	weights_file = join(base, 'yolo-fastest-1.1_coco.weights')

	batch_size = 4
	random.seed(3)  # deterministic bbox colors
	network, class_names, class_colors = darknet.load_network(
	    # args.config_file,
	    # args.data_file,
	    # args.weights,
	    cfg_file,
	    data_file,
	    weights_file,
	    batch_size=batch_size
	)

	# image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
	image_names = filenames
	images = [cv2.imread(image) for image in image_names]
	# images, detections,  = image_detection(network, images, class_names,
	#                                 class_colors, batch_size=batch_size)

	final_d={}
	for image_name in image_names:
		image, detections = image_detection(
		    image_name, network, class_names, class_colors, 0.1
		)
		# for name, image in zip(image_names, images):
		#     cv2.imwrite(name.replace(".jpg", "_det.jpg"), image)
		# print(detections)
		cv2.imshow(image_name.replace(".jpg", "_det.jpg"), image)
		cv2.imwrite(image_name.replace(".jpg", "_det.jpg"), image)
		"""
		[("car", "12.33", (220.30792236328125, 122.24264526367188, 14.454839706420898, 17.448427200317383)), 
		("car", "17.76", (61.90693664550781, 101.94390869140625, 10.29442310333252, 11.085121154785156)), 
		("car", "18.6", (225.34466552734375, 186.17745971679688, 11.648178100585938, 11.994532585144043)), 
		("train", "19.79", (141.77163696289062, 172.28277587890625, 178.4548797607422, 259.9874572753906))]
		"""
		det_objects = [det[0] for det in detections]
		from collections import Counter
		obj_dict = Counter(det_objects)
		cars = obj_dict.get('car', 0)
		final_d.update({image_name: cars})

	print(final_d)













