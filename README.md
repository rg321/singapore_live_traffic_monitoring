**SINGAPORE LIVE TRAFFIC MONITORING**

This project demonstrates live monitoring of road traffic using live images of roads of different areas of Singapore, provided by Singapore Government on their website -: 

**https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras.html**

**General Flow -:**

1) First, images are extracted using url provided on the webpage, regarding particular area.
2) Then, object-detection model (here we use Yolo-Fastest) is used to detect objects (mainly cars) from image.
3) Program prints number of cars detected in each image. Final output is a **dict**, corresponding to number of cars in each image, of a particular area.

**Instructions to run use this repo**

1) Installing dependencies

```bash
pip install requirements.txt
```

2) Installing YOLO model

Install any **darknet yolo** based model **(https://github.com/AlexeyAB/darknet)**.

We use **https://github.com/dog-qiuqiu/Yolo-Fastest**. Compile the code and download necessary weights. Please visit repo for complete instructions. In case you indent to do the same, do following change -:

2.1) after cloning repo **https://github.com/dog-qiuqiu/Yolo-Fastest** -:
```bash
mv Yolo-Fastest yolo_fastest
```

3) Run main.py to detect number of cars in each region.

```bash
PYTHONPATH=yolo_fastest python main.py
```

**Sample Output**

For example for area **Sentosa Gateway**, output is as follows -:
```
{'stg_20220723180617_0.jpg': 14, 'stg_20220723180614_1.jpg': 9}
```




