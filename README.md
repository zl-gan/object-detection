# Detection of Defects in Die Preparation Process.
## Introduction
The report is about a consultancy project and practicum for the course CDS590 Consultancy Project & Practicum, provided by School of Computer Sciences, USM. 
## Objectives
The main goal of the project is to create an automated solution to replace the manual classification of defects at the die preparation process. \
The objectives of the project are to accurately detect features and defects on NAND flash memory wafers using computer vision model, and to develop an API which can quickly return the class and coordinates of features and defects on NAND flash memory wafers. 
## Methodology
The methodology followed the CRISP-DM process model, which includes business understanding, data understanding, data preparation, modeling, evaluation, and deployment. \
Five object detection models were trained using transfer learning, namely YOLOv8, YOLO-NAS, SSD, RetinaNet, and Faster R-CNN. 
Hyperparameter tuning was performed to obtain the best performing model. 
## Deployment
The best performing object detection model is first exported from Pytorch .pt format to ONNX format for better interoperability with other libraries. \
The ONNX model is then read by OpenCV to perform the object detection task. \
An object class, together with various associated functions, is written to read the input image, feed the image to the object detection model, process the model output, and return the object class together with its bounding box. \
FastAPI, an API framework available in Python, is used to create the API. The API can be called by performing HTTP POST requests, where new AOI images are sent as the input, and the output is the class and coordinates of features and defects on the images. \
A simple website is also created using streamlit, where it accepts user input image and display the image with bouding box of objects detected. 
## Results
All the object detection models trained in this project can detect features and defects on NAND flash memory wafers with mAP at 50% IoU ranging from between 0.759 to 0.925. \
The YOLOv8 object detection model was found to be the best performing model in this study with mAP at 50% IoU of 0.925 during validation with test dataset. \
The best performing YOLOv8 model was deployed using an API and a simple webpage. The average response time for the POST request made was 0.5489s, which is a huge improvement over the reported time of 5 seconds required for manual operation. 
## Conclusion
This project demonstrates the feasibility and benefits of using computer vision and deep learning for defect detection on NAND memory wafers. \
The project also provides a valuable opportunity to apply data science and analytics skills in a real-world problem and to gain industry-related knowledge.
