**Introduction**
Object detection is a critical task in computer vision which enables machines to identify and
locate objects within images or video frames. Among the various object detection methods, the
You Only Look Once (YOLO) model stands out for its real-time performance and high
accuracy. YOLO is a convolutional neural network-based approach that processes images in a
single pass, making it exceptionally fast compared to traditional detection systems that rely on
sliding window approaches.
The YOLO model operates by dividing the input image into a grid and predicting bounding
boxes and class probabilities for each grid cell. This approach allows YOLO to detect multiple
objects of different classes in a single inference step, making it well-suited for real-time
applications such as autonomous driving, surveillance, and robotics [1].
In this report, we delve into the application of YOLO for face detection, a task crucial for
various domains, including security systems, human-computer interaction, and biometric
authentication. We undertake two main tasks:
**Basic YOLO Implementation for Face Detection** 
In this task, we begin by obtaining a face detection dataset from Kaggle, which contains
annotated images with bounding boxes around faces. We then implement a basic
YOLOv3 model using OpenCV and fine-tune it to detect faces specifically. The
objective is to update the fully connected component of the YOLO model to specialize
in detecting human faces. We train the model using the provided dataset and evaluate
its performance on a separate validation set.
**Part B: Development of Personalized YOLO for Face Detection**
Building upon the basic YOLO implementation, we explore innovative modifications
to the YOLO architecture to develop a personalized version optimized for face
detection. This task involves experimenting with various modifications, such as
removing certain pre-trained layers, adjusting network parameters, or introducing new
components. The goal is to create a streamlined version of YOLO tailored specifically
for detecting human faces with improved efficiency and accuracy.
Throughout both tasks, we aim to not only achieve accurate face detection but also optimize
the models for deployment on resource-constrained devices, such as mobile phones or edge
computing platforms. By leveraging the capabilities of YOLO and customizing it for face
detection, we seek to address the unique challenges posed by this task and pave the way for
practical applications in diverse real-world scenarios.
**Experiment Description** 
In this experiment, we aim to utilize the YOLO model for face detection. The basic YOLOv3
model is trained to detect a wide range of objects across 80 different classes. However, for our
specific task of face detection, we need to adapt the model to detect faces exclusively.
For Part A, we start by downloading a face detection dataset from Kaggle. We then implement
a basic YOLOv3 model and fine-tune it to detect faces using the provided dataset. The model
is modified to update only the fully connected component to specialize in face detection.
For Part B, we propose an innovative modification to the YOLO architecture to develop a
personalized version. We explore the impact of removing certain pre-trained layers from the
original YOLO network and contrast it with a single multi-layer perceptron. Additionally, we
aim to reduce the number of trainable parameters while maintaining or improving performance.
