# FaceSaysBMI
Predict BMI, age, and gender from facial images using deep learning.


<img src="https://res.cloudinary.com/b0tb1mho/image/upload/v1784646670/uu3wjohrqff38shoapql.webp"/>

## File structure

```
├── data
│   ├── bookid
│   ├── face
│   ├── meta
│   └── test
├── face2bmi_mt.py
├── face2bmi.py
├── full.csv
├── img
│   ├── avengers_comparison.jpg
│   ├── detect_predict_multi_faces.png
│   ├── model_structure.jpg
│   ├── mtcnn_face_alignment.jpg
│   ├── tensorboard_results.png
│   └── trump_comparison.jpg
├── models.py
├── multi_task_end_to_end.ipynb
├── notebook.ipynb
├── prediction.ipynb
├── README.md
├── requirements.txt
├── saved_model
│   └── models_vgg16_fc6.pkl
├── tasks.py
├── test.csv
├── train.csv
├── train_vggface_model_feature_extraction.ipynb
└── web_scrape.py
```

## Reference

* https://github.com/rcmalli/keras-vggface


# Face Says BMI


This is FUTURE 2.0! Now even a single picture of your
face can predict your BMI without knowing your
height & weight. Not even a single face, multiple faces
can be deteced in a single pisture to predict the AGE, SEX & BMI
of all the visible faces on that picture. Isn't that futuristic?


## Documentation

# Face detection and BMI/Age/Sex prediction

The model provides end-to-end capability of detecting faces and predicting the BMI, Age and Gender for each person in the same picture. 

The architecture of the model is described as below:

![Screenshot 2022-09-05 at 3 11 29 PM](https://user-images.githubusercontent.com/99794785/188419564-11d33f6b-eaed-468f-89c2-56aecc4b740a.png)



## Face detection

Face detection is done by `MTCNN`, which is able to detect multiple faces within an image and draw the bounding box for each faces.  

It serves two purposes for this project:

### 1) preprocess and align the facial features of image.

Prior model training, each image is preprocessed by `MTCNN` to extract faces and crop images to focus on the facial part. The cropped images are saved and used to train the model in later part.

Illustration of face alignment:

![Screenshot 2022-09-05 at 3 12 48 PM](https://user-images.githubusercontent.com/99794785/188419741-5803256f-2a97-46a4-b390-ad8074e80323.png)


### 2) enable prediction for multiple persons in the same image.

In inference phase, faces will be detected from the input image. For each face, it will go through the same preprocssing and make the predictions.

Illustration of ability to predict for multiple faces:

![Screenshot 2022-09-05 at 3 13 45 PM](https://user-images.githubusercontent.com/99794785/188419922-263e6b33-bf2c-46b1-9ca6-291704d93ec6.png)


## Multi-task prediction

In vanilla CNN architecture, convolution blocks are followed by the dense layers to make output the prediction. In a naive implementation, we can build 3 models to predict BMI, age and gender individually. However, there is a strong drawback that 3 models are required to be trained and serialized separately, which drastically increases the maintenance efforts.

|   |
|---|
|`[input image] => [VGG16] => [dense layers] => [BMI]`|
|`[input image] => [VGG16] => [dense layers] => [AGE]`|
|`[input image] => [VGG16] => [dense layers] => [SEX]`|

Since we are going to predict `BMI`, `Age`, `Sex` from the same image, we can share the same backbone for the three different prediction heads and hence only one model will be maintained.

|    |
|----|
|`[input image] => [VGG16] => [separate dense layers] x3 => weighted([BMI], [AGE], [SEX])`|

This is the most simplified multi-task learning structure, which assumed independent tasks and hence separate dense layers were used for each head. Other research such as `Deep Relationship Networks`, used `matrix priors` to model the relationship between tasks.

![](https://ruder.io/content/images/2017/05/relationship_networks.png)
_A Deep Relationship Network with shared convolutional and task-specific fully connected layers with matrix priors (Long and Wang, 2015)._
 
## Reference
 * MTCNN: [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
 * VGGFace: [https://github.com/rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)
## Tech Used

In this project Tech invloved as follows: 

MT-CNN, Cv2, VggFace, Keras, Tensorflow




For better understanding : 
 * MTCNN: [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
 * VGGFace: [https://github.com/rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)
## Demo

Here you can access the web-application.

<p align="center">
  <a href="https://youtu.be/wiHwQ7lal88">
    <img src="https://img.shields.io/badge/▶-Watch_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white">
  </a>
</p>

https://user-images.githubusercontent.com/99794785/188427670-e5f1c002-f0e1-40ed-9460-537ead4cea77.mov


## Screenshots

![Screenshot 2022-09-05 at 3 54 55 PM](https://user-images.githubusercontent.com/99794785/188428027-19eca9cb-69ce-4a02-8994-dfb4ee28727c.png)
![Screenshot 2022-09-05 at 3 55 01 PM](https://user-images.githubusercontent.com/99794785/188428065-2d3b2430-e07c-456c-980d-770a4746b61f.png)
![Screenshot 2022-09-05 at 3 55 07 PM](https://user-images.githubusercontent.com/99794785/188428074-ff3e08bf-e82b-4f58-85a2-f6b6a6b19367.png)
![Screenshot 2022-09-05 at 3 55 11 PM](https://user-images.githubusercontent.com/99794785/188428077-78b9ca14-8d86-4a12-a156-96d490bec544.png)
![Screenshot 2022-09-05 at 3 55 26 PM](https://user-images.githubusercontent.com/99794785/188428082-82b3d618-1a61-434e-814e-deaac9ca4629.png)

## Lessons Learned

Learned how to use MtCNN for reducing the processing time 
and increasing the efficiency to get a better optimised 
algorithm to process multiple CNN models simultaneously 
and get a unified output. 

VggFace library is pretty handy to process, allign, and 
agument human faces.

## Future Improvements

Due to lack of enough data set of asian specially Indian
faces, the model is not as accurate when applied on Indian
faces as expected. Providing a good ammount of data set, 
this model can predict bmi as efficiently as it does in 
case of western faces.


## License

This project is licensed under the [Apache-2.0](LICENSE).
See the LICENSE file for full license text.

---

## About phiUture


Welcome to phiUture — Beautiful Technology. Centered Around You. Shaping Tomorrow.

Inspired by the Golden Ratio (φ), our name represents beautiful, intelligent engineering, while the "U" stands for You—placing people at the center of every solution we build.

phiUture is an AI-first software company creating intelligent products, automation systems, and modern digital experiences. This channel documents the journey of building practical AI solutions, from concept to deployment.

```text
Here you'll find:
• AI applications and product demos
• AI agents and automation workflows
• Web and mobile app showcases
• Machine Learning and Data Engineering projects
• Product launches and development insights
• UI/UX and software engineering content
• Tutorials, experiments, and future innovations
```

| Personal | Business | Artist |
|----------|----------|--------|
| [![GitHub](https://img.shields.io/badge/GitHub-thebitanpaul-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thebitanpaul) | [![Website](https://img.shields.io/badge/Website-phiUture-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://phiuture.com) | [![YouTube](https://img.shields.io/badge/YouTube-thebitanpaul-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@thebitanpaul) |
| [![LinkedIn](https://img.shields.io/badge/LinkedIn-thebitanpaul-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thebitanpaul) | [![YouTube](https://img.shields.io/badge/YouTube-phiUture-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@phiuture) | [![Spotify](https://img.shields.io/badge/Spotify-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://open.spotify.com/artist/6ghDcCBlKzJIgm3e586jpV) |
| [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/thebitanpaul) | [![Google Play](https://img.shields.io/badge/Google_Play-Developer-34A853?style=for-the-badge&logo=googleplay&logoColor=white)](https://play.google.com/store/apps/dev?id=6358474525178045834&hl=en) | [![YouTube Music](https://img.shields.io/badge/YouTube_Music-FF0000?style=for-the-badge&logo=youtubemusic&logoColor=white)](https://music.youtube.com/playlist?list=OLAK5uy_km3cjEB2zl59Etcgv9UBKWw800O9G3NdE) |
| [![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://facebook.com/thebitanpaul) | [![Email](https://img.shields.io/badge/Business_Email-thephiuture%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thephiuture@gmail.com) | [![Amazon Music](https://img.shields.io/badge/Amazon_Music-46C3D0?style=for-the-badge&logo=amazonmusic&logoColor=white)](https://music.amazon.com/albums/B0G52QMYDC) |
| [![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/thebitanpaul_) |  | [![Apple Music](https://img.shields.io/badge/Apple_Music-FA243C?style=for-the-badge&logo=applemusic&logoColor=white)](https://music.apple.com/us/artist/thebitanpaul/1858534880) |
| [![Snapchat](https://img.shields.io/badge/Snapchat-FFFC00?style=for-the-badge&logo=snapchat&logoColor=000000)](https://snapchat.com/t/UgO0Iywr) |  | [![JioSaavn](https://img.shields.io/badge/JioSaavn-2BC5B4?style=for-the-badge&logo=jiosaavn&logoColor=white)](https://www.jiosaavn.com/artist/thebitanpaul-songs/zuo0NgC65gQ_) |
| [![Email](https://img.shields.io/badge/Personal_Email-thebitanpaul%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:thebitanpaul@gmail.com) |  |  |



<div align="center">

**2024 · © phiUture · All Rights Reserved**

</div>
