# PCA-and-Eigen-Faces
Face Recognition
Task1:
Load all 25 images in the Eigenfaces\Train
Display all original faces in an 5 by 5 grid.


![alt text](pictures/original.PNG)


Find mean face and perform PCA on training faces.
Display mean face.


![alt text](pictures/mean_face.PNG)


Display all eigen faces.
![alt text](pictures/eigen_faces.PNG)

Task2:
Select top k=2 eigenfaces (eigenvectors which corresponds to the largest eigen values)
Reconstruct training faces and display reconstructed faces.
Repeat process for k=5 and k=15 and k= 25.

![screenshot](https://github.com/ChahalSandeep/PCA-and-Eigen-Faces/edit/master/pictures/recons_2.PNG)
![alt text](pictures/recons_5.PNG)
![alt text](pictures/recons_15.PNG)


Task 3: 
Load all the test image from Eigenfaces\Test.
Project each image on k=2 eigen vectors and find if its face or not.
If face then find the closest training image(Euclidean distance) to claculate distance 
The image on thr right is its closest image in the eigenfaces space.
the right side is blank if it is non face.

Some example for For k= 2

![alt text](pictures/K_2.PNG)

![alt text](pictures/K_2_1.PNG)
![alt text](pictures/K_2_2.PNG)
![alt text](pictures/K_2_3.PNG)

Some example for For k= 5

![alt text](pictures/K_5.PNG)
![alt text](pictures/K_5_1.PNG)
![alt text](pictures/K_5_2.PNG)


Some example for For k= 15

![alt text](pictures/K_15.PNG)
![alt text](pictures/K_15_1.PNG)
![alt text](pictures/K_15_2.PNG)
![alt text](pictures/K_15_3.PNG)
![alt text](pictures/K_15_4.PNG)


Task 4: Plot the percentage classification error rate as a function of k. 

![alt text](pictures/plot.PNG)
