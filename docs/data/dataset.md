<h1 align=left> Datasets </h1>
<h2 align=left> Medieval Port Dataset </h2>
<h3 align=left> Camera Arrangement </h3>
This dataset consists of a light-field with 25 sub-aperture images, arranged as shown below. The color coding indicates the training/test/val split.

- Cameras are color coded as follows
  - <span style="color:#D5E8D4">*Training*</span>, <span style="color:#FFF2CC">*Validation*</span>, <span style="color:#F8CECC">*Test*</span>
- The *baseline* between cameras is **0.2 units**.
- Reference camera (12) is shown in dotted circle.
- The azimuth angle theta goes from right to left.
- Elevation angle phi goes from top to bottom.
<div class="column", align=center>
    <img src="../assets/MedPortDataset.png", width=800px>
  </div>
</div>

<h2 align=left> Residential Area Dataset </h2>
<h3 align=left> Camera Arrangement </h3>
This dataset has 3 scenes, each with 3x3 light-field images. 

***Coordinate system convention*** similar to the medieval_port dataset
```
x-forward, y-up and z-right
```
***Azimuth and elevation angles*** map to pixel coordinates on the ERP image as follows:

- $\phi = \frac{\pi}{2}$ at the top, $\phi = -\frac{\pi}{2}$ at the bottom. $\phi$ is a vertical rotation with x-z as 0, up direction is positive.
- $\theta = -\frac{3}{2}\times \pi$ at the left, $\theta = \frac{\pi}{2}$ at the right. Horizontal rotation, clockwise direction (from x-axis to z-axis, when seen from the top) is positive.

Check the [residential data loader](/data/residential.py) and [utils](/src/utils.py) for more details. Camera poses (together with RGB images) are provided as *.h5* files.


***Train/test/val split***
8 of the 9 views are used for training and 1 view is held out for testing.


<h2 align=left> Replica Dataset </h2>
<h3 align=left> Camera Arrangement </h3>
This dataset has 14 scenes, each with 9x9 light-field images. 

***Coordinate system convention*** similar to the medieval_port dataset
```
x-forward, y-left and z-up
```
***Azimuth and elevation angles*** map to pixel coordinates on the ERP image as follows:

- $\phi = 0$ at the top, $\phi = \pi$ at the bottom
- $\theta = -\pi$ at the left, $\theta = \pi$ at the right

There is no rotation between the images in the light field and the baseline between cameras is **0.1 units**.

***Train/test/val split***

Each 9x9 light field is split into training, validation, and test sets, as follows: 
```python
# Note that the camera indexing here is row-major order
testing = [4, 20, 22, 24, 36, 38, 42, 44, 56, 58, 60, 76]
trainining = [0, 2, 6, 8, 18, 26, 40, 54, 62, 72, 74, 80]
validation = [1, 5, 7, 37, 41, 43, 71, 77, 79]
```










