# Lane_Detection_in_Self_Driving_Cars 

Lane lines play a key role in indicating traffic flow and directing vehicles. Computer vision based lane detection is an essential technology for self-driving cars. In this project a lane detection system is proposed to detect lane lines in urban streets and highway roads under complex background. In order to nullify the distortions caused by the camera lenses we generate a distortion model by calibrating images against a known object reducing inaccurate measurements. Generalized filtering approach using Sobel operator in HLS colour space. A bird eye view of image is generated using perspective transformation. A special algorithm called sliding window algorithm is used to detect lane lines and later curve fitting is done using polynomial regression. The experimental results show that the proposed system for lane detection, self-calibration and vehicle offset estimation are effective and accurate for both straight and curved lanes.

# Introduction :

In the recent years there has been a large leap from Automatic Driving Systems to modern day Self-driving Cars. To determine the traffic flow and for vehicles to direct, lane lines play a key role. By determining the lane-lines we can assist driver using vehicle offset values from not crossing the lanes and have safe and secure journey.

# Objective :

The main objective of this project is to design a system using Open CV that can detect lane lines and estimate vehicular offset value with the help of lane curvature. We collected images of various lanes and then applied different Computer Vision techniques to clearly detect lane lines, and applied this to video (taken using camera mounted upon a car).

# Benefits :

This system can be used in real time with a camera mounted upon the car to detect lane lines, which will help Automatic Driving Systems in working efficiently and reduce accidents caused by misinterpreted lane detection.

# Working :

This system works on Computer Vision based techniques like Sobel filtering, Canny edge detection, Hough transform (for straight lane detection), Sliding window algorithm and Curve fitting (for both straight and curved lane detection). When a video is given as an input the code works in such a way that the whole lane detection system is overlaid on the video.

# Algorithm :

A.	For straight lane detection

    1.	Colour Filtering in HLS(Hue-Lightness-Saturation)
    2.	Region of Interest
    3.	Canny Edge Detection
    4.	Hough Line Detection
    5.	Line Filtering and Averaging
    6.	Overlay detected lane
    7.	Applying to video

B.	For Curved lane detection

    1.	Perspective Warp
    2.	Sobel Filtering
    3.	Histogram Peak Detection
    4.	Sliding Window Search
    5.	Curve Fitting
    6.	Overlay Detected Lane
    7.	Apply to Video

We will look into lane detection using Sliding window as it can detect both Straight and Curved lanes.

# Canny Edge Detection:

Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed. It has been widely applied in various computer vision systems. Canny has found that the requirements for the application of edge detection on diverse vision systems are relatively similar. Thus, an edge detection solution to address these requirements can be implemented in a wide range of situations. The general criteria for edge detection include:

1.	Detection of edge with low error rate, which means that the detection should accurately catch as many edges shown in the image as possible
2.	The edge point detected from the operator should accurately localize on the center of the edge.
3.	A given edge in the image should only be marked once, and where possible, image noise should not create false edges.
To satisfy these requirements Canny used the calculus of variations – a technique which finds
the function which optimizes a given functional. The optimal function in Canny's detector is described by the sum of four exponential terms, but it can be approximated by the first derivative of a Gaussian.
 
Among the edge detection methods developed so far, Canny edge detection algorithm is one of the most strictly defined methods that provides good and reliable detection. Owing to its optimality to meet with the three criteria for edge detection and the simplicity of process for implementation, it became one of the most popular algorithms for edge detection.

# Hough Line Transform :

In automated analysis of digital images, a sub problem often arises of detecting simple shapes, such as straight lines, circles or ellipses. In many cases an edge detector can be used as a pre-processing stage to obtain image points or image pixels that are on the desired curve in the image space. Due to imperfections in either the image data or the edge detector, however, there may be missing points or pixels on the desired curves as well as spatial deviations between the ideal line/circle/ellipse and the noisy edge points as they are obtained from the edge detector. For these reasons, it is often non-trivial to group the extracted edge features to an appropriate set of lines, circles or ellipses. The purpose of the Hough transform is to address this problem by making it possible to perform groupings of edge points into object candidates by performing an explicit voting procedure over a set of parameterized image objects .

The simplest case of Hough transform is detecting straight lines. In general, the straight
line y = mx + b can be represented as a point (b, m) in the parameter space. However, vertical lines pose a problem. They would give rise to unbounded values of the slope parameter m. Thus, for computational reasons, Duda and Hart proposed the use of the Hesse normal form ,where is the distance from the origin to the closest point on the straight line, and (theta) is the angle between the axis and the line connecting the origin with that closest point. It is therefore possible to associate with each line of the image a pair . The plane is sometimes referred to as Hough space for the set of straight lines in two dimensions. This representation makes the Hough transform conceptually very close to the two-dimensional Radon transform. (They can be seen as different ways of looking at the same transform.) Given a single point in the plane, then the set of all straight lines going through that point corresponds to a sinusoidal curve in the (r,θ) plane, which is unique to that point. A set of two or more points that form a straight line will produce sinusoids which cross at the (r,θ) for that line. Thus, the problem of detecting collinear points can be converted to the problem of
finding concurrent curves.


# Perspective Warp:

Detecting curved lanes in camera space is not very easy. What if we could get a bird’s eye view of the lanes? That can be done by applying a perspective transformation on the image. When human eyes see near things they look bigger as compare to those who are far away. This is called perspective in a general way. Whereas transformation is the transfer of an object from one state to another. So overall, the perspective transformation deals with the conversion of 3d world into 2d image. The same principle on which human vision works and the same principle on which the camera works.


# Sobel Filtering:

When we apply this mask on the image it prominent vertical edges. It simply works like as first order derivate and calculates the difference of pixel intensities in a edge region. As the center column is of zero so it does not include the original values of an image but rather it calculates the difference of right and left pixel values around that edge. Also the center values of both the first and third column is 2 and -2 respectively. This give more weight age to the pixel values around the edge region. This increase the edge intensity and it become enhanced comparatively to the original image.


# Histogram peak detection:

In an image processing context, the histogram of an image normally refers to a histogram of the pixel intensity values. This histogram is a graph showing the number of pixels in an image at each different intensity value found in that image. For an 8-bit grayscale image there are 256 different possible intensities, and so the histogram will graphically display 256 numbers showing the distribution of pixels amongst those grayscale values. Histograms can also be taken of colour images either individual histograms of red, green and blue channels can be taken, or a 3-D histogram can be produced, with the three axes representing the red, blue and green channels, and brightness at each point representing the pixel count. The exact output from the operation depends upon the implementation it may simply be a picture of the required histogram in a suitable image format, or it may be a data file of some sort representing the histogram statistics.


# Sliding Window Algorithm:

The sliding window algorithm will be used to differentiate between the left and right lane boundaries so that we can fit two different curves representing the lane boundaries. The algorithm itself is very simple. Starting from the initial position, the first window measures how many pixels are located inside the window. If the amount of pixels reaches a certain threshold, it shifts the next window to the average lateral position of the detected pixels. If not enough pixels are detected, the next window starts in the same lateral position. This continues until the windows reach the other edge of the image. The pixels that fall within the windows are given a marker. In the images below, the blue marked pixels represent the right lane, and the red ones represent the left.

# Curve Fitting:

It is the process of constructing a curve, or mathematical function, that has the best fit to a series
of data points, possibly subject to constraints. Curve fitting can involve either interpolation, where an exact fit to the data is required, or smoothing,[8][9] in which a "smooth" function is constructed that approximately fits the data. A related topic is regression analysis, which focuses more on questions of statistical inference such as how much uncertainty is present in a curve that is fit to data observed with random errors. Fitted curves can be used as an aid for data visualization, to infer values of a function where no data are available,[14] and to summarize the relationships among two or more variables. Extrapolation refers to the use of a fitted curve beyond the range of the observed data, and is subject to a degree of uncertainty since it may reflect the method used to construct the curve as much as it reflects the observed data.

# Conclusion.

Thus the whole lane detection system is implemented is implemented in Open CV and applied to video using Moviepy editor.
Results obtained are thus accurate for uniform curved and straight lanes.

# References :

opencv.org

embedded-vision.com

pyimagesearch.com

arxiv.org


