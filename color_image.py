# USAGE
# python bw2color_image.py -i images/room.jpg -p model/colorization_deploy_v2.prototxt -m model/colorization_release_v2.caffemodel -c model/pts_in_hull.npy

# import the necessary packages
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input black and white image")

ap.add_argument("-p", "--prototxt", type=str, required=True,
	help="path to Caffe prototxt file")

ap.add_argument("-m", "--model", type=str, required=True,
	help="path to Caffe pre-trained model")

ap.add_argument("-c", "--points", type=str, required=True,
	help="path to cluster center points")

args = vars(ap.parse_args())

# load serialized black & white colorizer model & cluster center points from disk
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# load the input image from disk, scale the pixel intensities to the
# range [0, 1], and then convert the image from the BGR to Lab color
# space
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resize the Lab image to 224x224 (the dimensions the colorization
# network accepts), split channels, extract the 'L' channel, and then
# perform mean centering
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# pass the L channel through the network which will *predict* the 'a'
# and 'b' channel values
print("Colorizing image...")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# resize the predicted 'ab' volume to the same dimensions as our
# input image
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# grab the 'L' channel from the *original* input image (not the
# resized one) and concatenate the original 'L' channel with the
# predicted 'ab' channels
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# convert the output image from the Lab color space to RGB, then
# clip any values that fall outside the range [0, 1]
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# the current colorized image is represented as a floating point
# data type in the range [0, 1] -- let's convert to an unsigned
# 8-bit integer representation in the range [0, 255]
colorized = (255 * colorized).astype("uint8")

#resizing output image
scale_percent = 20 # percent of original size
width = int(colorized.shape[1] * scale_percent / 100)
height = int(colorized.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(colorized, dim, interpolation = cv2.INTER_AREA)
resized_ip = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)



# show the original and output colorized images
cv2.imshow("Original", resized_ip)
cv2.imshow("Colorized", resized)
cv2.waitKey(0)



#(cv) C:>python Users\Tanya Goel\Documents\proj\bw-colorization\bw-colorizationbw2color_image.py -p  model/colorization_deploy_v2.prototxt -m model/colorization_release_v2.caffemodel -c model/pts_in_hull.npy -i images/robin_williams.jpg
