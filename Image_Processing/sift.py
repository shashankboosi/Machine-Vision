import cv2
import numpy as np

class SiftDetector:
    def __init__(self, norm="L2", params=None):
        self.detector = self.get_detector(params)
        self.norm = norm

    def get_detector(self, params):
        if params is None:
            params = {}
            params["n_features"] = 0
            params["n_octave_layers"] = 3
            params["contrast_threshold"] = 0.04
            params["edge_threshold"] = 10
            params["sigma"] = 1.6

        detector = cv2.xfeatures2d.SIFT_create(
            nfeatures=params["n_features"],
            nOctaveLayers=params["n_octave_layers"],
            contrastThreshold=params["contrast_threshold"],
            edgeThreshold=params["edge_threshold"],
            sigma=params["sigma"])

        return detector


# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    (x, y) = image.shape[:2]
    (cX, cY) = (x // 2, y // 2)
    return (cX, cY)


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    # Also add any picture from the directory Machine-Vision/Images and check the Image similarities
    OriginalImage = cv2.imread("../Images/Sift/road_sign.jpg")
    gray_image = cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = SiftDetector(params=None)

    # Store SIFT keypoints of original image in a Numpy array
    detector = sift.detector
    kp1, des1 = detector.detectAndCompute(gray_image, None)

    # Degrees with which to rotate image
    angles = [0, 45, 90]
    for angle in angles:
        rotatedOriginalImage = rotate(OriginalImage, angle)
        gray_image_1 = cv2.cvtColor(rotatedOriginalImage, cv2.COLOR_BGR2GRAY)

        detector = sift.detector
        kp2, des2 = detector.detectAndCompute(gray_image_1, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(gray_image, kp1, gray_image_1, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        file_extensions = '../OutputImages/Sift/Road-Sign-OriginalImage{}.jpg'.format(angle)
        cv2.imwrite(file_extensions, img3)
