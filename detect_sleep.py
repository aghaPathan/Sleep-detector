import cv2
import time
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import os


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def DlibVideoHandler(dlibFacePredictor):
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlibFacePredictor)

    video_capture = cv2.VideoCapture(0)
    while (True):

        # Capture frame-by-frame

        ret, frame = video_capture.read()
        if ret is True:
            # start = time.time()
            frame = dlibVideo(frame,detector,predictor,lStart, lEnd,rStart, rEnd)
            cv2.imshow('Video', frame)
            # print("FPS: ", round(1.0 / (time.time() - start)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            continue

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def dlibVideo(frame,detector,predictor,lStart, lEnd,rStart, rEnd):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global lock
    global start
    global threshold

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        eyes = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if eyes < threshold and lock == False:
            lock = True
            start = time.time()

        if eyes > threshold and lock == True:
            lock = False
            start = 0.0

        elif eyes < threshold and lock == True:
            # Drawing outline of eyes
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            if round(time.time() - start) > 5:
                dlibVideoAlert()

    return frame


def dlibVideoAlert():
    print ('Alert')


if __name__ == '__main__':
    global lock
    global start
    global threshold
    home = os.getcwd()
    threshold = 0.2  #This is the threshold for open eye
    lock = False
    start = 0.0

    dlibFacePredictor = os.path.join(home, 'shape_predictor_68_face_landmarks.dat')
    DlibVideoHandler(dlibFacePredictor)



