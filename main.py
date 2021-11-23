import cv2
import time
import numpy as np
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


face_cascPath = "opencv_face_eye_crop_model/lbpcascade_frontalface_improved.xml"
right_eyes_cascPath = "opencv_face_eye_crop_model/haarcascade_righteye_2splits.xml"
left_eyes_cascPath = "opencv_face_eye_crop_model/haarcascade_lefteye_2splits.xml"
faceCascade = cv2.CascadeClassifier(face_cascPath)
right_eyes_Cascade = cv2.CascadeClassifier(right_eyes_cascPath)
left_eyes_Cascade = cv2.CascadeClassifier(left_eyes_cascPath)
video_capture = cv2.VideoCapture(0)
SIZE = (96, 96)


pth = "eye_close_model/eyes_quant_mobilenet_v2.tflite"
interpreter = make_interpreter(*pth.split("@"))
interpreter.allocate_tensors()
# Model must be uint8 quantized
if common.input_details(interpreter, "dtype") != np.uint8:
    raise ValueError("Only support uint8 input type.")
params = common.input_details(interpreter, "quantization_parameters")
scale = params["scales"]
zero_point = params["zero_points"]
mean = 128.0
std = 128.0

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 255, 255)
thickness = 1
lineType = 2


def run(image):

    # Input data requires preprocessing
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter, normalized_input.astype(np.uint8))
    # Run inference
    interpreter.invoke()
    classes = classify.get_classes(interpreter, 1, 0)
    if classes != []:
        return classes[0].id
    return None


def main():
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw a rectangle around the faces
        t1, t2 = None, None
        for (x_f, y_f, w_f, h_f) in faces:

            cv2.rectangle(frame, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 2)
            faceROI = gray[y_f : y_f + h_f, x_f : x_f + w_f]
            left_eye = left_eyes_Cascade.detectMultiScale(
                faceROI, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            right_eye = right_eyes_Cascade.detectMultiScale(
                faceROI, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (eX, eY, eW, eH) in left_eye:
                ptA = (x_f + eX, y_f + eY)
                ptB = (x_f + eX + eW, y_f + eY + eH)
                left_eye_roi = faceROI[eY : eY + eH, eX : eX + eW]
                cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
                left_eye_roi = cv2.cvtColor(left_eye_roi, cv2.COLOR_GRAY2RGB)
                left_eye_roi = cv2.resize(
                    left_eye_roi, SIZE, interpolation=cv2.INTER_AREA
                )
                res = run(left_eye_roi)
                if res == 1:
                    cv2.putText(
                        frame,
                        "Right Eyes Open",
                        (400, 50),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType,
                    )
                elif res == 0:
                    cv2.putText(
                        frame,
                        "Right Eyes Closed!",
                        (400, 50),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType,
                    )
                break

            for (eX, eY, eW, eH) in right_eye:
                ptA = (x_f + eX, y_f + eY)
                ptB = (x_f + eX + eW, y_f + eY + eH)
                right_eye_roi = faceROI[eY : eY + eH, eX : eX + eW]
                cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
                right_eye_roi = cv2.cvtColor(right_eye_roi, cv2.COLOR_GRAY2RGB)
                right_eye_roi = cv2.resize(
                    right_eye_roi, SIZE, interpolation=cv2.INTER_AREA
                )
                res = run(right_eye_roi)
                if res == 1:
                    cv2.putText(
                        frame,
                        "Left Eyes Open",
                        (10, 50),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType,
                    )
                elif res == 0:
                    cv2.putText(
                        frame,
                        "Left Eyes Closed!",
                        (10, 50),
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType,
                    )
                break

        # Display the resulting frame

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


main()
