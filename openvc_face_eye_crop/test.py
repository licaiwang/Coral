import cv2
import time

face_cascPath = "lbpcascade_frontalface_improved.xml"
right_eyes_cascPath = "haarcascade_righteye_2splits.xml"
left_eyes_cascPath = "haarcascade_lefteye_2splits.xml"

faceCascade = cv2.CascadeClassifier(face_cascPath)
right_eyes_Cascade = cv2.CascadeClassifier(right_eyes_cascPath)
left_eyes_Cascade = cv2.CascadeClassifier(left_eyes_cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start = time.time()
    faces = faceCascade.detectMultiScale(
        gray, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    end = time.time()
    print(f"Crop Face Cost:{end - start}")

    # Draw a rectangle around the faces

    for (x_f, y_f, w_f, h_f) in faces:
        cv2.rectangle(frame, (x_f, y_f), (x_f + w_f, y_f + h_f), (0, 255, 0), 2)
        faceROI = gray[y_f : y_f + h_f, x_f : x_f + w_f]
        start = time.time()
        left_eye = left_eyes_Cascade.detectMultiScale(
            faceROI, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        end = time.time()
        print(f"Crop left eye Cost:{end - start}")
        start = time.time()
        right_eye = right_eyes_Cascade.detectMultiScale(
            faceROI, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        end = time.time()
        print(f"Crop right eye Cost:{end - start}")
        for (eX, eY, eW, eH) in left_eye:
            ptA = (x_f + eX, y_f + eY)
            ptB = (x_f + eX + eW, y_f + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
            break

        for (eX, eY, eW, eH) in right_eye:
            ptA = (x_f + eX, y_f + eY)
            ptB = (x_f + eX + eW, y_f + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
            break

    # Display the resulting frame
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
