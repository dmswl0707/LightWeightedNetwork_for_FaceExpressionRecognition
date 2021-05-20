import dlib
import cv2

detector = dlib.get_frontal_face_detector()
webcam = cv2.VideoCapture(0)
captured_num=0

while (webcam.isOpened()):
    ret, img = webcam.read()
    if ret == True:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # dets = detector(rgb_image)
        dets, scores, subdetectors = detector.run(rgb_image, 1, 0)

        # for det in dets:
        #    cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255,0,0), 3)
        for i, det in enumerate(dets):
            captured_num = captured_num + 1
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0, 0), 3)
            img_detect =img[det.top():det.bottom(), det.left():det.right()]
            cv2.imwrite('./detected/face' + str(captured_num) + '.png', img_detect)
            print("Detection {}, score: {}, face_type: {}".format(det, scores[i], subdetectors[i]))

        cv2.imshow("WEBCAM", img)

        if cv2.waitKey(1) == 27:
            break

webcam.release()
cv2.destroyAllWindows()