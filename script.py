import cv2
# def generate_dataset(img, id, img_id):
#     cv2.imwrite("data/user." + str(id) + "." + str(img_id) + ".jpg", img)


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf, eyeCascade, noseCascade):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(grey_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 2 is thickness of border
        id, confidence = clf.predict(grey_img[y:y + h, x:x + w])
        print("ID is : ", id, confidence)
        if confidence < float(65):
            if id == 1:
                cv2.putText(img, "Prahalad", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(img, "Authorized Face Detected", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                            1, cv2.LINE_AA)
            elif id == 3:
                cv2.putText(img, "Abhilash", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(img, "Authorized Face Detected", (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                            1, cv2.LINE_AA)
            # elif id == " ": cv2.putText(img, "User Not authorized", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            # color, 1, cv2.LINE_AA)
        else:
            color = (0, 0, 255)
            # print(color)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Restricted person/Unable to Detect", (400,450), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                        1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def facedetection(img, classifier, scaleFactor, minNeighbors, color, text):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(grey_img, scaleFactor, minNeighbors)
    # coords = []
    return faces, grey_img

    # for (x, y, w, h) in features:
    # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 2 is thickness of border
    # cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
    # coords = [x, y, w, h]


def recognize(img, clf, faceCascade, eyeCascade, noseCascade):  ## added , eyeCascade, noseCascade
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.3, 5, color['white'], "", clf, eyeCascade, noseCascade)  # defaults is 1.1 & 10
    return img


eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# mouthCascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

# radius = 1
# neighbors = 8
# grid_x = 8
# grid_y = 8
# radius, neighbors, grid_x, grid_y
# double threshold = DBL_MAX


clf = cv2.face.LBPHFaceRecognizer_create()  ##
clf.read("classifier.yml")  ##

video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
    _, img = video_capture.read()
    # img = detect(img, faceCascade, eyeCascade, noseCascade, img_id)  # , mouthCascade ##
    img = recognize(img, clf, faceCascade, eyeCascade, noseCascade)
    cv2.imshow("Face Detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # break loop with pressing q
        break

video_capture.release()
cv2.destroyAllWindows()
