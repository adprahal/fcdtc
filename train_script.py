import cv2
import radius as radius
from subprocess import call


def generate_dataset(img, id, img_id):
    cv2.imwrite("data/user." + str(id) + "." + str(img_id) + ".png", img)


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(grey_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 2 is thickness of border
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def detect(img, faceCascade, eyeCascade, noseCascade, img_id):  # , mouthCascade
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)
    return img


eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# mouthCascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
    _, img = video_capture.read()
    # print(img)
    img = detect(img, faceCascade, eyeCascade, noseCascade, img_id)  # , mouthCascade ##
    cv2.imshow("Face Detection", img)
    img_id += 1

    # call(["python", "classifier.py"])
    if cv2.waitKey(1) & 0xFF == ord('q'):  # break loop with pre    ssing q
        break

video_capture.release()
cv2.destroyAllWindows()
