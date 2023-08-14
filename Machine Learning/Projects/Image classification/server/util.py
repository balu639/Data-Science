import cv2
import joblib
import json
import numpy as np
import base64
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_face_if_valid(file_path,image_base64_data)

    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack( (scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32 , 1)))

        len_img_array = 32 * 32 * 3 + 32 * 32

        final_img = combined_img.reshape(1, len_img_array).astype(float)

        result.append(
            {
                "class": class_number_to_name(__model.predict(final_img)[0]),
                "class_probability": np.round(__model.predict_proba(final_img)*100,2).tolist()[0],
                "class_dictionary": __class_name_to_number

            }
        )

        
    return result

def load_artifacts():
    print("loading saved artifacts......")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/celebrity_class.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("./artifacts/image_classification.pkl", "rb") as f:
            __model = joblib.load(f)

    print("loading saved artifacts.....completed")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def get_cv2_image_from_b64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr= np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img

def get_b64_test_image_for_virat():
    with open('b64.txt') as f:
        return f.read()


def get_cropped_face_if_valid(img_path , image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_image_from_b64_string(image_base64_data)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces



if __name__ == "__main__":
    load_artifacts()
    # print(classify_image(get_b64_test_image_for_virat(), None))
    print(classify_image(None, './test_images/sharapova1.jpg'))
    print(classify_image(None, './test_images/virat_kohli1.jpg'))

