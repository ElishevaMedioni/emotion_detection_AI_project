
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

'''
One of the popular algorithms for facial detection is “haarcascade”. 
It is computationally less expensive, a fast algorithm, and gives high accuracy.
'''
# Loading the Haarcascade file
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Loading the model we build
classifier = load_model("model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# capture a video from the camera
cap = cv2.VideoCapture(0)

# list to count the recurrence of emotions
count_emotions = [1] * 7

while True:
    _, frame = cap.read()
    labels = []
    '''
    cv2.cvtColor() method is used to convert an image from one color space to another.
    
    Why do we convert image to grayscale in CNN?
    Colours are not relevant in object-identification
    In this case, converting a coloured image to a grayscale image will not matter,
    because eventually the model will be learning from the geometry present in the image. 
    The image-binarization will help in sharpening the image by identifying the light and dark areas.
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    '''
    detectMultiScale()
    Detects objects of different sizes in the input image. 
    The detected objects are returned as a list of rectangles.
    '''
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w] # this will return the cropped face from the image
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # counting the recurrence of emotions in the video
            if label == 'Angry':
                count_emotions[0] += 1
            if label == 'Disgust':
                count_emotions[1] += 1
            if label == 'Fear':
                count_emotions[2] += 1
            if label == 'Happy':
                count_emotions[3] += 1
            if label == 'Neutral':
                count_emotions[4] += 1
            if label == 'Sad':
                count_emotions[5] += 1
            if label == 'Surprise':
                count_emotions[6] += 1

            label_position = (x, y)
            # cv2.putText() method is used to draw a text string on any image
            # we are drawing the emotion label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #  cv2.imshow() method is used to display an image in a window
    # we are displaying the video in a window
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sum = 0
        for x in range(len(count_emotions)):
            sum += count_emotions[x]
        percentage_emotion = [0] * 7
        for y in range(len(count_emotions)):
            percentage_emotion[y] = round((count_emotions[y] / sum * 100), 2)

        emotion_list = ['angry ' + str(percentage_emotion[0]) + "%\n",
                        'disgust ' + str(percentage_emotion[1]) + "%\n",
                        'fear ' + str(percentage_emotion[2]) + "%\n",
                        'happy ' + str(percentage_emotion[3]) + "%\n",
                        'neutral ' + str(percentage_emotion[4]) + "%\n",
                        'sad ' + str(percentage_emotion[5]) + "%\n",
                        'surprise ' + str(percentage_emotion[6]) + "%\n"
                        ]

        for i in range(len(emotion_list)):
            print(emotion_list[i])

        # copy to a file
        file1 = open("MyFile.txt", "w")
        file1.write("Percentage of each emotion that was seen during this recording:\n")
        file1.writelines(emotion_list)
        file1.write("\nAI Project - Emotion Detection")
        file1.close()
        break

cap.release()
cv2.destroyAllWindows()
