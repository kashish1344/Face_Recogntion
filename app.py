from flask import Flask, render_template, Response
import os
# uncomment this line if you want to run your tensorflow model on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from imutils.video import VideoStream
import face_recognition
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2


app=Flask(__name__)
camera = cv2.VideoCapture(0)

def recognition_liveness(model_path, le_path, detector_folder, encodings, confidence=0.5):
    args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 
            'encodings':encodings, 'confidence':confidence}

    # load the encoded faces and names
    print('[INFO] loading encodings...')
    
    with open('model/face_detector/encoded_faces.pickle', 'rb') as file:
        encoded_data = pickle.loads(file.read())
        
        proto_path = 'model/face_detector/deploy.prototxt'
        model_path = 'model/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # Loading liveness detector model
        liveness_model = tf.keras.models.load_model('model/liveness/liveness.model')
        le = pickle.loads(open('model/liveness/label_encoder.pickle', 'rb').read())
    

    #     encoded_data = pickle.loads(file.read())
    # # load our serialized face detector from disk
    # print('[INFO] loading face detector...')
    # # proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
    # # model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
    # proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
    # model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
    # detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    # # load the liveness detector model and label encoder from disk
    # liveness_model = tf.keras.models.load_model(args['model'])
    # le = pickle.loads(open(args['le'], 'rb').read())
    
    
    # initialize the video stream and allow camera to warmup
    print('[INFO] starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(2) # wait camera to warmup
    # count the sequence that person appears
    # this is just to make sure of that person and to show how model works
    # you can delete this if you want
    sequence_count = 0 
    
    while True:
        # grab the frame from the threaded video stream
        # and resize it to have a maximum width of 600 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        # grab the frame dimensions and convert it to a blob
        # blob is used to preprocess image to be easy to read for NN
        # basically, it does mean subtraction and scaling
        # (104.0, 177.0, 123.0) is the mean of image in FaceNet
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network 
        # and obtain the detections and predictions
        detector_net.setInput(blob)
        detections = detector_net.forward()
        
        # iterate over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e. probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections
            if confidence > args['confidence']:
                # compute the (x,y) coordinates of the bounding box
                # for the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')
                
                # expand the bounding box a bit
                # (from experiment, the model works better this way)
                # and ensure that the bounding box does not fall outside of the frame
                startX = max(0, startX-20)
                startY = max(0, startY-20)
                endX = min(w, endX+20)
                endY = min(h, endY+20)
                
                # extract the face ROI and then preprocess it
                # in the same manner as our training data
                face = frame[startY:endY, startX:endX] # for liveness detection
                # expand the bounding box so that the model can recog easier
                face_to_recog = face # for recognition
                # some error occur here if my face is out of frame and comeback in the frame
                try:
                    face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
                except:
                    break
            
                # face recognition
                rgb = cv2.cvtColor(face_to_recog, cv2.COLOR_BGR2RGB)
                #rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb)
                # initialize the default name if it doesn't found a face for detected faces
                name = 'Unknown'
                # loop over the encoded faces (even it's only 1 face in one bounding box)
                # this is just a convention for other works with this kind of model
                for encoding in encodings:
                    matches = face_recognition.compare_faces(encoded_data['encodings'], encoding)
                    
                    # check whether we found a matched face
                    if True in matches:
                        # find the indexes of all matched faces then initialize a dict
                        # to count the total number of times each face was matched
                        matchedIdxs = [i for i, b in enumerate(matches) if b]
                        counts = {}
                        
                        # loop over matched indexes and count
                        for i in matchedIdxs:
                            name = encoded_data['names'][i]
                            counts[name] = counts.get(name, 0) + 1
                            
                        # get the name with the most count
                        name = max(counts, key=counts.get)
                            
                face = face.astype('float') / 255.0 
                face = tf.keras.preprocessing.image.img_to_array(face)
                # tf model require batch of data to feed in
                # so if we need only one image at a time, we have to add one more dimension
                # in this case it's the same with [face]
                face = np.expand_dims(face, axis=0)
            
                # pass the face ROI through the trained liveness detection model
                # to determine if the face is 'real' or 'fake'
                # predict return 2 value for each example (because in the model we have 2 output classes)
                # the first value stores the prob of being real, the second value stores the prob of being fake
                # so argmax will pick the one with highest prob
                # we care only first output (since we have only 1 input)
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = le.classes_[j] # get label of predicted class
                
                # draw the label and bounding box on the frame
                label = f'{label_name}: {preds[j]:.4f}'
                if name == 'Unknown' or label_name == 'fake':
                    sequence_count = 0
                else:
                    sequence_count += 1
                print(f'[INFO] {name}, {label_name}, seq: {sequence_count}')
                
                if label_name == 'fake':
                    color = (0,0,255)   #red
                if label_name == 'real':
                    color = (0,255,0)   #green
                    
                
                cv2.putText(frame, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, color,2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 4)
        
                
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
            # ret, buffer = cv2.imencode('.jpg', frame) #compress and store image to memory buffer
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #concat frame one by one and return frame



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognition_liveness('liveness.model', 'label_encoder.pickle', 'face_detector', '../face_recognition/encoded_faces.pickle', confidence=0.5), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__=='__main__':
    app.run(debug=True)