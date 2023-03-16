from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import face_recognition
import uvicorn
import io
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/form')
async def create_item(options_: list = Form(...)):
    options_ = [0 if option=='0.0' else 1 for option in options_]
    num_faces={1: 20,
    2: 20,
    3: 20,
    4: 19,
    5: 20,
    6: 20,
    7: 20,
    8: 19,
    9: 20,
    10: 20,
    11: 19,
    12: 19,
    13: 20,
    14: 20,
    15: 20,
    16: 19,
    17: 19,
    18: 20,
    19: 18,
    20: 20,
    21: 20,
    22: 20,
    23: 20,
    24: 20,
    25: 20,
    26: 20,
    27: 20,
    28: 19,
    29: 19,
    30: 20,
    31: 17,
    32: 20,
    33: 20,
    34: 19,
    35: 20,
    36: 20,
    37: 20,
    38: 20,
    39: 19,
    40: 18,
    41: 20,
    42: 19,
    43: 20,
    44: 20,
    45: 20,
    46: 20,
    47: 18,
    48: 9,
    49: 19,
    50: 20,
    51: 19,
    52: 19}
    
    global y
    y=[]
    for i in range(1,53):
        for item in num_faces[i]*[options_[i-1]]:
            y.append(item)
    return y

@app.post("/image/")
def something(image: bytes=File(...)):
    global y

    image_object = io.BytesIO(image)
    image_to_test = face_recognition.load_image_file(image_object)
    face_landmarks_list = face_recognition.face_landmarks(image_to_test)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)
    
    es = EarlyStopping(patience=3)
    reg_l1 = regularizers.L1(0.01)
    reg_l2 = regularizers.L2(0.01)
    
    global model
    def initialize_model():
        model = Sequential()
        model.add(Dense(units = 128, input_dim = 128, activation = 'relu'))
        model.add(Dropout(0.01))
        model.add(Dense(units = 64, activation = 'relu',
                        kernel_regularizer=reg_l1))
        model.add(Dropout(0.2))
        model.add(Dense(units = 50, activation = 'relu',
                        kernel_regularizer=reg_l2))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model
    
    model = initialize_model()
    X = pickle.load(open('face_encode', 'rb'))
    y = np.array(y) 
    model.fit(X, y, epochs=30, batch_size=32, verbose=1, shuffle=True, callbacks=[es]) 
    results=[]
    for encode in image_to_test_encoding:
        prediction=model.predict(encode.reshape(1,128))[0][0]  
        results.append(float(prediction))
    return list((results, face_landmarks_list))
   
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8005)