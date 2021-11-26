import cv2
import os
import  tensorflow as tf
import numpy as np
import time
gpus  = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)




def  predict():

        
        img = cv2.imread(f"test.png")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img= cv2.resize(img,(96,96))
      
        model = tf.lite.Interpreter(model_path="eyes_quant_mobilenet_v2.tflite")
        model.allocate_tensors()
        input_detail = model.get_input_details()[0] 
        output_detail = model.get_output_details()[0]
        x = np.expand_dims(img,0).astype(input_detail['dtype'])

        with tf.device('/gpu:0'):
            start = time.time()
            model.set_tensor(input_detail["index"],x)
            model.invoke()
            output = model.get_tensor(output_detail["index"])[0]
            end = time.time()
            print(f"Predicter cost --- {end -start}")


predict()
