from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

# classes intercept quantization_config parameter
# to delete before it crashes the application

class PatchedDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

class PatchedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)


# initialize FastAPI app
app = FastAPI(title="Traffic Sign Vision API", description="REST API for GTSRB Image Classification")

# load the trained model using relative path
MODEL_PATH = os.path.join("model", "latest_checkpoint.h5")

try:
    # pass interceptors into custom_objects parameter
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'Dense': PatchedDense, 'Conv2D': PatchedConv2D},
        compile=False  # We only need inference, not the optimizer
    )
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# define label dictionary
class_names = {
    0: 'Speed limit (20km/h)', 
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)', 
    9: 'No passing', 
    10: 'No passing veh over 3.5 tons', 
    11: 'Right-of-way at intersection', 
    12: 'Priority road', 
    13: 'Yield', 
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve left', 
    20: 'Dangerous curve right', 
    21: 'Double curve', 
    22: 'Bumpy road', 
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work', 
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing', 
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow', 
    31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right', 
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left', 
    40: 'Roundabout mandatory', 
    41: 'End of no passing', 
    42: 'End no passing veh > 3.5 tons'
}

# define predit endpoint
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG.")
    
    try:
        # read image file from request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # preprocess image exactly as during training
        image = image.resize((32, 32))  # resize to 32x32
        img_array = np.array(image)      # convert to np array
        img_array = img_array / 255.0    # rescale pixel values
        img_array = np.expand_dims(img_array, axis=0) # add batch dimension (1, 32, 32, 3)
        
        # run inference
        predictions = model.predict(img_array)
        predicted_class_id = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        
        # map numeric ID to readable label
        predicted_label = class_names.get(predicted_class_id, "Unknown Class")
        
        # return json response
        return {
            "filename": file.filename,
            "class_id": predicted_class_id,
            "label": predicted_label,
            "confidence": f"{confidence * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
