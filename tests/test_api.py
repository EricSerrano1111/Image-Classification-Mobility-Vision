import requests
import os

# URL of local FastAPI server
# Replace with your local or cloud endpoint
API_URL = "https://YOUR-PROJECT-ID.run.app/predict"

# exact path to a test image from data folder
IMAGE_PATH = r"..\data\Test\00001.png"  

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Could not find image at {IMAGE_PATH}")
else:
    print(f"Sending {IMAGE_PATH} to the API...")
    
    # open image in binary mode and send the POST request
    with open(IMAGE_PATH, "rb") as image_file:
        files = {"file": (os.path.basename(IMAGE_PATH), image_file, "image/png")}
        response = requests.post(API_URL, files=files)
    
    # print results
    if response.status_code == 200:
        print("\n API Success, here is the prediction:")
        print(response.json())
    else:
        print(f"\n API Error: {response.status_code}")
        print(response.text)