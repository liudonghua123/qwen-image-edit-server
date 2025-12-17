import requests
import json
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000/v1/images/generations"
API_KEY = "your_secret_api_key_here"

def main():
    # Load dummy input images if needed or use empty for text-only if supported by model in some mode (though this model is Image Edit)
    # The Qwen-Image-Edit model typically needs input images.
    # For this demo, we will create a dummy white image.
    
    img = Image.new('RGB', (512, 512), color = 'white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    payload = {
        "prompt": "A beautiful sunset over the ocean",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
        "input_images": [img_b64], # Providing one image as input
        "negative_prompt": "blurry",
        "guidance_scale": 7.5,
        "true_cfg_scale": 4.0,
        "num_inference_steps": 20
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print("Sending request to API...")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            b64_json = data["data"][0]["b64_json"]
            
            # Decode and save
            img_data = base64.b64decode(b64_json)
            with open("output_demo.png", "wb") as f:
                f.write(img_data)
            print("Success! Image saved to output_demo.png")
        else:
            print("No image data received.")
            print(data)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if response.text:
            print(f"Response: {response.text}")

if __name__ == "__main__":
    main()
