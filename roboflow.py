import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Load Image with PIL
img = cv2.imread("test_image3.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pilImage = Image.fromarray(image)

# Convert to JPEG Buffer
buffered = io.BytesIO()
pilImage.save(buffered, quality=100, format="JPEG")

# Build multipart form and post request
m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

response = requests.post("https://detect.roboflow.com/basking-shark/1?api_key=KMpxYEO1dkSSnJKdNKhR", data=m, headers={'Content-Type': m.content_type})

print(response)
print(response.json())