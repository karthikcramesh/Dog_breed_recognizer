from model_utils import is_dog
from PIL import Image
import requests
from io import BytesIO

# Test with a dog image
url = 'https://images.dog.ceo/breeds/husky/n02110185_10047.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content))
result = is_dog(image)
print('Dog image result:', result)

# Test with a cat image
url2 = 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-1.2.1&auto=format&fit=crop&w=300&q=80'
response2 = requests.get(url2)
image2 = Image.open(BytesIO(response2.content))
result2 = is_dog(image2)
print('Cat image result:', result2)