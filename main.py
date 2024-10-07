import boto3
import base64
from PIL import Image
from io import BytesIO

# Create SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Your Stable Diffusion endpoint name
endpoint_name = "stable-diffusion-endpoint"

# Text prompt for the Stable Diffusion model
prompt = "A scenic view of mountains during sunset"

# Prepare input payload
payload = {"prompt": prompt}

# Invoke SageMaker endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
response_body = response['Body'].read()
result = json.loads(response_body)

# The response contains a base64-encoded image
image_data = base64.b64decode(result['generated_image'])

# Open and save the image
image = Image.open(BytesIO(image_data))
image.show()
image.save("generated_image.png")