from transformers import StableDiffusionPipeline
import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

# Load Stable Diffusion Model from Hugging Face
stable_diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# AWS session setup
session = sagemaker.Session()

# Define IAM Role for SageMaker
role = 'arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/sagemaker-execution-role'  # Replace with your IAM role

# Define Hugging Face Model details
huggingface_model = HuggingFaceModel(
    model_data=None,  # No model.tar.gz required since we use Hugging Face integration
    role=role,
    transformers_version="4.6",
    pytorch_version="1.9",
    py_version="py38",
)

# Deploy the model
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.p2.xlarge",  # GPU instance for deep learning models
    endpoint_name="stable-diffusion-endpoint"
)