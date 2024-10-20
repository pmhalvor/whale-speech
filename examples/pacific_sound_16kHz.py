import boto3 
from botocore import UNSIGNED
from botocore.client import Config

s3_client = boto3.client('s3',
    aws_access_key_id='',
    aws_secret_access_key='', 
    config=Config(signature_version=UNSIGNED)
)

year = "2024"
month = "08"
bucket = 'pacific-sound-16khz'

for obj in s3_client.list_objects_v2(Bucket=bucket, Prefix=f'{year}/{month}')['Contents']:
    print(obj['Key'])