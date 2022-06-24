# deploy car-damage model

## deploy on cloud

server:
```shell script
# build docker and push to ecr
sh build_and_push.sh
# create sagemaker endpoint
!python create_endpoint.py \
--endpoint_ecr_image_path "726335585155.dkr.ecr.us-east-1.amazonaws.com/dalle" \
--endpoint_name 'dalle-mini-v3' \
--instance_type "ml.p3.2xlarge"
```

client:
```python
%%time
from boto3.session import Session
import json
data={"data": 'one austic child on the yard!',
     "bucket_name":"app-nlp-lab"} #save picture path
session = Session()
    
runtime = session.client("runtime.sagemaker")
response = runtime.invoke_endpoint(
    EndpointName='dalle-mini-v3',
    ContentType="application/json",
    Body=json.dumps(data),
)

result = json.loads(response["Body"].read())
print (result)
```

result:
```
{'result': 'success!'}
CPU times: user 57.1 ms, sys: 1.88 ms, total: 59 ms
Wall time: 2.03 s

```
