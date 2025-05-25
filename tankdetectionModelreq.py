# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com/yolo-nas-tank-class-pu7ui/1",
    api_key="abOEUBIRgGCLrXM3F4Fq"
)

# infer on a local image
result = CLIENT.infer("C:\\Users\\suyas\\Downloads\\tank-military-equipment-top-view-drone-footage_375001-2473.jpg", model_id="yolo-nas-tank-class-pu7ui/1")