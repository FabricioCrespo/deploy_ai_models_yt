"""
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
"""
#import time

import requests
import json



"""
credentials = CognitiveServicesCredentials(key)

client = ComputerVisionClient(
    endpoint=endpoint,
    credentials=credentials
)
"
"""

def ocr_space_file(filename, overlay=False, api_key='K88542578688957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    
    """
    api_key = "K88542578688957"
    url = "https://api.ocr.space/parse/image"

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post(url,
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

def read_image(filename):
    

    test_file = ocr_space_file(filename=filename, language='eng')

    # Convert the string to a dictionary
    json_data = json.loads(test_file)

    # Extract ParsedText
    parsed_text = json_data["ParsedResults"][0]["ParsedText"]

    return parsed_text