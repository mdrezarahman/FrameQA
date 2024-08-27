# ------------------------------------------------------------------------------
# Main driver for the web/REST interface
# ------------------------------------------------------------------------------
from flask import Flask, request  # type: ignore
from flask_restful import Api  # type: ignore
from flask_cors import CORS  # type: ignore
from typing import Any
from werkzeug.exceptions import HTTPException
from ai.conversation import Conversation
from ai.common.documents import DocDir

# Create the flask app
app = Flask(__name__)

# Make it CORS compatible
CORS(app)

# Create the API
api = Api(app)

# Create the conversation
conversation: Conversation = Conversation()

# Create the directory handler
documents: DocDir = DocDir(conversation.getStore())


def getBodyValue(key, *, default: Any = None, required: bool = False) -> Any:
    '''
    Gets an argument from the BODY
    '''
    if request.json and key in request.json:
        return request.json[key]
    if not required:
        return default
    raise HTTPException(f'Missing body parameter: {key}')


def getArgValue(key, *, default: Any = None, required: bool = False) -> Any:
    '''
    Gets an argument from the QUERY string
    '''
    if key in request.args:
        return request.args[key]

    if not required:
        return default
    raise HTTPException(f'Missing query parameter: {key}')
