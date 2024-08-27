# ------------------------------------------------------------------------------
# Supports the LOAD endpoint
# ------------------------------------------------------------------------------
from flask import request
from ..main import app, getBodyValue, documents


@app.route('/doc/load', methods=['POST', 'DELETE'])
def load():
    '''
    Handle the load endpoint
    '''
    if request.method == 'POST':
        '''
        Handle the doc/load endpoint
            POST - Load a set of documents
            Body:
                files=List[str]
        '''
        # Handle POST request

        # Get the list of files paths
        pathList = getBodyValue('files', required=True)

        # Check it
        if not isinstance(pathList, list):
            raise TypeError('Invalid pathList')

        # Add the files
        documents.addDocuments(pathList)

        # Done - return the status
        return documents.getStatus()

    elif request.method == 'DELETE':
        '''
        Handle the doc/load endpoint
            DELETE - Stop a load operation
        '''
        # Cancel any pending uploads
        documents.cancelUpload()

        # Return the status
        return documents.getStatus()
