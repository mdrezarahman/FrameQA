'''
Processes request for document dir
'''
from flask import request
from ..main import app, documents, getArgValue


@app.route('/doc/documents', methods=['GET', 'DELETE'])
def doc():
    if request.method == 'GET':
        '''
        Handle the /doc/documents endpoint
            GET - Gets a directory
            Returns:
                { "name": isDir, ...}
        '''
        # Get the path argument
        path = getArgValue('path')

        # Get the children
        children = documents.getChildren(path=path)

        # Get the list of children
        return children

    elif request.method == 'DELETE':
        '''
        Handle the remove document endpoint
            DELETE - Remove documents
        '''
        # Get the path argument
        pathList = getArgValue('path', required=True)

        # Check it
        if isinstance(pathList, list):
            documents.removeDocuments(pathList)
        else:
            documents.removeDocument(pathList)

        # Nothing to return
        return {}
