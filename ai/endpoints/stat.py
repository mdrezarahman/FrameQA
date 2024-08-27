# ------------------------------------------------------------------------------
# Supports the STAT endpoint
# ------------------------------------------------------------------------------
from .main import app, documents


@app.route('/stat', methods=['GET'])
def stat():
    '''
    Handle the stat endpoint
        GET - Gets the LoadStatus
        Returns:
            LoadStatus
    '''
    # Return the status
    return documents.getStatus()
