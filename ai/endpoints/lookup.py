# ------------------------------------------------------------------------------
# Lookup end point
# ------------------------------------------------------------------------------
from ai.common.schema import DocFilter
from typing import Any
from .main import app, getArgValue, conversation


@app.route('/lookup', methods=['GET'])
def lookup() -> Any:
    '''
    Handle the lookup endpoint
        GET - send instruction/receive answer
        Query:
            query=str
        Return:
            answer - json
    '''
    # Get the parameters
    query = getArgValue('query', required=True)
    filter = getArgValue('filter', default={})

    # Create a new doc filter
    docFilter: DocFilter = DocFilter(filter)

    # Run it
    answer = conversation.getLookup()(
        query,
        docFilter=docFilter
    )

    # Return our answers
    return answer
