# ------------------------------------------------------------------------------
# Conversational endpoint
# ------------------------------------------------------------------------------
from ai.common.schema import DocFilter
from typing import Any
from .main import app, getBodyValue, conversation


@app.route('/conv', methods=['POST'])
def conv() -> Any:
    '''
    Handle the conversational endpoint
        POST - send instruction/receive answer
        Body:
            query=str
            history=List[str]
            filter=DocFilter (as a Dictionary)
        Return:
            ConvResult
    '''
    # Get the parameters
    query = getBodyValue('query', required=True)
    history = getBodyValue('history', default=[])
    filter = getBodyValue('filter', default={})

    # Create a new doc filter
    docFilter: DocFilter = DocFilter(filter)

    # Run it
    answers = conversation.conversation(
        query,
        docFilter=docFilter,
        history=history
    )

    # Return our answers
    return answers
