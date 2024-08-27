# ------------------------------------------------------------------------------
# Supports the RELOAD endpoint
# ------------------------------------------------------------------------------
from .main import app

from ai.common.prompts import Prompts


@app.route('/reload', methods=['POST'])
def reload():
    '''
    Handle the reload endpoint. Reloads all the prompt strings
        POST - Reload prompts
        Returns:
            {}
    '''
    Prompts.loadPrompts()

    # Return our answers
    return {}
