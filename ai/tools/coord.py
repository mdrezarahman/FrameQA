# ------------------------------------------------------------------------------
# PyProj tool
# ------------------------------------------------------------------------------
import pyproj  # type: ignore
from aparavi import debug
from typing import Dict, List, Any
from ..common.prompts import Prompts
from ..conversation.conversation import Conversation

#
# Declare the name of this tool and what it does
#
name = 'coord'

#
# A conversion cache
#
cache: Dict[str, str] = {}

#
# Declare the invoke function
#


async def invokeTool(
        coord: List[Dict[str, Any]],
        *,
        metadata: Conversation.Metadata
) -> None:
    '''
    Uses the LLM to interpret the coordinates and convert them
    '''
    debug(f'        Source Coord      : {coord}')

    # Our target coordinate system
    targetSystem = 'epsg:4236'

    # Save the current one
    source_current = None

    # Define the source and target coordinate systems
    source_proj = None

    # Convert into abslute decimal degrees
    target_proj = pyproj.Proj(targetSystem)

    # Convert coordinates using PyProj
    for c in coord:
        # Run the documents through our question interface to get the answer
        if 'coordSystem' not in c:
            continue

        # Get the requested one
        requested = c['coordSystem']

        # If this is not currently it, reset
        if requested != source_current:
            source_current = None
            source_proj = None

        # If we don't have it mapped yet
        if requested not in cache:
            # Get the prompt
            prompt = Prompts.getPrompt('coord', {
                'system': requested
            })

            # Map it and save it in the cache
            cache[requested] = await metadata['chat'].achat(prompt)

        if source_proj is None:
            # Create the source CRS
            source_proj = pyproj.Proj(cache[requested])

            # Save the one we currently haveCreate the source CRS
            source_current = requested

        # Get the x,y positions
        x, y = c['lon'], c['lat']

        # If they are not valid
        if x is None or y is None:
            # Set to none
            lon = None
            lat = None
        else:
            # Convert them
            lon, lat = pyproj.transform(source_proj, target_proj, x, y)

        # Update the lat/on and coordinate system
        c['lon'] = lon
        c['lat'] = lat
        c['coordSystem'] = targetSystem

    # Output some debug info
    debug(f'        Converted Coordinates: {coord}')

    # Return the results - it will be a string or None
    return
