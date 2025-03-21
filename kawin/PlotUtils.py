import matplotlib.pyplot as plt

def _get_axis(ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        return ax
    else:
        return ax
    
def _adjust_kwargs(varName, defaultKwargs = {}, userKwargs = {}):
    '''
    Merges default kwargs with user input kwargs
    This will override any existing default kwarg with user kwarg and add user kwarg for any missing default kwarg
    '''
    # Search through all user kwargs (this ensures any kwarg not in defaultKwargs will be added)
    for p in userKwargs:
        # If user specifies a dict for the kwarg (based off varName, then get the variable specific kwarg)
        if isinstance(userKwargs[p], dict):
            # If the dict doesnt have the varName, then we don't add/override default kwargs
            if varName in userKwargs[p]:
                defaultKwargs[p] = userKwargs[p][varName]
        else:
            defaultKwargs[p] = userKwargs[p]
    return defaultKwargs