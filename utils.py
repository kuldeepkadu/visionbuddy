import time

def getDateTime():
    return time.strftime("%d%b%Y %H%M%S",time.gmtime())
