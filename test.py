# class   singleton():
#     def  __new__(cls, *args, **kwargs):
#
#         if not hasattr(cls,'_instance'):
def singleton(cls,*args,**kw):
    instances={}
    def getinstance():
        if cls not in instances:
            instances[cls]=cls(*args,**kw)
        return instances[cls]
    return getinstance





