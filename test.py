from time import ctime, sleep
def timefun_arg(pre="hello"):
    def timefun(func):
        def wrappedfunc():
            print("%s called at %s %s"%(func.__name__, ctime(), pre))
            return func()
        return wrappedfunc
    return timefun
@timefun_arg("itcast")
def foo():
    print("I am foo")
@timefun_arg("python")
def too():
    print("I am too")
foo()
sleep(2)
foo()
too()
sleep(2)
too()