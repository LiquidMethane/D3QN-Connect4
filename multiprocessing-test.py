from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')

    process_list = []
    for i in range(10):
        p = Process(target=f, args=('bob',))
        p.start()
        p.join()
        process_list.append(p)

    for p in process_list:
        print(p)
