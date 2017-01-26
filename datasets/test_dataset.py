import dataset

def test():
    dset = dataset.Dataset("YCB_Dataset")
    train_iterator = dset.train_iterator(40,True)
    for i in range(10):
        print i
        x,y = train_iterator.next()

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test", number=10))
