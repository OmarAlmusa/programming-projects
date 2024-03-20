import multiprocessing
import time


def do_something():
        print('sleeping for 1 second...')
        time.sleep(1)

if __name__ == '__main__':
    start = time.time()

    processes = []

    for _ in range(8):
        p = multiprocessing.Process(target=do_something)
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    finish = time.time()

    print(f'Finished in: {round(finish-start, 3)} second(s)')