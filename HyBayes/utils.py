import inspect
import os
import time


def mk_dir_if_not_exists(folder_address):
    chain = folder_address.split("/")
    folder_address = ""
    changed = False
    for newPart in chain:
        folder_address = os.path.join(folder_address, newPart)
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)
            changed = True
    return changed


last_time = 0


def printLine(depth=1):
    """
    A debugging tool!
    Prints the stack trace up to 'depth' number of calls in the stack with line number, file and function names,
     as well as the time from last printLine call.
    """
    global last_time
    return
    new_time = time.time()
    for i in range(1, depth + 1):
        info = inspect.stack()[i]
        for j in range(i - 1):
            print("\t", end="")
        print(f"Line {info.lineno} in {info.filename}, Function: {info.function}, time:{new_time - last_time}")
    last_time = new_time
