import sysv_ipc
import posix_ipc
import ctypes


class EyeFinderReader:
    num_longs = 2 * 30 * ctypes.sizeof(ctypes.c_long)
    sizeoflong = ctypes.sizeof(ctypes.c_long)

    def __enter__(self):
        # Connect to existing shared memory
        self.memory = sysv_ipc.SharedMemory(123456)
        # Connect to existing semaphore
        self.sem = posix_ipc.Semaphore("/capstone")
        return self

    def bytes_to_longs(self, bstr):
        tmp = [int.from_bytes(bstr[i * EyeFinderReader.sizeoflong:(i + 1) * EyeFinderReader.sizeoflong], byteorder='little')
               for i in range(int(len(bstr) / EyeFinderReader.sizeoflong))]
        return tmp

    def read(self):
        self.sem.acquire()
        memory_value = self.memory.read()
        self.sem.release()
        facial_features_list = self.bytes_to_longs(memory_value)
        return facial_features_list

    def clean(self):
        self.sem.unlink()
        self.memory.remove()

    def __exit__(self, exc_type, exc_value, traceback):
        # Clean up Semaphore
        self.sem.close()
        # Clean up shared memory
        try:
            self.memory.detach()
        except:
            pass

