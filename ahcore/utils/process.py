
import multiprocessing
import traceback


class Process(multiprocessing.Process):
    """
    Class which returns child Exceptions to Parent.
    Source: https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        """
        The main activity of the process, uses a Pipe to
        send exceptions to the parent process.
        """
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))

    @property
    def exception(self):
        """
        Property that contains exception information from the Process
        """
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception
