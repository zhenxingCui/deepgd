from IPython import get_ipython
from IPython.core.magic import register_line_cell_magic
from IPython.lib.backgroundjobs import BackgroundJobManager

_job_manager = BackgroundJobManager()

@register_line_cell_magic
def job(line, cell=None):
    _job_manager.new(cell or line, get_ipython().user_global_ns)

def get_job_manager():
    return _job_manager