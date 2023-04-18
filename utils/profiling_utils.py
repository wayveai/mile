
import contextlib
import cProfile
import pstats

from typing import Optional


@contextlib.contextmanager
def profile(enable: bool=True, how_many_lines_to_print: int=20, print_callers_of: Optional[str]=None,
            print_callees_of: Optional[str]=None, sort_by: str='time'):
    '''
    :param enable: whether to profile or not
    :param how_many_lines_to_print:  how_many_lines_to_print
    :param print_callers_of: print callers of a function
    :param print_callees_of: print callees of a function
    :param sort_by: sort results https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats
    :param dump_stats_file_path String: path where to output dump_stats file
    :return: context manager for profiling
    '''
    if not enable:
        yield
        return

    stats_file_path = 'prof'
    profiler_instance = cProfile.Profile()
    try:
        profiler_instance.enable()
        yield
    finally:
        profiler_instance.disable()
        profiler_instance.dump_stats(stats_file_path)

    assert sort_by in ["stdname", "calls", "time", "cumulative", "cumtime"]

    p = pstats.Stats(stats_file_path)
    p.strip_dirs().sort_stats(sort_by).print_stats(how_many_lines_to_print)
    if print_callers_of is not None:
        p.print_callers(print_callers_of)
    if print_callees_of is not None:
        p.print_callees(print_callees_of)
