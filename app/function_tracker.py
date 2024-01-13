import random
import string
import inspect
from functools import wraps
from collections import deque
import time

def generate_random_id(length=4):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def get_package_name(func):
    return inspect.getmodule(func).__name__

def format_arguments(args, kwargs, limit=60):
    def format_arg(arg):
        if isinstance(arg, (bool, int, float, str, list, dict)):
            return str(arg)
        return 'obj'

    args_repr = ', '.join(map(format_arg, args))
    kwargs_repr = ', '.join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
    all_args = ', '.join(filter(None, [args_repr, kwargs_repr]))

    return (all_args[:limit - 3] + "...") if len(all_args) > limit else all_args

class FunctionTracker:
    def __init__(self):
        self.call_log = deque()
        self.enabled = False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            rand_id = generate_random_id()
            package_name = get_package_name(func)
            formatted_args = format_arguments(args, kwargs)

            func_identifier = f"[{rand_id}]{package_name}.{func.__name__} ({formatted_args})"

            start_time = time.time()
            self.call_log.append((func_identifier, 'start', start_time))

            result = func(*args, **kwargs)

            end_time = time.time()
            self.call_log.append((func_identifier, 'end', end_time))

            return result
        return wrapper

    def enable(self):
        self.enabled = True

    def report(self, depth_threshold=3):
        stack = []
        res = []

        for func_name, action, timestamp in self.call_log:

            if action == 'start':
                stack.append((func_name, timestamp))
            elif action == 'end' and stack:
                start_func, start_time = stack.pop()
                if start_func == func_name:
                    elapsed_time = timestamp - start_time
                    call_level = len(stack)

                    if call_level <= depth_threshold:
                        indent = '    ' * call_level + "|-> " if stack else ""
                        res.append(f"{indent}{func_name}: takes {elapsed_time:.2f} sec")
                    elif call_level == depth_threshold + 1:
                        # Add an ellipsis to indicate omitted deeper calls
                        indent = '    ' * call_level + "|-> "
                        res.append(f"{indent}... (deeper calls omitted)")

            if len(stack) == 0:
                print('\n'.join(res[::-1]))
                print()

    def clear(self):
        self.call_log = deque()


global_tracker = FunctionTracker()

if __name__ == '__main__':
    # Initialize the tracker

    # Define and decorate your test functions with @tracker
    @global_tracker
    def function_a():
        time.sleep(0.1)

    @global_tracker
    def function_b():
        function_a()

    @global_tracker
    def function_c():
        function_b()

    # Run your test functions
    function_c()


    @global_tracker
    def deep_function(level):
        if level > 0:
            time.sleep(0.1)
            deep_function(level - 1)

    @global_tracker
    def deep_nested_call():
        deep_function(3)

    # Run the test
    deep_nested_call()


    @global_tracker
    def independent_function_1():
        time.sleep(0.2)

    @global_tracker
    def independent_function_2():
        time.sleep(0.3)

    # Run the test
    independent_function_1()
    independent_function_2()


    @global_tracker
    def recursive_function(n):
        if n <= 0:
            return
        time.sleep(0.1)
        recursive_function(n - 1)

    # Run the test
    recursive_function(6)


    @global_tracker
    def mixed_calls():
        independent_function_1()
        deep_nested_call()
        recursive_function(2)

    # Run the test
    mixed_calls()

    # Print the report
    global_tracker.report()
