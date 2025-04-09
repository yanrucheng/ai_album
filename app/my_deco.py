import time
import functools
from typing import Callable, Any, Optional, Dict, Union
import requests
import json

def retry_api(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying API calls that fail with specific error codes.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between retries in seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if response is an error dictionary with a code
                    if isinstance(result, dict) and 'code' in result and result['code'] not in [200, None]:
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                            continue
                        return result
                    
                    return result
                
                except (requests.exceptions.RequestException, 
                       requests.exceptions.ConnectionError,
                       requests.exceptions.Timeout) as e:
                    # Network-related errors that should be retried
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return {"error": str(e), "code": 503}  # Service Unavailable
                
                except (json.JSONDecodeError, 
                       TypeError, 
                       ValueError, 
                       AssertionError) as e:
                    # Syntax/validation errors that shouldn't be retried
                    return {"error": str(e), "code": 400}  # Bad Request
                
                except Exception as e:
                    # Other unexpected errors - we'll retry these
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return {"error": str(e), "code": 500}  # Internal Server Error
            
            return {"error": f"Max retries ({max_retries}) exceeded. Last error: {str(last_exception)}",
                    "code": 429}  # Too Many Requests
        return wrapper
    return decorator

def retry_geo_api(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for retrying geocoding API calls that fail with specific error codes.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between retries in seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if response indicates an error that should be retried
                    if isinstance(result, dict):
                        # AMap error format
                        if result.get('status') == '0' and result.get('infocode') not in ['10000', '10001']:
                            if attempt < max_retries - 1:
                                time.sleep(delay * (attempt + 1))  # Exponential backoff
                                continue
                            return result
                        # LocationIQ error format
                        elif 'error' in result and not isinstance(result.get('error'), str):
                            if attempt < max_retries - 1:
                                time.sleep(delay * (attempt + 1))
                                continue
                            return result
                    
                    return result
                
                except (requests.exceptions.RequestException,
                       requests.exceptions.ConnectionError,
                       requests.exceptions.Timeout) as e:
                    # Network-related errors that should be retried
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return {"status": "0", "error": str(e), "infocode": "503"}
                
                except Exception as e:
                    # Other unexpected errors - we'll retry these
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    return {"status": "0", "error": str(e), "infocode": "500"}
            
            return {"status": "0",
                    "error": f"Max retries ({max_retries}) exceeded. Last error: {str(last_exception)}",
                    "infocode": "429"}
        return wrapper
    return decorator
