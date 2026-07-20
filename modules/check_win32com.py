import platform
from importlib.util import find_spec


def check_excel_com() -> bool:

    if platform.system() != "Windows":
        return False

    if find_spec("win32com.client") is None:
        return False

    try:
        import win32com.client

        excel = win32com.client.Dispatch("Excel.Application")
        excel.Quit()

    except Exception:
        return False

    return True