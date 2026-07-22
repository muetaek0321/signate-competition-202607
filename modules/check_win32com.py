import platform
from importlib.util import find_spec


def check_win32com(target: str) -> bool:
    """win32comが使用可能かどうか判定

    Args:
        target(str): 対象とするファイル種別（Excel, Word, PowerPoint）

    Return:
        bool: 判定結果
    """
    # OSがWindowsかどうか判定
    if platform.system() != "Windows":
        return False

    # pywin32がインストールされているかどうか判定
    if find_spec("win32com.client") is None:
        return False

    # 対象のOffice製品が使用可能かどうか判定
    try:
        import win32com.client

        client = win32com.client.Dispatch(f"{target}.Application")
        client.Quit()

    except Exception:
        return False

    return True
