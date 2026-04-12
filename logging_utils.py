from datetime import datetime
import os


def create_debug_logger(debug_dir):
    log_path = os.path.join(
        debug_dir,
        f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    def write_debug_log(message, print_to_console=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(log_message)
        except Exception as exc:
            print(f"Log file write error: {exc}")
        if print_to_console:
            print(message)

    return log_path, write_debug_log
