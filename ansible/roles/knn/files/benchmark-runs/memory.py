import os

MEMORY_RESERVE_BYTES = 8 * 1024 * 1024 * 1024
MEMORY_RESERVE_RATIO = 0.10
MEMORY_ALLOWED_EXECUTION_BUFFER = 4 * 1024 * 1024 * 1024

def memory_info() -> tuple[int, int]:
    total = 0
    available = 0
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                key, value = line.split(":", maxsplit=1)
                if key == "MemTotal":
                    total = int(value.strip().split()[0]) * 1024
                elif key == "MemAvailable":
                    available = int(value.strip().split()[0]) * 1024
                if total and available:
                    break
    except FileNotFoundError:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available = total
    return total, available


def memory_reserve(total: int) -> int:
    return max(MEMORY_RESERVE_BYTES, int(total * MEMORY_RESERVE_RATIO))


def usable_memory(total: int) -> int:
    return max(0, total - memory_reserve(total))


def maximum_memory_reservation() -> int:
    total, _ = memory_info()
    # yes, this is -2xReserve
    return usable_memory(total) - memory_reserve(total)
