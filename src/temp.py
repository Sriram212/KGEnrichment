from util import write_table

# sample data: list of tuples (bytes_sent, bytes_received, elapsed_time)
measurements = [
    (1024, 2048, 0.123),
    (4096, 1024, 0.456),
    (8192, 8192, 0.789),
]

write_table(measurements, "results.txt", False)