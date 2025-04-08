

class Party:
    # each party should have a shared set to store the ciphertext pairs and the encrypted score
    # basic socket setting
    ip_addr = "0.0.0.0"
    port = 0
    

    def __init__(self, ip_addr, port):
        self.ip_addr = ip_addr
        self.port = port
        