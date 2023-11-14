class Service:
    def __init__(
        self,
        name: str,
        workload: int,
        arrival_rate: float,
        input_size: int,
        output_size: int,
        max_delay: float,
    ):
        """Class representing a service. A service k can be instanced in each
        Edge Server, and a different instance may be added for each base
        station redirecting traffic to the edge server.
        Args:
            name (str): Name of the service
            workload (int): Number of ops necessary to complete the request
            arrival_rate (float): Number of req per sec caused by this service
            input_size (int): Bytes of data per request made to this service
            output_size (int): Bytes of data per request returned by this service
            max_delay (float): Maximum delay that the service can tolerate
        """
        self.name = name
        self.workload = workload
        self.arrival_rate = arrival_rate
        self.input_size = input_size * 8  # bits
        self.output_size = output_size * 8  # bits
        self.max_delay = max_delay  # seconds

    def __repr__(self):
        return self.name
